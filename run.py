#!/usr/bin/env python3
"""
Runner script for medical imaging SageMaker pipelines.

Usage:
    python run.py setup-role                       # Create/verify IAM role
    python run.py status                           # Check all endpoint status

    # Segmentation
    python run.py train-segmentation               # Launch segmentation training job
    python run.py deploy-segmentation              # Deploy segmentation model
    python run.py invoke-segmentation <s3_uri>     # Invoke segmentation endpoint
    python run.py cleanup-segmentation             # Delete segmentation endpoint

    # Classification
    python run.py train-classification             # Launch classification training job
    python run.py deploy-classification            # Deploy classification model
    python run.py invoke-classification <s3_uri>   # Invoke classification endpoint
    python run.py cleanup-classification           # Delete classification endpoint

    python run.py --config config.local.yaml ...   # Use custom config file

All parameters are read from config.yaml (or config.local.yaml).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
import yaml


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file with environment variable overrides."""
    if config_path is None:
        # Prefer config_pvt.yaml, then config.local.yaml, fall back to config.yaml
        root = Path(__file__).parent
        pvt_config = root / "config_pvt.yaml"
        local_config = root / "config.local.yaml"
        default_config = root / "config.yaml"

        if pvt_config.exists():
            config_path = str(pvt_config)
        elif local_config.exists():
            config_path = str(local_config)
        else:
            config_path = str(default_config)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    env_overrides = {
        "aws.region": "AWS_DEFAULT_REGION",
        "aws.account_id": "AWS_ACCOUNT_ID",
        "aws.sagemaker_role": "AWS_SAGEMAKER_ROLE",
        "aws.s3_bucket": "AWS_TEST_BUCKET",
        "segmentation.data_s3": "SEGMENTATION_DATA_S3",
        "segmentation.model_s3": "SEGMENTATION_MODEL_S3",
        "segmentation.test_nifti_s3": "TEST_NIFTI_S3",
        "classification.data_s3": "CLASSIFICATION_DATA_S3",
        "classification.model_s3": "CLASSIFICATION_MODEL_S3",
        "classification.test_image_s3": "CLASSIFICATION_TEST_IMAGE_S3",
        "tracking.wandb_api_key": "WANDB_API_KEY",
    }

    for key_path, env_var in env_overrides.items():
        value = os.environ.get(env_var)
        if value:
            keys = key_path.split(".")
            obj = config
            for k in keys[:-1]:
                obj = obj[k]
            obj[keys[-1]] = value

    return config


def get_role(config: dict) -> str:
    """Get or create the SageMaker execution role."""
    role = config["aws"].get("sagemaker_role", "")
    if role:
        # If it's a full ARN, use as-is; otherwise resolve it
        if role.startswith("arn:aws:iam::"):
            return role
        # It's a role name, resolve to ARN
        iam = boto3.client("iam")
        response = iam.get_role(RoleName=role)
        return response["Role"]["Arn"]

    # Auto-create using utility
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import get_or_create_role
    return get_or_create_role()


def get_session(config: dict):
    """Create a SageMaker session."""
    import sagemaker
    region = config["aws"]["region"]
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def get_bucket(config: dict, session) -> str:
    """Get S3 bucket from config or SageMaker default."""
    bucket = config["aws"].get("s3_bucket", "")
    return bucket if bucket else session.default_bucket()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup_role(config: dict):
    """Create or verify the SageMaker execution role."""
    role = get_role(config)
    print(f"\nRole ARN: {role}")


def cmd_status(config: dict):
    """Check status of all endpoints."""
    region = config["aws"]["region"]
    sm_client = boto3.client("sagemaker", region_name=region)

    endpoints_to_check = [
        config["segmentation"]["deployment"]["endpoint_name"],
        config["classification"]["deployment"]["endpoint_name"],
    ]

    for ep_name in endpoints_to_check:
        try:
            desc = sm_client.describe_endpoint(EndpointName=ep_name)
            status = desc["EndpointStatus"]
            instances = desc.get("ProductionVariants", [{}])[0].get(
                "CurrentInstanceCount", "N/A"
            )
            print(f"  {ep_name}: {status} ({instances} instances)")
        except sm_client.exceptions.ClientError:
            print(f"  {ep_name}: NOT FOUND")


def cmd_train_segmentation(config: dict):
    """Launch a SageMaker segmentation training job."""
    from sagemaker.pytorch.estimator import PyTorch

    session = get_session(config)
    role = get_role(config)
    bucket = get_bucket(config, session)

    seg_config = config["segmentation"]
    train_config = seg_config["training"]
    tracking = config["tracking"]

    hyperparameters = {
        "model_name": train_config["model_name"],
        "batch_size": train_config["batch_size"],
        "epochs": train_config["epochs"],
        "lr": train_config["learning_rate"],
    }

    # Add tracking params
    if tracking["use_mlflow"]:
        hyperparameters["use_mlflow"] = True
        hyperparameters["mlflow_tracking_uri"] = tracking["mlflow_tracking_uri"]
        hyperparameters["mlflow_experiment_name"] = tracking["mlflow_experiment_name"]

    if tracking["use_wandb"]:
        hyperparameters["use_wandb"] = True
        hyperparameters["wandb_project"] = tracking["wandb_project"]
        hyperparameters["wandb_api_key"] = tracking["wandb_api_key"]

    data_path = seg_config["data_s3"]
    output_path = f"s3://{bucket}/segmentation_data/output"

    source_dir = str(
        Path(__file__).parent / "medical-image-segmentation" / "code" / "training"
    )

    estimator = PyTorch(
        entry_point="train_fsdp_all.py",
        source_dir=source_dir,
        role=role,
        instance_count=train_config["instance_count"],
        instance_type=train_config["instance_type"],
        framework_version=train_config["framework_version"],
        py_version=train_config["py_version"],
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name="medical-seg-train",
        keep_alive_period_in_seconds=1800,
        distribution={"pytorchddp": {"enabled": True}},
        sagemaker_session=session,
    )

    print(f"Launching training job...")
    print(f"  Model: {train_config['model_name']}")
    print(f"  Instance: {train_config['instance_type']} x {train_config['instance_count']}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_path}")

    estimator.fit({"training": data_path}, wait=False)
    job_name = estimator.latest_training_job.name
    print(f"\nTraining job started: {job_name}")
    print(f"Monitor at: https://{config['aws']['region']}.console.aws.amazon.com/sagemaker/home?region={config['aws']['region']}#/jobs/{job_name}")


def cmd_deploy_segmentation(config: dict):
    """Deploy segmentation model to an async SageMaker endpoint."""
    from sagemaker.pytorch import PyTorchModel
    from sagemaker.async_inference import AsyncInferenceConfig

    session = get_session(config)
    role = get_role(config)
    bucket = get_bucket(config, session)
    region = config["aws"]["region"]

    seg_config = config["segmentation"]
    deploy_config = seg_config["deployment"]

    model_data = seg_config["model_s3"]
    if not model_data:
        print("ERROR: segmentation.model_s3 is not set in config.")
        print("Run a training job first, then set the model artifact path.")
        sys.exit(1)

    endpoint_name = deploy_config["endpoint_name"]

    # Check if already exists
    sm_client = boto3.client("sagemaker", region_name=region)
    existing = sm_client.list_endpoints()["Endpoints"]
    if any(ep["EndpointName"] == endpoint_name for ep in existing):
        print(f"Endpoint '{endpoint_name}' already exists.")
        return

    # Resolve image URI
    import sagemaker
    image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=seg_config["training"]["framework_version"],
        py_version=seg_config["training"]["py_version"],
        instance_type=deploy_config["instance_type"],
        image_scope="inference",
    )

    async_config = AsyncInferenceConfig(
        output_path=f"s3://{bucket}/segmentation-inference/output",
        failure_path=f"s3://{bucket}/segmentation-inference/failures",
        max_concurrent_invocations_per_instance=deploy_config["max_concurrent_invocations"],
    )

    source_dir = str(
        Path(__file__).parent / "medical-image-segmentation" / "notebooks" / "deploy"
    )

    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        source_dir=source_dir,
        entry_point="inference.py",
        framework_version=seg_config["training"]["framework_version"],
        py_version=seg_config["training"]["py_version"],
        image_uri=image_uri,
        sagemaker_session=session,
        env={
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "300",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        },
    )

    print(f"Deploying endpoint: {endpoint_name}")
    print(f"  Model: {model_data}")
    print(f"  Instance: {deploy_config['instance_type']}")
    print(f"  Image: {image_uri}")

    pytorch_model.deploy(
        instance_type=deploy_config["instance_type"],
        initial_instance_count=deploy_config["initial_instance_count"],
        endpoint_name=endpoint_name,
        async_inference_config=async_config,
    )
    print(f"Endpoint '{endpoint_name}' deployed.")

    # Configure auto-scaling
    _setup_autoscaling(config, endpoint_name, deploy_config)


def _setup_autoscaling(config: dict, endpoint_name: str, deploy_config: dict):
    """Configure auto-scaling for the endpoint."""
    aas_client = boto3.client(
        "application-autoscaling", region_name=config["aws"]["region"]
    )
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=deploy_config["min_instance_count"],
        MaxCapacity=deploy_config["max_instance_count"],
    )

    aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": deploy_config["scale_target_backlog"],
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": deploy_config["scale_in_cooldown"],
            "ScaleOutCooldown": deploy_config["scale_out_cooldown"],
        },
    )
    print(f"Auto-scaling configured: min={deploy_config['min_instance_count']}, max={deploy_config['max_instance_count']}")


def cmd_invoke_segmentation(config: dict, s3_uri: str):
    """Invoke the segmentation endpoint with a NIfTI file."""
    region = config["aws"]["region"]
    session = get_session(config)
    bucket = get_bucket(config, session)
    endpoint_name = config["segmentation"]["deployment"]["endpoint_name"]

    runtime_client = boto3.client("sagemaker-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    # Upload request payload
    payload = json.dumps({"file_path": s3_uri})
    input_key = f"segmentation-inference/input/request_{int(time.time())}.json"
    s3_client.put_object(Bucket=bucket, Key=input_key, Body=payload)
    input_location = f"s3://{bucket}/{input_key}"

    print(f"Invoking endpoint: {endpoint_name}")
    print(f"  Input: {s3_uri}")

    response = runtime_client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        ContentType="application/json",
    )
    output_location = response["OutputLocation"]
    print(f"  Output location: {output_location}")

    # Poll for result
    parts = output_location.replace("s3://", "").split("/", 1)
    out_bucket, out_key = parts[0], parts[1]

    print("Waiting for result...")
    start_time = time.time()
    timeout = 600

    while time.time() - start_time < timeout:
        try:
            obj = s3_client.get_object(Bucket=out_bucket, Key=out_key)
            result = json.loads(obj["Body"].read().decode("utf-8"))
            elapsed = int(time.time() - start_time)
            print(f"\nResult received in {elapsed}s:")
            print(f"  Model: {result.get('model_name', 'N/A')}")
            print(f"  Original shape: {result.get('original_shape', 'N/A')}")
            print(f"  Segmented voxels: {result.get('segmented_voxels', 'N/A'):,}")
            print(f"  Segmentation coverage: {result.get('segmentation_percentage', 'N/A')}%")

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            return result
        except s3_client.exceptions.NoSuchKey:
            elapsed = int(time.time() - start_time)
            print(f"  Waiting... ({elapsed}s)", end="\r")
            time.sleep(10)

    print(f"\nTimeout after {timeout}s. Check CloudWatch logs.")
    sys.exit(1)


def cmd_cleanup_segmentation(config: dict):
    """Delete the segmentation endpoint and all associated resources."""
    region = config["aws"]["region"]
    endpoint_name = config["segmentation"]["deployment"]["endpoint_name"]
    _cleanup_endpoint(config, endpoint_name)


# ---------------------------------------------------------------------------
# Classification Commands
# ---------------------------------------------------------------------------

def cmd_train_classification(config: dict):
    """Launch a SageMaker classification training job."""
    from sagemaker.pytorch.estimator import PyTorch
    from sagemaker.inputs import TrainingInput

    session = get_session(config)
    role = get_role(config)
    bucket = get_bucket(config, session)

    cls_config = config["classification"]
    train_config = cls_config["training"]

    data_s3 = cls_config["data_s3"]
    if not data_s3:
        print("ERROR: classification.data_s3 is not set in config.")
        print("Set it to an S3 path containing train/ and valid/ subdirectories.")
        sys.exit(1)

    hyperparameters = {
        "batch_size": train_config["batch_size"],
        "epochs": train_config["epochs"],
        "learning_rate": train_config["learning_rate"],
        "model_name": train_config["model_name"],
        "num_classes": train_config["num_classes"],
        "val_interval": 1,
    }

    output_path = f"s3://{bucket}/classification/output"

    source_dir = str(
        Path(__file__).parent
        / "medical-image-classification"
        / "notebooks"
        / "02_sm_script_mode"
        / "code"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=source_dir,
        role=role,
        instance_count=train_config["instance_count"],
        instance_type=train_config["instance_type"],
        framework_version=train_config["framework_version"],
        py_version=train_config["py_version"],
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name="medical-cls-train",
        keep_alive_period_in_seconds=1800,
        sagemaker_session=session,
    )

    # Classification uses separate train/test channels
    data_s3 = data_s3.rstrip("/")
    train_input = TrainingInput(s3_data=f"{data_s3}/train")
    test_input = TrainingInput(s3_data=f"{data_s3}/valid")

    print(f"Launching classification training job...")
    print(f"  Model: {train_config['model_name']}")
    print(f"  Classes: {train_config['num_classes']}")
    print(f"  Instance: {train_config['instance_type']} x {train_config['instance_count']}")
    print(f"  Data: {data_s3}")
    print(f"  Output: {output_path}")

    estimator.fit({"train": train_input, "test": test_input}, wait=False)
    job_name = estimator.latest_training_job.name
    print(f"\nTraining job started: {job_name}")
    print(
        f"Monitor at: https://{config['aws']['region']}.console.aws.amazon.com"
        f"/sagemaker/home?region={config['aws']['region']}#/jobs/{job_name}"
    )


def cmd_deploy_classification(config: dict):
    """Deploy classification model to an async SageMaker endpoint."""
    from sagemaker.pytorch import PyTorchModel
    from sagemaker.async_inference import AsyncInferenceConfig

    session = get_session(config)
    role = get_role(config)
    bucket = get_bucket(config, session)
    region = config["aws"]["region"]

    cls_config = config["classification"]
    deploy_config = cls_config["deployment"]

    model_data = cls_config["model_s3"]
    if not model_data:
        print("ERROR: classification.model_s3 is not set in config.")
        print("Run a training job first, then set the model artifact path.")
        sys.exit(1)

    endpoint_name = deploy_config["endpoint_name"]

    # Check if already exists
    sm_client = boto3.client("sagemaker", region_name=region)
    existing = sm_client.list_endpoints()["Endpoints"]
    if any(ep["EndpointName"] == endpoint_name for ep in existing):
        print(f"Endpoint '{endpoint_name}' already exists.")
        return

    # Resolve image URI
    import sagemaker

    image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=cls_config["training"]["framework_version"],
        py_version=cls_config["training"]["py_version"],
        instance_type=deploy_config["instance_type"],
        image_scope="inference",
    )

    async_config = AsyncInferenceConfig(
        output_path=f"s3://{bucket}/classification-inference/output",
        failure_path=f"s3://{bucket}/classification-inference/failures",
        max_concurrent_invocations_per_instance=deploy_config["max_concurrent_invocations"],
    )

    source_dir = str(
        Path(__file__).parent
        / "medical-image-classification"
        / "notebooks"
        / "05_sagemaker_deployment"
    )

    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        source_dir=source_dir,
        entry_point="inference.py",
        framework_version=cls_config["training"]["framework_version"],
        py_version=cls_config["training"]["py_version"],
        image_uri=image_uri,
        sagemaker_session=session,
    )

    print(f"Deploying endpoint: {endpoint_name}")
    print(f"  Model: {model_data}")
    print(f"  Instance: {deploy_config['instance_type']}")
    print(f"  Image: {image_uri}")

    pytorch_model.deploy(
        instance_type=deploy_config["instance_type"],
        initial_instance_count=deploy_config["initial_instance_count"],
        endpoint_name=endpoint_name,
        async_inference_config=async_config,
    )
    print(f"Endpoint '{endpoint_name}' deployed.")

    # Configure auto-scaling
    _setup_autoscaling(config, endpoint_name, deploy_config)


def cmd_invoke_classification(config: dict, s3_uri: str):
    """Invoke the classification endpoint with a DICOM/image file."""
    region = config["aws"]["region"]
    session = get_session(config)
    bucket = get_bucket(config, session)
    endpoint_name = config["classification"]["deployment"]["endpoint_name"]

    runtime_client = boto3.client("sagemaker-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    # Upload request payload
    payload = json.dumps({"file_path": s3_uri})
    input_key = f"classification-inference/input/request_{int(time.time())}.json"
    s3_client.put_object(Bucket=bucket, Key=input_key, Body=payload)
    input_location = f"s3://{bucket}/{input_key}"

    print(f"Invoking endpoint: {endpoint_name}")
    print(f"  Input: {s3_uri}")

    response = runtime_client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        ContentType="application/json",
    )
    output_location = response["OutputLocation"]
    print(f"  Output location: {output_location}")

    # Poll for result
    parts = output_location.replace("s3://", "").split("/", 1)
    out_bucket, out_key = parts[0], parts[1]

    labels = [
        "Surgical_implant",
        "Vertebral_collapse",
        "Spondylolysthesis",
        "No_finding",
        "Foraminal_stenosis",
        "Other_lesions",
        "Disc_space_narrowing",
        "Osteophytes",
    ]

    print("Waiting for result...")
    start_time = time.time()
    timeout = 300

    while time.time() - start_time < timeout:
        try:
            obj = s3_client.get_object(Bucket=out_bucket, Key=out_key)
            result = json.loads(obj["Body"].read().decode("utf-8"))
            elapsed = int(time.time() - start_time)

            if "error" in result:
                print(f"\nERROR after {elapsed}s: {result['error']}")
                return result

            predictions = result["predictions"][0]
            predicted_idx = result["predicted_class"][0]
            confidence = result["confidence"][0]

            print(f"\nResult received in {elapsed}s:")
            print(f"  Predicted class: {labels[predicted_idx]}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  All probabilities:")
            for label, prob in zip(labels, predictions):
                bar = "█" * int(prob * 30)
                print(f"    {label:25s} {prob:.4f} {bar}")
            return result
        except s3_client.exceptions.NoSuchKey:
            elapsed = int(time.time() - start_time)
            print(f"  Waiting... ({elapsed}s)", end="\r")
            time.sleep(5)

    print(f"\nTimeout after {timeout}s. Check CloudWatch logs.")
    sys.exit(1)


def cmd_cleanup_classification(config: dict):
    """Delete the classification endpoint and all associated resources."""
    region = config["aws"]["region"]
    endpoint_name = config["classification"]["deployment"]["endpoint_name"]
    _cleanup_endpoint(config, endpoint_name)


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _cleanup_endpoint(config: dict, endpoint_name: str):
    """Delete an endpoint and all associated resources."""
    region = config["aws"]["region"]
    sm_client = boto3.client("sagemaker", region_name=region)
    aas_client = boto3.client("application-autoscaling", region_name=region)

    # Deregister auto-scaling
    try:
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
        aas_client.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        print(f"Deregistered auto-scaling: {endpoint_name}")
    except Exception as e:
        print(f"Auto-scaling cleanup: {e}")

    # Delete endpoint, config, and model
    try:
        desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = desc["EndpointConfigName"]
        config_desc = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        model_name = config_desc["ProductionVariants"][0]["ModelName"]

        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")

        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Deleted endpoint config: {endpoint_config_name}")

        sm_client.delete_model(ModelName=model_name)
        print(f"Deleted model: {model_name}")

    except sm_client.exceptions.ClientError as e:
        print(f"Cleanup error: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run medical imaging SageMaker pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config YAML file (default: config.local.yaml or config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("setup-role", help="Create/verify IAM role")
    subparsers.add_parser("status", help="Check endpoint status")

    # Segmentation commands
    subparsers.add_parser("train-segmentation", help="Launch segmentation training job")
    subparsers.add_parser("deploy-segmentation", help="Deploy segmentation model")

    invoke_seg_parser = subparsers.add_parser(
        "invoke-segmentation", help="Invoke segmentation endpoint"
    )
    invoke_seg_parser.add_argument("s3_uri", help="S3 URI to NIfTI file")

    subparsers.add_parser("cleanup-segmentation", help="Delete segmentation endpoint")

    # Classification commands
    subparsers.add_parser("train-classification", help="Launch classification training job")
    subparsers.add_parser("deploy-classification", help="Deploy classification model")

    invoke_cls_parser = subparsers.add_parser(
        "invoke-classification", help="Invoke classification endpoint"
    )
    invoke_cls_parser.add_argument("s3_uri", help="S3 URI to DICOM/image file")

    subparsers.add_parser("cleanup-classification", help="Delete classification endpoint")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)

    commands = {
        "setup-role": lambda: cmd_setup_role(config),
        "status": lambda: cmd_status(config),
        # Segmentation
        "train-segmentation": lambda: cmd_train_segmentation(config),
        "deploy-segmentation": lambda: cmd_deploy_segmentation(config),
        "invoke-segmentation": lambda: cmd_invoke_segmentation(config, args.s3_uri),
        "cleanup-segmentation": lambda: cmd_cleanup_segmentation(config),
        # Classification
        "train-classification": lambda: cmd_train_classification(config),
        "deploy-classification": lambda: cmd_deploy_classification(config),
        "invoke-classification": lambda: cmd_invoke_classification(config, args.s3_uri),
        "cleanup-classification": lambda: cmd_cleanup_classification(config),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
