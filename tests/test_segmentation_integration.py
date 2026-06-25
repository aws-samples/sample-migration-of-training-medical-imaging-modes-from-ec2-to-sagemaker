"""
Integration tests for medical-image-segmentation notebooks.

These tests execute segmentation training and deployment against real AWS services.

Run with: pytest -m "integration and segmentation" --timeout=3600

Environment variables:
    AWS_TEST_REGION: AWS region (default: us-east-1)
    AWS_SAGEMAKER_ROLE: SageMaker execution role ARN or name
    AWS_TEST_BUCKET: S3 bucket for test data (default: SageMaker default bucket)
    SEGMENTATION_DATA_S3: S3 URI to segmentation dataset
        (e.g., s3://bucket/segmentation_data)
    SEGMENTATION_MODEL_S3: S3 URI to a trained model.tar.gz for deployment tests
    WANDB_API_KEY: Weights & Biases API key (for lab3 tests)
"""

import os
import time
import json
import pytest
import boto3
import sagemaker
from pathlib import Path

from helpers import (
    NOTEBOOK_PATHS,
    SEGMENTATION_NOTEBOOKS,
    load_notebook,
    get_code_cells,
    execute_notebook,
    wait_for_training_job,
    wait_for_endpoint,
)


# ============================================================================
# Configuration
# ============================================================================

SEGMENTATION_DATA_S3 = os.environ.get("SEGMENTATION_DATA_S3", "")
SEGMENTATION_MODEL_S3 = os.environ.get("SEGMENTATION_MODEL_S3", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")


def requires_segmentation_data(func):
    """Skip test if segmentation data S3 path is not configured."""
    return pytest.mark.skipif(
        not SEGMENTATION_DATA_S3,
        reason="SEGMENTATION_DATA_S3 environment variable not set"
    )(func)


def requires_segmentation_model(func):
    """Skip if trained model path is not configured."""
    return pytest.mark.skipif(
        not SEGMENTATION_MODEL_S3,
        reason="SEGMENTATION_MODEL_S3 environment variable not set"
    )(func)


# ============================================================================
# Test: Lab 1 - Single GPU Training
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.training
@pytest.mark.expensive
class TestSingleGPUTraining:
    """Integration tests for Lab 1: Single GPU segmentation training."""

    @requires_segmentation_data
    def test_segresnet_training_completes(
        self,
        sagemaker_session,
        execution_role,
        aws_region,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Train SegResNet on single GPU with minimal epochs."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-seg-single-gpu-{int(time.time())}"

        # Use the managed PyTorch container
        estimator = PyTorch(
            entry_point="train_simple.py",
            source_dir=str(
                Path(__file__).parent.parent
                / "medical-image-segmentation"
                / "code"
                / "training"
            ),
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "model_name": "SegResNet",
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
            },
            keep_alive_period_in_seconds=0,
        )

        train_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "valid": valid_input},
            job_name=job_name,
            wait=False,
        )
        cleanup_training_jobs.append(job_name)

        status = wait_for_training_job(sagemaker_client, job_name, timeout=1800)
        assert status == "Completed", f"Training failed with status: {status}"

        # Verify model output
        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        assert "S3ModelArtifacts" in desc["ModelArtifacts"]

    @requires_segmentation_data
    def test_training_job_logs_metrics(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Verify training job produces expected log output."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-seg-metrics-{int(time.time())}"

        estimator = PyTorch(
            entry_point="train_simple.py",
            source_dir=str(
                Path(__file__).parent.parent
                / "medical-image-segmentation"
                / "code"
                / "training"
            ),
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "model_name": "SegResNet",
            },
            metric_definitions=[
                {"Name": "train:loss", "Regex": "train_loss=([0-9\\.]+)"},
                {"Name": "valid:dice", "Regex": "valid_dice=([0-9\\.]+)"},
            ],
        )

        train_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "valid": valid_input},
            job_name=job_name,
            wait=True,
        )
        cleanup_training_jobs.append(job_name)

        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        assert desc["TrainingJobStatus"] == "Completed"


# ============================================================================
# Test: Lab 2 - FSDP Multi-GPU Training
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.training
@pytest.mark.expensive
class TestFSDPMultiGPU:
    """Integration tests for Lab 2: FSDP multi-GPU training."""

    @requires_segmentation_data
    def test_fsdp_training_completes(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Train with FSDP on multi-GPU instance."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-seg-fsdp-{int(time.time())}"

        estimator = PyTorch(
            entry_point="train_fsdp.py",
            source_dir=str(
                Path(__file__).parent.parent
                / "medical-image-segmentation"
                / "code"
                / "training"
            ),
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.12xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            distribution={"pytorchddp": {"enabled": True}},
            hyperparameters={
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "model_name": "SegResNet",
            },
        )

        train_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "valid": valid_input},
            job_name=job_name,
            wait=False,
        )
        cleanup_training_jobs.append(job_name)

        status = wait_for_training_job(sagemaker_client, job_name, timeout=2400)
        assert status == "Completed", f"FSDP training failed: {status}"


# ============================================================================
# Test: Lab 3 - Weights & Biases Tracking
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.training
@pytest.mark.expensive
class TestWandbTracking:
    """Integration tests for Lab 3: W&B experiment tracking."""

    @requires_segmentation_data
    @pytest.mark.skipif(not WANDB_API_KEY, reason="WANDB_API_KEY not set")
    def test_wandb_training_completes(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Train with W&B tracking enabled."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-seg-wandb-{int(time.time())}"

        estimator = PyTorch(
            entry_point="train_simple.py",
            source_dir=str(
                Path(__file__).parent.parent
                / "medical-image-segmentation"
                / "code"
                / "training"
            ),
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "model_name": "SegResNet",
                "use_wandb": True,
                "wandb_project": "integration-tests",
                "wandb_api_key": WANDB_API_KEY,
            },
        )

        train_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "valid": valid_input},
            job_name=job_name,
            wait=True,
        )
        cleanup_training_jobs.append(job_name)

        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        assert desc["TrainingJobStatus"] == "Completed"


# ============================================================================
# Test: Lab 5 - Hyperparameter Optimization
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.training
@pytest.mark.expensive
class TestHyperparameterOptimization:
    """Integration tests for Lab 5: SageMaker HPO."""

    @requires_segmentation_data
    def test_hpo_tuning_job_launches(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Launch HPO tuner with minimal configuration."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.tuner import (
            HyperparameterTuner,
            ContinuousParameter,
            IntegerParameter,
        )
        from sagemaker.inputs import TrainingInput

        estimator = PyTorch(
            entry_point="train_simple.py",
            source_dir=str(
                Path(__file__).parent.parent
                / "medical-image-segmentation"
                / "code"
                / "training"
            ),
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "epochs": 1,
                "model_name": "SegResNet",
            },
            metric_definitions=[
                {"Name": "valid:dice", "Regex": "valid_dice=([0-9\\.]+)"},
            ],
        )

        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name="valid:dice",
            objective_type="Maximize",
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(0.0001, 0.01),
                "batch_size": IntegerParameter(1, 4),
            },
            max_jobs=2,
            max_parallel_jobs=2,
            strategy="Random",
        )

        tuning_job_name = f"test-seg-hpo-{int(time.time())}"

        train_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{SEGMENTATION_DATA_S3}/valid")

        tuner.fit(
            {"train": train_input, "valid": valid_input},
            job_name=tuning_job_name,
            wait=False,
        )

        # Verify tuning job was created
        time.sleep(10)
        desc = sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        assert desc["HyperParameterTuningJobStatus"] in (
            "InProgress",
            "Completed",
        )

        # Stop the tuning job to save costs
        try:
            sagemaker_client.stop_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )
        except Exception:
            pass  # May already be completed


# ============================================================================
# Test: Lab 7 - Model Deployment
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.deployment
@pytest.mark.expensive
class TestSegmentationDeployment:
    """Integration tests for Lab 7: Segmentation model deployment."""

    @requires_segmentation_model
    def test_async_endpoint_deploys(
        self,
        sagemaker_session,
        execution_role,
        test_bucket,
        sagemaker_client,
        cleanup_endpoints,
    ):
        """Deploy segmentation model to async endpoint."""
        from sagemaker.pytorch import PyTorchModel
        from sagemaker.async_inference import AsyncInferenceConfig

        endpoint_name = f"test-seg-async-{int(time.time())}"
        cleanup_endpoints.append(endpoint_name)

        image_uri = sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=sagemaker_session.boto_region_name,
            version="2.1.0",
            py_version="py310",
            instance_type="ml.g5.xlarge",
            image_scope="inference",
        )

        async_config = AsyncInferenceConfig(
            output_path=f"s3://{test_bucket}/test-seg-async-output/",
            failure_path=f"s3://{test_bucket}/test-seg-async-failures/",
            max_concurrent_invocations_per_instance=2,
        )

        model = PyTorchModel(
            model_data=SEGMENTATION_MODEL_S3,
            role=execution_role,
            source_dir=str(SEGMENTATION_NOTEBOOKS / "deploy"),
            entry_point="inference.py",
            framework_version="2.1.0",
            py_version="py310",
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
        )

        model.deploy(
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True,
        )

        status = wait_for_endpoint(sagemaker_client, endpoint_name)
        assert status == "InService"

    @requires_segmentation_model
    def test_async_endpoint_processes_inference(
        self,
        sagemaker_session,
        execution_role,
        test_bucket,
        sagemaker_client,
        cleanup_endpoints,
    ):
        """Deploy and invoke async endpoint with a test NIfTI file.

        Requires TEST_NIFTI_S3 environment variable pointing to a .nii.gz file.
        """
        test_nifti = os.environ.get("TEST_NIFTI_S3")
        if not test_nifti:
            pytest.skip("TEST_NIFTI_S3 environment variable not set")

        from sagemaker.pytorch import PyTorchModel
        from sagemaker.async_inference import AsyncInferenceConfig

        endpoint_name = f"test-seg-infer-{int(time.time())}"
        cleanup_endpoints.append(endpoint_name)

        image_uri = sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=sagemaker_session.boto_region_name,
            version="2.1.0",
            py_version="py310",
            instance_type="ml.g5.xlarge",
            image_scope="inference",
        )

        async_config = AsyncInferenceConfig(
            output_path=f"s3://{test_bucket}/test-seg-infer-output/",
            max_concurrent_invocations_per_instance=2,
        )

        model = PyTorchModel(
            model_data=SEGMENTATION_MODEL_S3,
            role=execution_role,
            source_dir=str(SEGMENTATION_NOTEBOOKS / "deploy"),
            entry_point="inference.py",
            framework_version="2.1.0",
            py_version="py310",
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
        )

        model.deploy(
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True,
        )

        wait_for_endpoint(sagemaker_client, endpoint_name)

        # Invoke with NIfTI file
        runtime_client = boto3.client("sagemaker-runtime")
        s3_client = boto3.client("s3")

        payload = json.dumps({"file_path": test_nifti})
        input_key = "test-seg-infer-input/request.json"
        s3_client.put_object(Bucket=test_bucket, Key=input_key, Body=payload)

        response = runtime_client.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=f"s3://{test_bucket}/{input_key}",
            ContentType="application/json",
        )

        output_location = response["OutputLocation"]
        assert output_location.startswith("s3://")

        # Wait for result (with timeout)
        output_bucket, output_key = output_location.replace("s3://", "").split("/", 1)
        start = time.time()
        while time.time() - start < 300:
            try:
                obj = s3_client.get_object(Bucket=output_bucket, Key=output_key)
                result = json.loads(obj["Body"].read().decode())
                assert "segmentation_shape" in result or "error" in result
                if "error" not in result:
                    assert result["segmented_voxels"] >= 0
                    assert "segmentation_mask_base64" in result
                return
            except s3_client.exceptions.NoSuchKey:
                time.sleep(10)

        pytest.fail("Async inference did not produce output within 5 minutes")


# ============================================================================
# Test: Auto-Scaling Configuration
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.deployment
@pytest.mark.expensive
class TestAutoScaling:
    """Test auto-scaling configuration for endpoints."""

    @requires_segmentation_model
    def test_scale_to_zero_configuration(
        self,
        sagemaker_session,
        execution_role,
        test_bucket,
        sagemaker_client,
        cleanup_endpoints,
    ):
        """Verify auto-scaling with scale-to-zero can be configured."""
        from sagemaker.pytorch import PyTorchModel
        from sagemaker.async_inference import AsyncInferenceConfig

        endpoint_name = f"test-seg-autoscale-{int(time.time())}"
        cleanup_endpoints.append(endpoint_name)

        image_uri = sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=sagemaker_session.boto_region_name,
            version="2.1.0",
            py_version="py310",
            instance_type="ml.g5.xlarge",
            image_scope="inference",
        )

        async_config = AsyncInferenceConfig(
            output_path=f"s3://{test_bucket}/test-autoscale-output/",
            max_concurrent_invocations_per_instance=2,
        )

        model = PyTorchModel(
            model_data=SEGMENTATION_MODEL_S3,
            role=execution_role,
            source_dir=str(SEGMENTATION_NOTEBOOKS / "deploy"),
            entry_point="inference.py",
            framework_version="2.1.0",
            py_version="py310",
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
        )

        model.deploy(
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True,
        )

        wait_for_endpoint(sagemaker_client, endpoint_name)

        # Configure auto-scaling
        aas_client = boto3.client("application-autoscaling")
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

        aas_client.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=0,
            MaxCapacity=3,
        )

        aas_client.put_scaling_policy(
            PolicyName=f"{endpoint_name}-scaling-policy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": 5.0,
                "CustomizedMetricSpecification": {
                    "MetricName": "ApproximateBacklogSizePerInstance",
                    "Namespace": "AWS/SageMaker",
                    "Dimensions": [
                        {"Name": "EndpointName", "Value": endpoint_name}
                    ],
                    "Statistic": "Average",
                },
                "ScaleInCooldown": 300,
                "ScaleOutCooldown": 60,
            },
        )

        # Verify scaling target registered
        targets = aas_client.describe_scalable_targets(
            ServiceNamespace="sagemaker",
            ResourceIds=[resource_id],
        )
        assert len(targets["ScalableTargets"]) == 1
        assert targets["ScalableTargets"][0]["MinCapacity"] == 0
        assert targets["ScalableTargets"][0]["MaxCapacity"] == 3

        # Cleanup auto-scaling
        aas_client.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
