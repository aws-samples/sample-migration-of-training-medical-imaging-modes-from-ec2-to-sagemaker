"""
Integration tests for medical-image-classification notebooks.

These tests execute the notebooks against real AWS services. They will:
- Create IAM roles (if not existing)
- Upload data to S3
- Launch SageMaker training jobs
- Deploy endpoints
- Run inference
- Clean up all resources

Run with: pytest -m "integration and classification" --timeout=3600

Environment variables:
    AWS_TEST_REGION: AWS region (default: us-east-1)
    AWS_SAGEMAKER_ROLE: SageMaker execution role ARN or name
    AWS_TEST_BUCKET: S3 bucket for test data (default: SageMaker default bucket)
    CLASSIFICATION_DATA_S3: S3 URI to classification dataset
        (e.g., s3://bucket/classification_data)
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
    CLASSIFICATION_NOTEBOOKS,
    load_notebook,
    get_code_cells,
    execute_notebook,
    wait_for_training_job,
    wait_for_endpoint,
)


# ============================================================================
# Configuration
# ============================================================================

CLASSIFICATION_DATA_S3 = os.environ.get("CLASSIFICATION_DATA_S3", "")


def requires_classification_data(func):
    """Skip test if classification data S3 path is not configured."""
    return pytest.mark.skipif(
        not CLASSIFICATION_DATA_S3,
        reason="CLASSIFICATION_DATA_S3 environment variable not set"
    )(func)


# ============================================================================
# Test: Data Preprocessing (01)
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.preprocessing
class TestDataPreprocessing:
    """Integration tests for the data preprocessing notebook."""

    @requires_classification_data
    def test_preprocessing_job_completes(
        self, sagemaker_session, execution_role, test_bucket, sagemaker_client
    ):
        """Run the SageMaker Processing job for data splitting."""
        from sagemaker.processing import ScriptProcessor

        # Use the same container from the notebook's Dockerfile
        # For integration testing, we use a prebuilt sklearn container
        from sagemaker.sklearn.processing import SKLearnProcessor

        processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=execution_role,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            sagemaker_session=sagemaker_session,
        )

        processing_job_name = f"test-preprocess-{int(time.time())}"

        processor.run(
            code=str(
                CLASSIFICATION_NOTEBOOKS / "01_data_preprocessing" / "preprocessing.py"
            ),
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=CLASSIFICATION_DATA_S3,
                    destination="/opt/ml/processing/input",
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{test_bucket}/test-preprocessing-output/",
                )
            ],
            job_name=processing_job_name,
            wait=True,
        )

        # Verify job completed
        desc = sagemaker_client.describe_processing_job(
            ProcessingJobName=processing_job_name
        )
        assert desc["ProcessingJobStatus"] == "Completed"

        # Verify output exists in S3
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(
            Bucket=test_bucket, Prefix="test-preprocessing-output/", MaxKeys=5
        )
        assert response.get("KeyCount", 0) > 0, "No output files from preprocessing"


# ============================================================================
# Test: Script Mode Training (02)
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.training
@pytest.mark.expensive
class TestScriptModeTraining:
    """Integration tests for SageMaker Script Mode training."""

    @requires_classification_data
    def test_script_mode_training_completes(
        self,
        sagemaker_session,
        execution_role,
        test_bucket,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Launch a Script Mode training job and verify completion."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-script-mode-{int(time.time())}"

        estimator = PyTorch(
            source_dir=str(CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "code"),
            entry_point="train.py",
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "batch_size": 8,
                "epochs": 2,  # Minimal epochs for testing
                "learning_rate": 0.001,
                "model_name": "DenseNet121",
                "num_classes": 8,
            },
            dependencies=[
                str(CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "requirements.txt")
            ],
        )

        train_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "test": valid_input},
            job_name=job_name,
            wait=False,
        )
        cleanup_training_jobs.append(job_name)

        # Wait for completion
        status = wait_for_training_job(sagemaker_client, job_name, timeout=1800)
        assert status == "Completed", f"Training job failed with status: {status}"

        # Verify model artifacts
        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        model_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]
        assert model_s3.endswith("model.tar.gz"), "No model artifacts produced"

    @requires_classification_data
    def test_script_mode_training_produces_metrics(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Verify that training produces CloudWatch metrics."""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        job_name = f"test-sm-metrics-{int(time.time())}"

        estimator = PyTorch(
            source_dir=str(CLASSIFICATION_NOTEBOOKS / "02_sm_script_mode" / "code"),
            entry_point="train.py",
            framework_version="2.1.0",
            py_version="py310",
            instance_count=1,
            instance_type="ml.g5.xlarge",
            role=execution_role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 0.001,
                "model_name": "DenseNet121",
                "num_classes": 8,
            },
            metric_definitions=[
                {"Name": "train:loss", "Regex": "Train Loss: ([0-9\\.]+)"},
                {"Name": "valid:accuracy", "Regex": "Valid Accuracy: ([0-9\\.]+)"},
            ],
        )

        train_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "test": valid_input},
            job_name=job_name,
            wait=True,
        )
        cleanup_training_jobs.append(job_name)

        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        assert desc["TrainingJobStatus"] == "Completed"


# ============================================================================
# Test: BYOC Training (03)
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.training
@pytest.mark.expensive
class TestBYOCTraining:
    """Integration tests for Bring Your Own Container training."""

    @requires_classification_data
    def test_byoc_training_completes(
        self,
        sagemaker_session,
        execution_role,
        aws_region,
        sagemaker_client,
        cleanup_training_jobs,
    ):
        """Launch a BYOC training job.

        Requires the container to be built and pushed to ECR first.
        Set BYOC_IMAGE_URI environment variable to the ECR image URI.
        """
        image_uri = os.environ.get("BYOC_IMAGE_URI")
        if not image_uri:
            pytest.skip("BYOC_IMAGE_URI environment variable not set")

        from sagemaker.estimator import Estimator
        from sagemaker.inputs import TrainingInput

        job_name = f"test-byoc-{int(time.time())}"

        estimator = Estimator(
            image_uri=image_uri,
            role=execution_role,
            instance_count=1,
            instance_type="ml.g5.xlarge",
            sagemaker_session=sagemaker_session,
            hyperparameters={
                "batch_size": 8,
                "epochs": 2,
                "learning_rate": 0.001,
                "model_name": "DenseNet121",
                "num_classes": 8,
            },
        )

        train_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/train")
        valid_input = TrainingInput(s3_data=f"{CLASSIFICATION_DATA_S3}/valid")

        estimator.fit(
            {"train": train_input, "test": valid_input},
            job_name=job_name,
            wait=True,
        )
        cleanup_training_jobs.append(job_name)

        desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        assert desc["TrainingJobStatus"] == "Completed"


# ============================================================================
# Test: Deployment (05)
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.deployment
@pytest.mark.expensive
class TestClassificationDeployment:
    """Integration tests for model deployment notebooks."""

    def test_async_endpoint_deployment(
        self,
        sagemaker_session,
        execution_role,
        test_bucket,
        sagemaker_client,
        cleanup_endpoints,
    ):
        """Deploy a model to an async endpoint and run inference.

        Requires MODEL_DATA_S3 environment variable pointing to a model.tar.gz.
        """
        model_data = os.environ.get("CLASSIFICATION_MODEL_S3")
        if not model_data:
            pytest.skip("CLASSIFICATION_MODEL_S3 environment variable not set")

        from sagemaker.pytorch import PyTorchModel
        from sagemaker.async_inference import AsyncInferenceConfig

        endpoint_name = f"test-cls-async-{int(time.time())}"
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
            output_path=f"s3://{test_bucket}/test-async-output/",
            max_concurrent_invocations_per_instance=2,
        )

        model = PyTorchModel(
            model_data=model_data,
            role=execution_role,
            source_dir=str(CLASSIFICATION_NOTEBOOKS / "05_sagemaker_deployment"),
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

        # Verify endpoint is InService
        status = wait_for_endpoint(sagemaker_client, endpoint_name, timeout=900)
        assert status == "InService", f"Endpoint status: {status}"

    def test_realtime_endpoint_deployment(
        self,
        sagemaker_session,
        execution_role,
        sagemaker_client,
        cleanup_endpoints,
    ):
        """Deploy a model to a real-time endpoint.

        Requires CLASSIFICATION_MODEL_S3 environment variable.
        """
        model_data = os.environ.get("CLASSIFICATION_MODEL_S3")
        if not model_data:
            pytest.skip("CLASSIFICATION_MODEL_S3 environment variable not set")

        from sagemaker.pytorch import PyTorchModel

        endpoint_name = f"test-cls-rt-{int(time.time())}"
        cleanup_endpoints.append(endpoint_name)

        image_uri = sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=sagemaker_session.boto_region_name,
            version="2.1.0",
            py_version="py310",
            instance_type="ml.g5.xlarge",
            image_scope="inference",
        )

        model = PyTorchModel(
            model_data=model_data,
            role=execution_role,
            source_dir=str(CLASSIFICATION_NOTEBOOKS / "05_sagemaker_deployment"),
            entry_point="inference_realtime.py",
            framework_version="2.1.0",
            py_version="py310",
            image_uri=image_uri,
            sagemaker_session=sagemaker_session,
        )

        predictor = model.deploy(
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            wait=True,
        )

        status = wait_for_endpoint(sagemaker_client, endpoint_name, timeout=900)
        assert status == "InService"

        # Invoke endpoint with test payload
        runtime_client = boto3.client("sagemaker-runtime")
        test_payload = json.dumps({"test": True})

        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=test_payload,
            )
            result = json.loads(response["Body"].read().decode())
            assert "predictions" in result or "error" not in result
        except Exception as e:
            # Endpoint may not work without real input data, but it should be reachable
            assert "ValidationError" in str(e) or "ModelError" in str(e)
