"""
Integration tests that exercise run.py commands.

These tests call the same functions that run.py exposes as CLI commands,
using config.yaml values (overridden by environment variables).

Run with:
    pytest tests/test_run_cli.py -m smoke                    # Config loading, no AWS calls
    pytest tests/test_run_cli.py -m integration              # All integration tests
    pytest tests/test_run_cli.py -m "integration and training"  # Training only
    pytest tests/test_run_cli.py -m "integration and deployment" # Deploy only

Environment variables (same as run.py):
    AWS_DEFAULT_REGION, AWS_ACCOUNT_ID, AWS_SAGEMAKER_ROLE, AWS_TEST_BUCKET
    SEGMENTATION_DATA_S3, SEGMENTATION_MODEL_S3, TEST_NIFTI_S3
    CLASSIFICATION_DATA_S3, CLASSIFICATION_MODEL_S3, CLASSIFICATION_TEST_IMAGE_S3
    WANDB_API_KEY
"""

import os
import sys
import time
import pytest
from pathlib import Path

# Add project root to path so we can import run module
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import run as run_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def config():
    """Load project config with env var overrides."""
    return run_module.load_config()


@pytest.fixture(scope="session")
def role(config):
    """Resolve SageMaker execution role."""
    return run_module.get_role(config)


@pytest.fixture(scope="session")
def session(config):
    """SageMaker session."""
    return run_module.get_session(config)


@pytest.fixture(scope="session")
def bucket(config, session):
    """S3 bucket for tests."""
    return run_module.get_bucket(config, session)


# ============================================================================
# Smoke Tests — Config loading, no AWS calls
# ============================================================================


@pytest.mark.smoke
class TestConfigLoading:
    """Verify config.yaml loads correctly."""

    def test_config_loads(self, config):
        """Config YAML should parse without errors."""
        assert "aws" in config
        assert "segmentation" in config
        assert "classification" in config
        assert "tracking" in config

    def test_config_has_required_aws_keys(self, config):
        """AWS section should have region, account_id, sagemaker_role, s3_bucket."""
        aws = config["aws"]
        assert "region" in aws
        assert "account_id" in aws
        assert "sagemaker_role" in aws
        assert "s3_bucket" in aws

    def test_config_segmentation_structure(self, config):
        """Segmentation config should have training and deployment sections."""
        seg = config["segmentation"]
        assert "data_s3" in seg
        assert "model_s3" in seg
        assert "training" in seg
        assert "deployment" in seg
        assert "model_name" in seg["training"]
        assert "endpoint_name" in seg["deployment"]

    def test_config_classification_structure(self, config):
        """Classification config should have training and deployment sections."""
        cls = config["classification"]
        assert "data_s3" in cls
        assert "model_s3" in cls
        assert "training" in cls
        assert "deployment" in cls
        assert "model_name" in cls["training"]
        assert "num_classes" in cls["training"]
        assert "endpoint_name" in cls["deployment"]

    def test_env_var_overrides(self):
        """Environment variables should override config values."""
        os.environ["SEGMENTATION_DATA_S3"] = "s3://test-bucket/test-data/"
        try:
            cfg = run_module.load_config()
            assert cfg["segmentation"]["data_s3"] == "s3://test-bucket/test-data/"
        finally:
            del os.environ["SEGMENTATION_DATA_S3"]


# ============================================================================
# Infrastructure Tests — Lightweight AWS calls
# ============================================================================


@pytest.mark.integration
class TestRoleSetup:
    """Verify IAM role creation/retrieval via run.py."""

    def test_get_role_returns_arn(self, role):
        """get_role() should return a valid IAM ARN."""
        assert role.startswith("arn:aws:iam::")
        assert "role/" in role

    def test_setup_role_command(self, config, capsys):
        """setup-role command should print the role ARN."""
        run_module.cmd_setup_role(config)
        captured = capsys.readouterr()
        assert "arn:aws:iam::" in captured.out


@pytest.mark.integration
class TestStatusCommand:
    """Verify the status command works."""

    def test_status_runs_without_error(self, config, capsys):
        """status command should not crash even with no endpoints."""
        run_module.cmd_status(config)
        captured = capsys.readouterr()
        # Should print something for each endpoint (either status or NOT FOUND)
        assert "NOT FOUND" in captured.out or "InService" in captured.out


# ============================================================================
# Segmentation Training Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.training
@pytest.mark.expensive
class TestSegmentationTraining:
    """Integration tests for segmentation training via run.py."""

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set",
    )
    def test_train_segmentation_launches_job(
        self, config, sagemaker_client, cleanup_training_jobs, capsys
    ):
        """train-segmentation should launch a SageMaker training job."""
        # Override to minimal epochs for testing
        config["segmentation"]["training"]["epochs"] = 1

        run_module.cmd_train_segmentation(config)
        captured = capsys.readouterr()

        assert "Training job started:" in captured.out

        # Extract job name and track for cleanup
        for line in captured.out.splitlines():
            if "Training job started:" in line:
                job_name = line.split(":")[-1].strip()
                cleanup_training_jobs.append(job_name)
                break


# ============================================================================
# Classification Training Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.training
@pytest.mark.expensive
class TestClassificationTraining:
    """Integration tests for classification training via run.py."""

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_DATA_S3"),
        reason="CLASSIFICATION_DATA_S3 not set",
    )
    def test_train_classification_launches_job(
        self, config, sagemaker_client, cleanup_training_jobs, capsys
    ):
        """train-classification should launch a SageMaker training job."""
        # Override to minimal epochs for testing
        config["classification"]["training"]["epochs"] = 1

        run_module.cmd_train_classification(config)
        captured = capsys.readouterr()

        assert "Training job started:" in captured.out

        # Extract job name and track for cleanup
        for line in captured.out.splitlines():
            if "Training job started:" in line:
                job_name = line.split(":")[-1].strip()
                cleanup_training_jobs.append(job_name)
                break


# ============================================================================
# Segmentation Deployment Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.segmentation
@pytest.mark.deployment
@pytest.mark.expensive
class TestSegmentationDeployment:
    """Integration tests for segmentation deployment via run.py."""

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_MODEL_S3"),
        reason="SEGMENTATION_MODEL_S3 not set",
    )
    def test_deploy_segmentation_creates_endpoint(
        self, config, sagemaker_client, cleanup_endpoints, capsys
    ):
        """deploy-segmentation should create an async endpoint."""
        # Use a unique test endpoint name
        test_endpoint = f"test-seg-deploy-{int(time.time())}"
        config["segmentation"]["deployment"]["endpoint_name"] = test_endpoint
        cleanup_endpoints.append(test_endpoint)

        run_module.cmd_deploy_segmentation(config)
        captured = capsys.readouterr()

        assert f"'{test_endpoint}' deployed" in captured.out

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_MODEL_S3")
        or not os.environ.get("TEST_NIFTI_S3"),
        reason="SEGMENTATION_MODEL_S3 or TEST_NIFTI_S3 not set",
    )
    def test_invoke_segmentation_returns_result(self, config, capsys):
        """invoke-segmentation should return segmentation results."""
        s3_uri = os.environ["TEST_NIFTI_S3"]
        result = run_module.cmd_invoke_segmentation(config, s3_uri)

        assert result is not None
        assert "error" not in result
        assert "segmentation_percentage" in result


# ============================================================================
# Classification Deployment Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.classification
@pytest.mark.deployment
@pytest.mark.expensive
class TestClassificationDeployment:
    """Integration tests for classification deployment via run.py."""

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_MODEL_S3"),
        reason="CLASSIFICATION_MODEL_S3 not set",
    )
    def test_deploy_classification_creates_endpoint(
        self, config, sagemaker_client, cleanup_endpoints, capsys
    ):
        """deploy-classification should create an async endpoint."""
        test_endpoint = f"test-cls-deploy-{int(time.time())}"
        config["classification"]["deployment"]["endpoint_name"] = test_endpoint
        cleanup_endpoints.append(test_endpoint)

        run_module.cmd_deploy_classification(config)
        captured = capsys.readouterr()

        assert f"'{test_endpoint}' deployed" in captured.out

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_MODEL_S3")
        or not os.environ.get("CLASSIFICATION_TEST_IMAGE_S3"),
        reason="CLASSIFICATION_MODEL_S3 or CLASSIFICATION_TEST_IMAGE_S3 not set",
    )
    def test_invoke_classification_returns_result(self, config, capsys):
        """invoke-classification should return predictions."""
        s3_uri = os.environ["CLASSIFICATION_TEST_IMAGE_S3"]
        result = run_module.cmd_invoke_classification(config, s3_uri)

        assert result is not None
        assert "error" not in result
        assert "predictions" in result
        assert "predicted_class" in result
        assert "confidence" in result


# ============================================================================
# Cleanup Tests
# ============================================================================


@pytest.mark.integration
class TestCleanupCommands:
    """Verify cleanup commands don't crash on non-existent endpoints."""

    def test_cleanup_segmentation_handles_missing(self, config, capsys):
        """cleanup-segmentation should handle non-existent endpoint gracefully."""
        config["segmentation"]["deployment"]["endpoint_name"] = "nonexistent-endpoint-xyz"
        run_module.cmd_cleanup_segmentation(config)
        # Should not raise — just print error messages

    def test_cleanup_classification_handles_missing(self, config, capsys):
        """cleanup-classification should handle non-existent endpoint gracefully."""
        config["classification"]["deployment"]["endpoint_name"] = "nonexistent-endpoint-xyz"
        run_module.cmd_cleanup_classification(config)
        # Should not raise — just print error messages
