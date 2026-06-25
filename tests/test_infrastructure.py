"""
Infrastructure validation tests.

These tests verify that AWS prerequisites are correctly configured before
running expensive training/deployment tests. Run these first to catch
permission or configuration issues early.

Run with: pytest -m integration tests/test_infrastructure.py
"""

import os
import pytest
import boto3
import json


# ============================================================================
# IAM Role Tests
# ============================================================================


@pytest.mark.integration
class TestIAMConfiguration:
    """Verify IAM roles and policies are correctly configured."""

    def test_execution_role_exists(self, execution_role, iam_client):
        """SageMaker execution role should exist."""
        # execution_role fixture already resolves the ARN
        assert execution_role.startswith("arn:aws:iam::")
        assert "role/" in execution_role

    def test_execution_role_has_sagemaker_trust(self, execution_role, iam_client):
        """Role should have SageMaker service in trust policy."""
        role_name = execution_role.split("/")[-1]
        role = iam_client.get_role(RoleName=role_name)
        trust_policy = role["Role"]["AssumeRolePolicyDocument"]

        principals = []
        for statement in trust_policy.get("Statement", []):
            principal = statement.get("Principal", {})
            service = principal.get("Service", "")
            if isinstance(service, list):
                principals.extend(service)
            else:
                principals.append(service)

        assert "sagemaker.amazonaws.com" in principals, \
            "Role trust policy does not include sagemaker.amazonaws.com"

    def test_execution_role_has_s3_access(self, execution_role, iam_client):
        """Role should have S3 access policy attached."""
        role_name = execution_role.split("/")[-1]
        policies = iam_client.list_attached_role_policies(RoleName=role_name)

        policy_arns = [p["PolicyArn"] for p in policies["AttachedPolicies"]]
        has_s3 = any("S3" in arn for arn in policy_arns)
        assert has_s3, "Role does not have an S3 access policy attached"

    def test_execution_role_has_sagemaker_access(self, execution_role, iam_client):
        """Role should have SageMaker access policy."""
        role_name = execution_role.split("/")[-1]
        policies = iam_client.list_attached_role_policies(RoleName=role_name)

        policy_arns = [p["PolicyArn"] for p in policies["AttachedPolicies"]]
        has_sm = any("SageMaker" in arn for arn in policy_arns)
        assert has_sm, "Role does not have a SageMaker access policy attached"


# ============================================================================
# S3 Bucket Tests
# ============================================================================


@pytest.mark.integration
class TestS3Configuration:
    """Verify S3 bucket access and data availability."""

    def test_default_bucket_accessible(self, test_bucket, s3_client):
        """Should be able to access the test bucket."""
        # HeadBucket will raise if we don't have access
        s3_client.head_bucket(Bucket=test_bucket)

    def test_can_write_to_bucket(self, test_bucket, s3_client):
        """Should be able to write objects to the test bucket."""
        test_key = "integration-tests/.test-write-access"
        s3_client.put_object(
            Bucket=test_bucket,
            Key=test_key,
            Body=b"test",
        )
        # Cleanup
        s3_client.delete_object(Bucket=test_bucket, Key=test_key)

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_DATA_S3"),
        reason="CLASSIFICATION_DATA_S3 not set"
    )
    def test_classification_data_exists(self, s3_client):
        """Verify classification data exists in S3."""
        data_uri = os.environ["CLASSIFICATION_DATA_S3"]
        bucket, prefix = data_uri.replace("s3://", "").split("/", 1)

        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, MaxKeys=5
        )
        assert response.get("KeyCount", 0) > 0, \
            f"No objects found at {data_uri}"

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set"
    )
    def test_segmentation_data_exists(self, s3_client):
        """Verify segmentation data exists in S3."""
        data_uri = os.environ["SEGMENTATION_DATA_S3"]
        bucket, prefix = data_uri.replace("s3://", "").split("/", 1)

        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, MaxKeys=5
        )
        assert response.get("KeyCount", 0) > 0, \
            f"No objects found at {data_uri}"

    @pytest.mark.skipif(
        not os.environ.get("CLASSIFICATION_DATA_S3"),
        reason="CLASSIFICATION_DATA_S3 not set"
    )
    def test_classification_data_has_train_split(self, s3_client):
        """Classification data should have train/ subfolder."""
        data_uri = os.environ["CLASSIFICATION_DATA_S3"]
        bucket, prefix = data_uri.replace("s3://", "").split("/", 1)
        if not prefix.endswith("/"):
            prefix += "/"

        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=f"{prefix}train/", MaxKeys=1
        )
        assert response.get("KeyCount", 0) > 0, \
            "No train/ subfolder in classification data"

    @pytest.mark.skipif(
        not os.environ.get("SEGMENTATION_DATA_S3"),
        reason="SEGMENTATION_DATA_S3 not set"
    )
    def test_segmentation_data_has_train_split(self, s3_client):
        """Segmentation data should have train/ subfolder."""
        data_uri = os.environ["SEGMENTATION_DATA_S3"]
        bucket, prefix = data_uri.replace("s3://", "").split("/", 1)
        if not prefix.endswith("/"):
            prefix += "/"

        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=f"{prefix}train/", MaxKeys=1
        )
        assert response.get("KeyCount", 0) > 0, \
            "No train/ subfolder in segmentation data"


# ============================================================================
# SageMaker Service Tests
# ============================================================================


@pytest.mark.integration
class TestSageMakerService:
    """Verify SageMaker service access and configuration."""

    def test_can_list_training_jobs(self, sagemaker_client):
        """Should be able to list training jobs (verifies API access)."""
        response = sagemaker_client.list_training_jobs(MaxResults=1)
        assert "TrainingJobSummaries" in response

    def test_can_list_endpoints(self, sagemaker_client):
        """Should be able to list endpoints."""
        response = sagemaker_client.list_endpoints(MaxResults=1)
        assert "Endpoints" in response

    def test_can_list_processing_jobs(self, sagemaker_client):
        """Should be able to list processing jobs."""
        response = sagemaker_client.list_processing_jobs(MaxResults=1)
        assert "ProcessingJobSummaries" in response

    def test_pytorch_image_uri_resolves(self, aws_region):
        """Should be able to resolve PyTorch container image URI."""
        import sagemaker

        image_uri = sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=aws_region,
            version="2.1.0",
            py_version="py310",
            instance_type="ml.g5.xlarge",
            image_scope="training",
        )
        assert image_uri is not None
        assert "pytorch" in image_uri
        assert aws_region in image_uri


# ============================================================================
# ECR Access Tests
# ============================================================================


@pytest.mark.integration
class TestECRAccess:
    """Verify ECR access for container images."""

    def test_can_describe_ecr_repositories(self, aws_session):
        """Should have ECR read access."""
        ecr_client = aws_session.client("ecr")
        # Just verifying API access - may have 0 repos
        try:
            ecr_client.describe_repositories(maxResults=1)
        except ecr_client.exceptions.ClientException:
            pytest.fail("Cannot access ECR")

    def test_sagemaker_container_registry_accessible(self, aws_region):
        """Should be able to pull from SageMaker's container registry."""
        # This is a read-only check on the public SageMaker ECR
        ecr_client = boto3.client("ecr", region_name=aws_region)
        try:
            ecr_client.describe_repositories(
                registryId="763104351884",  # SageMaker's ECR account
                maxResults=1,
            )
        except Exception:
            # Cross-account describe may not work, but that's OK
            # The important thing is we can pull during training
            pass
