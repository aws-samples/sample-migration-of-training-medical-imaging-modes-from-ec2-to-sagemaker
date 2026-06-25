"""
Shared pytest fixtures for notebook integration tests.

Usage:
    # Run only smoke tests (no AWS resources created):
    pytest -m smoke

    # Run full integration tests (creates real AWS resources - costs money):
    pytest -m integration

    # Run only classification tests:
    pytest -m classification

    # Run everything except expensive GPU/endpoint tests:
    pytest -m "not expensive"
"""

import os
import time
import pytest
import boto3


# ============================================================================
# AWS Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def aws_region():
    """AWS region for integration tests."""
    return os.environ.get("AWS_TEST_REGION", "us-east-1")


@pytest.fixture(scope="session")
def aws_session(aws_region):
    """Boto3 session for integration tests."""
    return boto3.Session(region_name=aws_region)


@pytest.fixture(scope="session")
def sagemaker_client(aws_session):
    """SageMaker client."""
    return aws_session.client("sagemaker")


@pytest.fixture(scope="session")
def s3_client(aws_session):
    """S3 client."""
    return aws_session.client("s3")


@pytest.fixture(scope="session")
def iam_client(aws_session):
    """IAM client."""
    return aws_session.client("iam")


@pytest.fixture(scope="session")
def sagemaker_session(aws_session):
    """SageMaker SDK session."""
    import sagemaker
    return sagemaker.Session(boto_session=aws_session)


@pytest.fixture(scope="session")
def execution_role():
    """SageMaker execution role ARN.

    Set via AWS_SAGEMAKER_ROLE environment variable, or defaults to the
    role used in the notebooks.
    """
    role = os.environ.get(
        "AWS_SAGEMAKER_ROLE",
        "AmazonSageMaker-ExecutionRole-sgm"
    )
    # If it's just a role name, resolve the full ARN
    if not role.startswith("arn:"):
        iam = boto3.client("iam")
        try:
            response = iam.get_role(RoleName=role)
            role = response["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            pytest.skip(f"IAM role '{role}' does not exist. Set AWS_SAGEMAKER_ROLE.")
    return role


@pytest.fixture(scope="session")
def test_bucket(sagemaker_session):
    """S3 bucket for test data and artifacts."""
    bucket = os.environ.get("AWS_TEST_BUCKET", sagemaker_session.default_bucket())
    return bucket


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_endpoints(sagemaker_client):
    """Track and clean up any SageMaker endpoints created during tests."""
    created_endpoints = []
    yield created_endpoints

    # Cleanup after all tests
    for endpoint_name in created_endpoints:
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Cleaned up endpoint: {endpoint_name}")
        except Exception as e:
            print(f"Failed to clean up endpoint {endpoint_name}: {e}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_training_jobs(sagemaker_client):
    """Track training jobs for verification (cannot delete, but can stop)."""
    created_jobs = []
    yield created_jobs

    # Stop any still-running jobs
    for job_name in created_jobs:
        try:
            desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            if desc["TrainingJobStatus"] in ("InProgress", "Stopping"):
                sagemaker_client.stop_training_job(TrainingJobName=job_name)
                print(f"Stopped training job: {job_name}")
        except Exception as e:
            print(f"Could not stop training job {job_name}: {e}")
