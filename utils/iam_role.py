"""
Shared utility for creating or retrieving the SageMaker execution role.

Usage from any notebook:
    import sys
    sys.path.insert(0, "<path-to-project-root>")
    from utils import get_or_create_role

    role = get_or_create_role()

When to use get_or_create_role() vs sagemaker.get_execution_role():
    - Use get_or_create_role() when running notebooks OUTSIDE of SageMaker
      (e.g., local machines, EC2 instances, CI/CD pipelines). It creates the
      IAM role with the correct policies if it doesn't already exist, so you
      don't need to manually set up roles in the AWS console first.
    - Use sagemaker.get_execution_role() only when running INSIDE a SageMaker
      managed environment (Studio, notebook instances) where a role is already
      attached to the instance. It simply retrieves the pre-assigned role ARN
      from instance metadata — it cannot create roles and will fail outside
      SageMaker-managed environments.
"""

import json
import boto3


def get_or_create_role(
    role_name: str = "AmazonSageMaker-ExecutionRole-sgm",
    managed_policies: list = None,
) -> str:
    """
    Get or create a SageMaker execution role with the required policies.

    If the role already exists, returns its ARN. Otherwise, creates it with
    the default managed policies and an inline MLflow policy.

    Parameters
    ----------
    role_name : str
        Name of the IAM role. Default: "AmazonSageMaker-ExecutionRole-sgm"
    managed_policies : list, optional
        List of managed policy ARNs to attach. If None, uses a sensible default set.

    Returns
    -------
    str
        The ARN of the role.
    """
    if managed_policies is None:
        managed_policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonRekognitionFullAccess",
            "arn:aws:iam::aws:policy/ComprehendFullAccess",
            "arn:aws:iam::aws:policy/ComprehendMedicalFullAccess",
            "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicFullAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerCanvasFullAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerCanvasAIServicesAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerCanvasSMDataScienceAssistantAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerCanvasDataPrepFullAccess",
        ]

    # Inline policy for MLflow
    mlflow_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["sagemaker-mlflow:*"],
                "Resource": "*",
            }
        ],
    }

    # Trust policy — allows SageMaker service to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    iam = boto3.client("iam")

    try:
        existing = iam.get_role(RoleName=role_name)
        role_arn = existing["Role"]["Arn"]
        print(f"Role already exists: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        # Create role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker execution role for medical imaging training",
        )
        role_arn = response["Role"]["Arn"]
        print(f"Created role: {role_arn}")

        # Attach managed policies
        for policy_arn in managed_policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            print(f"  Attached: {policy_arn.split('/')[-1]}")

        # Add inline MLflow policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="mlflow-policy",
            PolicyDocument=json.dumps(mlflow_policy),
        )
        print("  Added inline policy: mlflow-policy")

    return role_arn
