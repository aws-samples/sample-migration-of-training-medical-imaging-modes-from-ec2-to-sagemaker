"""
Utility to get or create a SageMaker execution role with required policies.
"""

import json
import boto3


def get_or_create_sagemaker_role(role_name="AmazonSageMaker-ExecutionRole-sgm"):
    """Create a SageMaker execution role if it doesn't exist, and return its ARN.

    Args:
        role_name: Name of the IAM role to create or retrieve.

    Returns:
        The ARN of the role.
    """
    iam = boto3.client("iam")

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

    try:
        existing = iam.get_role(RoleName=role_name)
        role_arn = existing["Role"]["Arn"]
        print(f"Role already exists: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker execution role for nnU-Net training",
        )
        role_arn = response["Role"]["Arn"]
        print(f"Created role: {role_arn}")

        for policy_arn in managed_policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            print(f"  Attached: {policy_arn.split('/')[-1]}")

        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="mlflow-policy",
            PolicyDocument=json.dumps(mlflow_policy),
        )
        print("  Added inline policy: mlflow-policy")

    print(f"\nUsing role: {role_arn}")
    return role_arn


if __name__ == "__main__":
    role = get_or_create_sagemaker_role()
