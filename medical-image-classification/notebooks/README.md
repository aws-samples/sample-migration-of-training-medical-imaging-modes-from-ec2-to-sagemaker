# Medical Image Classification Notebooks

SageMaker-based pipeline for preprocessing and training medical image classification models.

## Execution Role
Ensure that the SageMaker execution role has the following permissions:
- AmazonS3FullAccess
- AmazonElasticContainerRegistryPublicFullAccess
- AmazonSagemakerFullAccess
- 

MLFlow Policy
```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": [
				"sagemaker-mlflow:*"
			],
			"Resource": "*"
		}
	]
}
```


## Prerequisites

- AWS Account with SageMaker access
- Docker installed locally
- AWS CLI configured

## ECR Authentication

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

## Notebooks

### 01_data_preprocessing

Splits medical image datasets into train/test/validation sets using SageMaker Processing.

**Key Steps:**
- Build custom Docker image for preprocessing
- Configure S3 input/output paths
- Run SageMaker Processing job with 70/20/10 split
- Verify processed data structure

**Input Structure:**
```
data/
├── class_1/
│   ├── image_1.png
│   └── ...
└── class_2/
    └── ...
```

**Output Structure:**
```
data/
├── train/
├── test/
└── val/
```

## Usage

1. Update S3 bucket names and paths in notebooks
2. Run notebooks sequentially
3. Monitor SageMaker jobs in AWS Console
