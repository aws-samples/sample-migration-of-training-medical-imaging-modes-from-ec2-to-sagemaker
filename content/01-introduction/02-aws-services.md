---
title: "AWS Services Overview"
weight: 12
---

## Core Services

### Amazon SageMaker

Fully managed ML platform for building, training, managing and deploying models.

**Key Features:**
- **Training Jobs**: Managed infrastructure for model training
- **Processing Jobs**: Data preprocessing at scale
- **Endpoints**: Real-time and batch inference
- **Experiments**: Track and compare training runs
- **Model Registry**: Version and manage models

**Examples of Instance Types:**
- `ml.g5.xlarge`: 1 GPU, 24GB memory
- `ml.g5.12xlarge`: 4 GPUs, 96GB memory 
- `ml.g5.48xlarge`: 8 GPUs, 192GB memory 

### Amazon EC2

Virtual compute instances for custom ML workloads.

**GPU Instances:**
- `g5.xlarge`: 1 A10G GPU, 24GB memory
- `g5.12xlarge`: 4 A10G GPUs, 96GB memory
- `p4d.24xlarge`: 8 A100 GPUs, 320GB memory

**Use Cases:**
- Baseline training experiments
- Custom environments
- Interactive development
- On prem development experience with integration with popular IDEs 

### Amazon S3

Object storage for datasets and model artifacts.

**Best Practices:**
- Use S3 for all training data
- Enable versioning for datasets
- Use S3 Select for efficient data access
- Organize with prefixes: `s3://bucket/train/`, `s3://bucket/val/`

### Amazon Elastic Container Registry (ECR)

Container registry for custom Docker images.

**Workflow:**
1. Build custom training container
2. Push to ECR
3. Use the container as base images to run training jobs

### AWS IAM

Identity and access management.

**Required Permissions:**
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`

## Service Comparison

| Feature | EC2 | SageMaker |
|---------|-----|-----------|
| Infrastructure Management | Manual | Automatic |
| Scaling | Manual | Automatic |
| Cost | Pay for uptime | Pay per job |
| Monitoring | CloudWatch | Built-in + CloudWatch |
| Distributed Training | Manual setup | Built-in |
| Spot Instances | Manual | Automatic failover |
| Best For | Prototyping | Production |

## When to Use What?

### Use EC2 When:
- Rapid prototyping and experimentation
- Need full control over environment
- Running Jupyter notebooks interactively
- Cost-sensitive with long-running workloads

### Use SageMaker When:
- Production training pipelines
- Need distributed training
- Want automatic infrastructure management
- Require experiment tracking and model registry
- Building end-to-end ML workflows

## Workshop Architecture

In this workshop, you'll use:

1. **SageMaker Notebook Instance**: Development environment
2. **Amazon S3**: Store datasets and model artifacts
3. **SageMaker Training Jobs**: Train models at scale
4. **Amazon ECR**: Store custom containers
5. **CloudWatch**: Monitor training jobs

All infrastructure is provisioned via CloudFormation for consistency.
