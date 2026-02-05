# Medical Imaging on AWS SageMaker Workshop

End-to-end workshops for medical image classification and segmentation using Amazon SageMaker, covering data preprocessing, training strategies, and production deployment.

## Overview

This repository contains two comprehensive workshops:

1. **Medical Image Classification** - Train classification models from EC2 to SageMaker with custom containers
2. **Medical Image Segmentation** - 3D segmentation with single/multi-GPU distributed training using MONAI

## Repository Structure

```
medical-imaging-sagemaker-workshop/
├── medical-image-classification/
│   └── notebooks/
│       ├── 00_ec2_training/              # Baseline EC2 training
│       ├── 01_data_preprocessing/        # SageMaker Processing
│       ├── 02_sm_script_mode/            # SageMaker Script Mode
│       ├── 03_sagemaker_byoc/            # Bring Your Own Container
│       └── 04_sagemaker_byoc_mlflow/     # BYOC + MLflow
└── medical-image-segmentation/
    ├── code/
    │   ├── training/                     # Training scripts (DDP/FSDP)
    │   ├── models/                       # Model definitions
    │   └── scripts/                      # Build scripts
    └── notebooks/
        ├── lab1_single_gpu_training.ipynb
        ├── lab2_fsdp_multi_gpu.ipynb
        ├── lab3_wandb_experiment_tracking.ipynb
        ├── lab4_ddp_unified_tracking.ipynb
        └── lab5_hyperparameter_optimization.ipynb
```

## Prerequisites

### AWS Requirements
- AWS Account with SageMaker access
- IAM role with permissions:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
  - `AmazonElasticContainerRegistryPublicFullAccess`
- S3 bucket for data and model artifacts

### Local Development
- Python 3.10+
- Docker (for custom containers)
- AWS CLI configured
- 16GB+ GPU memory (for local testing)

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd medical-imaging-sagemaker-workshop
```

### 2. ECR Authentication
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

### 3. Choose Your Workshop
- **Classification**: `cd medical-image-classification/notebooks`
- **Segmentation**: `cd medical-image-segmentation/notebooks`

## Workshop 1: Medical Image Classification

Progressive journey from EC2 to production SageMaker deployment.

### Lab 00: EC2 Training (Baseline)
**Duration**: 30 minutes

Train classification models on EC2 to establish baseline performance.

### Lab 01: Data Preprocessing
**Duration**: 45 minutes  
**Instance**: ml.m5.xlarge

Split datasets into train/test/val using SageMaker Processing.

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

### Lab 02: SageMaker Script Mode
**Duration**: 45 minutes  
**Instance**: ml.g5.xlarge

Use managed PyTorch containers with minimal code changes.

**Key Features:**
- Automatic infrastructure provisioning
- Built-in distributed training support
- Model artifact management
- CloudWatch integration

### Lab 03: Bring Your Own Container (BYOC)
**Duration**: 60 minutes  
**Instance**: ml.g5.xlarge

Full control with custom Docker containers.

**Benefits:**
- Custom dependencies and library versions
- Complete environment control
- Reproducible training environments
- Support for any ML framework

### Lab 04: BYOC + MLflow
**Duration**: 60 minutes  
**Instance**: ml.g5.xlarge

Add experiment tracking with MLflow integration.

**Features:**
- Automatic hyperparameter logging
- Metric tracking over time
- Model versioning
- Experiment comparison

## Workshop 2: Medical Image Segmentation

Production-scale 3D medical image segmentation with MONAI.

### Lab 1: Single GPU Training
**Duration**: 30 minutes  
**Instance**: ml.g5.xlarge (1 GPU)

Train SegResNet for 3D medical image segmentation.

**Models Available:**
- SegResNet (~5M params, 2GB memory)
- UNet (~31M params, 8GB memory)
- SwinUNETR (~62M params, 20GB memory)

### Lab 2: Multi-GPU FSDP
**Duration**: 45-60 minutes  
**Instance**: ml.g5.12xlarge (4 GPUs)

Scale training with Fully Sharded Data Parallel.

**Benefits:**
- Train larger models (SwinUNETR)
- Reduced memory per GPU
- Near-linear scaling efficiency
- Automatic gradient sharding

### Lab 3: Weights & Biases Tracking
**Duration**: 30 minutes  
**Instance**: ml.g5.xlarge

Advanced experiment tracking and visualization.

**Features:**
- Real-time metric visualization
- Hyperparameter comparison
- Model artifact versioning
- Team collaboration

### Lab 4: DDP Unified Tracking
**Duration**: 45 minutes  
**Instance**: ml.g5.12xlarge (4 GPUs)

Combine TensorBoard, MLflow, and WandB with DDP.

### Lab 5: Hyperparameter Optimization
**Duration**: 60+ minutes  
**Instance**: Multiple

Automated hyperparameter tuning with SageMaker.

**Strategies:**
- Bayesian optimization
- Random search
- Grid search
- Cost-effective tuning

## Cost Optimization

### Instance Recommendations

| Use Case | Instance | GPUs | Cost/Hour | Notes |
|----------|----------|------|-----------|-------|
| Prototyping | ml.g5.xlarge | 1 | $1.41 | Single GPU development |
| Classification | ml.g5.2xlarge | 1 | $1.52 | Production training |
| Segmentation (Multi-GPU) | ml.g5.12xlarge | 4 | $7.09 | FSDP/DDP training |
| Budget Multi-GPU | ml.g4dn.12xlarge | 4 | $4.89 | Cost-effective option |

### Best Practices

1. **Use Warm Pools**
   ```python
   keep_alive_period_in_seconds=1800
   ```

2. **Enable Spot Instances** (Save up to 70%)
   ```python
   use_spot_instances=True,
   max_wait=7200
   ```

3. **Cache Dependencies**
   ```python
   environment={"PIP_CACHE_DIR": "/opt/ml/sagemaker/warmpoolcache/pip"}
   ```

4. **Optimize Batch Size**
   - Single GPU: batch_size=2-4
   - Multi-GPU: batch_size=2 per GPU

## Data Preparation

### Classification Data
```
s3://bucket/classification_data/
├── train/
│   ├── class_1/
│   └── class_2/
├── test/
└── val/
```

### Segmentation Data
```
s3://bucket/segmentation_data/
├── train/
│   └── subject_001/
│       ├── img.nii.gz
│       └── label.nii.gz
└── valid/
```

## Training Strategies

### Single GPU
```python
estimator = PyTorch(
    entry_point="train.py",
    instance_type="ml.g5.xlarge",
    instance_count=1
)
```

### Data Parallel (DDP)
```python
estimator = PyTorch(
    entry_point="train_ddp.py",
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    distribution={"pytorchddp": {"enabled": True}}
)
```

### Fully Sharded (FSDP)
```python
estimator = PyTorch(
    entry_point="train_fsdp.py",
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    distribution={"pytorchddp": {"enabled": True}}
)
```

## Experiment Tracking

### TensorBoard (Built-in)
```bash
tensorboard --logdir=./output
```

### MLflow
```python
hyperparameters = {
    "use_mlflow": True,
    "mlflow_tracking_uri": "arn:aws:sagemaker:...",
    "mlflow_experiment_name": "medical-imaging"
}
```

### Weights & Biases
```python
hyperparameters = {
    "use_wandb": True,
    "wandb_project": "medical-imaging",
    "wandb_api_key": "your-api-key"
}
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Use gradient checkpointing
- Switch to FSDP for multi-GPU
- Enable mixed precision (FP16)

### Slow Training
- Increase batch size
- Use multiple GPUs
- Enable mixed precision
- Optimize data loading (num_workers)

### Data Loading Issues
- Verify S3 paths and permissions
- Check data format (NIfTI for segmentation)
- Ensure correct folder structure

### Container Issues
- Test locally with Docker first
- Check CloudWatch logs
- Verify IAM permissions
- Inspect container: `docker run -it <image> /bin/bash`

## Deployment

### Build Custom Container
```bash
cd code/scripts
./build_and_push.sh <image-name> <region>
```

### Deploy to Endpoint
```python
predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1
)
```

## Resources

### Documentation
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [MONAI Documentation](https://docs.monai.io/)
- [PyTorch on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html)

### AWS Services
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
- [Medical Imaging on AWS](https://aws.amazon.com/health/solutions/medical-imaging/)
- [AWS Pricing Calculator](https://calculator.aws)

### Examples
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)

## Support

For issues and questions:
- Open a GitHub issue
- Check CloudWatch logs for training jobs
- Review AWS SageMaker documentation
- Consult workshop-specific READMEs

## License

This project is licensed under the MIT License.

## Acknowledgments

Built with:
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - ML platform
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [MONAI](https://monai.io/) - Medical imaging framework
- [MLflow](https://mlflow.org/) - Experiment tracking
- [Weights & Biases](https://wandb.ai/) - ML operations
