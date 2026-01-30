# Medical Image Segmentation on AWS SageMaker

Train 3D medical image segmentation models on Amazon SageMaker using MONAI and PyTorch with single-GPU and multi-GPU distributed training.

## Overview

This workshop demonstrates end-to-end medical image segmentation workflows on AWS SageMaker, from single-GPU prototyping to production-scale multi-GPU training with experiment tracking.

### Key Features

- **Multiple Training Strategies**: Single GPU, DDP, and FSDP
- **State-of-the-art Models**: SegResNet, SwinUNETR, UNet
- **Experiment Tracking**: TensorBoard, MLflow, Weights & Biases
- **Production Ready**: Docker containers, SageMaker integration
- **Cost Optimized**: Warm pools, spot instances support

## Architecture

```
medical-image-segmentation/
├── code/
│   ├── training/          # Training scripts
│   │   ├── train_simple.py           # Single GPU
│   │   ├── train_ddp.py              # Multi-GPU DDP
│   │   ├── train_fsdp.py             # Multi-GPU FSDP
│   │   ├── train_*_mlflow.py         # MLflow tracking
│   │   ├── train_*_tensorboard.py    # TensorBoard tracking
│   │   ├── train_*_wandb.py          # WandB tracking
│   │   └── model_def.py              # Model definitions
│   ├── models/            # Model architectures
│   ├── scripts/           # Build and deployment scripts
│   └── docker/            # Docker configurations
└── notebooks/             # Jupyter notebooks
    ├── lab1_single_gpu_training.ipynb
    ├── lab2_fsdp_multi_gpu.ipynb
    ├── lab3_wandb_experiment_tracking.ipynb
    ├── lab4_ddp_unified_tracking.ipynb
    └── lab5_hyperparameter_optimization.ipynb
```

## Prerequisites

### AWS Requirements
- AWS Account with SageMaker access
- IAM role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
- S3 bucket for data and model artifacts

### Local Development
- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 16GB+ GPU memory recommended

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd medical-image-segmentation

# Install dependencies
pip install -r notebooks/requirements.txt
```

### 2. Prepare Data

Organize your medical imaging data in S3:

```
s3://your-bucket/segmentation_data/
├── train/
│   ├── subject_001/
│   │   ├── img.nii.gz
│   │   └── label.nii.gz
│   └── subject_002/
│       ├── img.nii.gz
│       └── label.nii.gz
└── valid/
    └── subject_001/
        ├── img.nii.gz
        └── label.nii.gz
```

### 3. Run Training

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/lab1_single_gpu_training.ipynb
```

**Option B: Python Script**
```bash
python code/training/train_simple.py \
    --model_name SegResNet \
    --data ./data \
    --batch_size 2 \
    --epochs 10 \
    --lr 0.0001
```

**Option C: Docker**
```bash
cd code/docker
docker-compose up
```

## Workshop Labs

### Lab 1: Single GPU Training
**Duration**: 30 minutes  
**Instance**: ml.g5.xlarge (1 GPU)

Learn the basics of medical image segmentation on SageMaker:
- Setup SageMaker training jobs
- Train SegResNet model
- Monitor with TensorBoard
- Save and deploy models

[Open Lab 1](notebooks/lab1_single_gpu_training.ipynb)

### Lab 2: Multi-GPU Training with FSDP
**Duration**: 45-60 minutes  
**Instance**: ml.g5.12xlarge (4 GPUs)

Scale training with Fully Sharded Data Parallel:
- Understand FSDP vs DDP
- Train large models (SwinUNETR)
- Optimize memory usage
- Compare performance metrics

[Open Lab 2](notebooks/lab2_fsdp_multi_gpu.ipynb)

### Lab 3: Experiment Tracking with WandB
**Duration**: 30 minutes  
**Instance**: ml.g5.xlarge

Advanced experiment tracking and visualization:
- Setup Weights & Biases
- Track hyperparameters and metrics
- Visualize training progress
- Compare experiments

[Open Lab 3](notebooks/lab3_wandb_experiment_tracking.ipynb)

### Lab 4: Unified Tracking (DDP)
**Duration**: 45 minutes  
**Instance**: ml.g5.12xlarge

Combine all tracking tools with DDP:
- TensorBoard + MLflow + WandB
- Multi-GPU distributed training
- Comprehensive monitoring

[Open Lab 4](notebooks/lab4_ddp_unified_tracking.ipynb)

### Lab 5: Hyperparameter Optimization
**Duration**: 60+ minutes  
**Instance**: Multiple

Automated hyperparameter tuning:
- SageMaker Automatic Model Tuning
- Bayesian optimization
- Cost-effective tuning strategies

[Open Lab 5](notebooks/lab5_hyperparameter_optimization.ipynb)

## Models

### SegResNet (Default)
- **Parameters**: ~5M
- **Memory**: ~2GB
- **Speed**: Fast
- **Use Case**: Prototyping, quick iterations

### SwinUNETR
- **Parameters**: ~62M
- **Memory**: ~20GB
- **Speed**: Moderate
- **Use Case**: High accuracy requirements

### UNet
- **Parameters**: ~31M
- **Memory**: ~8GB
- **Speed**: Fast
- **Use Case**: Classic baseline

## Training Strategies

### Single GPU (train_simple.py)
```python
estimator = PyTorch(
    entry_point="train_simple.py",
    instance_type="ml.g5.xlarge",
    instance_count=1
)
```

### DDP - Data Parallel (train_ddp.py)
```python
estimator = PyTorch(
    entry_point="train_ddp.py",
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    distribution={"pytorchddp": {"enabled": True}}
)
```

### FSDP - Fully Sharded (train_fsdp.py)
```python
estimator = PyTorch(
    entry_point="train_fsdp.py",
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    distribution={"pytorchddp": {"enabled": True}}
)
```

## Experiment Tracking

### TensorBoard (Always Enabled)
```bash
tensorboard --logdir=./output
```

### MLflow
```python
hyperparameters = {
    "use_mlflow": True,
    "mlflow_tracking_uri": "arn:aws:sagemaker:...",
    "mlflow_experiment_name": "medical-segmentation"
}
```

### Weights & Biases
```python
hyperparameters = {
    "use_wandb": True,
    "wandb_project": "medical-segmentation",
    "wandb_api_key": "your-api-key"
}
```

## Cost Optimization

### Instance Comparison

| Instance | GPUs | GPU Memory | Cost/Hour | Use Case |
|----------|------|------------|-----------|----------|
| ml.g5.xlarge | 1 | 24GB | $1.41 | Prototyping |
| ml.g5.2xlarge | 1 | 24GB | $1.52 | Single GPU training |
| ml.g5.12xlarge | 4 | 96GB | $7.09 | Multi-GPU training |
| ml.g4dn.12xlarge | 4 | 64GB | $4.89 | Budget multi-GPU |

### Best Practices

1. **Use Warm Pools**: Keep instances alive for 30 minutes
   ```python
   keep_alive_period_in_seconds=1800
   ```

2. **Enable Spot Instances**: Save up to 70%
   ```python
   use_spot_instances=True,
   max_wait=7200
   ```

3. **Optimize Batch Size**: Maximize GPU utilization
   - Single GPU: batch_size=2-4
   - Multi-GPU: batch_size=2 per GPU

4. **Cache Dependencies**: Reduce startup time
   ```python
   environment={"PIP_CACHE_DIR": "/opt/ml/sagemaker/warmpoolcache/pip"}
   ```

## Deployment

### Build Docker Image
```bash
cd code/scripts
./build_and_push.sh medical-image-segmentation us-east-1
```

### Deploy to SageMaker Endpoint
```python
predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1
)
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Use gradient checkpointing
- Switch to FSDP for multi-GPU
- Use mixed precision (FP16)

### Slow Training
- Increase batch size
- Use multiple GPUs
- Enable mixed precision
- Optimize data loading (num_workers)

### Data Loading Issues
- Verify S3 paths and permissions
- Check data format (NIfTI expected)
- Ensure train/valid folder structure

## Dependencies

```
monai==1.3.2
torch==2.4.1
numpy==1.26.4
sagemaker>=2.200.0
tensorboard
mlflow==3.0.0
wandb
SimpleITK
nibabel
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License.

## Resources

- [MONAI Documentation](https://docs.monai.io/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [AWS SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [Medical Imaging on AWS](https://aws.amazon.com/health/solutions/medical-imaging/)

## Support

For issues and questions:
- Open a GitHub issue
- Check AWS SageMaker documentation
- Review CloudWatch logs for training jobs

## Acknowledgments

Built with:
- [MONAI](https://monai.io/) - Medical imaging framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - ML platform
