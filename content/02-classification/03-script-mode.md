---
title: "SageMaker Script Mode"
weight: 23
---

## Overview

Migrate your EC2 training to SageMaker Script Mode with minimal code changes. SageMaker handles infrastructure, monitoring, and artifact management automatically.

## What is Script Mode?

Script Mode lets you use your existing PyTorch training scripts with managed SageMaker infrastructure:

- **Minimal changes**: Add environment variable handling
- **Managed containers**: Use AWS-maintained PyTorch images
- **Automatic scaling**: Multi-GPU and multi-node support
- **Built-in monitoring**: CloudWatch metrics and logs

## Lab: Train with SageMaker Script Mode

### Step 1: Adapt Training Script

Modify your EC2 script to use SageMaker environment variables:

```python
# train.py
import os
import argparse
import torch
from monai.networks.nets import DenseNet121

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL', './data/val'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    return parser.parse_args()

def train(args):
    # Model
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6).cuda()
    
    # Training loop (same as EC2)
    # ...
    
    # Save model to SageMaker model directory
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__ == '__main__':
    args = parse_args()
    train(args)
```

**Key Changes:**
- Use `SM_MODEL_DIR` for model output
- Use `SM_CHANNEL_*` for input data paths
- Accept hyperparameters as arguments

### Step 2: Create SageMaker Estimator

Open: `notebooks/classification/03-sagemaker-script-mode.ipynb`

```python
from sagemaker.pytorch import PyTorch

# Define estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='code',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.4.0',
    py_version='py311',
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-4
    }
)

# Start training
estimator.fit({
    'train': f's3://{bucket}/classification/processed/train/',
    'val': f's3://{bucket}/classification/processed/val/'
})
```

### Step 3: Monitor Training

```python
# View real-time logs
estimator.logs()

# Check training job status
estimator.latest_training_job.describe()

# Output:
# TrainingJobStatus: InProgress
# SecondaryStatus: Training
# TrainingStartTime: 2026-02-15 18:30:00
```

### Step 4: Retrieve Model Artifacts

```python
# Model automatically saved to S3
model_data = estimator.model_data
print(f"Model artifacts: {model_data}")

# Download model
!aws s3 cp {model_data} ./model.tar.gz
!tar -xzf model.tar.gz
```

## SageMaker Environment Variables

SageMaker provides these environment variables to your script:

| Variable | Description | Example |
|----------|-------------|---------|
| `SM_MODEL_DIR` | Model output directory | `/opt/ml/model` |
| `SM_CHANNEL_TRAIN` | Training data path | `/opt/ml/input/data/train` |
| `SM_CHANNEL_VAL` | Validation data path | `/opt/ml/input/data/val` |
| `SM_NUM_GPUS` | Number of GPUs | `1` |
| `SM_HOSTS` | List of hosts | `['algo-1']` |
| `SM_CURRENT_HOST` | Current host | `algo-1` |

## Advanced Features

### Multi-GPU Training

Enable distributed data parallel automatically:

```python
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.g5.12xlarge',  # 4 GPUs
    instance_count=1,
    distribution={'pytorchddp': {'enabled': True}}
)
```

Your script automatically uses all GPUs with minimal changes:

```python
import torch.distributed as dist

# SageMaker sets up distributed training
if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
```

### Spot Instances

Save up to 70% with managed spot instances:

```python
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.g5.xlarge',
    use_spot_instances=True,
    max_wait=7200,  # Max wait time in seconds
    max_run=3600    # Max training time
)
```

SageMaker automatically:
- Handles spot interruptions
- Resumes training from checkpoints
- Falls back to on-demand if needed

### Warm Pools

Reduce startup time for iterative experiments:

```python
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.g5.xlarge',
    keep_alive_period_in_seconds=1800  # Keep instance alive for 30 min
)
```

## Cost Comparison

### EC2 Training
- Instance: g5.xlarge
- Training time: 10 minutes
- Idle time: 50 minutes (forgot to terminate)
- **Cost**: $1.41/hr × 1 hr = **$1.41**

### SageMaker Script Mode
- Instance: ml.g5.xlarge
- Training time: 10 minutes
- Idle time: 0 (automatic termination)
- **Cost**: $1.41/hr × 0.167 hr = **$0.24**

**Savings: 83%** (no idle time)

## Monitoring and Debugging

### CloudWatch Metrics

Automatic metrics:
- GPU utilization
- Memory usage
- Disk I/O
- Network throughput

```python
# View metrics in CloudWatch
import boto3

cw = boto3.client('cloudwatch')
metrics = cw.get_metric_statistics(
    Namespace='/aws/sagemaker/TrainingJobs',
    MetricName='GPUUtilization',
    Dimensions=[{'Name': 'TrainingJobName', 'Value': estimator.latest_training_job.name}],
    StartTime=start_time,
    EndTime=end_time,
    Period=60,
    Statistics=['Average']
)
```

### CloudWatch Logs

```python
# Stream logs
estimator.logs()

# Or view in console:
# CloudWatch > Log Groups > /aws/sagemaker/TrainingJobs
```

## Key Takeaways

✅ Minimal code changes from EC2  
✅ Automatic infrastructure management  
✅ Built-in distributed training support  
✅ Cost savings with automatic termination  
✅ Integrated monitoring and logging  

## Next Steps

For even more control, build custom Docker containers with SageMaker BYOC.
