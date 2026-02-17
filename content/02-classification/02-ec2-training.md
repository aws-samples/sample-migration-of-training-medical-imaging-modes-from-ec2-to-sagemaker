---
title: "EC2 Baseline Training"
weight: 22
---

## Overview

Establish baseline performance by training a classification model on EC2. This provides a reference point for comparing SageMaker approaches.

## Why Start with EC2?

- **Familiar environment**: Standard PyTorch training loop
- **Full control**: Direct access to GPU and filesystem
- **Interactive**: Easy debugging with Jupyter
- **Baseline metrics**: Reference for optimization

## Lab: Train DenseNet121 on EC2

### Step 1: Launch EC2 Instance (Optional)

If not using SageMaker notebook with GPU:

```bash
# Launch g5.xlarge instance with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g5.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx
```

### Step 2: Review Training Script

Open: `notebooks/classification/02-ec2-training.ipynb`

Key components:

```python
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity

# Model definition
model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=6
).cuda()

# Data loading
train_transforms = Compose([
    LoadImage(image_only=True),
    Resize((64, 64)),
    ScaleIntensity()
])

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
        images, labels = batch[0].cuda(), batch[1].cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Step 3: Run Training

```python
# Download data from S3
!aws s3 sync s3://{bucket}/classification/processed/ ./data/

# Run training
%run train.py --data ./data --epochs 10 --batch_size 32
```

### Step 4: Monitor Training

```python
# Training output
Epoch 1/10: Loss=1.234, Acc=0.456
Epoch 2/10: Loss=0.987, Acc=0.623
Epoch 3/10: Loss=0.765, Acc=0.734
...
Epoch 10/10: Loss=0.234, Acc=0.912

# Validation metrics
Val Loss: 0.345
Val Accuracy: 0.887
Val AUC: 0.923
```

### Step 5: Save Model

```python
# Save model locally
torch.save(model.state_dict(), 'model.pth')

# Upload to S3
!aws s3 cp model.pth s3://{bucket}/models/ec2-baseline/
```

## Training Results

### Expected Performance

| Metric | Value |
|--------|-------|
| Training Time | ~10 minutes |
| Final Train Acc | ~91% |
| Final Val Acc | ~88% |
| Val AUC | ~0.92 |
| GPU Utilization | ~60% |

### Resource Usage

```python
# Monitor GPU
!nvidia-smi

# Output:
# GPU 0: A10G (24GB)
# Memory Used: 4.2 GB / 24 GB
# GPU Utilization: 58%
```

## Challenges with EC2 Training

### 1. Manual Infrastructure Management
- Must provision and terminate instances
- Configure security groups and networking
- Install dependencies manually

### 2. No Built-in Experiment Tracking
- Manual logging of hyperparameters
- No automatic metric comparison
- Difficult to reproduce experiments

### 3. Limited Scalability
- Manual setup for multi-GPU training
- No automatic distributed training
- Difficult to run parallel experiments

### 4. Cost Management
- Easy to forget running instances
- No automatic spot instance handling
- Pay for idle time

## Comparison: EC2 vs SageMaker

| Aspect | EC2 | SageMaker |
|--------|-----|-----------|
| Setup Time | 15-30 min | 2-5 min |
| Infrastructure | Manual | Automatic |
| Distributed Training | Manual | Built-in |
| Experiment Tracking | Manual | Built-in |
| Cost | Pay for uptime | Pay per job |
| Spot Instances | Manual | Automatic |

## Key Takeaways

✅ EC2 provides full control and familiar environment  
✅ Good for prototyping and baseline experiments  
⚠️ Requires manual infrastructure management  
⚠️ Limited scalability without additional setup  

## Next Steps

Now that you have a baseline, migrate to SageMaker Script Mode for managed training infrastructure.
