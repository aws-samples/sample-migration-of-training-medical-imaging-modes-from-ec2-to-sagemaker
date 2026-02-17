---
title: "Single GPU Training"
weight: 31
---

## Overview

Train 3D medical image segmentation models using MONAI on a single GPU. This establishes baseline performance before scaling to multi-GPU.

## Why MONAI?

MONAI (Medical Open Network for AI) is purpose-built for medical imaging:

- **Domain-specific transforms**: Medical image preprocessing
- **3D architectures**: SegResNet, UNet, SwinUNETR
- **Metrics**: Dice score, Hausdorff distance
- **Optimized**: Fast 3D convolutions and data loading

## Lab: Train SegResNet

### Step 1: Review Model Architecture

```python
from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=3,        # 3D volumes
    in_channels=1,         # Single modality (CT or MRI)
    out_channels=1,        # Binary segmentation
    init_filters=16,       # Base number of filters
    dropout_prob=0.2       # Regularization
)

# Model size: ~5M parameters
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 2: Data Loading

```python
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Resized, ScaleIntensityRanged, RandFlipd
)

# Define transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64)),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,  # CT Hounsfield units
        b_min=0, b_max=1
    ),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0)
])

# Create dataset
train_files = [
    {"image": "s3://bucket/train/subj_001/img.nii.gz",
     "mask": "s3://bucket/train/subj_001/label.nii.gz"},
    # ...
]

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
```

### Step 3: Training Loop

Open: `notebooks/segmentation/lab1_single_gpu_training.ipynb`

```python
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda")
model = SegResNet(...).to(device)

# Loss and metrics
loss_function = DiceLoss(sigmoid=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TensorBoard
writer = SummaryWriter()

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch["image"].to(device)
            val_masks = val_batch["mask"].to(device)
            val_outputs = model(val_images)
            dice_metric(y_pred=val_outputs, y=val_masks)
    
    # Log metrics
    mean_dice = dice_metric.aggregate().item()
    writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)
    writer.add_scalar("Dice/val", mean_dice, epoch)
    
    print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Dice={mean_dice:.4f}")
```

### Step 4: Run on SageMaker

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_simple.py',
    source_dir='../sample-migration-of-training-medical-imaging-modes-from-ec2-to-sagemaker/medical-image-segmentation/code/training',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.4.0',
    py_version='py311',
    hyperparameters={
        'model_name': 'SegResNet',
        'epochs': 10,
        'batch_size': 2,
        'lr': 1e-4,
        'val_interval': 2
    }
)

estimator.fit({'training': f's3://{bucket}/segmentation/processed/'})
```

### Step 5: Monitor with TensorBoard

```python
# Download TensorBoard logs
!aws s3 sync {estimator.model_data.replace('model.tar.gz', 'tensorboard/')} ./logs/

# Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./logs/
```

## Training Results

### Expected Performance

| Metric | Value |
|--------|-------|
| Training Time | ~30 min/epoch |
| Final Dice Score | 0.85-0.90 |
| GPU Memory | ~4 GB |
| GPU Utilization | ~70% |

### Sample Output

```
Epoch 1/10: Loss=0.4523, Dice=0.6234
Epoch 2/10: Loss=0.3456, Dice=0.7123
Epoch 3/10: Loss=0.2789, Dice=0.7845
...
Epoch 10/10: Loss=0.1234, Dice=0.8756
```

## Dice Score Explained

Dice coefficient measures overlap between prediction and ground truth:

```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```

- **0.0**: No overlap
- **0.5**: Moderate overlap
- **0.8+**: Good segmentation
- **1.0**: Perfect overlap

## Visualization

```python
import matplotlib.pyplot as plt
from monai.visualize import plot_2d_or_3d_image

# Visualize predictions
model.eval()
with torch.no_grad():
    test_input = test_batch["image"].to(device)
    test_output = model(test_input)
    
    # Plot middle slice
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_input[0, 0, :, :, 32].cpu(), cmap='gray')
    plt.title('Input Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(test_batch["mask"][0, 0, :, :, 32], cmap='gray')
    plt.title('Ground Truth')
    
    plt.subplot(1, 3, 3)
    plt.imshow(test_output[0, 0, :, :, 32].cpu() > 0.5, cmap='gray')
    plt.title('Prediction')
    plt.show()
```

## Optimization Tips

### Memory Optimization

If you encounter OOM errors:

```python
# 1. Reduce batch size
batch_size = 1

# 2. Reduce spatial size
spatial_size = (96, 96, 48)  # Instead of (128, 128, 64)

# 3. Enable gradient checkpointing
from torch.utils.checkpoint import checkpoint
```

### Speed Optimization

```python
# 1. Increase num_workers
train_loader = DataLoader(..., num_workers=8)

# 2. Enable mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = loss_function(outputs, masks)
```

## Key Takeaways

✅ MONAI simplifies medical image segmentation  
✅ Single GPU training establishes baseline  
✅ TensorBoard provides real-time monitoring  
✅ Dice score is the primary metric  

## Next Steps

Scale to multi-GPU training with FSDP for faster training and larger models.
