---
title: "Experiment Tracking"
weight: 33
---

## Overview

Track experiments systematically with Weights & Biases (WandB) to compare hyperparameters, visualize metrics, and collaborate with teams.

## Why Experiment Tracking?

Without tracking:
- ❌ Forget which hyperparameters were used
- ❌ Can't compare experiments easily
- ❌ Lose track of best models
- ❌ Difficult to reproduce results

With tracking:
- ✅ Automatic logging of hyperparameters
- ✅ Real-time metric visualization
- ✅ Compare multiple runs
- ✅ Model versioning and artifacts
- ✅ Team collaboration

## Tracking Tools Comparison

| Feature | TensorBoard | MLflow | WandB |
|---------|-------------|--------|-------|
| Setup | Easy | Medium | Easy |
| Visualization | Good | Basic | Excellent |
| Collaboration | Limited | Good | Excellent |
| Model Registry | No | Yes | Yes |
| Hyperparameter Tracking | Manual | Automatic | Automatic |
| Cloud Hosting | Self-hosted | Self-hosted | Free tier |
| Best For | Local dev | Enterprise | Teams |

## Lab: Integrate Weights & Biases

### Step 1: Setup WandB

```bash
# Install
pip install wandb

# Login (get API key from wandb.ai)
wandb login
```

### Step 2: Initialize WandB in Training Script

```python
import wandb
from monai.networks.nets import SegResNet

# Initialize WandB
wandb.init(
    project="medical-image-segmentation",
    name="segresnet-fsdp-4gpu",
    config={
        "model": "SegResNet",
        "epochs": 10,
        "batch_size": 2,
        "lr": 1e-4,
        "optimizer": "Adam",
        "instance_type": "ml.g5.12xlarge",
        "num_gpus": 4
    }
)

# Log model architecture
model = SegResNet(...)
wandb.watch(model, log="all", log_freq=10)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        # ... training code ...
        
        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
    
    # Validation
    dice_score = validate(model, val_loader)
    wandb.log({
        "val/dice": dice_score,
        "epoch": epoch
    })
    
    # Log images
    if epoch % 5 == 0:
        wandb.log({
            "predictions": wandb.Image(prediction_image),
            "ground_truth": wandb.Image(ground_truth_image)
        })

# Save model artifact
wandb.save("model.pth")
wandb.finish()
```

### Step 3: Pass WandB API Key to SageMaker

Open: `notebooks/segmentation/lab3_wandb_experiment_tracking.ipynb`

```python
import os
from sagemaker.pytorch import PyTorch

# Get WandB API key
wandb_api_key = os.environ.get('WANDB_API_KEY')  # Set in environment

estimator = PyTorch(
    entry_point='train_fsdp_wandb.py',
    source_dir='../sample-migration-of-training-medical-imaging-modes-from-ec2-to-sagemaker/medical-image-segmentation/code/training',
    role=role,
    instance_type='ml.g5.12xlarge',
    instance_count=1,
    framework_version='2.4.0',
    py_version='py311',
    distribution={'pytorchddp': {'enabled': True}},
    hyperparameters={
        'model_name': 'SegResNet',
        'epochs': 10,
        'batch_size': 2,
        'lr': 1e-4,
        'use_wandb': True,
        'wandb_project': 'medical-segmentation',
        'wandb_api_key': wandb_api_key
    }
)

estimator.fit({'training': f's3://{bucket}/segmentation/processed/'})
```

### Step 4: View Results in WandB Dashboard

Navigate to `https://wandb.ai/<username>/medical-image-segmentation`

You'll see:
- **Overview**: Summary of all runs
- **Charts**: Loss curves, Dice scores over time
- **System**: GPU utilization, memory usage
- **Logs**: Console output
- **Files**: Model artifacts

## Advanced WandB Features

### Hyperparameter Sweeps

Automatically search for best hyperparameters:

```python
# sweep_config.yaml
program: train_fsdp_wandb.py
method: bayes
metric:
  name: val/dice
  goal: maximize
parameters:
  lr:
    min: 0.00001
    max: 0.001
  batch_size:
    values: [1, 2, 4]
  dropout:
    min: 0.1
    max: 0.5
```

```python
# Run sweep
import wandb

sweep_id = wandb.sweep(sweep_config, project="medical-segmentation")
wandb.agent(sweep_id, function=train, count=10)
```

### Compare Experiments

```python
# In WandB dashboard, select multiple runs and click "Compare"
# Visualize:
# - Parallel coordinates plot
# - Scatter plots (lr vs dice score)
# - Table view of all hyperparameters
```

### Model Registry

```python
# Log model as artifact
artifact = wandb.Artifact('segresnet-best', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)

# Later, download best model
artifact = wandb.use_artifact('segresnet-best:latest')
artifact_dir = artifact.download()
```

## MLflow Integration

For enterprise environments, use MLflow:

```python
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("arn:aws:sagemaker:us-east-1:123456789:mlflow-tracking-server/mlflow")
mlflow.set_experiment("medical-segmentation")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model": "SegResNet",
        "epochs": 10,
        "batch_size": 2,
        "lr": 1e-4
    })
    
    # Training loop
    for epoch in range(epochs):
        # ... training ...
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        mlflow.log_metric("val_dice", dice_score, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

## Unified Tracking: TensorBoard + MLflow + WandB

Track with all three simultaneously:

```python
from torch.utils.tensorboard import SummaryWriter
import mlflow
import wandb

# Initialize all
writer = SummaryWriter()
mlflow.start_run()
wandb.init(project="medical-segmentation")

# Log to all
def log_metrics(metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)
        mlflow.log_metric(key, value, step=step)
        wandb.log({key: value, "step": step})

# Usage
log_metrics({"train/loss": loss.item(), "val/dice": dice_score}, epoch)
```

## Best Practices

### 1. Consistent Naming
```python
# Use hierarchical names
wandb.log({
    "train/loss": train_loss,
    "train/dice": train_dice,
    "val/loss": val_loss,
    "val/dice": val_dice,
    "system/gpu_memory": gpu_memory
})
```

### 2. Log System Metrics
```python
import psutil
import GPUtil

wandb.log({
    "system/cpu_percent": psutil.cpu_percent(),
    "system/memory_percent": psutil.virtual_memory().percent,
    "system/gpu_utilization": GPUtil.getGPUs()[0].load * 100
})
```

### 3. Save Checkpoints as Artifacts
```python
# Save best model
if dice_score > best_dice:
    best_dice = dice_score
    torch.save(model.state_dict(), "best_model.pth")
    wandb.save("best_model.pth")
```

### 4. Log Predictions
```python
# Visualize predictions periodically
if epoch % 5 == 0:
    fig = visualize_predictions(model, val_batch)
    wandb.log({"predictions": wandb.Image(fig)})
    plt.close(fig)
```

## Cost Considerations

### WandB Pricing
- **Free**: Unlimited runs, 100GB storage
- **Team**: $50/user/month, unlimited storage
- **Enterprise**: Custom pricing

### MLflow
- **Self-hosted**: EC2 costs (~$50/month for t3.medium)
- **SageMaker MLflow**: $1.50/hour when active

### TensorBoard
- **Free**: Self-hosted, no cloud costs
- **TensorBoard.dev**: Free public hosting

## Key Takeaways

✅ WandB provides best visualization and collaboration  
✅ MLflow better for enterprise model registry  
✅ TensorBoard good for local development  
✅ Can use all three simultaneously  

## Next Steps

Optimize hyperparameters automatically with SageMaker Hyperparameter Tuning.
