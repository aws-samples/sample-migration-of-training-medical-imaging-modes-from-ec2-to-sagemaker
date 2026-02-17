---
title: "Multi-GPU FSDP Training"
weight: 32
---

## Overview

Scale training to multiple GPUs using Fully Sharded Data Parallel (FSDP). Train larger models faster with efficient memory usage.

## FSDP vs DDP

### Data Distributed Parallel (DDP)
- Replicates model on each GPU
- Each GPU processes different data
- Memory: Full model per GPU
- Best for: Models that fit in single GPU memory

### Fully Sharded Data Parallel (FSDP)
- Shards model parameters across GPUs
- Each GPU holds only a portion of the model
- Memory: Model size / num_GPUs
- Best for: Large models (SwinUNETR, large batch sizes)

## Lab: Train with FSDP

### Step 1: Understand FSDP Architecture

```python
# Without FSDP (DDP)
# GPU 0: Full Model (20GB) + Data Batch 1
# GPU 1: Full Model (20GB) + Data Batch 2
# GPU 2: Full Model (20GB) + Data Batch 3
# GPU 3: Full Model (20GB) + Data Batch 4
# Total Memory: 80GB

# With FSDP
# GPU 0: Model Shard 1 (5GB) + Data Batch 1
# GPU 1: Model Shard 2 (5GB) + Data Batch 2
# GPU 2: Model Shard 3 (5GB) + Data Batch 3
# GPU 3: Model Shard 4 (5GB) + Data Batch 4
# Total Memory: 20GB (can train larger models!)
```

### Step 2: Review FSDP Training Script

Key differences from single GPU:

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def setup():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_fsdp(rank, world_size):
    setup()
    
    # Create model and wrap with FSDP
    model = SegResNet(...)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank
    )
    
    # Distributed sampler ensures each GPU gets different data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        for batch in train_loader:
            images = batch["image"].cuda()
            masks = batch["mask"].cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Only rank 0 logs metrics
        if rank == 0:
            print(f"Epoch {epoch}: Loss={loss.item()}")
    
    cleanup()
```

### Step 3: Launch Multi-GPU Training

Open: `notebooks/segmentation/lab2_fsdp_multi_gpu.ipynb`

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_fsdp.py',
    source_dir='../sample-migration-of-training-medical-imaging-modes-from-ec2-to-sagemaker/medical-image-segmentation/code/training',
    role=role,
    instance_type='ml.g5.12xlarge',  # 4 GPUs
    instance_count=1,
    framework_version='2.4.0',
    py_version='py311',
    distribution={
        'pytorchddp': {
            'enabled': True
        }
    },
    hyperparameters={
        'model_name': 'SegResNet',
        'epochs': 10,
        'batch_size': 2,  # Per GPU
        'lr': 1e-4
    }
)

estimator.fit({'training': f's3://{bucket}/segmentation/processed/'})
```

### Step 4: Monitor Multi-GPU Training

```python
# View logs from all GPUs
estimator.logs()

# Output shows logs from each GPU:
# [algo-1-gpu-0] Epoch 1: Loss=0.4523
# [algo-1-gpu-1] Epoch 1: Loss=0.4501
# [algo-1-gpu-2] Epoch 1: Loss=0.4534
# [algo-1-gpu-3] Epoch 1: Loss=0.4512
```

## Performance Comparison

### Single GPU (ml.g5.xlarge)
- **GPUs**: 1
- **Batch size**: 2
- **Time per epoch**: ~30 minutes
- **Cost per epoch**: $0.71
- **Total (10 epochs)**: $7.10

### Multi-GPU FSDP (ml.g5.12xlarge)
- **GPUs**: 4
- **Batch size**: 8 (2 per GPU)
- **Time per epoch**: ~10 minutes
- **Cost per epoch**: $1.18
- **Total (10 epochs)**: $11.80

**Speedup**: 3x faster  
**Cost increase**: 66% more  
**Cost per sample**: 45% cheaper

## Training Larger Models

FSDP enables training models that don't fit on a single GPU:

```python
from monai.networks.nets import SwinUNETR

# SwinUNETR: ~62M parameters, ~20GB memory
model = SwinUNETR(
    img_size=(128, 128, 64),
    in_channels=1,
    out_channels=1,
    feature_size=48,  # Larger features
    use_checkpoint=True  # Gradient checkpointing
)

# Wrap with FSDP
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# Now fits on 4 GPUs with ~5GB per GPU
```

## FSDP Sharding Strategies

```python
from torch.distributed.fsdp import ShardingStrategy

# 1. FULL_SHARD (default)
# - Shards parameters, gradients, and optimizer states
# - Maximum memory savings
# - Slightly slower communication

# 2. SHARD_GRAD_OP
# - Shards gradients and optimizer states only
# - Parameters replicated
# - Faster but uses more memory

# 3. NO_SHARD (same as DDP)
# - No sharding
# - Fastest but highest memory usage

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD  # Choose strategy
)
```

## Debugging Multi-GPU Training

### Check GPU Utilization

```python
# Add to training script
import subprocess

if rank == 0:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
```

### Verify Data Distribution

```python
# Ensure each GPU processes different data
print(f"Rank {rank}: Processing batch indices {train_sampler.indices[:10]}")

# Output:
# Rank 0: Processing batch indices [0, 4, 8, 12, ...]
# Rank 1: Processing batch indices [1, 5, 9, 13, ...]
# Rank 2: Processing batch indices [2, 6, 10, 14, ...]
# Rank 3: Processing batch indices [3, 7, 11, 15, ...]
```

### Synchronize Metrics

```python
# Aggregate metrics across GPUs
def reduce_metric(metric, world_size):
    """Average metric across all GPUs"""
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    return metric / world_size

# Usage
loss_tensor = torch.tensor(loss.item()).cuda()
avg_loss = reduce_metric(loss_tensor, world_size)

if rank == 0:
    print(f"Average loss across GPUs: {avg_loss.item()}")
```

## Best Practices

### 1. Batch Size Scaling
```python
# Scale batch size with number of GPUs
batch_size_per_gpu = 2
total_batch_size = batch_size_per_gpu * num_gpus

# Adjust learning rate accordingly
lr = base_lr * (total_batch_size / base_batch_size)
```

### 2. Gradient Accumulation
```python
# Simulate larger batch sizes
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    outputs = model(batch["image"])
    loss = loss_function(outputs, batch["mask"]) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Checkpointing
```python
# Save checkpoints from rank 0 only
if rank == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f'checkpoint_epoch_{epoch}.pth')
```

## Key Takeaways

✅ FSDP enables training larger models  
✅ 3x speedup with 4 GPUs  
✅ Memory efficient: model sharded across GPUs  
✅ Minimal code changes from single GPU  

## Next Steps

Add comprehensive experiment tracking with Weights & Biases to compare training runs.
