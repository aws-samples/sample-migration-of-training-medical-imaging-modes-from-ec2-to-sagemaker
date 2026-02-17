---
title: "Hyperparameter Optimization"
weight: 34
---

## Overview

Automate hyperparameter search with SageMaker Automatic Model Tuning to find optimal configurations without manual experimentation.

## Why Automated Tuning?

Manual tuning:
- ❌ Time-consuming (days/weeks)
- ❌ Requires ML expertise
- ❌ May miss optimal combinations
- ❌ Expensive (many failed experiments)

Automated tuning:
- ✅ Systematic search
- ✅ Bayesian optimization
- ✅ Parallel experiments
- ✅ Cost-effective

## Tuning Strategies

### 1. Grid Search
- Tests all combinations
- Exhaustive but expensive
- Best for: 2-3 hyperparameters

### 2. Random Search
- Random sampling
- Faster than grid search
- Best for: Initial exploration

### 3. Bayesian Optimization (Recommended)
- Learns from previous trials
- Focuses on promising regions
- Best for: Production tuning

## Lab: Hyperparameter Tuning

### Step 1: Define Hyperparameter Ranges

```python
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter
)

hyperparameter_ranges = {
    'lr': ContinuousParameter(1e-5, 1e-3, scaling_type='Logarithmic'),
    'batch_size': CategoricalParameter([1, 2, 4]),
    'dropout': ContinuousParameter(0.1, 0.5),
    'init_filters': CategoricalParameter([8, 16, 32])
}
```

### Step 2: Define Objective Metric

```python
# Metric to optimize (must be logged in training script)
objective_metric_name = 'val:dice'
objective_type = 'Maximize'  # or 'Minimize' for loss

# Training script must print:
# val:dice=0.8756
```

### Step 3: Create Base Estimator

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_fsdp.py',
    source_dir='../sample-migration-of-training-medical-imaging-modes-from-ec2-to-sagemaker/medical-image-segmentation/code/training',
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.4.0',
    py_version='py311',
    hyperparameters={
        'model_name': 'SegResNet',
        'epochs': 10,
        'val_interval': 2
    }
)
```

### Step 4: Create Tuning Job

Open: `notebooks/segmentation/lab5_hyperparameter_optimization.ipynb`

```python
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {'Name': 'val:dice', 'Regex': 'val:dice=([0-9\\.]+)'},
        {'Name': 'train:loss', 'Regex': 'train:loss=([0-9\\.]+)'}
    ],
    max_jobs=20,           # Total experiments
    max_parallel_jobs=4,   # Run 4 at a time
    objective_type=objective_type,
    strategy='Bayesian'    # Bayesian optimization
)

# Start tuning
tuner.fit({'training': f's3://{bucket}/segmentation/processed/'})
```

### Step 5: Monitor Tuning Job

```python
# Check status
tuner.describe()

# View best training job
best_job = tuner.best_training_job()
print(f"Best job: {best_job}")

# Get best hyperparameters
best_hyperparameters = tuner.best_estimator().hyperparameters()
print(f"Best hyperparameters: {best_hyperparameters}")
```

## Tuning Job Results

### Sample Output

```python
# After 20 trials:
Best Training Job: segmentation-tuning-job-020
Best Dice Score: 0.9123

Best Hyperparameters:
- lr: 0.000234
- batch_size: 2
- dropout: 0.23
- init_filters: 16

Improvement over baseline: +4.2%
```

### Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Get all training jobs
analytics = tuner.analytics()
df = analytics.dataframe()

# Plot learning rate vs dice score
plt.figure(figsize=(10, 6))
plt.scatter(df['lr'], df['val:dice'], alpha=0.6)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Dice Score')
plt.title('Learning Rate vs Dice Score')
plt.show()

# Plot parallel coordinates
from pandas.plotting import parallel_coordinates
parallel_coordinates(df, 'val:dice', colormap='viridis')
plt.show()
```

## Cost Optimization

### Strategy 1: Early Stopping

Stop poor-performing trials early:

```python
from sagemaker.tuner import EarlyStoppingType

tuner = HyperparameterTuner(
    estimator=estimator,
    # ... other parameters ...
    early_stopping_type=EarlyStoppingType.AUTO
)

# SageMaker stops trials that are unlikely to beat current best
# Saves ~30% on tuning costs
```

### Strategy 2: Warm Start

Reuse knowledge from previous tuning jobs:

```python
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

warm_start_config = WarmStartConfig(
    warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
    parents=['previous-tuning-job-name']
)

tuner = HyperparameterTuner(
    estimator=estimator,
    # ... other parameters ...
    warm_start_config=warm_start_config
)
```

### Strategy 3: Spot Instances

Use spot instances for tuning:

```python
estimator = PyTorch(
    # ... other parameters ...
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)

# Savings: Up to 70% on compute costs
```

### Cost Comparison

**Manual Tuning:**
- 20 experiments × 30 min × $1.41/hr = **$14.10**
- Time: 10 hours (sequential)

**Automated Tuning (4 parallel):**
- 20 experiments × 30 min × $1.41/hr = **$14.10**
- Time: 2.5 hours (parallel)
- **Speedup: 4x**

**With Spot + Early Stopping:**
- ~12 experiments (early stopping) × 30 min × $0.42/hr (spot) = **$2.52**
- **Savings: 82%**

## Advanced Tuning

### Multi-Objective Optimization

Optimize for multiple metrics:

```python
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='val:dice',
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {'Name': 'val:dice', 'Regex': 'val:dice=([0-9\\.]+)'},
        {'Name': 'inference:latency', 'Regex': 'inference:latency=([0-9\\.]+)'}
    ],
    # Optimize for both accuracy and speed
)
```

### Conditional Hyperparameters

Some hyperparameters depend on others:

```python
# Example: Only tune dropout if using it
hyperparameter_ranges = {
    'use_dropout': CategoricalParameter(['True', 'False']),
    'dropout': ContinuousParameter(0.1, 0.5),  # Only used if use_dropout=True
}

# Handle in training script:
if args.use_dropout == 'True':
    model = SegResNet(dropout_prob=args.dropout)
else:
    model = SegResNet(dropout_prob=0.0)
```

### Transfer Learning Tuning

Fine-tune hyperparameters for transfer learning:

```python
hyperparameter_ranges = {
    'freeze_encoder': CategoricalParameter(['True', 'False']),
    'encoder_lr': ContinuousParameter(1e-6, 1e-4),
    'decoder_lr': ContinuousParameter(1e-5, 1e-3)
}
```

## Best Practices

### 1. Start Small
```python
# Initial exploration: 10 jobs, wide ranges
tuner_v1 = HyperparameterTuner(
    max_jobs=10,
    hyperparameter_ranges={
        'lr': ContinuousParameter(1e-5, 1e-2)
    }
)

# Refinement: 20 jobs, narrow ranges
tuner_v2 = HyperparameterTuner(
    max_jobs=20,
    hyperparameter_ranges={
        'lr': ContinuousParameter(1e-4, 5e-4)  # Narrowed based on v1
    }
)
```

### 2. Log All Metrics
```python
# In training script, log everything
print(f"val:dice={dice_score:.4f}")
print(f"val:loss={val_loss:.4f}")
print(f"train:loss={train_loss:.4f}")
print(f"inference:time={inference_time:.4f}")
```

### 3. Use Checkpointing
```python
# Save checkpoints for long training jobs
if epoch % 5 == 0:
    checkpoint_path = f"/opt/ml/checkpoints/checkpoint-epoch-{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
```

### 4. Validate Ranges
```python
# Test extreme values manually first
test_configs = [
    {'lr': 1e-5, 'batch_size': 1},  # Minimum
    {'lr': 1e-3, 'batch_size': 4},  # Maximum
]

for config in test_configs:
    # Run quick test to ensure no crashes
    pass
```

## Interpreting Results

### Convergence Plot

```python
# Plot best objective over time
best_scores = []
for i in range(len(df)):
    best_scores.append(df['val:dice'][:i+1].max())

plt.plot(best_scores)
plt.xlabel('Trial Number')
plt.ylabel('Best Dice Score')
plt.title('Tuning Convergence')
plt.show()
```

### Hyperparameter Importance

```python
# Correlation between hyperparameters and objective
correlations = df[['lr', 'batch_size', 'dropout', 'val:dice']].corr()['val:dice']
print(correlations.sort_values(ascending=False))

# Output:
# val:dice        1.000000
# lr              0.456789  # Strong positive correlation
# dropout        -0.234567  # Weak negative correlation
# batch_size      0.123456  # Weak positive correlation
```

## Key Takeaways

✅ Bayesian optimization finds optimal hyperparameters  
✅ Parallel jobs reduce tuning time  
✅ Early stopping and spot instances reduce costs  
✅ Systematic approach beats manual tuning  

## Next Steps

Deploy your best model to production with SageMaker endpoints.
