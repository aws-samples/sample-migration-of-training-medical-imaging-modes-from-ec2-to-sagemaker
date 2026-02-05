# SageMaker BYOC with MLflow

## Overview
This directory demonstrates training medical image classification models using a custom Docker container with MLflow integration on Amazon SageMaker.

## What's Different?
This builds on the BYOC approach by adding:
- **MLflow Integration**: Automatic experiment tracking
- **Metric Logging**: Track training metrics over time
- **Model Versioning**: Automatic model versioning with MLflow
- **Artifact Management**: Store models and metadata together

## Files
- `train.ipynb` - Training notebook with MLflow
- `Dockerfile` - Container with MLflow dependencies
- `build_and_push.sh` - Build and push script
- `src/train_mlflow.py` - Training script with MLflow tracking
- `src/requirements.txt` - Dependencies including MLflow

## MLflow Benefits

### Experiment Tracking
- Automatically log hyperparameters
- Track metrics (loss, accuracy) over time
- Compare multiple training runs
- Visualize training progress

### Model Registry
- Version control for models
- Track model lineage
- Store model metadata
- Manage model lifecycle

## Setup

### 1. Build Container with MLflow
```bash
cd /path/to/04_sagemaker_byoc_mlflow/
docker build -t sm-training-byoc-mlflow:latest .
```

### 2. Push to ECR
```bash
./build_and_push.sh <account-id> <region>
```

### 3. Configure MLflow (Optional)
Set up MLflow tracking server:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

## Usage in Training Script

### Log Parameters
```python
import mlflow

mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
mlflow.log_param("model_name", "DenseNet121")
```

### Log Metrics
```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_acc = validate()
    
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
```

### Log Model
```python
mlflow.pytorch.log_model(model, "model")
```

### Log Artifacts
```python
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("training_curves.png")
```

## Viewing Results

### Local MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns
```
Access at http://localhost:5000

### Remote MLflow Server
Configure tracking URI to point to your MLflow server.

## Next Steps
- Set up MLflow tracking server on EC2 or ECS
- Integrate with model registry
- Automate model deployment based on metrics
- Set up experiment comparison dashboards

## Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow on AWS](https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/)
