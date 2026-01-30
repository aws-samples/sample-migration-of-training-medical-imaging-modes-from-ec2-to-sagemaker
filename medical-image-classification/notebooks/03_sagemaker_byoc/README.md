# SageMaker BYOC (Bring Your Own Container)

## Overview
This directory contains resources for training medical image classification models using custom Docker containers on Amazon SageMaker.

## Why Use BYOC?

### Advantages
- **Full Control**: Complete control over the training environment
- **Custom Dependencies**: Use any library versions you need
- **Reproducibility**: Exact same environment everywhere
- **Flexibility**: Support for any ML framework or custom code

### When to Use BYOC
- You need specific library versions not available in managed containers
- You have proprietary code or dependencies
- You want to replicate your local development environment
- You need specialized ML frameworks or tools

## Files in this Directory
- `train.ipynb` - Training notebook with custom container
- `Dockerfile` - Container definition
- `build_and_push.sh` - Script to build and push to ECR
- `src/` - Training code directory
  - `train.py` - Main training script
  - `model_def.py` - Model architectures
  - `utils.py` - Helper functions
  - `requirements.txt` - Python dependencies

## Container Requirements

### SageMaker Contract
Your container must:
1. Accept hyperparameters as command-line arguments
2. Read training data from `/opt/ml/input/data/<channel>/`
3. Save model artifacts to `/opt/ml/model/`
4. Write output to `/opt/ml/output/`
5. Exit with code 0 on success, non-zero on failure

### Directory Structure
```
/opt/ml/
├── input/
│   ├── config/
│   │   ├── hyperparameters.json
│   │   └── resourceconfig.json
│   └── data/
│       ├── train/
│       └── test/
├── model/          # Save your model here
├── output/         # Logs and failure info
└── code/           # Your training code
```

### Environment Variables
SageMaker provides:
- `SM_MODEL_DIR`: Where to save model
- `SM_CHANNEL_TRAIN`: Training data location
- `SM_CHANNEL_TEST`: Test data location
- `SM_NUM_GPUS`: Number of GPUs available
- `SM_HOSTS`: List of hosts in distributed training

## Setup Instructions

### 1. Build Docker Image
```bash
cd /path/to/03_sagemaker_byoc/
docker build -t sm-training-byoc:latest .
```

### 2. Test Locally (Optional)
```bash
docker run --rm \
  -v $(pwd)/data:/opt/ml/input/data \
  -v $(pwd)/model:/opt/ml/model \
  sm-training-byoc:latest \
  --epochs 1 --batch-size 16
```

### 3. Push to Amazon ECR
```bash
# Make script executable
chmod +x build_and_push.sh

# Run build and push script
./build_and_push.sh <your-account-id> <region>
```

The script will:
- Authenticate with ECR
- Create repository if needed
- Tag and push your image

### 4. Update Notebook
Update the `image_uri` in `train.ipynb`:
```python
image_uri = '<account-id>.dkr.ecr.<region>.amazonaws.com/sm-training-byoc:latest'
```

## Dockerfile Best Practices

### Base Image
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```
Choose a base image that matches your requirements.

### Install Dependencies
```dockerfile
COPY src/requirements.txt /opt/ml/code/
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt
```

### Copy Training Code
```dockerfile
COPY src/ /opt/ml/code/
WORKDIR /opt/ml/code
```

### Set Entry Point
```dockerfile
ENTRYPOINT ["python", "train.py"]
```

## Troubleshooting

### Common Issues

1. **Container fails to start**
   - Check CloudWatch logs
   - Verify ENTRYPOINT is correct
   - Test container locally first

2. **Permission errors**
   - Ensure IAM role has ECR permissions
   - Check S3 bucket permissions
   - Verify container runs as correct user

3. **Out of memory**
   - Reduce batch size
   - Use larger instance type
   - Optimize data loading

4. **Data not found**
   - Verify S3 paths are correct
   - Check data channel names match
   - Ensure data is in correct format

### Debugging Tips

1. **Add verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Test locally first:**
   ```bash
   docker run -it --entrypoint /bin/bash sm-training-byoc:latest
   ```

3. **Check CloudWatch Logs:**
   - Navigate to SageMaker console
   - Find your training job
   - View logs in CloudWatch

4. **Inspect container:**
   ```bash
   docker run --rm -it sm-training-byoc:latest /bin/bash
   ```

## Cost Optimization

### Tips
- Use appropriate instance types (don't over-provision)
- Enable Spot instances for cost savings
- Use `keep_alive_period_in_seconds` for iterative development
- Clean up unused ECR images
- Stop training jobs that aren't progressing

### Spot Instances
```python
estimator = PyTorch(
    ...
    use_spot_instances=True,
    max_wait=7200,  # Maximum time to wait for spot
    max_run=3600,   # Maximum training time
)
```

## Advanced Topics

### Distributed Training
For multi-GPU or multi-node training:
```python
estimator = PyTorch(
    ...
    instance_count=2,
    distribution={'pytorchddp': {'enabled': True}}
)
```

### Custom Metrics
Log metrics that SageMaker can track:
```python
print(f"[epoch {epoch}] train_loss: {loss:.4f}")
```

### Model Registry
Save model with metadata:
```python
import json
model_info = {
    'model_name': 'densenet121',
    'accuracy': 0.95,
    'framework': 'pytorch'
}
with open('/opt/ml/model/model_info.json', 'w') as f:
    json.dump(model_info, f)
```

## Next Steps
- Implement distributed training
- Add custom metrics and logging
- Set up CI/CD for container builds
- Deploy model to SageMaker endpoints
- Integrate with MLflow for experiment tracking

## Resources
- [SageMaker Containers](https://github.com/aws/sagemaker-containers)
- [Docker Documentation](https://docs.docker.com/)
- [Amazon ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [SageMaker Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)
