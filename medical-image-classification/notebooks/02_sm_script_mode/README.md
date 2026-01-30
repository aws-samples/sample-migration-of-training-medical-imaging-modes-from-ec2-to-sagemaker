# SageMaker Script Mode Training

## Overview
This notebook demonstrates training a medical image classification model using Amazon SageMaker Script Mode with PyTorch.

## What is Script Mode?
Script Mode allows you to use your own training scripts with minimal modifications while leveraging SageMaker's managed infrastructure for:
- Automatic provisioning and scaling
- Distributed training
- Model artifact management
- Integration with other AWS services

## Files in this Directory
- `train.ipynb` - Main training notebook
- `code/train.py` - Training script executed by SageMaker
- `code/model_def.py` - Model architecture definitions
- `code/utils.py` - Utility functions
- `code/label_dict.json` - Class label mappings
- `requirements.txt` - Python dependencies

## Prerequisites
1. AWS account with SageMaker access
2. Training data in S3 (organized by class)
3. IAM role with SageMaker permissions
4. SageMaker Python SDK installed

## Key Concepts

### Training Script Structure
Your training script (`code/train.py`) should:
- Parse command-line arguments for hyperparameters
- Load data from `/opt/ml/input/data/`
- Save model to `/opt/ml/model/`
- Use environment variables for distributed training

### Data Channels
SageMaker provides data through channels:
- `train` → `/opt/ml/input/data/train/`
- `test` → `/opt/ml/input/data/test/`

### Model Artifacts
Trained models are automatically uploaded to S3 from `/opt/ml/model/`

## Usage

1. **Prepare your data**: Upload to S3 in the correct structure
2. **Configure hyperparameters**: Edit the hyperparameters dictionary
3. **Run the notebook**: Execute cells sequentially
4. **Monitor training**: View logs in CloudWatch or notebook output
5. **Retrieve model**: Model artifacts saved to S3

## Cost Optimization Tips
- Use `keep_alive_period_in_seconds` for iterative development
- Choose appropriate instance types (ml.g5.xlarge for GPU)
- Use Spot instances for cost savings (add `use_spot_instances=True`)
- Clean up endpoints after testing

## Troubleshooting

### Common Issues
1. **Permission errors**: Verify IAM role has S3 and SageMaker permissions
2. **Data not found**: Check S3 paths and bucket permissions
3. **Out of memory**: Reduce batch size or use larger instance
4. **Training fails**: Check CloudWatch logs for detailed errors

### Debugging
- Add print statements in your training script
- Use `wait=False` in `estimator.fit()` to continue notebook execution
- Check `/opt/ml/output/failure` for error details

## Next Steps
- Experiment with different hyperparameters
- Try distributed training with `instance_count > 1`
- Deploy model using SageMaker endpoints
- Set up hyperparameter tuning jobs

## Resources
- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/)
- [PyTorch on SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html)
- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)
