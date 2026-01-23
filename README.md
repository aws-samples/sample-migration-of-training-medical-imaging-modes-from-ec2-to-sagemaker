# Medical Image Classification Pipeline with AWS Integration

A comprehensive medical image classification system that leverages deep learning models (DenseNet121 and Vision Transformer) to analyze and classify medical images, with seamless integration across AWS services including Amazon Elastic Compute Cloud (Amazon EC2), Amazon SageMaker, and Amazon Elastic Container Registry (Amazon ECR).

This project provides an end-to-end solution for medical image classification, supporting both local development on EC2 instances and production deployment through Amazon SageMaker. The system includes data preprocessing, model training, evaluation, and visualization components, with support for multiple deep learning architectures optimized for medical imaging tasks.

Key features include:
- Flexible model architecture support (DenseNet121 and Vision Transformer)
- Automated data preprocessing and splitting
- Comprehensive evaluation metrics including ROC curves and confusion matrices
- Multiple deployment options (EC2, SageMaker Script Mode, SageMaker Custom Container)
- GPU acceleration support for training
- Integration with AWS services for scalable deployment

## Repository Structure
```
medical-image-classification/
├── ec2/                           # EC2 instance training components
│   ├── train.py                  # Main training script
│   ├── model_def.py             # Model architecture definitions
│   └── generate_roc_curve.py    # Evaluation metrics generation
├── sm_byoc/                      # SageMaker Bring Your Own Container
│   ├── Dockerfile               # Container definition
│   └── src/                     # Source code for container
├── sm_custom_image/             # SageMaker custom training image
│   └── app/                     # Application code and container setup
├── sm_preprocessing/            # Data preprocessing container
│   └── app/                     # Preprocessing implementation
└── sm_scriptmode/              # SageMaker script mode implementation
    └── app/                    # Training scripts and requirements
```

## Usage Instructions
### Prerequisites
- Python 3.11+
- CUDA-capable GPU (for training)
- AWS Account with appropriate permissions
- Docker (for container builds)

Required Python packages:
```
torch
monai==1.4.0
torchvision
itk
numpy
tensorboard
einops
```

### Installation

#### Local Development (EC2)
```bash
# Clone the repository
git clone <repository-url>
cd medical-image-classification

# Install dependencies
pip install -r ec2/requirements.txt
```

#### SageMaker Deployment
```bash
# Build and push containers
cd sm_custom_image/app
./build_and_push.sh

cd ../../sm_preprocessing/app
./build_and_push.sh
```

### Quick Start

1. Preprocess your data:
```bash
python sm_preprocessing/app/preprocessing.py \
  --input-dir /path/to/raw/images \
  --output-dir /path/to/processed/data
```

2. Train the model:
```bash
cd ec2
./run_train.sh
```

### More Detailed Examples

Training with custom parameters:
```bash
python train.py \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model_name DenseNet121 \
  --num_epochs 10 \
  --val_interval 1
```

### Troubleshooting

Common Issues:
1. CUDA Out of Memory
   - Reduce batch size
   - Monitor GPU memory usage with `nvidia-smi`

2. Container Build Failures
   - Ensure Docker daemon is running
   - Check AWS credentials are configured
   - Verify ECR repository exists

Debug Mode:
```bash
# Enable debug logging
export PYTORCH_DEBUG=1
```

## Data Flow
The system processes medical images through a pipeline of preprocessing, training, and evaluation stages.

```ascii
Raw Images → Preprocessing → Training → Evaluation
[Input] → [Split/Transform] → [Model Training] → [Metrics/Visualization]
```

Component Interactions:
1. Preprocessing container splits data into train/test/validation sets
2. Training script loads preprocessed data using PyTorch DataLoaders
3. Model performs forward/backward passes during training
4. Evaluation scripts generate ROC curves and confusion matrices
5. TensorBoard logs training metrics and visualizations

## Infrastructure

![Infrastructure diagram](./docs/infra.svg)

### ECR Repositories
- `sm-training-byoc`: Custom training container repository
- `sm-preprocessing`: Data preprocessing container repository

### Docker Images
- Base Image: `pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`
- Custom training image with additional dependencies:
  - nibabel
  - monai
  - simpleitk
  - smdebug
  - torch
  - torchvision
  - itk