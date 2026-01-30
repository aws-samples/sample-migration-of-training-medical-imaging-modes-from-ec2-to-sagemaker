# EC2 Training Tutorial

Train medical image classification models (DenseNet121 or Vision Transformer) directly on EC2 instances using MONAI and PyTorch.

## Overview

This notebook demonstrates local training on EC2 with GPU acceleration, implementing a complete training pipeline with data loading, model training, validation, and TensorBoard logging.

## Prerequisites

- EC2 instance with GPU (e.g., g4dn.xlarge, p3.2xlarge)
- Python 3.10+
- CUDA-capable GPU
- Preprocessed medical image dataset with train/valid/test splits

## Required Dependencies

```bash
pip install torch torchvision monai==1.4.0 tensorboard tqdm numpy pillow
```

## Data Structure

Expected directory structure:
```
data/
├── train/
│   ├── class_1/
│   │   ├── image_001.png
│   │   └── ...
│   └── class_2/
│       └── ...
├── valid/
│   └── ...
└── test/
    └── ...
```

Requires `label_dict.json` mapping class names to numeric labels.

## Configuration

Key parameters to modify in Step 2:

```python
data_dir = '/home/ubuntu/data/vindr-spinexr-subset'
model_dir = '/home/ubuntu/data/spine-model'
output_dir = '/home/ubuntu/data/spine-output'

model_name = 'DenseNet121'  # or 'ViT'
learning_rate = 0.001
batch_size = 32
num_epochs = 3
val_interval = 10
early_stopping_rounds = 10
img_size = (256, 256, 1)
```

## Model Architectures

### DenseNet121
- 2D convolutional network
- Single channel input (grayscale medical images)
- Efficient feature reuse through dense connections

### Vision Transformer (ViT)
- Patch-based transformer architecture
- 16x16 patch size
- 12 layers, 12 attention heads
- Hidden size: 768

## Training Features

- **Early Stopping**: Stops training if validation loss doesn't improve for 10 epochs
- **TensorBoard Logging**: Real-time metrics visualization
- **Model Checkpointing**: Saves best model based on validation loss
- **Progress Tracking**: TQDM progress bars for training/validation
- **GPU Acceleration**: Automatic CUDA device detection

## Outputs

- **Model Weights**: `{output_dir}/{model_name}/model.pth`
- **Best Checkpoint**: `{model_dir}/{model_name}/best_model_{epoch}.pth`
- **TensorBoard Logs**: `{output_dir}/{model_name}/logs/`
- **Training Logs**: `logs/log_{timestamp}/app.log`

## Monitoring Training

View training metrics in real-time:
```bash
tensorboard --logdir={output_dir}/{model_name}/logs
```

Access at `http://localhost:6006`

## Execution Steps

1. **Setup**: Import libraries, configure logging, check GPU
2. **Configure**: Set paths and hyperparameters
3. **Define Model**: Create model architecture
4. **Load Data**: Build datasets and data loaders
5. **Training Functions**: Define train/validation loops
6. **Train**: Execute training with early stopping
7. **Save**: Persist final model weights

## Performance Tips

- Adjust `batch_size` based on GPU memory (reduce if OOM errors occur)
- Use `num_workers=16` for faster data loading on multi-core CPUs
- Monitor GPU utilization with `nvidia-smi`
- Increase `val_interval` for faster training (less frequent validation)

## Troubleshooting

**CUDA Out of Memory**
- Reduce batch_size to 16 or 8
- Use smaller image size

**Slow Data Loading**
- Reduce num_workers if CPU-bound
- Ensure data is on fast storage (NVMe SSD)

**Model Not Converging**
- Adjust learning_rate (try 0.0001 or 0.01)
- Increase num_epochs
- Check data preprocessing and labels
