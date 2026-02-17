---
title: "Dataset Exploration"
weight: 14
---

## Workshop Datasets

This workshop uses two datasets representing common medical imaging tasks.

## Classification Dataset: MedNIST

A simplified medical imaging dataset for classification tasks. We will use the [MedMNIST](https://medmnist.com/) dataset. 

### Overview
- **Task**: Multi-class classification
- **Classes**: 6 categories (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT)
- **Images**: ~58,000 grayscale images
- **Size**: 64x64 pixels
- **Format**: PNG

### Structure
```
medmnist/
├── train/
│   ├── AbdomenCT/
│   │   ├── 000001.png
│   │   └── ...
│   ├── BreastMRI/
│   └── ...
├── val/
└── test/
```

### Sample Images

| Class | Description | Count |
|-------|-------------|-------|
| AbdomenCT | Abdominal CT scans | ~10,000 |
| BreastMRI | Breast MRI scans | ~8,000 |
| CXR | Chest X-rays | ~10,000 |
| ChestCT | Chest CT scans | ~10,000 |
| Hand | Hand X-rays | ~10,000 |
| HeadCT | Head CT scans | ~10,000 |

### Download
You can download the data from the above website and then push the dataset to an s3 bucket. Say you downloaded the data in the directory `medmnist`.

# Copy the data to a s3 bucket
```
!aws s3 sync s3://{bucket}/{prefix} ./data/classification/
```

## Segmentation Dataset: Medical Decathlon

3D medical image segmentation dataset. Download the dataset from the Medical Decathlon [page](http://medicaldecathlon.com/dataaws/). There are multiple segmentation. Pick a dataset of your choice.

### Overview
- **Task**: Binary segmentation (organ/tumor)
- **Modality**: CT/MRI volumes
- **Format**: NIfTI (.nii.gz)
- **Size**: Variable (typically 512x512x150-300 slices)
- **Annotations**: Voxel-level masks

### Structure
```
segmentation/
├── train/
│   ├── subject_001/
│   │   ├── img.nii.gz      # 3D volume
│   │   └── label.nii.gz    # Segmentation mask
│   ├── subject_002/
│   └── ...
└── valid/
    └── subject_001/
        ├── img.nii.gz
        └── label.nii.gz
```

### Data Characteristics
Typical characteristics of a CT scan dataset.

| Property | Value |
|----------|-------|
| Dimensions | 3D (H x W x D) |
| Typical Size | 512 x 512 x 200 |
| Voxel Spacing | 0.5-1.0 mm |
| Intensity Range | -1024 to 3071 HU (CT) |
| File Size | 50-200 MB per volume |

### Upload the dataset

```
!aws s3 sync ./data/segmentation/ s3://medical-imaging-workshop-data/segmentation/
```

## Data Preprocessing

### Classification Preprocessing

```python
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    Resize, ScaleIntensity, RandFlip, RandRotate
)

train_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((64, 64)),
    ScaleIntensity(),
    RandFlip(prob=0.5),
    RandRotate(range_x=15, prob=0.5)
])
```

### Segmentation Preprocessing

```python
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandFlipd
)

train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Resized(keys=["image", "mask"], spatial_size=(128, 128, 64)),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0)
])
```

## Explore the Data

Open the exploration notebook:

```bash
jupyter notebook notebooks/00-data-exploration.ipynb
```

This notebook shows:
- Loading and visualizing images
- Data distribution analysis
- Preprocessing pipeline examples
- Data augmentation effects

## Data Storage Strategy

### S3 Organization

```
s3://medical-imaging-workshop-<account>/
├── classification/
│   ├── raw/              # Original data
│   ├── processed/        # Preprocessed splits
│   └── artifacts/        # Model outputs
└── segmentation/
    ├── raw/
    ├── processed/
    └── artifacts/
```

### Best Practices

1. **Use S3 for all data**: Don't store large datasets on notebook instances
2. **Preprocess once**: Save preprocessed data to S3
3. **Version datasets**: Use S3 versioning or date prefixes
4. **Optimize formats**: Use efficient formats (NIfTI for 3D, optimized PNGs for 2D)

## Next Steps

Now that you understand the datasets, proceed to Module 2 to start training classification models.
