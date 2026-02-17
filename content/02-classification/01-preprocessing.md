---
title: "Data Preprocessing"
weight: 21
---

## Overview

Use SageMaker Processing to split and preprocess medical imaging data at scale.

## Why SageMaker Processing?

- **Scalable**: Process large datasets in parallel
- **Managed**: No infrastructure management
- **Reproducible**: Containerized processing scripts
- **Cost-effective**: Pay only for processing time

## Lab: Preprocess MedNIST Dataset

### Step 1: Review Processing Script

The preprocessing script splits data into train/val/test sets:

```python
# preprocessing.py
import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test"""
    classes = os.listdir(input_dir)
    
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        images = os.listdir(cls_path)
        
        # Split: 70% train, 15% val, 15% test
        train, temp = train_test_split(images, train_size=train_ratio)
        val, test = train_test_split(temp, train_size=0.5)
        
        # Copy to output directories
        for split, files in [('train', train), ('val', val), ('test', test)]:
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(split_dir, f)
                )

if __name__ == "__main__":
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    split_dataset(input_dir, output_dir)
```

### Step 2: Create Processing Job

Open notebook: `notebooks/classification/01-data-preprocessing.ipynb`

```python
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()

# Create processor
processor = ScriptProcessor(
    image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-cpu-py311',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    command=['python3']
)

# Run processing job
processor.run(
    code='preprocessing.py',
    inputs=[
        ProcessingInput(
            source=f's3://{bucket}/classification/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{bucket}/classification/processed/'
        )
    ]
)
```

### Step 3: Monitor Job

```python
# Check job status
processor.jobs[-1].describe()

# View logs
processor.jobs[-1].wait(logs=True)
```

### Step 4: Verify Output

```python
# List processed data
!aws s3 ls s3://{bucket}/classification/processed/ --recursive

# Expected structure:
# s3://bucket/classification/processed/train/AbdomenCT/
# s3://bucket/classification/processed/train/BreastMRI/
# s3://bucket/classification/processed/val/...
# s3://bucket/classification/processed/test/...
```

## Processing Job Details

### Instance Types

| Instance | vCPUs | Memory | Cost/Hour | Use Case |
|----------|-------|--------|-----------|----------|
| ml.m5.large | 2 | 8 GB | $0.115 | Small datasets |
| ml.m5.xlarge | 4 | 16 GB | $0.23 | Medium datasets |
| ml.m5.4xlarge | 16 | 64 GB | $0.922 | Large datasets |

### Cost Estimation

For MedNIST dataset (~58,000 images):
- Instance: ml.m5.xlarge
- Duration: ~5 minutes
- Cost: ~$0.02

## Advanced: Custom Processing Container

For complex preprocessing, build a custom container:

```dockerfile
FROM python:3.11-slim

RUN pip install numpy pandas scikit-learn pillow

COPY preprocessing.py /opt/ml/code/
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "preprocessing.py"]
```

Build and push:
```bash
docker build -t preprocessing:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag preprocessing:latest <account>.dkr.ecr.us-east-1.amazonaws.com/preprocessing:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/preprocessing:latest
```

## Key Takeaways

✅ SageMaker Processing scales data preprocessing  
✅ Reproducible with containerized scripts  
✅ Integrates seamlessly with S3  
✅ Cost-effective for large datasets  

## Next Steps

With preprocessed data ready, proceed to train a baseline model on EC2.
