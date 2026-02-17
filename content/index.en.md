---
title: "Training Medical Imaging Models on AWS"
chapter: false
weight: 1
---

# Training Medical Imaging Models on AWS

Welcome to this comprehensive workshop on building, training, and deploying medical imaging models using AWS services. You'll learn how to migrate from traditional EC2-based training to production-scale distributed training on Amazon SageMaker.

## Workshop Overview

This hands-on workshop takes you through a progressive journey:

1. **Start Simple**: Train classification models on EC2 to understand the baseline
2. **Migrate to SageMaker**: Leverage managed infrastructure with Script Mode
3. **Customize Everything**: Build custom containers for complete control
4. **Scale Up**: Move to 3D segmentation with multi-GPU distributed training
5. **Production Ready**: Add experiment tracking, optimization, and deployment

## What You'll Learn

- Data preprocessing pipelines for medical imaging
- Training deep learning models on EC2 and SageMaker
- Custom Docker containers for ML workloads
- Distributed training strategies (DDP and FSDP)
- Experiment tracking with MLflow and Weights & Biases
- Hyperparameter optimization at scale
- Cost optimization techniques

## Prerequisites

- Basic Python and PyTorch knowledge
- Familiarity with AWS services (EC2, S3, IAM)
- Understanding of deep learning concepts
- AWS account with appropriate permissions

## Workshop Duration

**Total Time**: 4-5 hours

- Module 1: Introduction (30 min)
- Module 2: Classification Track (2 hours)
- Module 3: Segmentation Track (2 hours)
- Module 4: Production Deployment (30 min)

## Architecture

The workshop infrastructure includes SageMaker notebook instances, S3 storage, and training compute resources provisioned via CloudFormation.

## Let's Get Started!

Click **Next** to begin with the introduction module.
