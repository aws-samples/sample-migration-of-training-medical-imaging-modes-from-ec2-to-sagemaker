---
title: "Production Deployment"
chapter: true
weight: 40
---

# Module 4: Production Deployment

## Overview

Deploy trained models to production with SageMaker endpoints for real-time and batch inference.

## Learning Objectives

- Deploy models to SageMaker endpoints
- Implement batch inference
- Monitor model performance
- Optimize inference costs

## Duration

30 minutes

## Deployment Options

### Real-Time Endpoints
- Low latency (<100ms)
- Always-on infrastructure
- Auto-scaling support
- Best for: Interactive applications

### Batch Transform
- Process large datasets
- No persistent infrastructure
- Cost-effective
- Best for: Bulk processing

### Serverless Inference
- Pay per request
- Auto-scaling to zero
- Cold start latency
- Best for: Intermittent traffic

## Key Takeaways

✅ Multiple deployment options for different use cases  
✅ Built-in monitoring and auto-scaling  
✅ Cost optimization with serverless and batch  
✅ Production-ready with minimal code  

## Workshop Complete!

Congratulations! You've completed the medical imaging workshop and learned:

1. ✅ Data preprocessing with SageMaker Processing
2. ✅ Training progression from EC2 to SageMaker
3. ✅ Custom containers for full control
4. ✅ Multi-GPU distributed training with FSDP
5. ✅ Experiment tracking with WandB/MLflow
6. ✅ Automated hyperparameter optimization
7. ✅ Production deployment strategies

## Next Steps

- Explore [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- Read [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- Join [AWS ML Community](https://aws.amazon.com/machine-learning/community/)
- Build your own medical imaging projects!

## Resources

- [Workshop GitHub Repository](https://github.com/aws-samples/training-medical-imaging-models-on-sagemaker)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [MONAI Documentation](https://docs.monai.io/)
- [AWS Healthcare & Life Sciences](https://aws.amazon.com/health/)

## Cleanup

Don't forget to delete resources to avoid charges:

```bash
# Delete SageMaker endpoints
aws sagemaker delete-endpoint --endpoint-name medical-imaging-endpoint

# Delete S3 bucket contents
aws s3 rm s3://medical-imaging-workshop-<account>/ --recursive

# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name medical-imaging-workshop
```

Thank you for participating!
