#!/bin/bash

# Build and push Docker image to Amazon ECR

set -e

# Configuration
IMAGE_NAME=${1:-"medical-image-segmentation"}
AWS_REGION=${2:-"us-east-1"}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

echo "Building and pushing ${IMAGE_NAME} to ${ECR_REPO}"

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${AWS_REGION}

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build image
docker build -t ${IMAGE_NAME}:latest .

# Tag image
docker tag ${IMAGE_NAME}:latest ${ECR_REPO}:latest

# Push image
docker push ${ECR_REPO}:latest

echo "Successfully pushed ${ECR_REPO}:latest"
