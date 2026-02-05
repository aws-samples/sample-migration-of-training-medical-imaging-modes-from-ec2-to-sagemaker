AWS_ACCOUNT_ID=$1
AWS_REGION=$2
AWS_ECR_REPO=sm-training-byoc

IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_REPO:latest

docker build -t $IMAGE .
# docker run -it -p 8080:8080 $IMAGE 
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
# Push the image to ECR

aws ecr create-repository --repository-name $AWS_ECR_REPO
# 
docker push $IMAGE