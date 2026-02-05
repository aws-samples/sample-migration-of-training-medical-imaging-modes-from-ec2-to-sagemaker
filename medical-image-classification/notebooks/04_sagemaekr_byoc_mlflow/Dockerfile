FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker

RUN pip install --no-cache-dir \
    nibabel \
    monai==1.4.0 \
    simpleitk \
    smdebug \
    torch \
    torchvision \
    itk


COPY src/train.py /opt/program/train.py

ENV SAGEMAKER_PROGRAM /opt/program/train.py