#!/bin/bash

# Local training script for train_simple.py

# Set default values
DATA_DIR=${1:-"./data"}
OUTPUT_DIR=${2:-"./output"}
MODEL_NAME=${3:-"SegResNet"}
BATCH_SIZE=${4:-2}
EPOCHS=${5:-10}
LR=${6:-0.0001}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run training
python train_simple.py \
    --model_name "$MODEL_NAME" \
    --data "$DATA_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR"
