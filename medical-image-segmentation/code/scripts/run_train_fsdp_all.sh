#!/bin/bash

# Configuration
MODEL_NAME="SwinUNETR"
DATA_DIR="/home/ubuntu/data/segmentation_data"
OUTPUT_DIR="./output"
BATCH_SIZE=2
EPOCHS=10
LEARNING_RATE=1e-4

# Optional: WandB settings
USE_WANDB=false
WANDB_PROJECT="medical-segmentation-fsdp"
WANDB_API_KEY=""

# Optional: MLFlow settings
USE_MLFLOW=false
MLFLOW_TRACKING_URI=""
MLFLOW_EXPERIMENT_NAME="medical-segmentation-fsdp"

# Build command
CMD="python training/train_fsdp_all.py \
    --model_name $MODEL_NAME \
    --data $DATA_DIR \
    --out_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE"

# Add WandB flags if enabled
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY=$WANDB_API_KEY
fi

# Add MLFlow flags if enabled
if [ "$USE_MLFLOW" = true ]; then
    CMD="$CMD --use_mlflow --mlflow_experiment_name $MLFLOW_EXPERIMENT_NAME"
    [ -n "$MLFLOW_TRACKING_URI" ] && CMD="$CMD --mlflow_tracking_uri $MLFLOW_TRACKING_URI"
fi

# Run training
echo "Starting FSDP training..."
echo "Command: $CMD"
eval $CMD
