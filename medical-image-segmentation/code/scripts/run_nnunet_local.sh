#!/bin/bash

# Local run script for nnU-Net v2 pipeline
#
# Usage:
#   ./run_nnunet_local.sh DATA_DIR OUTPUT_DIR [STAGES]
#
# Example:
#   ./run_nnunet_local.sh ./data ./output "preprocess,train"

DATA_DIR=${1:-"./data"}
OUTPUT_DIR=${2:-"./output"}
STAGES=${3:-"preprocess,train,evaluate"}

mkdir -p "$OUTPUT_DIR"

export SM_CHANNEL_TRAINING="$DATA_DIR"
export SM_MODEL_DIR="$OUTPUT_DIR"
export nnUNet_raw="/tmp/nnUNet_raw"
export nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
export nnUNet_results="/tmp/nnUNet_results"

echo "=========================================="
echo "nnU-Net v2 Local Run"
echo "=========================================="
echo "Data:   $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Stages: $STAGES"
echo "=========================================="

python training/nnunet/nnunet_pipeline.py \
    --stages "$STAGES" \
    --data "$DATA_DIR" \
    --out_dir "$OUTPUT_DIR"
