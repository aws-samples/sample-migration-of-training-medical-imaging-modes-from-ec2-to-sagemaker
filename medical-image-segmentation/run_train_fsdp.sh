#!/bin/bash

python code/train_fsdp_all.py \
  --model_name SwinUNETR \
  --data ./data \
  --out_dir ./output \
  --batch_size 2 \
  --epochs 10 \
  --lr 1e-4
