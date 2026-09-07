#!/bin/bash
# Train metal surface defect classifier
# Usage: bash scripts/run_train.sh /path/to/data swin_base

DATA_DIR=${1:-"/path/to/your/dataset"}
MODEL=${2:-"swin_base"}

python src/train.py \
    --data_dir    "$DATA_DIR" \
    --model       "$MODEL" \
    --num_classes 20 \
    --img_size    256 \
    --batch_size  32 \
    --epochs      100 \
    --lr          1e-4 \
    --weight_decay 1e-4 \
    --patience    15 \
    --output_dir  outputs \
    --num_workers 4 \
    --amp
