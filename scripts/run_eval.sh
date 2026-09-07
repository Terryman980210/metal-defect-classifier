#!/bin/bash
# Evaluate trained model
# Usage: bash scripts/run_eval.sh /path/to/data swin_base

DATA_DIR=${1:-"/path/to/your/dataset"}
MODEL=${2:-"swin_base"}
CKPT="outputs/${MODEL}/best.pth"

python src/evaluate.py \
    --model      "$MODEL" \
    --ckpt       "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --num_classes 20 \
    --img_size   256 \
    --output_dir "outputs/${MODEL}/eval"

python src/plot_history.py \
    --history "outputs/${MODEL}/history.json" \
    --output  "outputs/${MODEL}/history.png"
