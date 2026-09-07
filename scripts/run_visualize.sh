#!/bin/bash
# Visualize XAI (Attention Map or GradCAM++)
# Usage: bash scripts/run_visualize.sh swin_base /path/to/image.png

MODEL=${1:-"swin_base"}
IMG=${2:-"sample.png"}
CKPT="outputs/${MODEL}/best.pth"
CLASSES="outputs/${MODEL}/eval/classes.json"

python src/visualize.py \
    --model           "$MODEL" \
    --ckpt            "$CKPT" \
    --img             "$IMG" \
    --class_names_file "$CLASSES" \
    --output_dir      "outputs/${MODEL}/vis"
