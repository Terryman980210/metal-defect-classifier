"""
Evaluate trained model on test set.
Outputs: accuracy, per-class F1, confusion matrix PNG, classification report.

Usage:
    python src/evaluate.py \
        --model swin_base \
        --ckpt outputs/swin_base/best.pth \
        --data_dir /path/to/data \
        --output_dir outputs/swin_base
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import timm

from dataset import build_dataloaders

MODEL_CONFIGS = {
    "swin_base":       "swin_base_patch4_window8_256",
    "coatnet_2":       "coatnet_2_rw_224",
    "convnextv2_base": "convnextv2_base",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=list(MODEL_CONFIGS))
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--output_dir",  type=str, default="outputs/eval")
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm_arr, class_names, output_path, normalize=True):
    if normalize:
        cm_arr = cm_arr.astype(float) / (cm_arr.sum(axis=1, keepdims=True) + 1e-8)

    n = len(class_names)
    figsize = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(
        cm_arr, annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""),
                 fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Confusion Matrix] Saved → {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_dataloaders(
        data_dir    = args.data_dir,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = 4,
    )

    # Save class names for later use
    with open(output_dir / "classes.json", "w") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    # Load model
    model = timm.create_model(
        MODEL_CONFIGS[args.model],
        pretrained=False,
        num_classes=args.num_classes,
        in_chans=3,
    )
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)

    preds, labels = predict_all(model, test_loader, device)

    # Metrics
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4)

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy (macro): {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("\n--- Per-class Report ---")
    print(report)

    # Save report
    with open(output_dir / "report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm_arr = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm_arr, class_names,
                          output_dir / "confusion_matrix.png", normalize=True)
    plot_confusion_matrix(cm_arr, class_names,
                          output_dir / "confusion_matrix_raw.png", normalize=False)

    # Results JSON
    results = {"accuracy": acc, "f1_macro": f1}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
