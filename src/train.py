"""
Metal Surface Defect Classification
Supports: Swin Transformer, CoAtNet, ConvNeXt V2
Input: Grayscale 256×256 images, ~20 classes
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import timm
from tqdm import tqdm

from dataset import build_dataloaders
from utils import (
    AverageMeter, accuracy, save_checkpoint,
    EarlyStopping, setup_logger
)


def parse_args():
    parser = argparse.ArgumentParser(description="Metal Surface Defect Classifier")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root dir with train/val/test subdirectories")
    parser.add_argument("--model",      type=str, default="swin_base",
                        choices=["swin_base", "coatnet_2", "convnextv2_base"],
                        help="Model architecture")
    parser.add_argument("--num_classes",type=int, default=20)
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--patience",   type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--amp",        action="store_true", default=True,
                        help="Use Automatic Mixed Precision")
    return parser.parse_args()


MODEL_CONFIGS = {
    "swin_base": {
        "timm_name": "swin_base_patch4_window8_256",
        "description": "Swin Transformer Base (256x256 native window)",
    },
    "coatnet_2": {
        "timm_name": "coatnet_2_rw_224",
        "description": "CoAtNet-2 (CNN + Transformer hybrid)",
    },
    "convnextv2_base": {
        "timm_name": "convnextv2_base",
        "description": "ConvNeXt V2 Base",
    },
}


def build_model(model_name: str, num_classes: int, img_size: int) -> nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    model = timm.create_model(
        cfg["timm_name"],
        pretrained=True,
        num_classes=num_classes,
        in_chans=3,          # grayscale → 3ch複製で転移学習を活かす
    )
    print(f"[Model] {cfg['description']}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    losses = AverageMeter()
    top1   = AverageMeter()

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(logits, labels)
        losses.update(loss.item(), imgs.size(0))
        top1.update(acc1,         imgs.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.3f}")

    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    losses = AverageMeter()
    top1   = AverageMeter()

    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(enabled=amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        acc1 = accuracy(logits, labels)
        losses.update(loss.item(), imgs.size(0))
        top1.update(acc1,         imgs.size(0))

    return losses.avg, top1.avg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- DataLoader ----------
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir   = args.data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    # ---------- Model ----------
    model = build_model(args.model, args.num_classes, args.img_size)
    model = model.to(device)

    # ---------- Loss / Optimizer ----------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=args.amp)
    stopper   = EarlyStopping(patience=args.patience,
                               ckpt_path=output_dir / "best.pth")

    # ---------- Training loop ----------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.amp)
        va_loss, va_acc = evaluate(
            model, val_loader, criterion, device, args.amp)

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"time={elapsed:.1f}s"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(model, output_dir / "best.pth")

        if stopper(va_loss, model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ---------- Test ----------
    logger.info("=== Test Evaluation ===")
    model.load_state_dict(torch.load(output_dir / "best.pth"))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device, args.amp)
    logger.info(f"Test loss={te_loss:.4f}  Test acc={te_acc:.4f}")

    # save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
