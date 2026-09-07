"""Utility functions: metrics, logging, checkpointing, early stopping."""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("defect_cls")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------ #
#  Checkpointing
# ------------------------------------------------------------------ #

def save_checkpoint(model: nn.Module, path: Path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------------ #
#  Early Stopping
# ------------------------------------------------------------------ #

class EarlyStopping:
    """
    Stop training when validation loss does not improve for `patience` epochs.
    Saves the best model automatically.
    """

    def __init__(self, patience: int = 15, ckpt_path: Path = None, delta: float = 1e-4):
        self.patience  = patience
        self.ckpt_path = ckpt_path
        self.delta     = delta
        self.best_loss = float("inf")
        self.counter   = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            if self.ckpt_path:
                save_checkpoint(model, self.ckpt_path)
        else:
            self.counter += 1

        return self.counter >= self.patience
