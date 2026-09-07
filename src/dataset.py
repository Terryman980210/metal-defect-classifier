"""
Dataset utilities for metal surface defect classification.
Expects directory structure:
    data_dir/
        train/
            class_A/  *.png (or jpg/bmp)
            class_B/
            ...
        val/
            class_A/
            ...
        test/
            class_A/
            ...

Grayscale images are converted to 3-channel by replication
so that ImageNet pretrained weights can be used.
"""

from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np


# ---- ImageNet stats (used even for grayscale→3ch images) ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    """
    Training: moderate augmentation suitable for industrial texture images.
    Val/Test: center-crop only.
    """
    if is_train:
        return transforms.Compose([
            # Grayscale → 3ch (replication)
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Class-balanced sampler to handle class imbalance.
    Each class is sampled with equal probability per epoch.
    """
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    weights_per_class = 1.0 / class_counts
    sample_weights = torch.tensor(
        [weights_per_class[t] for t in targets], dtype=torch.float
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    data_dir: str,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(
        data_dir / "train",
        transform=build_transforms(img_size, is_train=True),
    )
    val_ds = datasets.ImageFolder(
        data_dir / "val",
        transform=build_transforms(img_size, is_train=False),
    )
    test_ds = datasets.ImageFolder(
        data_dir / "test",
        transform=build_transforms(img_size, is_train=False),
    )

    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None
    shuffle = sampler is None  # shuffle only if no sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds.classes
