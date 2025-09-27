"""Dataset utilities for Plant Disease Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def create_datasets(processed_dir: Path, img_size: int) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, eval_tf = get_transforms(img_size)
    train_ds = datasets.ImageFolder(root=str(processed_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(processed_dir / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(root=str(processed_dir / "test"), transform=eval_tf)
    return train_ds, val_ds, test_ds


def create_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    processed_dir = Path(processed_dir)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_ds, val_ds, test_ds = create_datasets(processed_dir, img_size)
    classes = train_ds.classes

    common_kwargs = dict(num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common_kwargs)

    return train_loader, val_loader, test_loader, classes
