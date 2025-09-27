"""Train a plant disease classifier with PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_model
from utils import (
    append_metrics_csv,
    compute_metrics,
    confusion_matrix_dict,
    dump_json,
    save_checkpoint,
    set_seed,
)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    mixed_precision: bool,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    losses = []
    all_preds, all_targets = [], []

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, (imgs, targets) in enumerate(progress, start=1):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=mixed_precision):
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

        if log_every and step % log_every == 0:
            progress.set_postfix(loss=np.mean(losses))

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> Tuple[Dict[str, float], list[int], list[int]]:
    model.eval()
    losses = []
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(dataloader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, all_targets, all_preds


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer: optim.Optimizer, cfg: dict):
    if not cfg:
        return None
    name = cfg.get("name", "").lower()
    if name == "reduce_on_plateau":
        kwargs = {
            "mode": cfg.get("mode", "max"),
            "factor": cfg.get("factor", 0.5),
            "patience": cfg.get("patience", 2),
            "min_lr": cfg.get("min_lr", 1e-6),
        }
        if "verbose" in inspect.signature(ReduceLROnPlateau).parameters:
            kwargs["verbose"] = True
        return ReduceLROnPlateau(optimizer, **kwargs)
    return None


def main(config_path: str) -> None:
    cfg = load_config(Path(config_path))

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    scheduler_cfg = cfg.get("scheduler", {})
    save_cfg = cfg.get("save", {})
    logging_cfg = cfg.get("logging", {})

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    batch_size = int(data_cfg.get("batch_size", 32))
    img_size = int(data_cfg.get("img_size", 224))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader, val_loader, test_loader, classes = create_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )

    num_classes = len(classes)
    backbone = train_cfg.get("backbone", "tf_efficientnet_b0")
    pretrained = bool(train_cfg.get("pretrained", True))
    model = create_model(backbone_name=backbone, num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    use_mixed_precision = bool(train_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = GradScaler(enabled=use_mixed_precision)

    epochs = int(train_cfg.get("epochs", 20))
    log_every = int(logging_cfg.get("print_every", 0))

    out_dir = Path(save_cfg.get("out_dir", "outputs"))
    model_dir = Path(save_cfg.get("model_dir", out_dir / "models"))
    metrics_dir = Path(save_cfg.get("metrics_dir", out_dir / "metrics"))
    log_dir = Path(save_cfg.get("log_dir", out_dir / "logs"))
    best_model_path = model_dir / save_cfg.get("save_best_as", "best_model.pth")
    history_csv = metrics_dir / "training_history.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixed_precision=use_mixed_precision,
            log_every=log_every,
        )
        val_metrics, val_targets, val_preds = evaluate_epoch(model, val_loader, criterion, device, desc="val")

        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics.get("f1", val_metrics.get("accuracy", 0)))

        print(
            f"Train - loss: {train_metrics['loss']:.4f} | acc: {train_metrics['accuracy']:.4f} | f1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Val   - loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f} | f1: {val_metrics['f1']:.4f}"
        )

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_metrics_csv(history_csv, record)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "classes": classes,
                    "backbone": backbone,
                    "img_size": img_size,
                    "config": cfg,
                    "best_f1": best_f1,
                    "val_metrics": val_metrics,
                    "confusion_matrix": confusion_matrix_dict(val_targets, val_preds, list(range(num_classes))),
                },
                best_model_path,
            )
            print(f"Saved best model to {best_model_path}")

    if best_epoch == -1:
        raise RuntimeError("Training finished without saving a best model. Check your validation split.")

    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_targets, test_preds = evaluate_epoch(model, test_loader, criterion, device, desc="test")
    print("Test metrics:", test_metrics)

    dump_json(
        {
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "test_metrics": test_metrics,
            "classes": classes,
        },
        metrics_dir / "test_summary.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plant disease classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
