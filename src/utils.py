"""Utility helpers for training and evaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


def confusion_matrix_dict(y_true, y_pred, labels) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"confusion_matrix": cm.tolist(), "labels": labels}


def save_checkpoint(state: dict, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


def dump_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_metrics_csv(path: str | Path, record: dict) -> None:
    import pandas as pd  # local import to avoid hard dependency during inference

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {k: [v] for k, v in record.items()}
    df = pd.DataFrame(record)
    if path.exists():
        df_prev = pd.read_csv(path)
        df = pd.concat([df_prev, df], ignore_index=True)
    df.to_csv(path, index=False)
