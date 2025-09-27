"""Plot training and validation curves from the training history CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml


TRAIN_COLUMNS = {
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
}


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_history_path(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    save_cfg = config.get("save", {})
    metrics_dir = Path(save_cfg.get("metrics_dir", "outputs/metrics"))
    return metrics_dir / "training_history.csv"


def resolve_output_dir(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    save_cfg = config.get("save", {})
    return Path(save_cfg.get("metrics_dir", "outputs/metrics"))


def check_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        msg = (
            "Missing required columns in training history. "
            f"Expected at least {required}, but missing {missing}."
        )
        raise ValueError(msg)


def plot_curves(df: pd.DataFrame, output_dir: Path, dpi: int = 150) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = df.get("epoch")
    if epochs is None:
        epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axes[0].plot(epochs, df["train_accuracy"], label="Train", marker="o")
    axes[0].plot(epochs, df["val_accuracy"], label="Validation", marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Loss plot
    axes[1].plot(epochs, df["train_loss"], label="Train", marker="o")
    axes[1].plot(epochs, df["val_loss"], label="Validation", marker="o")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Training Curves", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {"training_curves": output_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy and loss curves from training history")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (for default paths)")
    parser.add_argument("--history", help="Optional explicit path to training_history.csv")
    parser.add_argument("--out_dir", help="Optional output directory for plots")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    history_path = resolve_history_path(config, args.history)
    output_dir = resolve_output_dir(config, args.out_dir)

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find training history CSV at {history_path}")

    df = pd.read_csv(history_path)
    check_columns(df, TRAIN_COLUMNS - {"epoch"})
    plot_paths = plot_curves(df, output_dir, dpi=args.dpi)

    for label, path in plot_paths.items():
        print(f"Saved {label} plot to {path}")


if __name__ == "__main__":
    main()
