"""Split the PlantVillage dataset into train/val/test folders.

Usage
-----
python src/prepare_data.py --config config.yaml [--clean]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import yaml
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def copy_split(src_dir: Path, dst_dir: Path, filenames: Iterable[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists():
            shutil.copy2(src, dst)


def clear_processed_dir(processed_dir: Path) -> None:
    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def prepare(raw_dir: Path, processed_dir: Path, val_split: float, test_split: float, seed: int) -> None:
    random.seed(seed)
    classes = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {raw_dir}. Make sure the PlantVillage dataset is downloaded.")

    print(f"Found {len(classes)} classes")
    for cls_dir in classes:
        images = list_images(cls_dir)
        if not images:
            print(f"Warning: no images found for class {cls_dir.name}, skipping.")
            continue

        train_and_val, test = train_test_split(images, test_size=test_split, random_state=seed, shuffle=True)
        relative_val_split = val_split / max(1e-6, (1 - test_split))
        train, val = train_test_split(train_and_val, test_size=relative_val_split, random_state=seed, shuffle=True)

        copy_split(cls_dir, processed_dir / "train" / cls_dir.name, train)
        copy_split(cls_dir, processed_dir / "val" / cls_dir.name, val)
        copy_split(cls_dir, processed_dir / "test" / cls_dir.name, test)
    print("Data prepared. Train/val/test folders populated under", processed_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage data splits.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--clean", action="store_true", help="Remove previous processed splits before creating new ones")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    data_cfg = cfg.get("data", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw/PlantVillage")).resolve()
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed")).resolve()
    val_split = float(data_cfg.get("val_split", 0.15))
    test_split = float(data_cfg.get("test_split", 0.05))
    seed = int(data_cfg.get("seed", 42))

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        clear_processed_dir(processed_dir)

    prepare(raw_dir, processed_dir, val_split=val_split, test_split=test_split, seed=seed)


if __name__ == "__main__":
    main()
