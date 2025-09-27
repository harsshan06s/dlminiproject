"""Create a balanced subset of the PlantVillage dataset.

Example
-------
python src/create_subset.py --raw_dir data/raw/PlantVillage --out_dir data/subset --n_per_class 500
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def copy_subset(images: Iterable[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, dst_dir / img_path.name)


def make_subset(raw_dir: Path, out_dir: Path, n_per_class: int, seed: int) -> None:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {raw_dir}")

    for cls_dir in class_dirs:
        images = list_images(cls_dir)
        if not images:
            print(f"Skipping {cls_dir.name}: no images found")
            continue

        sample_count = min(n_per_class, len(images))
        chosen = random.sample(images, sample_count)
        copy_subset(chosen, out_dir / cls_dir.name)
        print(f"Copied {sample_count} images to subset for class {cls_dir.name}")
    print(f"Subset created at {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced subset of PlantVillage images")
    parser.add_argument("--raw_dir", default="data/raw/PlantVillage", help="Input directory with class folders")
    parser.add_argument("--out_dir", default="data/subset", help="Output directory for subset")
    parser.add_argument("--n_per_class", type=int, default=500, help="Maximum images per class in subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Remove existing subset directory before creating new one")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    make_subset(raw_dir, out_dir, n_per_class=args.n_per_class, seed=args.seed)


if __name__ == "__main__":
    main()
