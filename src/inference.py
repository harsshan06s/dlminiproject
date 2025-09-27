"""Inference helpers for the trained model."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import yaml

from model import create_model


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = checkpoint.get("config", {})
    classes = checkpoint.get("classes")
    if not classes:
        raise KeyError("Checkpoint missing class names")

    backbone = checkpoint.get("backbone", cfg.get("train", {}).get("backbone", "tf_efficientnet_b0"))
    img_size = checkpoint.get("img_size", cfg.get("data", {}).get("img_size", 224))

    model = create_model(backbone_name=backbone, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes, {"img_size": img_size, "backbone": backbone}


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(image: Image.Image, model: torch.nn.Module, classes: list[str], device: torch.device, img_size: int) -> dict:
    tf = build_transform(img_size)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    return {
        "class": classes[top_idx],
        "prob": float(probs[top_idx]),
        "scores": {classes[i]: float(score) for i, score in enumerate(probs)},
    }


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def create_annotated_image(image: Image.Image, text: str) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, annotated.width // 18))
    except (OSError, IOError):
        font = ImageFont.load_default()

    margin = 12
    x, y = margin, margin
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    rect_coords = (
        x - margin // 2,
        y - margin // 2,
        x + text_width + margin // 2,
        y + text_height + margin // 2,
    )
    draw.rectangle(rect_coords, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return annotated


def render_annotated_image(image: Image.Image, text: str, image_format: str = "JPEG") -> bytes:
    annotated = create_annotated_image(image, text)
    buffer = io.BytesIO()
    annotated.save(buffer, format=image_format)
    return buffer.getvalue()


def save_annotated_image(image: Image.Image, text: str, output_path: Path) -> None:
    annotated = create_annotated_image(image, text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--model", required=True, help="Path to trained checkpoint (pth)")
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--config", default="config.yaml", help="Optional config override to select device")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--annotate", help="Optional path to save the image with prediction text overlayed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = {}
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}

    device = torch.device(cfg.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.model)

    model, classes, metadata = load_model(checkpoint_path, device)
    img = load_image(Path(args.image))
    result = predict(img, model, classes, device=device, img_size=metadata["img_size"])
    result["backbone"] = metadata["backbone"]

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.annotate:
        annotate_path = Path(args.annotate)
        label_text = f"{result['class']} ({result['prob'] * 100:.1f}%)"
        save_annotated_image(img, label_text, annotate_path)
        print(f"Saved annotated image to {annotate_path}")


if __name__ == "__main__":
    main()
