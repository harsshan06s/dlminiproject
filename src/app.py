"""Flask app for serving plant disease predictions."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
import yaml

from inference import load_model, predict, render_annotated_image

app = Flask(__name__)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


CONFIG = load_config(Path("config.yaml"))
DEVICE = torch.device(CONFIG.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(CONFIG.get("save", {}).get("model_dir", "outputs/models")) / CONFIG.get("save", {}).get(
    "save_best_as", "best_model.pth"
)
MODEL, CLASSES, META = load_model(MODEL_PATH, DEVICE)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "device": str(DEVICE), "backbone": META.get("backbone")})


@app.route("/", methods=["GET", "POST"])
def index():
    error: str | None = None
    result_view: dict | None = None
    scores_view: list[dict[str, float]] = []
    image_data: str | None = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or uploaded.filename == "":
            error = "Please choose an image before submitting."
        else:
            try:
                image_bytes = uploaded.read()
                if not image_bytes:
                    raise ValueError("Empty file uploaded.")
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:  # pylint: disable=broad-except
                error = f"Could not process the image: {exc}"
            else:
                result = predict(image, MODEL, CLASSES, device=DEVICE, img_size=META["img_size"])
                label_text = f"{result['class']} ({result['prob'] * 100:.1f}%)"
                annotated_bytes = render_annotated_image(image, label_text)
                image_data = base64.b64encode(annotated_bytes).decode("utf-8")

                result_view = {
                    "label": result["class"],
                    "confidence": result["prob"] * 100,
                }
                scores_view = [
                    {"label": label, "confidence": score * 100}
                    for label, score in sorted(result["scores"].items(), key=lambda item: item[1], reverse=True)
                ]

    return render_template(
        "index.html",
        error=error,
        result=result_view,
        scores=scores_view,
        image_data=image_data,
        backbone=META.get("backbone"),
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.json and "image_base64" in request.json:
        import base64

        image_bytes = base64.b64decode(request.json["image_base64"])
    else:
        return jsonify({"error": "No image provided. Upload a file field named 'image' or supply image_base64."}), 400

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Invalid image data: {exc}"}), 400

    result = predict(image, MODEL, CLASSES, device=DEVICE, img_size=META["img_size"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
