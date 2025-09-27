# Plant Disease Detection (PyTorch)

End-to-end pipeline for training and serving a convolutional neural network that classifies plant diseases from the PlantVillage dataset. The project covers data preparation, model training with timm backbones, single-image inference, and a lightweight Flask API for batch-free predictions.

## Directory layout

```
plant-disease-detection/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/PlantVillage/   # drop Kaggle PlantVillage class folders here
│   ├── processed/          # auto-populated with train/val/test splits
│   └── subset/             # optional balanced subset
├── src/
│   ├── prepare_data.py
│   ├── create_subset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   ├── inference.py
│   ├── app.py
│   └── plot_metrics.py
└── outputs/
    ├── models/
    ├── logs/
    └── metrics/
```

## Quickstart

1. **Create environment & install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   > Windows note: if the `python` command isn’t registered, swap it for `py -3` in the snippets above.

2. **Download PlantVillage** from Kaggle and unzip so each class has its own folder under `data/raw/PlantVillage` (e.g., `Apple___Apple_scab`).

https://www.kaggle.com/datasets/emmarex/plantdisease

3. **Prepare stratified splits**
   ```bash
   python src/prepare_data.py --config config.yaml --clean
   ```

4. **Train the classifier**
   ```bash
   python src/train.py --config config.yaml
   ```
   Checkpoints and metrics are written to `outputs/models` and `outputs/metrics`.

5. **Plot accuracy & loss curves**
   ```bash
   python src/plot_metrics.py --config config.yaml
   ```
   Generates `outputs/metrics/training_curves.png` (throws an error if training hasn’t produced a history yet).

6. **Run inference on an image**
   ```bash
   python src/inference.py --model outputs/models/best_model.pth --image path/to/sample.jpg --config config.yaml
   ```
   Add `--annotate outputs/predictions/sample_annotated.jpg` if you want a copy of the image with the predicted label (and confidence) stamped on top.

7. **Serve predictions with Flask**
   ```bash
   python src/app.py
   ```
   Open `http://localhost:5000/` for a built-in web interface where you can upload an image and see the annotated prediction inline. Programmatic clients can continue to POST to `http://localhost:5000/predict` with a form field named `image`.

## Configuration

`config.yaml` centralizes hyper-parameters and paths:

- `data`: input/output directories, image size, loader workers, data split ratios, random seed.
- `train`: epochs, optimizer parameters, timm backbone, device preference, and whether to use mixed precision.
- `scheduler`: optional learning rate scheduling (ReduceLROnPlateau by default).
- `save`: output folders for checkpoints, logs, and metrics.
- `logging`: console frequency and optional experiment tracking hooks.

Modify values to experiment with larger image sizes, different backbones, or CPU-only training. The checkpoint stores backbone and image size, so inference stays consistent even if you tweak the config later.

## Optional: create a smaller subset

Need a quicker experiment? Sample a balanced subset first and then regenerate splits.

```bash
python src/create_subset.py --raw_dir data/raw/PlantVillage --out_dir data/subset --n_per_class 500 --clean
# Update config.yaml -> data.raw_dir: "data/subset"
python src/prepare_data.py --config config.yaml --clean
```

## Tips to boost accuracy

- **Upgrade the backbone**: try `tf_efficientnet_b3`, `tf_efficientnet_b4`, or transformers like `swin_base_patch4_window7_224`. Update `train.backbone` in `config.yaml`.
- **Increase image size**: bump `data.img_size` to 299 or 380 (ensure GPU memory allows it).
- **Longer training & augmentation**: raise `train.epochs`, tweak augmentations in `src/dataset.py`, or enable CutMix/MixUp via timm (e.g., wrap the model with `timm.data.mixup.Mixup`).
- **Fine-tune schedule**: lower `train.lr`, add cosine annealing, or introduce warm restarts.
- **Ensembling**: train multiple backbones and average softmax scores at inference time.

## Outputs

- `outputs/models/best_model.pth`: checkpoint with model weights, optimizer state, class labels, backbone metadata, and validation metrics.
- `outputs/metrics/training_history.csv`: epoch-wise train/val metrics.
- `outputs/metrics/training_curves.png`: side-by-side plots of accuracy and loss across epochs.
- `outputs/metrics/test_summary.json`: summary of the best checkpoint evaluated on the held-out test split.

## Supported classes & data split

The current project is trained on six PlantVillage labels:

- `Pepper__bell___Bacterial_spot`
- `Pepper__bell___healthy`
- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`
- `Tomato__Target_Spot`

After running `prepare_data.py`, the splits contain:

| Split | Images | Class distribution |
| --- | ---: | --- |
| Train | 4,374 | 797 / 1,182 / 800 / 800 / 121 / 674 |
| Val | 695 | 150 / 222 / 150 / 150 / 23 / – |
| Test | 232 | 50 / 74 / 50 / 50 / 8 / – |

*(Classes listed in order above; a dash means the class wasn’t sampled into that split because of rounding when applying the 5% test ratio to this subset.)*

## Current training results

- Best epoch: **9** (saved automatically as `best_model.pth`).
- Validation F1: **1.000** at epoch 9.
- Test performance (`outputs/metrics/test_summary.json`): accuracy **0.996**, F1 **0.997**, loss **0.057**.

Because the model only knows the six labels above, it will always predict one of them. For out-of-scope leaves (different crops/diseases), treat low confidence scores as “unknown” or expand the training set with additional classes.

## API usage

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/leaf.jpg"
```

For JSON clients, send `{ "image_base64": "..." }` instead of multipart form data.

A simple health check is provided at `GET /healthz`, and the root page now doubles as a lightweight demo UI for quick manual testing.

## Reproducibility

- Set `data.seed` in `config.yaml` to keep data splits stable.
- The training script seeds PyTorch, NumPy, and Python’s RNG.
- Mixed precision is enabled automatically when CUDA is available; disable by setting `train.mixed_precision: false`.

## Next steps

- Plug in experiment tracking (Weights & Biases, TensorBoard) via `logging` section.
- Add test-time augmentation in `src/inference.py` for more robust predictions.
- Package the project for deployment (Dockerfile, gunicorn + nginx) or convert the Flask app to FastAPI for async inference.
