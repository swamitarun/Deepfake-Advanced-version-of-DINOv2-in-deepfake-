# Deepfake Detection with DINOv2

A self-supervised deepfake detection system using **DINOv2** (Vision Transformer) as a pretrained feature extractor with a custom **MLP classifier**.

---

## 🧠 Architecture

```
Image (any size)
  → [Optional] MTCNN Face Detection (crop face region)
  → Resize & Normalize (224×224, ImageNet stats)
  → DINOv2 ViT-S/14 (FROZEN backbone)
     ├── Patch Embedding (16×16 → 384-dim)
     ├── [CLS] Token + Positional Encoding
     └── 12× Transformer Encoder Blocks
  → CLS Token Feature Vector (384-dim)
  → MLP Classifier (TRAINABLE)
     ├── Linear(384, 256) → ReLU → Dropout
     ├── Linear(256, 128) → ReLU → Dropout
     └── Linear(128, 2) → [REAL, FAKE]
  → Prediction + Confidence Score
```

## 🔑 Key Difference from BYOL

| Aspect | BYOL | DINOv2 |
|--------|------|--------|
| Architecture | CNN-based (ResNet) | Transformer-based (ViT) |
| SSL Method | Twin networks + augmentation | Self-distillation + masking |
| Training | Need to run SSL pretraining | Use pretrained weights directly |
| Features | Learned via momentum encoder | Already powerful, frozen |

---

## 📁 Project Structure

```
Deepfake_DINOv2_01/
├── configs/config.yaml          # All hyperparameters
├── data/
│   ├── raw/real/                # Place real images here
│   ├── raw/fake/                # Place fake images here
│   └── faces/                   # Face-cropped images (auto-generated)
├── src/
│   ├── data/                    # Dataset, transforms, video loader
│   ├── models/                  # DINOv2 extractor + MLP classifier
│   ├── training/                # Training loop + early stopping
│   ├── evaluation/              # Metrics + visualization
│   └── utils/                   # Face detection, helpers, plotting
├── scripts/
│   ├── prepare_data.py          # Dataset preparation
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict_image.py         # Single image prediction
│   └── predict_video.py         # Video prediction
├── models/checkpoints/          # Saved model weights
└── results/plots/               # Training curves, ROC, confusion matrix
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place images in `data/raw/`:
```
data/raw/
├── real/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── fake/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

Check dataset:
```bash
python scripts/prepare_data.py
```

### 3. Train
```bash
# Full training
python scripts/train.py

# Debug mode (small dataset, 3 epochs)
python scripts/train.py --debug

# Custom settings
python scripts/train.py --epochs 50 --batch-size 64 --lr 0.0005
```

### 4. Evaluate
```bash
python scripts/evaluate.py
```

### 5. Predict
```bash
# Single image
python scripts/predict_image.py --image path/to/image.jpg

# With face detection
python scripts/predict_image.py --image path/to/image.jpg --face-detect

# Video
python scripts/predict_video.py --video path/to/video.mp4 --num-frames 32
```

---

## ⚙️ Configuration

All settings are in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.dino_variant` | `dinov2_vits14` | DINOv2 model size |
| `model.freeze_backbone` | `true` | Freeze DINOv2 weights |
| `training.batch_size` | `32` | Training batch size |
| `training.epochs` | `30` | Maximum epochs |
| `training.learning_rate` | `0.001` | Classifier learning rate |
| `training.early_stopping_patience` | `7` | Early stopping patience |
| `face_detection.enabled` | `false` | Use MTCNN face cropping |

---

## 📊 Evaluation Metrics

- **Accuracy** — Overall correctness
- **Precision** — Of predicted fakes, how many are truly fake
- **Recall** — Of actual fakes, how many were detected
- **F1-Score** — Harmonic mean of precision and recall
- **AUC-ROC** — Area under the ROC curve
- **Confusion Matrix** — Visual breakdown of predictions

---

## 🎯 Training Strategy

1. **Freeze DINOv2** — Use pretrained features (no backbone training)
2. **Train MLP only** — Fast convergence, fewer parameters
3. **CrossEntropyLoss** — Standard binary classification loss
4. **Adam optimizer** — With cosine annealing LR schedule
5. **Early stopping** — Prevent overfitting
6. **Data augmentation** — Random crop, flip, color jitter

---

## 📹 Video Pipeline

```
Video → Extract N frames (evenly spaced)
      → Each frame → [Optional Face Crop] → DINOv2 → MLP → Prediction
      → Aggregate all frame predictions
      → Final REAL/FAKE decision (mean probability or majority vote)
```
