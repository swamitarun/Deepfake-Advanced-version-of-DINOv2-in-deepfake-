# DIVOG50 — Advanced Deepfake Detection (Built on DINOv2)

> **Created by avoub**

**DIVOG50** (also referred to as **DINO-G50**) is a heavily modified and extended deepfake detection system built on top of Facebook's **DINOv2** Vision Transformer. While the original DINOv2 is a general-purpose **Self-Supervised Learning (SSL)** image encoder, DIVOG50 leverages the SSL-pretrained backbone and introduces a series of architectural and training innovations specifically designed for high-accuracy deepfake detection in both images and videos.

---

## 🔄 What We Changed: DINOv2 → DIVOG50

The table below summarises every modification made to the original DINOv2 to produce DIVOG50:

| Component | Original DINOv2 | DIVOG50 Changes |
|-----------|----------------|----------------|
| **Input** | Single image | **Dual-input**: full image + MTCNN face crop processed in parallel |
| **Backbone** | Fully frozen ViT-B/14 | Last **2 transformer blocks unfrozen** for task-specific fine-tuning |
| **Feature extraction** | CLS token only (768-dim) | CLS token (768-dim) per branch; supports multi-token pooling (1152-dim per branch) |
| **Feature fusion** | — | Dual branches **concatenated** → 1536-dim joint representation |
| **Classifier head** | Linear probe | **3-layer MLP** with BatchNorm, GELU activations and Dropout (1536→512→256→2) |
| **Loss function** | Standard CE | CrossEntropyLoss + **label smoothing** (0.1) + **class-balanced weights** |
| **Optimizer** | N/A (frozen) | **AdamW** with **cosine annealing** LR schedule and **LR warmup** |
| **Training stability** | — | **Mixed-precision (AMP)** training with gradient clipping (max norm 1.0) |
| **Video support** | Image only | **Temporal LSTM aggregator** + **attention-weighted frame aggregator** |
| **Face detection** | — | **MTCNN** (FaceNet-PyTorch) pipeline with configurable margin and confidence threshold |
| **Data augmentation** | — | Random crop, horizontal flip, colour jitter, and optional advanced augmentations |
| **Feature caching** | — | Pre-computed DINOv2 SSL features cached to disk for fast repeated training |
| **Deployment** | — | **Flask REST API**, **web app** (DeepShield AI), and **Hugging Face Space** |
| **Real-world testing** | — | Dedicated `real_world_test/test_real.py` for end-to-end video testing |

---

## 🧠 DIVOG50 Architecture

```
Image (any size)
  ┌────────────────────────────────────────────────────┐
  │  Branch A — Full Image                             │
  │    Resize & Normalize (224×224, ImageNet stats)    │
  │    DINOv2 ViT-B/14 (SSL pre-trained, last 2        │
  │    blocks fine-tuned)                              │
  │      ├── Patch Embedding (14×14 → 768-dim)         │
  │      ├── [CLS] Token + Positional Encoding         │
  │      └── 12× Transformer Encoder Blocks            │
  │    → CLS Feature Vector (768-dim)                  │
  └────────────────────────────────────────────────────┘
           │
           ├──── concatenate (1536-dim) ────┐
           │                               │
  ┌────────────────────────────────────────────────────┐
  │  Branch B — MTCNN Face Crop (shared weights)       │
  │    Same DINOv2 SSL extractor (weight-tied)         │
  │    → CLS Feature Vector (768-dim)                  │
  └────────────────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────▼───────────┐
  │  MLP Classifier (TRAINABLE)                        │
  │    Linear(1536, 512) → BatchNorm → GELU → Dropout  │
  │    Linear(512, 256)  → BatchNorm → GELU → Dropout  │
  │    Linear(256, 2)    → [REAL, FAKE]                │
  └────────────────────────────────────────────────────┘
  → Prediction + Confidence Score
```

### Video Pipeline (Temporal DIVOG50)

```
Video → Extract N frames (evenly spaced)
      → Each frame → [MTCNN Face Crop] → DINOv2 SSL Dual Extractor → Frame Features
      → Choose aggregation mode:
          ┌── LSTM temporal model (learns frame-to-frame inconsistencies)
          ├── Attention-weighted aggregation (learns most informative frames)
          ├── Mean probability averaging (fast baseline)
          └── Majority vote across frames (robust baseline)
      → Final REAL / FAKE decision + confidence score
```

---

## 🔑 DIVOG50 vs Original DINOv2 vs BYOL

| Aspect | BYOL | DINOv2 (original) | DIVOG50 (ours) |
|--------|------|-------------------|----------------|
| Architecture | CNN (ResNet) | ViT-B/14 | ViT-B/14 + dual-input MLP |
| SSL Method | Momentum encoder | Self-distillation + masking | SSL (DINOv2 self-distillation) + task fine-tuning |
| Input | Single image | Single image | Full image **and** face crop |
| Fine-tuning | Full | None (frozen) | Last 2 blocks + MLP head |
| Video support | No | No | LSTM + attention aggregation |
| Face pipeline | No | No | MTCNN auto face crop |
| Deployment | No | No | Flask API + web app + HF Space |

---

## 📁 Project Structure

This project contains **two experiment runs**:
- **Epoch 2** — original folders (`scripts/`, `results/`, `hf_webapp/`, `configs/config.yaml`)
- **Epoch 10** — extended folders (`scripts2/`, `results2/`, `hf_webapp2/`, `configs/config2.yaml`)

```
DIVOG50/
├── configs/
│   ├── config.yaml              # Epoch 2 — hyperparameters (epochs=2)
│   └── config2.yaml             # Epoch 10 — hyperparameters (epochs=10)
├── data/
│   ├── raw/real/                # Place real images here
│   ├── raw/fake/                # Place fake images here
│   └── faces/                   # Face-cropped images (auto-generated by MTCNN)
├── src/
│   ├── models/
│   │   ├── dino_extractor.py    # DINOv2 SSL ViT-B/14 wrapper with partial unfreezing
│   │   ├── classifier.py        # DeepfakeClassifier: dual-input MLP on DINOv2 SSL features
│   │   └── temporal_model.py    # LSTM + attention aggregators for video
│   ├── data/                    # Dataset, transforms, video loader, feature caching
│   ├── training/
│   │   └── trainer.py           # AMP trainer, LR warmup, early stopping, AUC/F1 tracking
│   ├── evaluation/
│   │   └── evaluator.py         # Full metrics suite (Acc, Prec, Recall, F1, AUC-ROC, CM)
│   └── utils/
│       ├── face_detect.py       # MTCNN face detector
│       ├── helpers.py           # Config loading, device setup, seed
│       └── visualization.py     # Training curves, ROC, confusion matrix plots
│
├── scripts/                     # ── Epoch 2 scripts ──
│   ├── prepare_data.py          # Dataset preparation & integrity check
│   ├── train.py                 # Full training script (AMP, dual-input, class weights)
│   ├── train_fast.py            # Fast MLP-only training on cached SSL features
│   ├── evaluate.py              # Evaluation on test set
│   ├── predict_image.py         # Single image prediction
│   ├── predict_video.py         # Video prediction with LSTM / attention / mean / majority
│   ├── cache_features.py        # Pre-compute and cache DINOv2 SSL features
│   ├── extract_frames.py        # Extract frames from video files
│   └── api.py                   # Flask REST API server
├── results/                     # ── Epoch 2 outputs ──
│   ├── logs/                    # Training logs (training.log, training_log.json)
│   └── plots/                   # Training curves, ROC curves, confusion matrices
│
├── scripts2/                    # ── Epoch 10 scripts ──
│   ├── prepare_data.py
│   ├── train.py
│   ├── train_fast.py
│   ├── evaluate.py
│   ├── predict_image.py
│   ├── predict_video.py
│   ├── cache_features.py
│   ├── extract_frames.py
│   └── api.py
├── results2/                    # ── Epoch 10 outputs ──
│   ├── logs/                    # Training logs (training.log, training_log.json)
│   └── plots/                   # Training curves, ROC curves, confusion matrices
│
├── hf_webapp/                   # ── Epoch 2 Hugging Face Space ──
│   ├── app.py                   # FastAPI backend for Hugging Face Spaces
│   ├── Dockerfile
│   ├── requirements.txt
│   └── static/
├── hf_webapp2/                  # ── Epoch 10 Hugging Face Space ──
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── static/
│
├── real_world_test/
│   └── test_real.py             # End-to-end real-world video deepfake testing
├── models/checkpoints/          # Epoch 2 saved model weights (best_model.pth)
├── models2/checkpoints/         # Epoch 10 saved model weights (best_model.pth)
└── requirements.txt             # Python dependencies
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
│   └── ...
└── fake/
    ├── img001.jpg
    └── ...
```

Check dataset integrity:
```bash
python scripts/prepare_data.py
```

### 3. Train DIVOG50

**Epoch 2 run** (uses `configs/config.yaml`, outputs to `results/`):
```bash
# Full dual-input training (full image + face crop, 2 epochs)
python scripts/train.py

# Debug mode (100 samples, 3 epochs)
python scripts/train.py --debug

# Custom GPU / hyperparameters
python scripts/train.py --gpu 1 --epochs 2 --batch-size 64

# Advanced augmentations
python scripts/train.py --advanced-aug

# Fast MLP-only training on cached SSL features
python scripts/cache_features.py   # pre-compute once
python scripts/train_fast.py       # trains in minutes
```

**Epoch 10 run** (uses `configs/config2.yaml`, outputs to `results2/`):
```bash
# Full dual-input training (10 epochs)
python scripts2/train.py

# Fast MLP-only training on cached SSL features
python scripts2/cache_features.py
python scripts2/train_fast.py
```

### 4. Evaluate
```bash
# Epoch 2
python scripts/evaluate.py

# Epoch 10
python scripts2/evaluate.py
```

### 5. Predict
```bash
# Single image
python scripts/predict_image.py --image path/to/image.jpg

# With face detection
python scripts/predict_image.py --image path/to/image.jpg --face-detect

# Video — LSTM temporal aggregation (default)
python scripts/predict_video.py --video path/to/video.mp4

# Video — choose aggregation mode
python scripts/predict_video.py --video video.mp4 --aggregation lstm
python scripts/predict_video.py --video video.mp4 --aggregation weighted
python scripts/predict_video.py --video video.mp4 --aggregation mean
python scripts/predict_video.py --video video.mp4 --aggregation majority
```

### 6. Real-World Video Testing
```bash
# Place .mp4 / .avi / .mov files in real_world_test/tests/
python real_world_test/test_real.py
```

### 7. Run the API Server
```bash
# Epoch 2
python scripts/api.py
# POST /predict  — accepts an image/video, returns JSON verdict

# Epoch 10
python scripts2/api.py
```

---

## ⚙️ Configuration

**Epoch 2** settings are in `configs/config.yaml`; **Epoch 10** settings are in `configs/config2.yaml`.

| Parameter | Epoch 2 (`config.yaml`) | Epoch 10 (`config2.yaml`) | Description |
|-----------|------------------------|--------------------------|-------------|
| `model.dino_variant` | `dinov2_vitb14` | `dinov2_vitb14` | DINOv2 SSL backbone size (ViT-B/14, 768-dim) |
| `model.freeze_backbone` | `true` | `true` | Freeze most SSL backbone weights |
| `model.unfreeze_last_n_blocks` | `2` | `2` | Fine-tune last N transformer blocks |
| `model.dual_input` | `true` | `true` | Enable dual-input (full image + face crop) |
| `training.batch_size` | `64` | `64` | Training batch size |
| `training.epochs` | **`2`** | **`10`** | Maximum training epochs |
| `training.learning_rate` | `0.0003` | `0.0003` | MLP classifier learning rate |
| `training.backbone_lr_factor` | `0.05` | `0.05` | Backbone LR = `lr × factor` (much smaller) |
| `training.weight_decay` | `0.01` | `0.01` | AdamW weight decay |
| `training.warmup_epochs` | `3` | `3` | LR warmup epochs |
| `training.early_stopping_patience` | `10` | `10` | Early stopping patience |
| `training.label_smoothing` | `0.1` | `0.1` | Cross-entropy label smoothing |
| `paths.checkpoint_dir` | `models/checkpoints` | `models2/checkpoints` | Model checkpoint output directory |
| `paths.log_dir` | `results/logs` | `results2/logs` | Training log output directory |
| `paths.plot_dir` | `results/plots` | `results2/plots` | Plot output directory |
| `face_detection.enabled` | `true` | `true` | Use MTCNN face cropping |
| `face_detection.confidence_threshold` | `0.9` | `0.9` | MTCNN detection threshold |
| `face_detection.margin` | `40` | `40` | Pixels to expand around detected face |

---

## 📊 Evaluation Metrics

DIVOG50 reports a comprehensive set of metrics after evaluation:

- **Accuracy** — Overall correctness across REAL and FAKE classes
- **Precision** — Of all predicted fakes, how many are truly fake
- **Recall** — Of all actual fakes, how many were correctly detected
- **F1-Score** — Harmonic mean of precision and recall
- **AUC-ROC** — Area under the ROC curve (main benchmark metric)
- **Confusion Matrix** — Full visual breakdown of TP / TN / FP / FN
- **Per-class breakdown** — Precision, Recall, F1 for both REAL and FAKE separately
- **ROC Curve plot** — Epoch 2: `results/plots/` · Epoch 10: `results2/plots/`

---

## 🎯 Training Strategy

1. **SSL-pretrained DINOv2 backbone** — SSL features (DINOv2 self-distillation + masking) used as a powerful frozen representation
2. **Dual-input** — Full image and face crop processed through the same ViT-B/14 (weight-tied SSL branches)
3. **Partial unfreezing** — Last 2 transformer blocks and the normalization layer are trainable; the rest of the SSL backbone is frozen
4. **Differential learning rates** — Backbone LR is 20× smaller than MLP head LR to avoid catastrophic forgetting of SSL features
5. **Class-balanced loss** — CrossEntropyLoss with inverse-frequency class weights + label smoothing (0.1)
6. **AdamW + cosine annealing** — Weight decay 0.01, cosine decay after warmup
7. **LR warmup** — 3-epoch linear ramp-up for stable early training
8. **Mixed-precision (AMP)** — FP16 forward/backward pass with gradient scaler for speed
9. **Gradient clipping** — Max norm 1.0 to prevent exploding gradients
10. **Early stopping** — Stops when validation accuracy does not improve for 10 epochs

---

## 📹 Temporal Video Aggregation (DIVOG50 Innovation)

Standard deepfake video detectors process each frame independently and average the results, which **ignores temporal inconsistencies** — a key signal in deepfakes (flickering, temporal jitter, blending artifacts across frames).

DIVOG50 addresses this with two temporal aggregation strategies:

### LSTM Temporal Model
```
Frame SSL features (T × 2304)
  → Feature projection (2304 → 512, LayerNorm + GELU)
  → LSTM (512 → 256 hidden, learns temporal patterns)
  → Final hidden state
  → Classifier (256 → 128 → 2)
  → REAL / FAKE
```
The LSTM learns patterns like "abrupt feature changes between frames 5 and 6" or "oscillating texture typical of GAN-generated content".

### Weighted Attention Aggregator
```
Frame SSL features (T × 2304)
  → Attention network (per-frame scalar weight via Tanh)
  → Softmax normalization across frames
  → Weighted sum → aggregated feature (2304)
  → Classifier → REAL / FAKE
```
More interpretable than LSTM — you can visualise which frames the model found most suspicious.

---

## 🌐 Deployment

### DeepShield AI — Epoch 2 (`hf_webapp/`)
A self-contained FastAPI backend deployed on Hugging Face Spaces (Docker) trained for **2 epochs**. It exposes:
- `POST /predict` — accepts a video file (mp4/mov/avi, max 30 MB), returns JSON:
  ```json
  {
    "fake_probability": 0.87,
    "real_probability": 0.13,
    "verdict": "FAKE",
    "frame_count": 16
  }
  ```

### DeepShield AI — Epoch 10 (`hf_webapp2/`)
Same interface as above, but backed by the **10-epoch** model checkpoint (`models2/checkpoints/best_model.pth`) for improved accuracy.

### Flask REST API
- `scripts/api.py` — Epoch 2 local Flask server
- `scripts2/api.py` — Epoch 10 local Flask server

---

## �� Authors & Credits

Tarun Swami

- **Tarun Swami** — Architecture design, DIVOG50 SSL modifications, training pipeline, web app, deployment
- Built upon new fined tune using **DINOv2** by Meta AI Research (Facebook Research)  — SSL pre-training via self-distillation with no labels
- Face detection via **FaceNet-PyTorch** (MTCNN implementation)
