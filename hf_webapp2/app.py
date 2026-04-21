"""
DeepShield AI — Full-Stack FastAPI Backend
Serves the frontend UI + deepfake detection API from one HF Space.
Self-contained version with exact architectural parity to test_real.py 
"""

import os
import sys
import uuid
import shutil
import logging
import tempfile
from pathlib import Path
from functools import lru_cache

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# EXACT PARITY MODEL DEFINITIONS (Copied from src/ to be standalone)
# -------------------------------------------------------------

class DINOv2Extractor(nn.Module):
    def __init__(self, variant: str = 'dinov2_vitb14'):
        super().__init__()
        self.embed_dim = 768
        logger.info(f"Loading {variant} from torch.hub ...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', variant, pretrained=True,
        )
        logger.info("DINOv2 loaded.")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepfakeDetector(nn.Module):
    def __init__(self, dual_input: bool = True):
        super().__init__()
        self.dual_input = dual_input
        self.extractor = DINOv2Extractor('dinov2_vitb14')
        feat_dim = 1536 if dual_input else 768
        self.classifier = MLPClassifier(feat_dim)

    def forward(self, full_image: torch.Tensor, face_crop: torch.Tensor = None) -> torch.Tensor:
        full_feat = self.extractor(full_image)
        if self.dual_input:
            face_feat = self.extractor(face_crop if face_crop is not None else full_image)
            features  = torch.cat([full_feat, face_feat], dim=1)
        else:
            features = full_feat
        return self.classifier(features)

# -------------------------------------------------------------
# APP SETTINGS & SETUP
# -------------------------------------------------------------

app = FastAPI(
    title="DeepShield AI",
    description="DINO-G50 deepfake detector — full-stack web app",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = Path("best_model.pth")
MAX_FRAMES = 20
MAX_FILE_MB = 30
MAX_DURATION_SEC = 60

# MTCNN face detector setup to mimic src/utils/face_detect.py precisely
try:
    MTCNN_DETECTOR = MTCNN(
        image_size=224,
        margin=40,
        keep_all=False,
        post_process=False,
        device='cpu'
    )
    logger.info("MTCNN face detector initialized.")
except Exception as e:
    MTCNN_DETECTOR = None
    logger.warning(f"MTCNN init failed (will use fallback): {e}")

# Exact transform replication
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_face_crop(img: Image.Image) -> Image.Image:
    if MTCNN_DETECTOR is None:
        return None
    try:
        boxes, probs = MTCNN_DETECTOR.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]
        if best_prob < 0.9:
            return None
            
        box = boxes[best_idx]
        w, h = img.size
        x1, y1, x2, y2 = [int(b) for b in box]
        margin = 40
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face = img.crop((x1, y1, x2, y2))
        return face.resize((224, 224), Image.LANCZOS)
    except Exception:
        pass
    return None

@lru_cache(maxsize=1)
def load_model() -> DeepfakeDetector:
    # First check default path, then fallback if possible
    ckpt_path_to_load = None
    if not CHECKPOINT_PATH.exists():
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models2/checkpoints/best_model.pth')
        if os.path.exists(fallback_path):
            ckpt_path_to_load = fallback_path
        else:
            raise RuntimeError("best_model.pth not found. Upload it to this HF Space.")
    else:
        ckpt_path_to_load = str(CHECKPOINT_PATH)

    logger.info(f"Loading checkpoint on {DEVICE} from {ckpt_path_to_load} ...")
    ckpt = torch.load(ckpt_path_to_load, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    # Determine architecture
    mlp_w = state.get("classifier.net.0.weight", None)
    dual = (mlp_w.shape[1] == 1536) if mlp_w is not None else True

    model = DeepfakeDetector(dual_input=dual).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Model ready. dual_input={dual}, device={DEVICE}")
    return model

def extract_frames(video_path: str, temp_dir: str, num_frames: int = MAX_FRAMES) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, (total if total > 0 else 300) // num_frames)
    indices = set(range(0, total if total > 0 else 300, step))
    
    saved = []
    for i in range(total if total > 0 else 300):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = os.path.join(temp_dir, f"frame_{len(saved):03d}.jpg")
            Image.fromarray(rgb).save(path)
            saved.append(path)
            if len(saved) >= num_frames: break
    cap.release()
    return saved


def run_inference(model: DeepfakeDetector, frame_paths: list) -> dict:
    fake_probs = []
    with torch.no_grad():
        for fpath in frame_paths:
            try:
                img = Image.open(fpath).convert("RGB")
                t_img = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                t_face = t_img
                
                if model.dual_input:
                    face_crop = detect_face_crop(img)
                    if face_crop is not None:
                        t_face = TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)

                logits = model(t_img, t_face if model.dual_input else None)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                fake_probs.append(prob)
            except Exception as e:
                logger.warning(f"Skipping frame {fpath}: {e}")

    if not fake_probs:
        raise ValueError("No frames could be processed.")

    video_fake_prob = float(np.mean(fake_probs))
    is_fake = video_fake_prob > 0.5
    avg_real = 1.0 - video_fake_prob

    return {
        "verdict": "FAKE" if is_fake else "REAL",
        "fake_probability": round(video_fake_prob * 100, 1),
        "real_probability": round(avg_real * 100, 1),
        "frame_count": len(fake_probs),
        "confidence": round(max(video_fake_prob, avg_real) * 100, 1),
        "per_frame_scores": [round(p * 100, 1) for p in fake_probs],
    }

# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")

@app.get("/health")
def health_check():
    try:
        model_loaded = CHECKPOINT_PATH.exists() or os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models2/checkpoints/best_model.pth'))
    except:
        model_loaded = False
        
    return {
        "status": "ok",
        "model": "DINO-G50 Deepfake Detector",
        "device": str(DEVICE),
        "model_loaded": model_loaded,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""

    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported type '{ext}'. Use: {allowed_exts}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_MB} MB.")

    job_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.gettempdir()) / f"deepshield_{job_id}"
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / f"input{ext}"

    try:
        with open(video_path, "wb") as f:
            f.write(content)
        del content

        model = load_model()
        logger.info(f"[{job_id}] Processing: {file.filename} ({size_mb:.1f} MB)")

        if ext in {".mp4", ".mov", ".avi", ".mkv"}:
            frame_paths = extract_frames(str(video_path), str(frames_dir))
            if not frame_paths:
                raise HTTPException(422, "No frames could be extracted from video.")
        else:
            img_path = frames_dir / f"frame_0000{ext}"
            shutil.copy(video_path, img_path)
            frame_paths = [str(img_path)]

        result = run_inference(model, frame_paths)
        result["filename"] = file.filename
        result["file_size_mb"] = round(size_mb, 2)
        result["job_id"] = job_id

        logger.info(f"[{job_id}] Result: {result['verdict']} ({result['fake_probability']}% fake)")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"[{job_id}] Cleanup done.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
