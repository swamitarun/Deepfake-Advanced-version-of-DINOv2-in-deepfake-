"""
predict_video.py — Advanced video deepfake prediction with temporal modeling.

Usage:
    python scripts/predict_video.py --video path/to/video.mp4
    python scripts/predict_video.py --video video.mp4 --aggregation lstm
    python scripts/predict_video.py --video video.mp4 --aggregation weighted
    python scripts/predict_video.py --video video.mp4 --aggregation mean

Aggregation modes:
    - lstm:     LSTM temporal model (learns temporal inconsistencies)
    - weighted: Attention-weighted averaging (learns frame importance)
    - mean:     Simple probability averaging (baseline)
    - majority: Majority vote across frames (baseline)
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, get_device
from src.data.transforms import get_inference_transforms
from src.data.video_loader import VideoFrameExtractor
from src.models.classifier import DualInputDeepfakeClassifier
from src.models.temporal_model import TemporalLSTMAggregator, WeightedAttentionAggregator
from src.utils.face_detect import FaceDetector


def predict_video(
    video_path: str,
    model: DualInputDeepfakeClassifier,
    transform,
    device: torch.device,
    num_frames: int = 16,
    face_detector: Optional[FaceDetector] = None,
    aggregation: str = 'lstm',
    temporal_model=None,
    dual_input: bool = True,
) -> dict:
    """
    Predict whether a video is real or fake.

    Temporal aggregation modes:
    1. LSTM: Extract features from all frames → LSTM → single prediction
    2. Weighted: Learn attention weights for each frame → weighted average
    3. Mean: Average softmax probabilities across frames
    4. Majority: Most common per-frame prediction wins

    Args:
        video_path: Path to video file.
        model: Trained DualInputDeepfakeClassifier.
        transform: Preprocessing transforms.
        device: Inference device.
        num_frames: Frames to extract.
        face_detector: Optional face detector.
        aggregation: 'lstm', 'weighted', 'mean', or 'majority'.
        temporal_model: Pretrained TemporalLSTMAggregator or WeightedAttentionAggregator.
        dual_input: Whether model uses dual input.

    Returns:
        Dict with prediction, confidence, and per-frame results.
    """
    logger = logging.getLogger(__name__)

    # ---- Step 1: Extract frames ----
    extractor = VideoFrameExtractor(num_frames=num_frames)
    pil_frames = extractor.extract_to_pil(video_path)

    if not pil_frames:
        return {'error': 'No frames extracted'}

    logger.info(f"Extracted {len(pil_frames)} frames from {video_path}")

    # ---- Step 2: Process each frame ----
    all_features = []
    all_probs = []
    all_preds = []
    frame_results = []

    model.eval()
    with torch.no_grad():
        for i, frame in enumerate(pil_frames):
            # Get face crop
            face_crop = None
            if dual_input:
                if face_detector:
                    face_crop = face_detector.detect_and_crop(frame)
                if face_crop is None:
                    face_crop = frame.copy()

            # Preprocess
            full_tensor = transform(frame).unsqueeze(0).to(device)
            face_tensor = None
            if face_crop is not None:
                face_tensor = transform(face_crop).unsqueeze(0).to(device)

            # For LSTM/weighted: extract features (before classifier)
            if aggregation in ('lstm', 'weighted'):
                features = model.extract_features(full_tensor, face_tensor)
                all_features.append(features)

            # For all modes: get per-frame predictions
            logits = model(full_tensor, face_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()
            label = 'REAL' if pred.item() == 0 else 'FAKE'

            all_probs.append([prob_real, prob_fake])
            all_preds.append(pred.item())
            frame_results.append({
                'frame': i,
                'prediction': label,
                'prob_real': prob_real,
                'prob_fake': prob_fake,
            })

    # ---- Step 3: Aggregate ----
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    if aggregation == 'lstm' and temporal_model is not None:
        # Stack features into sequence: (1, T, feat_dim)
        frame_seq = torch.cat(all_features, dim=0).unsqueeze(0).to(device)
        temporal_model.eval()
        with torch.no_grad():
            result = temporal_model.predict(frame_seq)
        final_pred = result['labels'][0]
        confidence = result['confidence'][0].item()

    elif aggregation == 'weighted' and temporal_model is not None:
        frame_seq = torch.cat(all_features, dim=0).unsqueeze(0).to(device)
        temporal_model.eval()
        with torch.no_grad():
            logits = temporal_model(frame_seq)
            probs_out = torch.softmax(logits, dim=1)
            pred_out = torch.argmax(probs_out, dim=1)
        final_pred = 'REAL' if pred_out.item() == 0 else 'FAKE'
        confidence = probs_out.max().item()

    elif aggregation == 'mean':
        avg_probs = all_probs.mean(axis=0)
        final_pred = 'REAL' if avg_probs[0] > avg_probs[1] else 'FAKE'
        confidence = max(avg_probs)

    elif aggregation == 'majority':
        real_count = (all_preds == 0).sum()
        fake_count = (all_preds == 1).sum()
        final_pred = 'REAL' if real_count > fake_count else 'FAKE'
        confidence = max(real_count, fake_count) / len(all_preds)

    else:
        # Fallback to mean if temporal model not available
        avg_probs = all_probs.mean(axis=0)
        final_pred = 'REAL' if avg_probs[0] > avg_probs[1] else 'FAKE'
        confidence = max(avg_probs)
        if aggregation in ('lstm', 'weighted'):
            logger.warning(
                f"Temporal model not loaded, falling back to mean aggregation"
            )

    return {
        'video': video_path,
        'prediction': final_pred,
        'confidence': float(confidence),
        'avg_prob_real': float(all_probs[:, 0].mean()),
        'avg_prob_fake': float(all_probs[:, 1].mean()),
        'total_frames': len(pil_frames),
        'frames_real': int((all_preds == 0).sum()),
        'frames_fake': int((all_preds == 1).sum()),
        'aggregation': aggregation,
        'frame_results': frame_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced Video Deepfake Prediction")
    parser.add_argument('--video', type=str, required=True, help='Video path')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num-frames', type=int, default=None)
    parser.add_argument('--no-face-detect', action='store_true')
    parser.add_argument('--aggregation', type=str, default=None,
                       choices=['lstm', 'weighted', 'mean', 'majority'])
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()
    logger = logging.getLogger(__name__)
    device = get_device(config['device'])

    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        return

    # ---- Model ----
    model_config = config['model']
    dual_input = model_config.get('dual_input', True)
    pooling_mode = model_config.get('pooling_mode', 'multi')

    model = DualInputDeepfakeClassifier(
        dino_variant=model_config['dino_variant'],
        hidden_dims=model_config['classifier']['hidden_dims'],
        num_classes=model_config['classifier']['num_classes'],
        dropout=model_config['classifier']['dropout'],
        freeze_backbone=model_config['freeze_backbone'],
        unfreeze_last_n_blocks=model_config['unfreeze_last_n_blocks'],
        pooling_mode=pooling_mode,
        dual_input=dual_input,
    )

    checkpoint_path = args.checkpoint or os.path.join(
        config['paths']['checkpoint_dir'], 'best_model.pth'
    )
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ---- Temporal model (if using LSTM/weighted aggregation) ----
    aggregation = args.aggregation or config['video'].get('aggregation', 'mean')
    temporal_model = None

    if aggregation in ('lstm', 'weighted'):
        # Compute feature dim for temporal model
        single_feat = 1152 if pooling_mode == 'multi' else 384
        total_feat = single_feat * 2 if dual_input else single_feat
        video_config = config['video']

        if aggregation == 'lstm':
            temporal_model = TemporalLSTMAggregator(
                input_dim=total_feat,
                hidden_dim=video_config.get('lstm_hidden_dim', 256),
                num_layers=video_config.get('lstm_num_layers', 1),
            ).to(device)
            # Try loading temporal model checkpoint
            temporal_ckpt = os.path.join(
                config['paths']['checkpoint_dir'], 'temporal_model.pth'
            )
            if os.path.exists(temporal_ckpt):
                temporal_model.load_state_dict(torch.load(temporal_ckpt, map_location=str(device)))
                logger.info("Loaded temporal LSTM model")
            else:
                logger.warning("No temporal model checkpoint found, falling back to mean")
                temporal_model = None
        elif aggregation == 'weighted':
            temporal_model = WeightedAttentionAggregator(
                input_dim=total_feat,
            ).to(device)
            temporal_ckpt = os.path.join(
                config['paths']['checkpoint_dir'], 'temporal_model.pth'
            )
            if os.path.exists(temporal_ckpt):
                temporal_model.load_state_dict(torch.load(temporal_ckpt, map_location=str(device)))
                logger.info("Loaded temporal attention model")
            else:
                logger.warning("No temporal model checkpoint, falling back to mean")
                temporal_model = None

    # ---- Face detector ----
    face_detector = None
    if not args.no_face_detect and dual_input:
        face_config = config['face_detection']
        face_detector = FaceDetector(
            margin=face_config['margin'],
            confidence_threshold=face_config['confidence_threshold'],
            image_size=config['data']['image_size'],
        )

    # ---- Predict ----
    transform = get_inference_transforms(config['data']['image_size'])
    num_frames = args.num_frames or config['video']['num_frames']

    result = predict_video(
        video_path=args.video,
        model=model,
        transform=transform,
        device=device,
        num_frames=num_frames,
        face_detector=face_detector,
        aggregation=aggregation,
        temporal_model=temporal_model,
        dual_input=dual_input,
    )

    # ---- Display ----
    print("\n" + "=" * 60)
    print("   ADVANCED DINOv2 — VIDEO PREDICTION")
    print("=" * 60)
    print(f"\n  Video:          {result['video']}")
    print(f"  Prediction:     {result['prediction']}")
    print(f"  Confidence:     {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"  Avg P(Real):    {result['avg_prob_real']:.4f}")
    print(f"  Avg P(Fake):    {result['avg_prob_fake']:.4f}")
    print(f"  Frames total:   {result['total_frames']}")
    print(f"  Frames → REAL:  {result['frames_real']}")
    print(f"  Frames → FAKE:  {result['frames_fake']}")
    print(f"  Aggregation:    {result['aggregation']}")

    print("\n  Per-frame:")
    print("  " + "-" * 45)
    for fr in result['frame_results']:
        bar = "█" * int(fr['prob_fake'] * 20)
        print(
            f"  Frame {fr['frame']:3d}: {fr['prediction']:4s} "
            f"(Fake: {fr['prob_fake']:.3f}) {bar}"
        )
    print("=" * 60)


if __name__ == '__main__':
    main()
