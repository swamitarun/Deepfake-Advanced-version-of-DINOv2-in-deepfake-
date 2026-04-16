"""
predict_image.py — Single image deepfake prediction (dual-input).

Usage:
    python scripts/predict_image.py --image path/to/image.jpg
    python scripts/predict_image.py --image path/to/image.jpg --no-face-detect

Pipeline:
    Image → [Full Image + Face Crop] → DINOv2 (shared) → Multi-token features
    → Concatenate → Advanced Classifier → REAL/FAKE + confidence
"""

import os
import sys
import argparse
import logging

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, get_device
from src.data.transforms import get_inference_transforms
from src.models.classifier import DualInputDeepfakeClassifier
from src.utils.face_detect import FaceDetector


def predict_single_image(
    image_path: str,
    model: DualInputDeepfakeClassifier,
    transform,
    device: torch.device,
    face_detector=None,
    dual_input: bool = True,
) -> dict:
    """
    Predict whether a single image is real or fake.

    Args:
        image_path: Path to the image file.
        model: Trained DualInputDeepfakeClassifier.
        transform: Preprocessing transforms.
        device: Device for inference.
        face_detector: FaceDetector for cropping face region.
        dual_input: If True, passes both full image and face crop.

    Returns:
        Dict with prediction, confidence, and probabilities.
    """
    # Load full image
    full_image = Image.open(image_path).convert('RGB')

    # Get face crop
    face_crop = None
    if dual_input:
        if face_detector:
            face_crop = face_detector.detect_and_crop(full_image)
        if face_crop is None:
            face_crop = full_image.copy()  # Fallback

    # Preprocess
    full_tensor = transform(full_image).unsqueeze(0).to(device)
    face_tensor = None
    if face_crop is not None:
        face_tensor = transform(face_crop).unsqueeze(0).to(device)

    # Predict
    result = model.predict(full_tensor, face_tensor)

    return {
        'image': image_path,
        'prediction': result['labels'][0],
        'confidence': result['confidence'][0].item(),
        'prob_real': result['prob_real'][0].item(),
        'prob_fake': result['prob_fake'][0].item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict deepfake on a single image")
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--no-face-detect', action='store_true',
                       help='Disable face detection (use full image for both branches)')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()
    logger = logging.getLogger(__name__)
    device = get_device(config['device'])

    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
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
    result = predict_single_image(
        image_path=args.image,
        model=model,
        transform=transform,
        device=device,
        face_detector=face_detector,
        dual_input=dual_input,
    )

    print("\n" + "=" * 50)
    print("   ADVANCED DINOv2 — IMAGE PREDICTION")
    print("=" * 50)
    print(f"\n  Image:      {result['image']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    print(f"  P(Real):    {result['prob_real']:.4f}")
    print(f"  P(Fake):    {result['prob_fake']:.4f}")
    print(f"  Mode:       {'Dual input (full + face)' if dual_input else 'Single input'}")
    print("=" * 50)


if __name__ == '__main__':
    main()
