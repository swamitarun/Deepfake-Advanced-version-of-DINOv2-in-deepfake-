"""
prepare_data.py — Dataset preparation script.

Usage:
    python scripts/prepare_data.py --config configs/config.yaml

This script:
    1. Scans data/raw/ for real/ and fake/ subdirectories
    2. (Optional) Runs face detection and crops faces
    3. Reports dataset statistics
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging
from src.utils.face_detect import FaceDetector


def count_images(directory: str) -> dict:
    """Count images in real/ and fake/ subdirectories."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    counts = {}

    for class_name in ['real', 'fake']:
        class_dir = Path(directory) / class_name
        if class_dir.exists():
            count = sum(1 for f in class_dir.iterdir()
                       if f.suffix.lower() in extensions)
            counts[class_name] = count
        else:
            counts[class_name] = 0

    return counts


def crop_faces(config: dict):
    """Run face detection on all images and save cropped faces."""
    logger = logging.getLogger(__name__)

    raw_dir = config['data']['raw_dir']
    faces_dir = config['data']['faces_dir']
    face_config = config['face_detection']

    detector = FaceDetector(
        margin=face_config['margin'],
        confidence_threshold=face_config['confidence_threshold'],
        image_size=config['data']['image_size'],
        device=config.get('device', 'cpu'),
    )

    for class_name in ['real', 'fake']:
        src_dir = Path(raw_dir) / class_name
        dst_dir = Path(faces_dir) / class_name
        os.makedirs(dst_dir, exist_ok=True)

        if not src_dir.exists():
            logger.warning(f"Source directory not found: {src_dir}")
            continue

        image_files = [f for f in src_dir.iterdir()
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

        logger.info(f"Processing {len(image_files)} {class_name} images...")

        for img_path in tqdm(image_files, desc=f"Face crop [{class_name}]"):
            save_path = dst_dir / img_path.name
            if save_path.exists():
                continue
                
            try:
                face = detector.detect_and_crop(str(img_path))
                if face is not None:
                    face.save(str(save_path))
            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for DINOv2 deepfake detection")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--crop-faces', action='store_true',
                       help='Run face detection and crop faces')
    parser.add_argument('--fix-corrupted', action='store_true',
                       help='Delete corrupted images found during integrity check')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config['paths']['log_dir'])

    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("DATASET PREPARATION")
    logger.info("=" * 50)

    raw_dir = config['data']['raw_dir']

    # ---- Step 1: Full integrity check ----
    from src.data.integrity import check_dataset_integrity
    integrity = check_dataset_integrity(
        data_dir=raw_dir,
        fix_corrupted=args.fix_corrupted,
    )

    if not integrity['is_healthy']:
        logger.error("Dataset has issues. Fix them before training.")
        if not args.fix_corrupted and integrity['total_corrupted'] > 0:
            logger.info("TIP: Run with --fix-corrupted to auto-delete corrupted images")
        return

    # ---- Step 2: Face cropping (optional) ----
    if args.crop_faces or config['face_detection']['enabled']:
        logger.info("\nRunning face detection and cropping...")
        crop_faces(config)
        faces_dir = config['data']['faces_dir']
        face_counts = count_images(faces_dir)
        logger.info(f"\nFace-cropped images saved to {faces_dir}")
        logger.info(f"  Real faces: {face_counts['real']}")
        logger.info(f"  Fake faces: {face_counts['fake']}")

    logger.info("\n✅ Dataset preparation complete!")


if __name__ == '__main__':
    main()
