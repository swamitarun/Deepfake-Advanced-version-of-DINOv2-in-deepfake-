"""
cache_features.py — Extract DINOv2 features ONCE and save to disk.

This runs the backbone on all images one time, saving the 768-dim CLS
tokens.  Afterwards you can train the small MLP head in seconds per epoch
instead of hours.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/cache_features.py --gpu 1
    CUDA_VISIBLE_DEVICES=1 python scripts/cache_features.py --gpu 1 --batch-size 128
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, get_device

# ImageNet normalization (DINOv2 standard)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_features(model, image_dir, device, batch_size=64):
    """Extract features for all images in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = sorted([f for f in Path(image_dir).iterdir()
                    if f.suffix.lower() in extensions])

    all_features = []
    all_names = []

    # Process in batches
    for i in tqdm(range(0, len(files), batch_size),
                  desc=f"  {Path(image_dir).name}", ncols=80):
        batch_files = files[i:i + batch_size]
        batch_imgs = []

        for fpath in batch_files:
            try:
                img = Image.open(fpath).convert('RGB')
                batch_imgs.append(TRANSFORM(img))
                all_names.append(fpath.name)
            except Exception as e:
                continue

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            features = model(batch_tensor)  # (B, 768)

        all_features.append(features.cpu().numpy())

    if all_features:
        return np.concatenate(all_features, axis=0), all_names
    return np.array([]), []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', default='data/cached_features')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg = load_config(args.config)
    setup_logging(log_dir=cfg['paths']['log_dir'])
    logger = logging.getLogger(__name__)
    device = get_device(cfg['device'])

    logger.info("=" * 50)
    logger.info("  FEATURE CACHING — Extract DINOv2 features once")
    logger.info("=" * 50)

    # Load DINOv2 backbone (frozen, no classifier)
    variant = cfg['model']['dino_variant']
    logger.info(f"Loading {variant} ...")
    backbone = torch.hub.load('facebookresearch/dinov2', variant, pretrained=True)
    backbone = backbone.to(device)
    backbone.eval()
    logger.info("Backbone loaded and set to eval mode.")

    # Output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw_dir = cfg['data']['raw_dir']
    faces_dir = cfg['data'].get('faces_dir', 'data/faces')

    # Extract features for raw images
    for class_name in ['real', 'fake']:
        # --- Raw features ---
        raw_class_dir = os.path.join(raw_dir, class_name)
        if os.path.exists(raw_class_dir):
            logger.info(f"\nExtracting RAW features for [{class_name}] ...")
            feats, names = extract_features(backbone, raw_class_dir, device, args.batch_size)
            np.save(str(out_dir / f'raw_{class_name}_features.npy'), feats)
            np.save(str(out_dir / f'raw_{class_name}_names.npy'), np.array(names))
            logger.info(f"  Saved {len(feats)} features → {out_dir / f'raw_{class_name}_features.npy'}")

        # --- Face features (if available) ---
        face_class_dir = os.path.join(faces_dir, class_name)
        if os.path.exists(face_class_dir):
            n_faces = len([f for f in Path(face_class_dir).iterdir()])
            if n_faces > 0:
                logger.info(f"\nExtracting FACE features for [{class_name}] ...")
                feats, names = extract_features(backbone, face_class_dir, device, args.batch_size)
                np.save(str(out_dir / f'face_{class_name}_features.npy'), feats)
                np.save(str(out_dir / f'face_{class_name}_names.npy'), np.array(names))
                logger.info(f"  Saved {len(feats)} features → {out_dir / f'face_{class_name}_features.npy'}")

    logger.info("\n✅ Feature caching complete!")
    logger.info(f"   Saved to: {out_dir}")
    logger.info("   Now run: python scripts/train_fast.py --gpu 1")


if __name__ == '__main__':
    main()
