"""
Feature Caching — Run DINOv2 once, save features as .npy, train classifier without GPU-heavy backbone.

=================================================================================
WHY FEATURE CACHING:
=================================================================================

Problem: Every training epoch re-runs DINOv2 on every image. DINOv2 is the
most expensive part (22M params, transformer forward pass). But if the
backbone is FROZEN (or we only fine-tune the last 2 blocks), the features
for frozen layers don't change between epochs — we're wasting GPU time!

Solution: Run DINOv2 ONCE on the entire dataset, save the feature vectors
as .npy files. Then train the classifier using these cached features.

Speed improvement: ~10-20x faster training per epoch because we skip
the entire DINOv2 forward pass.

When to cache:
- Backbone is fully frozen → features NEVER change → always cache
- Backbone is partially fine-tuned → DON'T cache (features change each epoch)

=================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.data.transforms import get_val_transforms
from src.models.dino_extractor import DINOv2Extractor

logger = logging.getLogger(__name__)


# ============================================================
# Step 1: Extract and cache features
# ============================================================

def cache_features(
    data_dir: str,
    cache_dir: str,
    dino_variant: str = 'dinov2_vits14',
    pooling_mode: str = 'multi',
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
    dual_input: bool = True,
    faces_dir: str = None,
):
    """
    Run DINOv2 on all images and save features to disk.

    Saves:
        cache_dir/features.npy  — (N, feat_dim) feature matrix
        cache_dir/labels.npy    — (N,) label array
        cache_dir/paths.npy     — (N,) image path array
        cache_dir/face_features.npy — (N, feat_dim) face features (if dual_input)

    Args:
        data_dir: Dataset directory with real/ and fake/.
        cache_dir: Directory to save cached features.
        dino_variant: DINOv2 model variant.
        pooling_mode: 'multi' or 'cls'.
        image_size: Input image size.
        batch_size: Batch size for feature extraction.
        num_workers: DataLoader workers.
        device: Device to run DINOv2 on.
        dual_input: If True, also extract face crop features.
        faces_dir: Pre-cropped faces directory.
    """
    os.makedirs(cache_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FEATURE CACHING — DINOv2 One-Pass Extraction")
    print("=" * 60)

    # ---- Load DINOv2 extractor (frozen, no classifier) ----
    extractor = DINOv2Extractor(
        variant=dino_variant,
        freeze=True,               # Always freeze for caching
        unfreeze_last_n_blocks=0,   # Fully frozen for consistent caching
        pooling_mode=pooling_mode,
    ).to(device)
    extractor.eval()

    feat_dim = extractor.get_feature_dim()
    print(f"  Model: {dino_variant} | Pooling: {pooling_mode} | Feat dim: {feat_dim}")

    # ---- Prepare dataset ----
    transform = get_val_transforms(image_size)

    # Simple dataset for feature extraction (no augmentation)
    from src.data.dataset import DeepfakeDataset
    dataset = DeepfakeDataset(data_dir=data_dir, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- Extract features ----
    all_features = []
    all_labels = []
    all_paths = []

    print(f"  Extracting features from {len(dataset)} images...\n")

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Extracting", ncols=80):
            images = images.to(device)
            features = extractor(images)  # (B, feat_dim)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    # ---- Save to disk ----
    np.save(os.path.join(cache_dir, 'features.npy'), all_features)
    np.save(os.path.join(cache_dir, 'labels.npy'), all_labels)

    print(f"\n  Saved features: {all_features.shape} to {cache_dir}/")
    print(f"  Labels: {all_labels.shape}")
    print(f"  Real: {(all_labels == 0).sum()}, Fake: {(all_labels == 1).sum()}")

    # ---- Also cache face features if dual_input ----
    if dual_input and faces_dir and os.path.exists(os.path.join(faces_dir, 'real')):
        print(f"\n  Extracting face crop features from {faces_dir}...")
        face_dataset = DeepfakeDataset(data_dir=faces_dir, transform=transform)
        face_loader = DataLoader(
            face_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        face_features = []
        with torch.no_grad():
            for images, _ in tqdm(face_loader, desc="  Face features", ncols=80):
                images = images.to(device)
                feats = extractor(images)
                face_features.append(feats.cpu().numpy())

        face_features = np.concatenate(face_features, axis=0)
        np.save(os.path.join(cache_dir, 'face_features.npy'), face_features)
        print(f"  Saved face features: {face_features.shape}")

    print(f"\n  ✅ Feature caching complete!")
    print("=" * 60 + "\n")


# ============================================================
# Step 2: Dataset that loads cached features
# ============================================================

class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-computed DINOv2 features from .npy files.
    Used for fast classifier-only training (no backbone computation).

    WHY: Training the classifier on cached features is ~10-20x faster
    because we skip the DINOv2 forward pass entirely.
    """

    def __init__(self, cache_dir: str, dual_input: bool = False):
        """
        Args:
            cache_dir: Directory containing features.npy and labels.npy.
            dual_input: If True, also load face_features.npy.
        """
        self.dual_input = dual_input

        features_path = os.path.join(cache_dir, 'features.npy')
        labels_path = os.path.join(cache_dir, 'labels.npy')

        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Cached features not found at {features_path}. "
                f"Run: python scripts/cache_features.py"
            )

        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

        self.face_features = None
        if dual_input:
            face_path = os.path.join(cache_dir, 'face_features.npy')
            if os.path.exists(face_path):
                self.face_features = np.load(face_path)
            else:
                # Fallback: duplicate full features as face features
                self.face_features = self.features.copy()

        logger.info(
            f"Loaded cached features: {self.features.shape} "
            f"({(self.labels == 0).sum()} real, {(self.labels == 1).sum()} fake)"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.features[idx])
        label = int(self.labels[idx])

        if self.dual_input and self.face_features is not None:
            face_feat = torch.FloatTensor(self.face_features[idx])
            # Concatenate full + face features
            combined = torch.cat([feat, face_feat], dim=0)
            return combined, label

        return feat, label


def main():
    parser = argparse.ArgumentParser(description="Cache DINOv2 features")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='data/cached_features')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config['paths']['log_dir'])
    set_seed(config['training']['seed'])
    device = get_device(config['device'])

    model_config = config['model']
    data_dir = args.data_dir or config['data']['raw_dir']

    cache_features(
        data_dir=data_dir,
        cache_dir=args.cache_dir,
        dino_variant=model_config['dino_variant'],
        pooling_mode=model_config.get('pooling_mode', 'multi'),
        image_size=config['data']['image_size'],
        batch_size=args.batch_size,
        num_workers=config['data']['num_workers'],
        device=str(device),
        dual_input=model_config.get('dual_input', True),
        faces_dir=config['data'].get('faces_dir'),
    )


if __name__ == '__main__':
    main()
