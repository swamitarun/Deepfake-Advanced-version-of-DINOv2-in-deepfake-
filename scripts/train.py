"""
train.py — Advanced training script for DINOv2 deepfake detector.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --debug          # Quick test (100 samples, 3 epochs)
    python scripts/train.py --use-cache      # Train from cached features (fast)
    python scripts/train.py --advanced-aug   # Use deepfake-specific augmentations

This script uses:
    - DualInputDeepfakeClassifier (full image + face crop)
    - Multi-token pooling (CLS + Mean + Max)
    - Partial fine-tuning (last 2 blocks)
    - Separate learning rates for backbone vs classifier
    - LR warmup + gradient clipping
    - Dataset integrity check before training
"""

import os
import sys
import argparse
import logging
import json
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.integrity import check_dataset_integrity
from src.models.classifier import DualInputDeepfakeClassifier
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train Advanced DINOv2 Deepfake Detector")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: 100 samples, 3 epochs, no face detect')
    parser.add_argument('--advanced-aug', action='store_true',
                       help='Use advanced deepfake-specific augmentations')
    parser.add_argument('--use-cache', action='store_true',
                       help='Train using cached features (fast, no backbone)')
    parser.add_argument('--cache-dir', type=str, default='data/cached_features',
                       help='Directory with cached features')
    parser.add_argument('--skip-integrity', action='store_true',
                       help='Skip dataset integrity check')
    args = parser.parse_args()

    # ---- Load Config ----
    config = load_config(args.config)

    # Apply overrides
    if args.data_dir:
        config['data']['raw_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # ---- Setup ----
    setup_logging(log_dir=config['paths']['log_dir'])
    logger = logging.getLogger(__name__)

    set_seed(config['training']['seed'])
    device = get_device(config['device'])

    logger.info("=" * 60)
    logger.info("   ADVANCED DINOv2 DEEPFAKE DETECTOR — TRAINING")
    logger.info("=" * 60)

    # ---- Model Configuration ----
    model_config = config['model']
    dual_input = model_config.get('dual_input', True)
    pooling_mode = model_config.get('pooling_mode', 'multi')

    # In debug mode, simplify everything for speed
    if args.debug:
        logger.info("🐛 DEBUG MODE ENABLED: 100 samples, 3 epochs, single input")
        dual_input = False  # Skip face detection for speed

    logger.info(f"  Dual input: {dual_input}")
    logger.info(f"  Pooling mode: {pooling_mode}")
    logger.info(f"  Unfreeze blocks: {model_config['unfreeze_last_n_blocks']}")
    if args.advanced_aug:
        logger.info(f"  Augmentations: ADVANCED (blur, JPEG, noise)")
    else:
        logger.info(f"  Augmentations: Standard")

    # ---- Cached Feature Training (fast mode) ----
    if args.use_cache:
        _train_from_cache(args, config, device, logger)
        return

    # ---- Dataset Integrity Check ----
    data_dir = config['data']['raw_dir']
    if not args.skip_integrity and not args.debug:
        integrity = check_dataset_integrity(data_dir)
        if not integrity['is_healthy']:
            logger.error("Dataset integrity check FAILED. Fix issues or use --skip-integrity")
            return
    elif args.debug:
        logger.info("  Skipping integrity check in debug mode")

    # ---- Data ----
    faces_dir = config['data'].get('faces_dir', None)
    image_size = config['data']['image_size']

    # Choose transform: advanced or standard
    if args.advanced_aug:
        from src.data.augmentations import get_advanced_train_transforms
        train_transform = get_advanced_train_transforms(image_size)
    else:
        train_transform = get_train_transforms(image_size)

    val_transform = get_val_transforms(image_size)

    max_samples = 100 if args.debug else None  # Debug uses 100 samples per class

    # Face detector for on-the-fly cropping
    face_detector = None
    if dual_input and config['face_detection']['enabled']:
        has_precomputed = (
            faces_dir and os.path.exists(os.path.join(faces_dir, 'real'))
        )
        if not has_precomputed:
            from src.utils.face_detect import FaceDetector
            face_config = config['face_detection']
            face_detector = FaceDetector(
                margin=face_config['margin'],
                confidence_threshold=face_config['confidence_threshold'],
                image_size=image_size,
            )
            logger.info("Using on-the-fly face detection for dual input")
        else:
            logger.info(f"Using pre-computed faces from {faces_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['data']['num_workers'],
        seed=config['training']['seed'],
        max_samples_per_class=max_samples,
        dual_input=dual_input,
        faces_dir=faces_dir,
        face_detector=face_detector,
    )

    # ---- Model ----
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

    # ---- Optimizer with Parameter Groups ----
    train_config = config['training']
    base_lr = train_config['learning_rate']
    backbone_lr_factor = train_config.get('backbone_lr_factor', 0.1)

    param_groups = model.get_param_groups(
        backbone_lr_factor=backbone_lr_factor,
        base_lr=base_lr,
    )

    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=train_config['weight_decay'],
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=train_config['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")

    # ---- Scheduler ----
    epochs = 3 if args.debug else train_config['epochs']
    warmup_epochs = 1 if args.debug else train_config.get('warmup_epochs', 3)

    if train_config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs),
        )
    elif train_config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config['step_size'],
            gamma=train_config['gamma'],
        )
    else:
        scheduler = None

    # ---- Training ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        epochs=epochs,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        early_stopping_patience=train_config['early_stopping_patience'],
        dual_input=dual_input,
        warmup_epochs=warmup_epochs,
    )

    history = trainer.fit()

    # ---- Save Training Log as JSON ----
    log_path = os.path.join(config['paths']['log_dir'], 'training_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  Training log saved to: {log_path}")

    # ---- Plot Training Curves ----
    plot_training_curves(
        history=history,
        save_dir=config['paths']['plot_dir'],
    )

    logger.info("\n✅ Training complete!")
    logger.info(f"   Best model: {config['paths']['checkpoint_dir']}/best_model.pth")
    logger.info(f"   Plots: {config['paths']['plot_dir']}/")
    logger.info(f"   Log: {log_path}")


def _train_from_cache(args, config, device, logger):
    """
    Train classifier using cached features (skips DINOv2 backbone entirely).
    ~10-20x faster than full pipeline.
    """
    from scripts.cache_features import CachedFeatureDataset
    from torch.utils.data import DataLoader, random_split
    from src.models.classifier import AdvancedClassifierHead

    logger.info("=" * 60)
    logger.info("  CACHED FEATURE TRAINING (Fast Mode)")
    logger.info("=" * 60)

    model_config = config['model']
    train_config = config['training']
    dual_input = model_config.get('dual_input', True)
    pooling_mode = model_config.get('pooling_mode', 'multi')

    # Calculate feature dim
    base_dim = 384 if model_config['dino_variant'] == 'dinov2_vits14' else 768
    feat_dim = base_dim * 3 if pooling_mode == 'multi' else base_dim
    total_dim = feat_dim * 2 if dual_input else feat_dim

    # Load cached features
    dataset = CachedFeatureDataset(args.cache_dir, dual_input=dual_input)

    # Split
    total = len(dataset)
    train_size = int(total * config['data']['train_split'])
    val_size = total - train_size

    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(train_config['seed']),
    )

    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False)

    # Just the classifier head (no backbone)
    classifier = AdvancedClassifierHead(
        input_dim=total_dim,
        hidden_dims=model_config['classifier']['hidden_dims'],
        num_classes=model_config['classifier']['num_classes'],
        dropout=model_config['classifier']['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 3 if args.debug else train_config['epochs']

    logger.info(f"  Feature dim: {total_dim}")
    logger.info(f"  Train: {train_size}, Val: {val_size}")
    logger.info(f"  Epochs: {epochs}")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        classifier.train()
        correct = total_count = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

        train_acc = 100.0 * correct / total_count

        # Validate
        classifier.eval()
        correct = total_count = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = classifier(feats)
                correct += (logits.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

        val_acc = 100.0 * correct / total_count
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(classifier.state_dict(), os.path.join(
                config['paths']['checkpoint_dir'], 'cached_classifier.pth'
            ))

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% "
            f"{'✓ BEST' if is_best else ''}"
        )

    logger.info(f"\n✅ Cached training complete! Best Val Acc: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
