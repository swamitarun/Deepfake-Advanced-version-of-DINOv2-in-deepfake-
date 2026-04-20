"""
train.py — Clean DINOv2 Deepfake Detector Training Script

Usage:
    python scripts/train.py                        # standard run
    python scripts/train.py --gpu 1               # use GPU 1
    python scripts/train.py --gpu 1 --debug       # quick test (3 epochs, 100 samples)
    python scripts/train.py --gpu 1 --advanced-aug
"""

import os
import sys
import json
import logging
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import DeepfakeClassifier
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/config.yaml')
    parser.add_argument('--gpu',        type=int,   default=None)
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch-size', type=int,   default=None)
    parser.add_argument('--debug',      action='store_true')
    parser.add_argument('--advanced-aug', action='store_true')
    parser.add_argument('--no-amp',     action='store_true')
    parser.add_argument('--skip-integrity', action='store_true')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU {args.gpu}")

    cfg = load_config(args.config)
    if args.epochs:    cfg['training']['epochs']     = args.epochs
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size

    setup_logging(log_dir=cfg['paths']['log_dir'])
    logger = logging.getLogger(__name__)
    set_seed(cfg['training']['seed'])
    device = get_device(cfg['device'])

    # ---- Data ----
    data_cfg  = cfg['data']
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    dual      = model_cfg['dual_input']

    if args.debug:
        cfg['training']['epochs'] = 3
        dual = False
        logger.info("DEBUG MODE: 3 epochs, no face detection")

    if args.advanced_aug:
        from src.data.augmentations import get_advanced_train_transforms
        train_transform = get_advanced_train_transforms(data_cfg['image_size'])
        logger.info("Using advanced augmentations")
    else:
        train_transform = get_train_transforms(data_cfg['image_size'])

    val_transform = get_val_transforms(data_cfg['image_size'])

    # Dataset integrity check
    from src.data.integrity import check_dataset_integrity
    if not args.skip_integrity and not args.debug:
        result = check_dataset_integrity(data_cfg['raw_dir'])
        if not result['is_healthy']:
            logger.error("Dataset integrity check FAILED. Use --skip-integrity to bypass.")
            return

    # Face detector
    face_detector = None
    if dual and cfg['face_detection']['enabled']:
        faces_dir = data_cfg.get('faces_dir', None)
        has_precomputed = faces_dir and os.path.exists(os.path.join(faces_dir, 'real'))
        if not has_precomputed:
            from src.utils.face_detect import FaceDetector
            fd_cfg = cfg['face_detection']
            face_detector = FaceDetector(
                margin=fd_cfg['margin'],
                confidence_threshold=fd_cfg['confidence_threshold'],
                image_size=data_cfg['image_size'],
                device=str(device),
            )
        else:
            logger.info(f"Using pre-computed faces from {faces_dir}")

    max_samples = 100 if args.debug else None

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_cfg['raw_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=train_cfg['batch_size'],
        train_split=data_cfg['train_split'],
        val_split=data_cfg['val_split'],
        test_split=data_cfg['test_split'],
        num_workers=data_cfg['num_workers'],
        seed=train_cfg['seed'],
        max_samples_per_class=max_samples,
        dual_input=dual,
        faces_dir=data_cfg.get('faces_dir'),
        face_detector=face_detector,
    )

    # ---- Model ----
    model = DeepfakeClassifier(
        dino_variant=model_cfg['dino_variant'],
        freeze_backbone=model_cfg['freeze_backbone'],
        unfreeze_last_n_blocks=model_cfg['unfreeze_last_n_blocks'],
        dual_input=dual,
    )

    # ---- Optimizer ----
    param_groups = model.get_param_groups(
        base_lr=train_cfg['learning_rate'],
        backbone_lr_factor=train_cfg['backbone_lr_factor'],
    )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=train_cfg['weight_decay']
    )

    # ---- Scheduler ----
    epochs = train_cfg['epochs']
    warmup = train_cfg['warmup_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup)
    )

    # ---- Loss: CrossEntropy with label smoothing + class weights ----
    try:
        ds = train_loader.dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        real_n = ds.class_counts.get(0, 1)
        fake_n = ds.class_counts.get(1, 1)
    except Exception:
        real_n, fake_n = 5728, 48847

    total_n   = real_n + fake_n
    w_real    = total_n / (2.0 * max(1, real_n))
    w_fake    = total_n / (2.0 * max(1, fake_n))
    weights   = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=train_cfg['label_smoothing'],
    )
    logger.info(f"Class weights: Real={w_real:.2f}, Fake={w_fake:.2f}")

    # ---- Train ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=str(device),
        epochs=epochs,
        checkpoint_dir=cfg['paths']['checkpoint_dir'],
        early_stopping_patience=train_cfg['early_stopping_patience'],
        dual_input=dual,
        warmup_epochs=warmup,
        use_amp=not args.no_amp,
    )

    history = trainer.fit()

    # ---- Save log ----
    os.makedirs(cfg['paths']['log_dir'], exist_ok=True)
    log_path = os.path.join(cfg['paths']['log_dir'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history=history, save_dir=cfg['paths']['plot_dir'])

    print(f"\nDone! Best model -> {cfg['paths']['checkpoint_dir']}/best_model.pth")
    print(f"Training log   -> {log_path}")


if __name__ == '__main__':
    main()
