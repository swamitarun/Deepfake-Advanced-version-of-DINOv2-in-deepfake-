"""
evaluate.py — Evaluate the trained model on the test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --gpu 1
"""

import os
import sys
import argparse
import logging

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import DeepfakeClassifier
from src.evaluation.evaluator import Evaluator
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 Deepfake Detector")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # ---- Setup ----
    config = load_config(args.config)
    setup_logging(log_dir=config['paths']['log_dir'])
    logger = logging.getLogger(__name__)
    set_seed(config['training']['seed'])
    device = get_device(config['device'])

    logger.info("=" * 50)
    logger.info("  DINOv2 DEEPFAKE DETECTOR — EVALUATION")
    logger.info("=" * 50)

    # ---- Checkpoint & Config Override ----
    model_cfg = config['model']
    dual_input = model_cfg.get('dual_input', True)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        if os.path.exists(os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')):
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
        elif os.path.exists(os.path.join(config['paths']['checkpoint_dir'], 'best_mlp.pth')):
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_mlp.pth')
        else:
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint FIRST to check if dual_input should be overridden
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    if checkpoint_path.endswith('best_mlp.pth') and 'feat_dim' in checkpoint:
        feat_dim = checkpoint['feat_dim']
        # If feat_dim is 768, it was trained on single input. If 1536, dual input.
        if feat_dim == 768:
            dual_input = False
            logger.info("Fast checkpoint was trained on RAW features only (768-dim). Disabled dual_input for evaluation.")
        elif feat_dim == 1536:
            dual_input = True
            logger.info("Fast checkpoint was trained on DUAL features (1536-dim). Enabled dual_input for evaluation.")

    # ---- Data ----
    data_cfg = config['data']
    image_size = data_cfg['image_size']

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    face_detector = None
    if dual_input and config['face_detection']['enabled']:
        faces_dir = data_cfg.get('faces_dir', None)
        has_precomputed = faces_dir and os.path.exists(os.path.join(faces_dir, 'real'))
        if not has_precomputed:
            from src.utils.face_detect import FaceDetector
            fd_cfg = config['face_detection']
            face_detector = FaceDetector(
                margin=fd_cfg['margin'],
                confidence_threshold=fd_cfg['confidence_threshold'],
                image_size=image_size,
                device=str(device),
            )

    _, _, test_loader = create_dataloaders(
        data_dir=data_cfg['raw_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['training']['batch_size'],
        train_split=data_cfg['train_split'],
        val_split=data_cfg['val_split'],
        test_split=data_cfg['test_split'],
        num_workers=data_cfg['num_workers'],
        seed=config['training']['seed'],
        dual_input=dual_input,
        faces_dir=data_cfg.get('faces_dir'),
        face_detector=face_detector,
    )

    # ---- Model ----
    model = DeepfakeClassifier(
        dino_variant=model_cfg['dino_variant'],
        freeze_backbone=model_cfg['freeze_backbone'],
        unfreeze_last_n_blocks=model_cfg['unfreeze_last_n_blocks'],
        dual_input=dual_input,
    )

    if checkpoint_path.endswith('best_mlp.pth'):
        # Loading from Fast Training
        state_dict = checkpoint['model_state_dict']
        # The FastMLP has 'net.0.weight' etc. Our DeepfakeClassifier's MLP is in model.classifier.net
        # So we can load the state dict directly into model.classifier
        model.classifier.load_state_dict(state_dict)
        logger.info(f"Loaded Fast MLP checkpoint epoch {checkpoint.get('epoch', '?')} (Full DINOv2 Backbone is untouched)")
    else:
        # Loading from Slow Training
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded full model checkpoint epoch {checkpoint.get('epoch', '?')}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded checkpoint.")

    # ---- Evaluate ----
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=str(device),
        dual_input=dual_input,
    )

    metrics = evaluator.evaluate()
    evaluator.print_report()

    # ---- Plots ----
    plot_dir = config['paths']['plot_dir']
    plot_confusion_matrix(cm=metrics['confusion_matrix'], save_dir=plot_dir)

    roc_data = evaluator.get_roc_data()
    if roc_data:
        fpr, tpr = roc_data
        plot_roc_curve(fpr=fpr, tpr=tpr, auc_score=metrics['auc_roc'] / 100, save_dir=plot_dir)

    logger.info(f"\n✅ Evaluation complete! Plots: {plot_dir}/")


if __name__ == '__main__':
    main()
