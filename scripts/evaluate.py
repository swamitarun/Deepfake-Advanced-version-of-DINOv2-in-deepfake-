"""
evaluate.py — Evaluate the advanced model on the test set.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
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
from src.models.classifier import DualInputDeepfakeClassifier
from src.evaluation.evaluator import Evaluator
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate Advanced DINOv2 Detector")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    # ---- Setup ----
    config = load_config(args.config)
    setup_logging(log_dir=config['paths']['log_dir'])
    logger = logging.getLogger(__name__)
    set_seed(config['training']['seed'])
    device = get_device(config['device'])

    logger.info("=" * 60)
    logger.info("   ADVANCED DINOv2 DEEPFAKE DETECTOR — EVALUATION")
    logger.info("=" * 60)

    # ---- Config ----
    model_config = config['model']
    dual_input = model_config.get('dual_input', True)
    pooling_mode = model_config.get('pooling_mode', 'multi')

    # ---- Data ----
    data_dir = args.data_dir or config['data']['raw_dir']
    faces_dir = config['data'].get('faces_dir', None)
    image_size = config['data']['image_size']

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

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

    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['data']['num_workers'],
        seed=config['training']['seed'],
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

    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(
        config['paths']['checkpoint_dir'], 'best_model.pth'
    )
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train the model first: python scripts/train.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")

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
