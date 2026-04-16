"""
Evaluator — Updated for dual-input model.
"""

import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates the trained DeepfakeClassifier on a test set.
    Supports both single-input and dual-input models.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None,
        dual_input: bool = True,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names or ['REAL', 'FAKE']
        self.dual_input = dual_input

        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.metrics: Dict = {}

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run evaluation on the test set."""
        self.model.eval()
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

        logger.info("Running evaluation on test set...")
        pbar = tqdm(self.test_loader, desc="Evaluating", ncols=100)

        for batch in pbar:
            if self.dual_input:
                full_images, face_crops, labels = batch
                full_images = full_images.to(self.device)
                face_crops = face_crops.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(full_images, face_crops)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images, None)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            self.all_labels.extend(labels.cpu().numpy())
            self.all_preds.extend(preds.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())

        self.all_labels = np.array(self.all_labels)
        self.all_preds = np.array(self.all_preds)
        self.all_probs = np.array(self.all_probs)

        self.metrics = self._compute_metrics()
        return self.metrics

    def _compute_metrics(self) -> Dict:
        """Compute all classification metrics."""
        metrics = {}

        metrics['accuracy'] = accuracy_score(self.all_labels, self.all_preds) * 100
        metrics['precision'] = precision_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        ) * 100
        metrics['recall'] = recall_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        ) * 100
        metrics['f1_score'] = f1_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        ) * 100

        metrics['per_class_precision'] = precision_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        ) * 100
        metrics['per_class_recall'] = recall_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        ) * 100
        metrics['per_class_f1'] = f1_score(
            self.all_labels, self.all_preds, average=None, zero_division=0
        ) * 100

        try:
            metrics['auc_roc'] = roc_auc_score(
                self.all_labels, self.all_probs[:, 1]
            ) * 100
        except ValueError:
            metrics['auc_roc'] = 0.0

        metrics['confusion_matrix'] = confusion_matrix(self.all_labels, self.all_preds)

        try:
            fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_probs[:, 1])
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        except ValueError:
            metrics['roc_curve'] = None

        return metrics

    def print_report(self):
        """Print formatted evaluation report."""
        if not self.metrics:
            logger.warning("No metrics available. Run evaluate() first.")
            return

        print("\n" + "=" * 60)
        print("    EVALUATION REPORT — Advanced DINOv2 Deepfake Detector")
        print("=" * 60)

        print(f"\n  Overall Accuracy:    {self.metrics['accuracy']:.2f}%")
        print(f"  Macro Precision:     {self.metrics['precision']:.2f}%")
        print(f"  Macro Recall:        {self.metrics['recall']:.2f}%")
        print(f"  Macro F1-Score:      {self.metrics['f1_score']:.2f}%")
        print(f"  AUC-ROC:             {self.metrics['auc_roc']:.2f}%")

        print("\n  Per-Class Metrics:")
        print("  " + "-" * 45)
        print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("  " + "-" * 45)
        for i, name in enumerate(self.class_names):
            print(
                f"  {name:<10} "
                f"{self.metrics['per_class_precision'][i]:>9.2f}% "
                f"{self.metrics['per_class_recall'][i]:>9.2f}% "
                f"{self.metrics['per_class_f1'][i]:>9.2f}%"
            )

        print("\n  Confusion Matrix:")
        cm = self.metrics['confusion_matrix']
        print(f"  {'':>10} {'Pred REAL':>12} {'Pred FAKE':>12}")
        print(f"  {'True REAL':<10} {cm[0][0]:>12} {cm[0][1]:>12}")
        print(f"  {'True FAKE':<10} {cm[1][0]:>12} {cm[1][1]:>12}")
        print("\n" + "=" * 60)

        print("\n  Detailed Report:")
        print(classification_report(
            self.all_labels, self.all_preds, target_names=self.class_names,
        ))

    def get_roc_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (fpr, tpr) for ROC curve plotting."""
        if self.metrics.get('roc_curve'):
            return (
                self.metrics['roc_curve']['fpr'],
                self.metrics['roc_curve']['tpr'],
            )
        return None
