"""
Visualization utilities — training curves, confusion matrix, ROC curve.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for servers without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

# --- Style Configuration ---
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: str = 'results/plots',
    filename: str = 'training_curves.png',
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Training history dict with keys:
                 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_dir: Directory to save the plot.
        filename: Output filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss Plot ---
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'o-', color='#e94560', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', color='#0f3460', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend(framealpha=0.8)
    ax1.grid(True)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Accuracy Plot ---
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'o-', color='#e94560', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 's-', color='#0f3460', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend(framealpha=0.8)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_dir: str = 'results/plots',
    filename: str = 'confusion_matrix.png',
):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (numpy array).
        class_names: List of class labels.
        save_dir: Directory to save the plot.
        filename: Output filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        class_names = ['REAL', 'FAKE']

    fig, ax = plt.subplots(figsize=(7, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='RdPu')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    # Annotate cells with values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = '#fff' if cm[i, j] > thresh else '#333'
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', color=color, fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_dir: str = 'results/plots',
    filename: str = 'roc_curve.png',
):
    """
    Plot ROC curve with AUC score.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc_score: Area under the curve score.
        save_dir: Directory to save the plot.
        filename: Output filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, color='#e94560', linewidth=2.5,
            label=f'ROC Curve (AUC = {auc_score:.4f})')

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='#666', linestyle='--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Deepfake Detection')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"ROC curve saved to {save_path}")
