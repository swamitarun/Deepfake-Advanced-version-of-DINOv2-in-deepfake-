"""
Advanced Trainer — Handles dual-input training with parameter groups.

=================================================================================
FEATURES:
=================================================================================

1. Dual-input support: Unpacks (full_image, face_crop, label) from dataloader
2. Separate learning rates: Backbone fine-tune layers get lower LR
3. Gradient clipping: Prevents exploding gradients during fine-tuning
4. LR Warmup: Ramps learning rate from 0 to target over first N epochs
5. GPU memory logging: Tracks peak GPU memory each epoch
6. Epoch timing: Records time per epoch in history
=================================================================================
"""

import os
import time
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training pipeline for the Advanced DeepfakeClassifier.

    Supports both single-input (image, label) and dual-input
    (full_image, face_crop, label) dataloaders.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, optimizer, ...)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        epochs: int = 30,
        checkpoint_dir: str = 'models/checkpoints',
        early_stopping_patience: int = 7,
        dual_input: bool = True,
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 3,
    ):
        """
        Args:
            model: DeepfakeClassifier model (dual or single input).
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer instance (may have param groups with different LRs).
            scheduler: Optional learning rate scheduler.
            criterion: Loss function (default: CrossEntropyLoss).
            device: Device to train on.
            epochs: Maximum training epochs.
            checkpoint_dir: Where to save model checkpoints.
            early_stopping_patience: Stop if no improvement for N epochs.
            dual_input: If True, expects 3-tuple from dataloader.
            max_grad_norm: Maximum gradient norm for clipping.
            warmup_epochs: Number of epochs to linearly ramp LR from 0 to target.
                          WHY: During the first few epochs, the randomly initialized
                          classifier produces large/random gradients. If these flow
                          back to the unfrozen backbone layers at full LR, they can
                          corrupt the pretrained weights. Warmup starts with tiny LR
                          and gradually increases, giving the classifier time to
                          stabilize before the backbone receives meaningful updates.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.patience = early_stopping_patience
        self.dual_input = dual_input
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs

        # Store target LRs for warmup
        self.target_lrs = [pg['lr'] for pg in optimizer.param_groups]

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': [],
            'gpu_memory_mb': [],
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dictionary.
        """
        logger.info("=" * 60)
        logger.info("STARTING ADVANCED TRAINING")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dual input: {self.dual_input}")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Early stopping patience: {self.patience}")
        logger.info(f"  Gradient clipping: max_norm={self.max_grad_norm}")
        logger.info(f"  LR warmup: {self.warmup_epochs} epochs")

        # Log learning rates per param group
        for i, pg in enumerate(self.optimizer.param_groups):
            name = pg.get('name', f'group_{i}')
            logger.info(f"  LR [{name}]: {pg['lr']}")

        logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # --- LR Warmup ---
            # Linearly ramp LR from 0 to target over warmup_epochs
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                for i, pg in enumerate(self.optimizer.param_groups):
                    pg['lr'] = self.target_lrs[i] * warmup_factor
                logger.info(f"  Warmup epoch {epoch}/{self.warmup_epochs}: LR factor={warmup_factor:.2f}")

            # --- Train ---
            train_loss, train_acc = self.train_one_epoch(epoch)

            # --- Validate ---
            val_loss, val_acc = self.validate(epoch)

            # --- Record history ---
            current_lr = self.optimizer.param_groups[-1]['lr']  # Classifier LR
            epoch_time = time.time() - epoch_start

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # --- GPU memory logging ---
            gpu_mem_mb = 0.0
            if self.device == 'cuda' or 'cuda' in str(self.device):
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats()
            self.history['gpu_memory_mb'].append(gpu_mem_mb)

            # --- Learning rate step (skip during warmup) ---
            if self.scheduler and epoch > self.warmup_epochs:
                self.scheduler.step()

            # --- Log epoch summary ---
            logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | "
                f"GPU: {gpu_mem_mb:.0f}MB"
            )

            # --- Check for improvement ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

            # --- Early stopping ---
            if self.patience_counter >= self.patience:
                logger.info(
                    f"\n⚠ Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch} (Val Acc: {self.best_val_acc:.2f}%)"
                )
                break

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"  Best epoch: {self.best_epoch}")
        logger.info(f"  Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"  Best Val Acc: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)

        return self.history

    def train_one_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch with dual-input support.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, accuracy_percent)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            ncols=100,
        )

        for batch in pbar:
            # ---- Unpack batch (dual or single input) ----
            if self.dual_input:
                full_images, face_crops, labels = batch
                full_images = full_images.to(self.device)
                face_crops = face_crops.to(self.device)
                labels = labels.to(self.device)
            else:
                images, labels = batch
                full_images = images.to(self.device)
                face_crops = None
                labels = labels.to(self.device)

            # ---- Forward pass ----
            logits = self.model(full_images, face_crops)
            loss = self.criterion(logits, labels)

            # ---- Backward pass ----
            self.optimizer.zero_grad()
            loss.backward()

            # ---- Gradient clipping ----
            # WHY: When fine-tuning backbone layers, gradients can become very
            # large (the pretrained weights are sensitive to updates). Clipping
            # limits the maximum gradient norm to prevent catastrophic updates.
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            # ---- Track metrics ----
            running_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.1f}%',
            })

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """
        Validate with dual-input support.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, accuracy_percent)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False,
            ncols=100,
        )

        for batch in pbar:
            if self.dual_input:
                full_images, face_crops, labels = batch
                full_images = full_images.to(self.device)
                face_crops = face_crops.to(self.device)
                labels = labels.to(self.device)
            else:
                images, labels = batch
                full_images = images.to(self.device)
                face_crops = None
                labels = labels.to(self.device)

            logits = self.model(full_images, face_crops)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'val_acc': self.best_val_acc,
            'history': self.history,
            'dual_input': self.dual_input,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)

        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        self.best_epoch = checkpoint.get('epoch', 0)

        logger.info(
            f"Loaded checkpoint epoch {self.best_epoch} "
            f"(Val Acc: {self.best_val_acc:.2f}%)"
        )
