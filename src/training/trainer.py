"""
Clean Trainer for DINOv2 Deepfake Classifier.
Supports AMP, LR warmup, early stopping, AUC/F1 tracking.
"""

import os
import time
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Compatible with both old and new PyTorch versions
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_KWARG = {'device_type': 'cuda'}
    SCALER_ARGS = ('cuda',)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_KWARG = {}
    SCALER_ARGS = ()

logger = logging.getLogger(__name__)


def _compute_metrics(all_probs: list, all_labels: list):
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score, f1_score
        labels = np.array(all_labels)
        probs  = np.array(all_probs)
        preds  = (probs >= 0.5).astype(int)
        auc = float(roc_auc_score(labels, probs))
        f1  = float(f1_score(labels, preds, zero_division=0))
        return auc, f1
    except Exception:
        return 0.0, 0.0


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        epochs: int = 50,
        checkpoint_dir: str = 'models/checkpoints',
        early_stopping_patience: int = 10,
        dual_input: bool = True,
        warmup_epochs: int = 3,
        use_amp: bool = True,
    ):
        self.model         = model.to(device)
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.criterion     = criterion or nn.CrossEntropyLoss()
        self.device        = device
        self.epochs        = epochs
        self.checkpoint_dir = checkpoint_dir
        self.patience      = early_stopping_patience
        self.dual_input    = dual_input
        self.warmup_epochs = warmup_epochs

        self.use_amp = use_amp and ('cuda' in str(device))
        self.scaler  = GradScaler(*SCALER_ARGS) if self.use_amp else None

        # Store target LRs for warmup
        self.target_lrs = [pg['lr'] for pg in optimizer.param_groups]

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List] = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
            'val_auc':    [], 'val_f1':    [],
            'lr':         [],
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 0
        self.patience_ctr = 0

    # ------------------------------------------------------------------
    def fit(self) -> Dict:
        logger.info("=" * 50)
        logger.info("TRAINING START")
        logger.info(f"  Epochs: {self.epochs} | Device: {self.device} | AMP: {self.use_amp}")
        logger.info("=" * 50)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Warmup LR
            if epoch <= self.warmup_epochs:
                factor = epoch / self.warmup_epochs
                for i, pg in enumerate(self.optimizer.param_groups):
                    pg['lr'] = self.target_lrs[i] * factor

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_auc, val_f1 = self._val_epoch(epoch)

            lr = self.optimizer.param_groups[-1]['lr']
            elapsed = time.time() - t0

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(lr)

            if self.scheduler and epoch > self.warmup_epochs:
                self.scheduler.step()

            print(f"\nEpoch [{epoch}/{self.epochs}]  ({elapsed:.0f}s)")
            print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
            print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%  AUC={val_auc:.4f}  F1={val_f1:.4f}")
            print(f"  LR={lr:.2e}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                self.patience_ctr = 0
                self._save(epoch)
                print(f"  ** Best model saved (val_acc={val_acc:.2f}%) **")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                    break

        logger.info(f"Training done. Best epoch {self.best_epoch}, val_acc={self.best_val_acc:.2f}%")
        return self.history

    # ------------------------------------------------------------------
    def _unpack(self, batch):
        if self.dual_input:
            imgs, faces, labels = batch
            return (
                imgs.to(self.device, non_blocking=True),
                faces.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )
        else:
            imgs, labels = batch
            return (
                imgs.to(self.device, non_blocking=True),
                None,
                labels.to(self.device, non_blocking=True),
            )

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = correct = total = 0

        pbar = tqdm(self.train_loader, desc=f"Train [{epoch}/{self.epochs}]", ncols=100)
        for batch in pbar:
            imgs, faces, labels = self._unpack(batch)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast(**AMP_DEVICE_KWARG):
                    logits = self.model(imgs, faces)
                    loss   = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(imgs, faces)
                loss   = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.1f}%')

        return total_loss / max(1, total), 100. * correct / max(1, total)

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        self.model.eval()
        total_loss = correct = total = 0
        all_probs: List[float] = []
        all_labels: List[int]  = []

        for batch in tqdm(self.val_loader, desc=f"Val   [{epoch}/{self.epochs}]", ncols=100, leave=False):
            imgs, faces, labels = self._unpack(batch)

            if self.use_amp:
                with autocast(**AMP_DEVICE_KWARG):
                    logits = self.model(imgs, faces)
                    loss   = self.criterion(logits, labels)
            else:
                logits = self.model(imgs, faces)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        auc, f1 = _compute_metrics(all_probs, all_labels)
        return total_loss / max(1, total), 100. * correct / max(1, total), auc, f1

    def _save(self, epoch: int):
        ckpt = {
            'epoch':      epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc':    self.best_val_acc,
            'history':    self.history,
        }
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            ckpt['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'best_model.pth'))
