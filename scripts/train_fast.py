"""
train_fast.py — Train ONLY the MLP classifier on cached DINOv2 features.

This skips the entire DINOv2 backbone during training.
Each epoch takes ~5 seconds instead of ~1 hour.

Usage:
    python scripts/train_fast.py --gpu 1
    python scripts/train_fast.py --gpu 1 --epochs 100
    python scripts/train_fast.py --gpu 1 --dual   # use face features too
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import load_config, setup_logging, set_seed, get_device
from src.utils.visualization import plot_training_curves


# ---- Dataset for cached features ----
class CachedFeatureDataset(Dataset):
    """Loads pre-extracted .npy feature files."""

    def __init__(self, cache_dir: str, dual: bool = False):
        cache_dir = cache_dir

        # Load raw features
        real_feats = np.load(os.path.join(cache_dir, 'raw_real_features.npy'))
        fake_feats = np.load(os.path.join(cache_dir, 'raw_fake_features.npy'))

        real_labels = np.zeros(len(real_feats), dtype=np.int64)
        fake_labels = np.ones(len(fake_feats), dtype=np.int64)

        self.features = np.concatenate([real_feats, fake_feats], axis=0)
        self.labels = np.concatenate([real_labels, fake_labels], axis=0)

        # Dual: also load face features and concat
        if dual:
            face_real_path = os.path.join(cache_dir, 'face_real_features.npy')
            face_fake_path = os.path.join(cache_dir, 'face_fake_features.npy')

            if os.path.exists(face_real_path) and os.path.exists(face_fake_path):
                face_real = np.load(face_real_path)
                face_fake = np.load(face_fake_path)

                # Match by index (only if same count)
                if len(face_real) == len(real_feats) and len(face_fake) == len(fake_feats):
                    face_all = np.concatenate([face_real, face_fake], axis=0)
                    self.features = np.concatenate([self.features, face_all], axis=1)
                    print(f"  Dual-input features loaded: {self.features.shape[1]}-dim")
                else:
                    print(f"  Face feature count mismatch — using raw only")
            else:
                print(f"  Face features not found — using raw only")

        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)

        print(f"  Loaded {len(self)} samples ({(self.labels==0).sum()} real, {(self.labels==1).sum()} fake)")
        print(f"  Feature dim: {self.features.shape[1]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---- Simple MLP (same as classifier.py) ----
class FastMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),

            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--cache-dir', default='data/cached_features')
    parser.add_argument('--dual', action='store_true', help='Use face+raw concat features')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg = load_config(args.config)
    setup_logging(log_dir=cfg['paths']['log_dir'])
    logger = logging.getLogger(__name__)
    set_seed(cfg['training']['seed'])
    device = get_device(cfg['device'])

    logger.info("=" * 50)
    logger.info("  FAST TRAINING — MLP on cached DINOv2 features")
    logger.info("=" * 50)

    # ---- Dataset ----
    dataset = CachedFeatureDataset(args.cache_dir, dual=args.dual)
    feat_dim = dataset.features.shape[1]

    # Split
    total = len(dataset)
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Balanced sampling for train
    train_labels = [dataset.labels[i].item() for i in train_ds.indices]
    counts = Counter(train_labels)
    total_train = len(train_labels)
    weight_map = {c: total_train / (2.0 * n) for c, n in counts.items()}
    sample_weights = torch.DoubleTensor([weight_map[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    logger.info(f"  Train: {train_size}  Val: {val_size}  Test: {test_size}")
    logger.info(f"  Feature dim: {feat_dim}")
    logger.info(f"  Balanced sampling: real={counts.get(0,0)}, fake={counts.get(1,0)}")

    # ---- Class weights for loss ----
    real_n = counts.get(0, 1)
    fake_n = counts.get(1, 1)
    w_real = total_train / (2.0 * max(1, real_n))
    w_fake = total_train / (2.0 * max(1, fake_n))
    weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    logger.info(f"  Class weights: real={w_real:.2f}, fake={w_fake:.2f}")

    # ---- Model ----
    model = FastMLP(feat_dim, num_classes=2, dropout=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  MLP params: {param_count:,}")
    logger.info(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")

    # ---- Training loop ----
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_ctr = 0
    ckpt_dir = cfg['paths']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        correct = total_count = 0
        epoch_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            epoch_loss += loss.item() * labels.size(0)

        train_acc = 100.0 * correct / total_count
        train_loss = epoch_loss / total_count

        # Validate
        model.eval()
        correct = total_count = 0
        val_loss_sum = 0.0
        all_probs = []
        all_labels_list = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss = criterion(logits, labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                val_loss_sum += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_labels_list.extend(labels.cpu().tolist())

        val_acc = 100.0 * correct / total_count
        val_loss = val_loss_sum / total_count

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels_list, all_probs)
        except:
            auc = 0.0

        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'feat_dim': feat_dim,
            }, os.path.join(ckpt_dir, 'best_mlp.pth'))
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or is_best or epoch == 1:
            print(f"Epoch [{epoch:3d}/{args.epochs}] ({elapsed:.1f}s)  "
                  f"Train: {train_acc:.1f}%  Val: {val_acc:.1f}%  AUC: {auc:.4f}  "
                  f"{'** BEST **' if is_best else ''}")

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}. Best: epoch {best_epoch} ({best_val_acc:.2f}%)")
            break

    total_time = time.time() - total_start

    # ---- Test ----
    print("\n" + "=" * 50)
    print("  FINAL TEST EVALUATION")
    print("=" * 50)

    # Load best model
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_mlp.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    correct = total_count = 0
    all_probs = []
    all_labels_list = []
    real_correct = real_total = fake_correct = fake_total = 0

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels_list.extend(labels.cpu().tolist())

            # Per-class
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            real_total += real_mask.sum().item()
            fake_total += fake_mask.sum().item()
            real_correct += (preds[real_mask] == 0).sum().item()
            fake_correct += (preds[fake_mask] == 1).sum().item()

    test_acc = 100.0 * correct / total_count
    try:
        from sklearn.metrics import roc_auc_score, f1_score, classification_report
        test_auc = roc_auc_score(all_labels_list, all_probs)
        test_preds = [1 if p >= 0.5 else 0 for p in all_probs]
        test_f1 = f1_score(all_labels_list, test_preds)
        print(classification_report(all_labels_list, test_preds, target_names=['REAL', 'FAKE']))
    except:
        test_auc = test_f1 = 0.0

    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Test AUC:       {test_auc:.4f}")
    print(f"  Test F1:        {test_f1:.4f}")
    print(f"  Real Accuracy:  {100.*real_correct/max(1,real_total):.1f}% ({real_correct}/{real_total})")
    print(f"  Fake Accuracy:  {100.*fake_correct/max(1,fake_total):.1f}% ({fake_correct}/{fake_total})")
    print(f"\n  Training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best epoch: {best_epoch} (val_acc={best_val_acc:.2f}%)")
    print("=" * 50)

    # Save
    log_path = os.path.join(cfg['paths']['log_dir'], 'fast_training_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history=history, save_dir=cfg['paths']['plot_dir'],
                        filename='fast_training_curves.png')

    print(f"\nCheckpoint: {ckpt_dir}/best_mlp.pth")
    print(f"Log: {log_path}")
    print(f"Plots: {cfg['paths']['plot_dir']}/")


if __name__ == '__main__':
    main()
