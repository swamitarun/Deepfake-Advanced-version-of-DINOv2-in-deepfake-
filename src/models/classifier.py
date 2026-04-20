"""
Clean Deepfake Classifier using DINOv2 dual-input.

Pipeline:
  full_image  -> DINOv2 -> CLS (768)  --|
                                        concat (1536) -> MLP -> [REAL / FAKE]
  face_crop   -> DINOv2 -> CLS (768)  --|

MLP: 1536 -> 512 -> 256 -> 2  (~2M params, trains fast, generalises well)
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_extractor import DINOv2Extractor

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Simple 3-layer MLP with BatchNorm, GELU and Dropout."""

    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepfakeClassifier(nn.Module):
    """
    DINOv2 dual-input deepfake detector.

    Both branches share the same extractor weights (tied weights).
    Features are concatenated then classified by a small MLP.
    """

    def __init__(
        self,
        dino_variant: str = 'dinov2_vitb14',
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
        dual_input: bool = True,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.dual_input = dual_input

        self.extractor = DINOv2Extractor(
            variant=dino_variant,
            freeze=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

        feat_dim = self.extractor.get_feature_dim()           # 768
        classifier_input = feat_dim * 2 if dual_input else feat_dim   # 1536 or 768

        self.classifier = MLPClassifier(classifier_input, num_classes, dropout)

        trainable_backbone = sum(p.numel() for p in self.extractor.parameters() if p.requires_grad)
        trainable_head     = sum(p.numel() for p in self.classifier.parameters())
        logger.info("=" * 50)
        logger.info("  DEEPFAKE CLASSIFIER SUMMARY")
        logger.info(f"  Backbone:    {dino_variant}  (trainable: {trainable_backbone:,})")
        logger.info(f"  Dual input:  {dual_input}  (classifier input: {classifier_input})")
        logger.info(f"  MLP params:  {trainable_head:,}")
        logger.info(f"  TOTAL train: {trainable_backbone + trainable_head:,}")
        logger.info("=" * 50)

    def forward(
        self,
        full_image: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        full_feat = self.extractor(full_image)

        if self.dual_input:
            face_feat = self.extractor(face_crop if face_crop is not None else full_image)
            features  = torch.cat([full_feat, face_feat], dim=1)
        else:
            features = full_feat

        return self.classifier(features)

    def predict(
        self,
        full_image: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> Dict:
        self.eval()
        with torch.no_grad():
            logits = self.forward(full_image, face_crop)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
        label_map = {0: 'REAL', 1: 'FAKE'}
        return {
            'labels':        [label_map[p.item()] for p in preds],
            'probabilities': probs,
            'prob_real':     probs[:, 0],
            'prob_fake':     probs[:, 1],
            'confidence':    probs.max(dim=1).values,
        }

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float):
        backbone_params   = [p for p in self.extractor.parameters() if p.requires_grad]
        classifier_params = list(self.classifier.parameters())
        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': base_lr * backbone_lr_factor, 'name': 'backbone'})
        groups.append({'params': classifier_params, 'lr': base_lr, 'name': 'classifier'})
        return groups
