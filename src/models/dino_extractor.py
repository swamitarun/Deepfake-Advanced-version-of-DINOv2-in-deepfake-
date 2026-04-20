"""
Simple DINOv2 Feature Extractor.
Returns the CLS token (768-dim) from dinov2_vitb14.
Optionally unfreezes the last N transformer blocks for fine-tuning.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DINOv2Extractor(nn.Module):
    """
    Loads DINOv2 ViT-B/14, freezes most of it, returns 768-dim CLS token.
    """

    def __init__(
        self,
        variant: str = 'dinov2_vitb14',
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 2,
    ):
        super().__init__()
        self.embed_dim = 768

        logger.info(f"Loading {variant} from torch.hub ...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', variant, pretrained=True,
        )
        logger.info("DINOv2 loaded.")

        if freeze:
            self._freeze(unfreeze_last_n_blocks)

    def _freeze(self, unfreeze_last_n: int):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            blocks = self.backbone.blocks
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, 'norm'):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        logger.info(f"Backbone: {total:,} total | {trainable:,} trainable | {total-trainable:,} frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # returns CLS token: (B, 768)

    def get_feature_dim(self) -> int:
        return self.embed_dim
