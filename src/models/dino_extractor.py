"""
Advanced DINOv2 Feature Extractor — Multi-Token Pooling with Partial Fine-Tuning

=================================================================================
WHAT CHANGED FROM STANDARD IMPLEMENTATION:
=================================================================================

Standard DINOv2 usage:
    Image → DINOv2 → CLS token (384-dim) → done

Our CUSTOM approach:
    Image → DINOv2 → Extract ALL tokens → 3 pooling strategies → concat (1152-dim)

The 3 pooling strategies:
    1. CLS token (384-dim):
       - The special [CLS] token that attends to all patches
       - Captures the GLOBAL image representation
       - This is what standard implementations use alone

    2. Mean Pooling of patch tokens (384-dim):
       - Average across all 256 spatial patch embeddings
       - Captures BROAD, distributed features across the entire image
       - Good for detecting overall texture inconsistencies in deepfakes

    3. Max Pooling of patch tokens (384-dim):
       - Maximum activation across all patches per feature dimension
       - Highlights the MOST SALIENT features from any spatial location
       - Good for detecting localized deepfake artifacts (blending boundaries,
         eye/mouth artifacts) because even if only ONE patch shows an artifact,
         max pooling will preserve that signal

    Final output: concat(CLS, Mean, Max) = 384 * 3 = 1152-dim

=================================================================================
PARTIAL FINE-TUNING:
=================================================================================

DINOv2 ViT-S/14 has 12 transformer blocks (indexed 0 to 11).

We FREEZE blocks 0-9 (early/mid layers that capture general visual features).
We UNFREEZE blocks 10-11 (last 2 layers that can adapt to deepfake-specific patterns).

WHY: Early layers learn universal features (edges, textures, shapes) that
transfer well. Later layers learn more task-specific representations. By
unfreezing the last 2 blocks, we allow the model to specialize its high-level
representations for deepfake detection while keeping the foundational features
intact. This is a common strategy called "partial fine-tuning" or "gradual
unfreezing".

=================================================================================
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DINOv2 variant configurations
# Each variant has a different embedding dimension and model size
DINOV2_CONFIGS = {
    'dinov2_vits14': {
        'embed_dim': 384,        # ViT-Small: 384-dim embeddings
        'num_blocks': 12,        # 12 transformer encoder blocks
        'num_heads': 6,          # 6 attention heads
        'patch_size': 14,        # 14x14 pixel patches
        'description': 'ViT-Small/14 (21M params)',
    },
    'dinov2_vitb14': {
        'embed_dim': 768,
        'num_blocks': 12,
        'num_heads': 12,
        'patch_size': 14,
        'description': 'ViT-Base/14 (86M params)',
    },
    'dinov2_vitl14': {
        'embed_dim': 1024,
        'num_blocks': 24,
        'num_heads': 16,
        'patch_size': 14,
        'description': 'ViT-Large/14 (300M params)',
    },
    'dinov2_vitg14': {
        'embed_dim': 1536,
        'num_blocks': 40,
        'num_heads': 24,
        'patch_size': 14,
        'description': 'ViT-Giant/14 (1.1B params)',
    },
}


class DINOv2Extractor(nn.Module):
    """
    Advanced DINOv2 feature extractor with multi-token pooling.

    Internal pipeline (for dinov2_vits14, 224x224 input):
    =====================================================

    Step 1 — Patch Creation:
        Input image (B, 3, 224, 224)
        Split into non-overlapping patches of size 14x14 pixels
        Number of patches: (224/14) × (224/14) = 16 × 16 = 256 patches

    Step 2 — Patch Embedding:
        Each 14×14×3 = 588 pixel patch is flattened and linearly projected
        to a 384-dimensional embedding vector
        Result: (B, 256, 384)

    Step 3 — CLS Token:
        A learnable [CLS] token (1, 384) is prepended to the patch sequence
        Result: (B, 257, 384) — 256 patches + 1 CLS

    Step 4 — Positional Encoding:
        Learnable position embeddings (257, 384) are ADDED to all tokens
        This injects spatial information so the model knows WHERE each patch
        came from in the original image

    Step 5 — Transformer Encoder (12 blocks):
        Each block contains:
        a) Multi-Head Self-Attention (6 heads):
           - Each token attends to ALL other tokens
           - Captures relationships between different image regions
           - 6 parallel attention heads capture different relationship patterns
        b) LayerNorm: Normalizes activations for stable training
        c) MLP: Two linear layers with GELU activation
           - Expands to 4× dim (384 → 1536), then projects back (1536 → 384)
           - Adds non-linear transformation capacity
        d) Residual connections: Skip connections around attention and MLP
           - Prevents vanishing gradients in deep networks

    Step 6 — Multi-Token Extraction (OUR CUSTOM PART):
        Instead of taking only [CLS], we extract:
        - CLS token: output[0, :, :]  → (B, 384)
        - Patch tokens: output[0, 1:, :]  → (B, 256, 384)
        - Mean of patches: mean(patches, dim=1) → (B, 384)
        - Max of patches: max(patches, dim=1) → (B, 384)
        - Concatenated: (B, 384*3) = (B, 1152)
    """

    # Pooling mode constants
    POOL_CLS_ONLY = 'cls'        # Standard: just CLS token (384-dim)
    POOL_MULTI = 'multi'          # Custom: CLS + mean + max (1152-dim)

    def __init__(
        self,
        variant: str = 'dinov2_vits14',
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 2,
        pooling_mode: str = 'multi',
    ):
        """
        Args:
            variant: DINOv2 model variant name.
            freeze: If True, freeze backbone parameters first.
            unfreeze_last_n_blocks: Number of last transformer blocks to
                                    keep trainable. Default=2 for partial fine-tuning.
            pooling_mode: 'cls' for CLS-only (standard), 'multi' for CLS+mean+max (custom).
        """
        super().__init__()

        if variant not in DINOV2_CONFIGS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(DINOV2_CONFIGS.keys())}"
            )

        self.variant = variant
        self.config = DINOV2_CONFIGS[variant]
        self.embed_dim = self.config['embed_dim']
        self.pooling_mode = pooling_mode

        # ---- Load pretrained DINOv2 from Facebook Research ----
        # torch.hub.load downloads the model definition and pretrained weights
        # These weights were trained with self-supervised learning (self-distillation)
        # on 142M images — they capture very strong visual representations
        logger.info(f"Loading DINOv2: {variant} ({self.config['description']})")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            variant,
            pretrained=True,
        )
        logger.info(f"DINOv2 loaded successfully. Embed dim: {self.embed_dim}")

        # ---- Freeze / Unfreeze strategy ----
        if freeze:
            self._freeze_backbone(unfreeze_last_n_blocks)

    def _freeze_backbone(self, unfreeze_last_n: int = 2):
        """
        Freeze/unfreeze strategy for partial fine-tuning.

        For dinov2_vits14 with unfreeze_last_n=2:
            - Blocks 0-9:  FROZEN  (general visual features — keep as-is)
            - Blocks 10-11: TRAINABLE (adapt to deepfake detection)
            - Patch embedding: FROZEN
            - CLS token: FROZEN
            - Position embedding: FROZEN
            - Final LayerNorm: TRAINABLE (part of the output processing)

        Args:
            unfreeze_last_n: Number of last transformer blocks to unfreeze.
        """
        # Step 1: Freeze EVERYTHING first
        for param in self.backbone.parameters():
            param.requires_grad = False

        total_params = sum(1 for p in self.backbone.parameters())
        logger.info(f"Frozen all {total_params} parameter tensors in DINOv2 backbone")

        # Step 2: Unfreeze the last N transformer blocks
        if unfreeze_last_n > 0:
            blocks = self.backbone.blocks
            total_blocks = len(blocks)
            unfreeze_from = max(0, total_blocks - unfreeze_last_n)

            for block_idx in range(unfreeze_from, total_blocks):
                for param in blocks[block_idx].parameters():
                    param.requires_grad = True

                logger.info(
                    f"  Unfroze block {block_idx}/{total_blocks - 1} "
                    f"(contains self-attention + MLP)"
                )

            # Also unfreeze the final LayerNorm that comes after all blocks
            # This norm is applied to the output before we extract features
            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                logger.info("  Unfroze final LayerNorm")

        # Log parameter summary
        params = self.get_num_params()
        logger.info(
            f"  Backbone: {params['total']:,} total | "
            f"{params['trainable']:,} trainable | "
            f"{params['frozen']:,} frozen"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-token features from DINOv2.

        For pooling_mode='multi' (our custom approach):
            1. Get ALL token outputs from DINOv2 (CLS + 256 patch tokens)
            2. Separate CLS token from patch tokens
            3. Apply 3 pooling strategies
            4. Concatenate into single feature vector

        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            If pooling='multi':  (B, embed_dim * 3) = (B, 1152) for vits14
            If pooling='cls':    (B, embed_dim) = (B, 384) for vits14
        """
        # ---- Enable gradients only if we have trainable params ----
        # (If backbone is fully frozen, we skip gradient computation for speed)
        grad_enabled = self._has_trainable_params()

        with torch.set_grad_enabled(grad_enabled):
            if self.pooling_mode == self.POOL_MULTI:
                # ---- CUSTOM: Multi-token extraction ----
                # Use get_intermediate_layers to access full token outputs
                # n=1 means get output from the LAST transformer block
                # return_class_token=True gives us both patch tokens and CLS token
                outputs = self.backbone.get_intermediate_layers(
                    x,
                    n=1,                    # Get output of the last block
                    return_class_token=True, # Also return the CLS token separately
                )

                # outputs is a list of tuples: [(patch_tokens, cls_token)]
                # patch_tokens: (B, num_patches, embed_dim) = (B, 256, 384)
                # cls_token: (B, embed_dim) = (B, 384)
                patch_tokens, cls_token = outputs[0]

                # ---- Pooling Strategy 1: CLS Token ----
                # The [CLS] token has attended to all patches through 12 layers
                # It represents a learned GLOBAL summary of the image
                # Shape: (B, 384)
                feat_cls = cls_token

                # ---- Pooling Strategy 2: Mean Pooling ----
                # Average all 256 patch embeddings
                # This captures DISTRIBUTED information across the whole image
                # Unlike CLS which is a single learned summary, mean pooling
                # gives equal weight to every spatial location
                # Shape: (B, 384)
                feat_mean = patch_tokens.mean(dim=1)

                # ---- Pooling Strategy 3: Max Pooling ----
                # For each of the 384 feature dimensions, take the maximum
                # value across all 256 patches
                # This HIGHLIGHTS the most activated features regardless of
                # spatial location — excellent for detecting localized artifacts
                # like face blending boundaries or eye distortions in deepfakes
                # Shape: (B, 384)
                feat_max = patch_tokens.max(dim=1).values

                # ---- Concatenate all 3 representations ----
                # Each captures different aspects:
                #   CLS  = global summary (what the model "thinks" overall)
                #   Mean = distributed average (broad texture/structure info)
                #   Max  = salient features (strongest local activations)
                # Shape: (B, 384*3) = (B, 1152)
                features = torch.cat([feat_cls, feat_mean, feat_max], dim=1)

            else:
                # ---- STANDARD: CLS token only ----
                # Uses the default forward() which returns just the CLS token
                # Shape: (B, 384)
                features = self.backbone(x)

        return features

    def _has_trainable_params(self) -> bool:
        """Check if any backbone parameter requires gradients."""
        return any(p.requires_grad for p in self.backbone.parameters())

    def get_feature_dim(self) -> int:
        """
        Return the output feature dimension based on pooling mode.

        multi mode: embed_dim × 3 (CLS + mean + max concatenated)
        cls mode:   embed_dim × 1 (CLS token only)
        """
        if self.pooling_mode == self.POOL_MULTI:
            return self.embed_dim * 3  # 384 * 3 = 1152
        return self.embed_dim  # 384

    def get_num_params(self) -> Dict[str, int]:
        """Return count of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
