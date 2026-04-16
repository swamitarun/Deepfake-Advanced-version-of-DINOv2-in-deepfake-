"""
Advanced Deepfake Classifier — Dual-Input Pipeline with Custom Architecture

=================================================================================
WHAT CHANGED FROM STANDARD IMPLEMENTATION:
=================================================================================

Standard approach:
    Single image → DINOv2 → CLS (384) → Linear→ReLU→Dropout → [REAL/FAKE]

Our CUSTOM approach:
    Full image  ─→ DINOv2 ─→ [CLS+Mean+Max] (1152) ─┐
                   (shared)                            ├→ concat (2304) → Advanced Classifier
    Face crop   ─→ DINOv2 ─→ [CLS+Mean+Max] (1152) ─┘

=================================================================================
WHY DUAL INPUT:
=================================================================================

Deepfake detection benefits from looking at BOTH:
1. FULL IMAGE: Captures context — background consistency, lighting, overall
   composition. Some deepfakes have artifacts at image boundaries or
   inconsistent lighting between the face and background.

2. FACE CROP: Focuses on facial details — skin texture, eye reflections,
   hair boundaries, mouth movements. Most deepfake artifacts concentrate
   in the face region.

By processing both through the SAME DINOv2 backbone (shared weights), we:
- Extract complementary features from different scales/regions
- Don't double the model parameters (shared backbone)
- Let the classifier learn which scale/region is more informative

=================================================================================
WHY LayerNorm + GELU (not BatchNorm + ReLU):
=================================================================================

LayerNorm vs BatchNorm:
- BatchNorm normalizes across the BATCH dimension — unstable with small batches
  and behaves differently at training vs inference time
- LayerNorm normalizes across the FEATURE dimension — consistent behavior
  regardless of batch size, works identically at training and inference
- DINOv2 itself uses LayerNorm internally, so our features are already in
  a "LayerNorm-friendly" distribution

GELU vs ReLU:
- ReLU(x) = max(0, x) — hard cutoff, kills all negative values
- GELU(x) = x × Φ(x) — smooth approximation, allows small negative gradients
- GELU is what DINOv2's transformer blocks use internally
- Using GELU in our classifier keeps feature activations in the same "space"
  as the backbone output, leading to smoother gradient flow
- Smoother than ReLU → better optimization landscape → faster convergence

=================================================================================
"""

import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn

from .dino_extractor import DINOv2Extractor

logger = logging.getLogger(__name__)


# ============================================================================
# Building Block: Single classifier layer
# ============================================================================

class ClassifierBlock(nn.Module):
    """
    Single block of the advanced classifier.

    Architecture: Linear → LayerNorm → GELU → Dropout

    Why this order?
    - Linear: Transforms features to a new dimension
    - LayerNorm: Stabilizes activations BEFORE the non-linearity
      (pre-norm is more stable than post-norm in deep networks)
    - GELU: Smooth non-linear activation
    - Dropout: Randomly zeroes features during training to prevent
      co-adaptation and overfitting
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        """
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            dropout: Dropout probability (fraction of features zeroed).
        """
        super().__init__()

        # Linear projection: Maps from in_features → out_features
        # This is a learned weight matrix W (out×in) + bias b (out)
        # output = input @ W.T + b
        self.linear = nn.Linear(in_features, out_features)

        # LayerNorm: Normalizes across the feature dimension
        # For input (B, out_features), it computes:
        #   normalized = (x - mean(x)) / sqrt(var(x) + eps)
        #   output = gamma * normalized + beta
        # gamma and beta are learnable scale/shift parameters
        self.norm = nn.LayerNorm(out_features)

        # GELU: Gaussian Error Linear Unit
        # GELU(x) = x × Φ(x) where Φ is the cumulative distribution
        # function of the standard normal distribution
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        # Unlike ReLU, it doesn't have a hard zero cutoff — allows small
        # negative values to pass through with diminished magnitude
        self.activation = nn.GELU()

        # Dropout: During training, randomly sets elements to zero
        # with probability p=dropout. Remaining elements are scaled
        # by 1/(1-p) to maintain expected values. At inference, no dropout.
        # This prevents the network from relying on any single feature,
        # forcing it to learn redundant, robust representations
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, in_features)
        Returns:
            Output tensor (B, out_features)
        """
        x = self.linear(x)       # (B, in) → (B, out): Linear transformation
        x = self.norm(x)          # (B, out) → (B, out): Normalize features
        x = self.activation(x)   # (B, out) → (B, out): Non-linear activation
        x = self.dropout(x)      # (B, out) → (B, out): Regularization
        return x


# ============================================================================
# Advanced Classifier Head
# ============================================================================

class AdvancedClassifierHead(nn.Module):
    """
    Advanced classification head using LayerNorm + GELU + Dropout blocks.

    Architecture for dual-input with dinov2_vits14:
        Input: 2304-dim (1152 from full image + 1152 from face crop)
        Block 1: Linear(2304, 512) → LayerNorm(512) → GELU → Dropout(0.3)
        Block 2: Linear(512, 256)  → LayerNorm(256) → GELU → Dropout(0.3)
        Block 3: Linear(256, 128)  → LayerNorm(128) → GELU → Dropout(0.3)
        Output:  Linear(128, 2)    → raw logits (no activation!)

    WHY no activation on the output layer?
    - CrossEntropyLoss internally applies log_softmax
    - Adding softmax before CrossEntropyLoss would compute log(softmax(softmax(x)))
      which is WRONG and leads to poor training
    - Raw logits → CrossEntropyLoss is the correct approach
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Input feature dimension (2304 for dual-input vits14).
            hidden_dims: List of hidden layer sizes. Default: [512, 256, 128].
            num_classes: Number of output classes (2 = real/fake).
            dropout: Dropout probability for each block.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # ---- Build the classifier blocks ----
        blocks = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            blocks.append(ClassifierBlock(prev_dim, hidden_dim, dropout))
            logger.debug(f"  Classifier block {i}: {prev_dim} → {hidden_dim}")
            prev_dim = hidden_dim

        self.blocks = nn.Sequential(*blocks)

        # ---- Output layer ----
        # Maps from last hidden dim to num_classes
        # NO activation — raw logits for CrossEntropyLoss
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # ---- Weight initialization ----
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with appropriate strategies.

        - Linear layers in ClassifierBlocks: Kaiming/He initialization
          (designed for layers followed by ReLU-family activations like GELU)
        - Output layer: Xavier/Glorot initialization
          (designed for layers without activation, good for logits)
        - Biases: All initialized to zero
        - LayerNorm: gamma=1, beta=0 (default PyTorch behavior)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming init for hidden layers, Xavier for output
                if module is self.output_layer:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(
                        module.weight, mode='fan_out', nonlinearity='relu'
                    )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vector (B, input_dim)
        Returns:
            Logits (B, num_classes)
        """
        x = self.blocks(x)           # (B, input_dim) → (B, last_hidden_dim)
        logits = self.output_layer(x) # (B, last_hidden_dim) → (B, num_classes)
        return logits


# ============================================================================
# Dual-Input Deepfake Classifier (MAIN MODEL)
# ============================================================================

class DualInputDeepfakeClassifier(nn.Module):
    """
    Advanced deepfake detection model with dual-input pipeline.

    ================================================================
    FULL PIPELINE:
    ================================================================

    Input: Two images per sample
        full_image: (B, 3, 224, 224)  — complete image
        face_crop:  (B, 3, 224, 224)  — MTCNN-cropped face region

    Step 1 — Feature Extraction (SHARED DINOv2 backbone):
        full_image → DINOv2 → [CLS + MeanPool + MaxPool] → (B, 1152)
        face_crop  → DINOv2 → [CLS + MeanPool + MaxPool] → (B, 1152)

        WHY shared weights? The same visual feature extractor processes
        both inputs. This means:
        - No extra parameters for a second backbone
        - Both feature sets live in the SAME feature space
        - The classifier can directly compare them

    Step 2 — Feature Fusion:
        Concatenate: [full_features, face_features] → (B, 2304)

        WHY concatenation (not addition)? Concatenation preserves ALL
        information from both branches. Addition would force them into the
        same space and lose the distinction between "full image says X"
        and "face crop says Y". The classifier can learn cross-branch
        relationships from concatenated features.

    Step 3 — Classification:
        (B, 2304) → AdvancedClassifierHead → (B, 2)

    Step 4 — Output:
        Logits [real_score, fake_score] → softmax → probabilities
    ================================================================
    """

    def __init__(
        self,
        dino_variant: str = 'dinov2_vits14',
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
        pooling_mode: str = 'multi',
        dual_input: bool = True,
    ):
        """
        Args:
            dino_variant: DINOv2 model variant.
            hidden_dims: Classifier hidden layer sizes.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            freeze_backbone: Whether to freeze most of DINOv2.
            unfreeze_last_n_blocks: Number of last blocks to keep trainable.
            pooling_mode: 'multi' for CLS+mean+max, 'cls' for CLS only.
            dual_input: If True, expects two images (full + face crop).
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.dual_input = dual_input
        self.pooling_mode = pooling_mode

        # ---- Feature Extractor (DINOv2 backbone) ----
        # Single backbone shared between both input branches
        self.feature_extractor = DINOv2Extractor(
            variant=dino_variant,
            freeze=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            pooling_mode=pooling_mode,
        )

        # Calculate total feature dimension
        single_feat_dim = self.feature_extractor.get_feature_dim()
        # If dual input: features from both branches are concatenated
        total_feat_dim = single_feat_dim * 2 if dual_input else single_feat_dim

        logger.info(f"Feature dimensions:")
        logger.info(f"  Per-branch: {single_feat_dim}")
        logger.info(f"  Total ({'dual' if dual_input else 'single'}): {total_feat_dim}")

        # ---- Advanced Classifier Head ----
        self.classifier = AdvancedClassifierHead(
            input_dim=total_feat_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Store config for logging
        self._config = {
            'variant': dino_variant,
            'dual_input': dual_input,
            'pooling_mode': pooling_mode,
            'feat_dim': total_feat_dim,
            'hidden_dims': hidden_dims,
            'unfreeze_blocks': unfreeze_last_n_blocks,
        }

        self._log_summary()

    def _log_summary(self):
        """Log detailed model summary."""
        backbone_params = self.feature_extractor.get_num_params()
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_trainable = (
            backbone_params['trainable'] + classifier_params
        )

        logger.info("=" * 60)
        logger.info("  ADVANCED DEEPFAKE CLASSIFIER — MODEL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Backbone:         {self._config['variant']}")
        logger.info(f"  Pooling mode:     {self._config['pooling_mode']}")
        logger.info(f"  Dual input:       {self._config['dual_input']}")
        logger.info(f"  Feature dim:      {self._config['feat_dim']}")
        logger.info(f"  Classifier dims:  {self._config['hidden_dims']}")
        logger.info(f"  Unfrozen blocks:  {self._config['unfreeze_blocks']}")
        logger.info(f"  ---")
        logger.info(f"  Backbone total:     {backbone_params['total']:,}")
        logger.info(f"  Backbone trainable: {backbone_params['trainable']:,}")
        logger.info(f"  Backbone frozen:    {backbone_params['frozen']:,}")
        logger.info(f"  Classifier params:  {classifier_params:,}")
        logger.info(f"  TOTAL trainable:    {total_trainable:,}")
        logger.info("=" * 60)

    def forward(
        self,
        full_image: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through dual-input pipeline.

        Args:
            full_image: Full images (B, 3, 224, 224)
            face_crop: Face-cropped images (B, 3, 224, 224).
                       If None and dual_input=True, uses full_image as fallback.

        Returns:
            Logits (B, num_classes)
        """
        # ---- Branch 1: Full image features ----
        # Process complete image through DINOv2
        # Gets global context: background, lighting, overall composition
        full_features = self.feature_extractor(full_image)  # (B, 1152)

        if self.dual_input:
            # ---- Branch 2: Face crop features ----
            # Process face region through the SAME DINOv2 backbone
            # Gets fine-grained facial details: texture, eyes, mouth, boundaries
            if face_crop is None:
                # Fallback: duplicate full image features if no face detected
                face_features = full_features
            else:
                face_features = self.feature_extractor(face_crop)  # (B, 1152)

            # ---- Feature Fusion: Concatenation ----
            # Combine both branch outputs into a single feature vector
            # The classifier will learn which branch is more informative
            # and how to combine them optimally
            combined = torch.cat([full_features, face_features], dim=1)  # (B, 2304)
        else:
            combined = full_features  # (B, 1152)

        # ---- Classification ----
        logits = self.classifier(combined)  # (B, 2)

        return logits

    def extract_features(
        self,
        full_image: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract feature vector WITHOUT classification.
        Useful for LSTM temporal model (video processing).

        Args:
            full_image: (B, 3, 224, 224)
            face_crop: (B, 3, 224, 224) or None

        Returns:
            Feature vector (B, feat_dim) — 2304 for dual, 1152 for single
        """
        with torch.no_grad():
            full_features = self.feature_extractor(full_image)

            if self.dual_input and face_crop is not None:
                face_features = self.feature_extractor(face_crop)
                return torch.cat([full_features, face_features], dim=1)

            return full_features

    def predict(
        self,
        full_image: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Predict with full confidence output.

        Args:
            full_image: (B, 3, 224, 224)
            face_crop: (B, 3, 224, 224) or None

        Returns:
            Dict with keys:
                - labels: List[str] — ['REAL', 'FAKE', ...]
                - predictions: Tensor — [0, 1, ...] (0=real, 1=fake)
                - probabilities: Tensor — (B, 2) softmax scores
                - confidence: Tensor — max probability per sample
                - prob_real: Tensor — P(real) per sample
                - prob_fake: Tensor — P(fake) per sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(full_image, face_crop)

            # Apply softmax to convert logits to probabilities
            # softmax(x_i) = exp(x_i) / sum(exp(x_j))
            # This ensures outputs sum to 1 and are in [0, 1]
            probs = torch.softmax(logits, dim=1)

            # Predicted class = class with highest probability
            preds = torch.argmax(probs, dim=1)

        label_map = {0: 'REAL', 1: 'FAKE'}
        labels = [label_map[p.item()] for p in preds]

        return {
            'labels': labels,
            'predictions': preds,
            'probabilities': probs,
            'confidence': probs.max(dim=1).values,
            'prob_real': probs[:, 0],
            'prob_fake': probs[:, 1],
        }

    def get_trainable_params(self) -> list:
        """Return only parameters that require gradients (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self, backbone_lr_factor: float = 0.1, base_lr: float = 0.001):
        """
        Create parameter groups with different learning rates.

        WHY different LRs?
        - Backbone (unfrozen blocks): Already has good weights from pretraining.
          We want to adjust them GENTLY → use low LR (base_lr × backbone_lr_factor)
        - Classifier: Randomly initialized. Needs to learn from scratch → use full LR

        Args:
            backbone_lr_factor: Multiplier for backbone LR (e.g., 0.1 = 10× lower)
            base_lr: Base learning rate for the classifier.

        Returns:
            List of param group dicts for the optimizer.
        """
        backbone_params = [
            p for p in self.feature_extractor.parameters() if p.requires_grad
        ]
        classifier_params = list(self.classifier.parameters())

        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_factor,
                'name': 'backbone_finetune',
            })

        param_groups.append({
            'params': classifier_params,
            'lr': base_lr,
            'name': 'classifier',
        })

        logger.info(f"Parameter groups:")
        for pg in param_groups:
            n_params = sum(p.numel() for p in pg['params'])
            logger.info(f"  {pg['name']}: {n_params:,} params, lr={pg['lr']}")

        return param_groups


# ============================================================================
# Backwards-compatible single-input classifier (kept for reference)
# ============================================================================

class SingleInputClassifier(DualInputDeepfakeClassifier):
    """
    Single-input version for backwards compatibility.
    Wraps DualInputDeepfakeClassifier with dual_input=False.
    """

    def __init__(self, **kwargs):
        kwargs['dual_input'] = False
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(full_image=x, face_crop=None)

    def predict(self, x: torch.Tensor) -> Dict:
        return super().predict(full_image=x, face_crop=None)
