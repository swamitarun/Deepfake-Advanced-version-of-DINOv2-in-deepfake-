"""
Temporal Aggregation Model — LSTM for Video Deepfake Detection

=================================================================================
WHY TEMPORAL MODELING FOR VIDEO:
=================================================================================

Standard approach: Process each frame independently → average predictions
    Frame_1 → REAL (0.7)
    Frame_2 → FAKE (0.6)
    Frame_3 → REAL (0.8)
    Average = 0.7 → REAL

Problem: This IGNORES temporal information. Deepfakes often have:
    - Frame-to-frame flickering (inconsistent face generation)
    - Temporal jitter (face position jumps between frames)
    - Blending inconsistencies that appear/disappear across frames
    - Inconsistent lighting changes

Our approach: Process the SEQUENCE of frame features through an LSTM
    [Frame_1_features, Frame_2_features, ..., Frame_N_features]
    → LSTM → learns temporal patterns → single prediction

=================================================================================
LSTM EXPLAINED:
=================================================================================

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN)
designed to process SEQUENCES while remembering long-range dependencies.

At each timestep t, the LSTM has:
    - x_t:  current input (frame features, 2304-dim)
    - h_t:  hidden state (what the LSTM "remembers" so far, 256-dim)
    - c_t:  cell state (long-term memory, 256-dim)

It uses 4 gates to control information flow:
    1. Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
       → Decides what to FORGET from previous cell state
       → σ output in [0,1]: 0=forget completely, 1=keep everything

    2. Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
       → Decides what NEW information to add to cell state

    3. Cell update:  c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
       → Creates candidate values to add

    4. Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
       → Decides what part of cell state to output

Cell update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Hidden state: h_t = o_t ⊙ tanh(c_t)

For video deepfake detection:
    - The LSTM processes frame features in temporal order
    - It can learn that "features changed abruptly between frames 5 and 6"
    - Or "face texture oscillates in a pattern typical of GAN-generated content"
    - The final hidden state h_T captures the temporal summary

=================================================================================
WEIGHTED ATTENTION ALTERNATIVE:
=================================================================================

We also provide a simpler attention-based weighted averaging:
    - Each frame gets a learned attention weight
    - Weights are computed from the frame features via a small network
    - Higher weight = frame is more informative for the decision
    - Output = weighted sum of frame features

This is less powerful than LSTM but:
    - Faster (no sequential processing)
    - More interpretable (you can see which frames matter most)
    - Works well when temporal order is less important

=================================================================================
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalLSTMAggregator(nn.Module):
    """
    LSTM-based temporal aggregation for video frame features.

    Pipeline:
        Frame features (N_frames, feat_dim)
        → LSTM → processes sequence temporally
        → Last hidden state (hidden_dim)
        → Classifier → [REAL, FAKE]

    For our dual-input DINOv2 with multi-token pooling:
        feat_dim = 2304 (1152 per branch × 2 branches)
        hidden_dim = 256 (configurable)
    """

    def __init__(
        self,
        input_dim: int = 2304,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Feature dimension per frame (2304 for dual-input vits14).
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            num_classes: Output classes (2 = real/fake).
            dropout: Dropout probability.
            bidirectional: If True, process sequence in both directions.
                          Doubles the effective hidden dim.

        WHY hidden_dim=256?
            - Frame features (2304-dim) contain rich information
            - The LSTM compresses the temporal sequence into a smaller summary
            - 256 is enough to capture temporal patterns without overfitting
            - Too large → overfitting on small video datasets
            - Too small → can't capture complex temporal patterns
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ---- Feature projection ----
        # Reduce feature dimension before LSTM to save memory and computation
        # 2304 → 512 is a 4.5× reduction
        # WHY: LSTM parameters scale with input_dim², so reducing it first
        # dramatically reduces parameter count and computation
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- LSTM layer ----
        # Processes the sequence of projected frame features
        # Input: (sequence_length, batch, 512)
        # Output: (sequence_length, batch, hidden_dim)
        # batch_first=True means input/output shape is (batch, seq_len, dim)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # Input shape: (B, T, D) not (T, B, D)
            dropout=dropout if num_layers > 1 else 0,  # Inter-layer dropout
            bidirectional=bidirectional,
        )

        # ---- Output classifier ----
        # Maps LSTM output to class predictions
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        # LSTM weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier initialization
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization
                # This helps prevent vanishing/exploding gradients in RNNs
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias initialization: zeros, except forget gate bias = 1
                # Setting forget gate bias to 1 makes the LSTM initially
                # "remember everything" rather than forgetting, which helps
                # with learning long-range dependencies
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                # Forget gate bias is in the second quarter of the bias vector
                start = n // 4
                end = n // 2
                nn.init.constant_(param.data[start:end], 1.0)

        # Linear layer initialization
        for module in [self.feature_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of frame features through LSTM.

        Args:
            frame_features: (B, T, feat_dim) where T is number of frames
                           For single video: (1, T, 2304)

        Returns:
            Logits: (B, num_classes)
        """
        B, T, D = frame_features.shape

        # Step 1: Project frame features to lower dimension
        # (B, T, 2304) → (B, T, 512)
        projected = self.feature_proj(frame_features)

        # Step 2: Process through LSTM
        # LSTM internally maintains hidden state (h) and cell state (c)
        # across the T timesteps
        # lstm_out: (B, T, hidden_dim) — output at each timestep
        # (h_n, c_n): final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(projected)

        # Step 3: Extract the LAST hidden state
        # h_n shape: (num_layers * num_directions, B, hidden_dim)
        # We want the output from the last layer
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # Forward: h_n[-2] processes sequence left→right
            # Backward: h_n[-1] processes sequence right→left
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Just the last layer's hidden state
            final_hidden = h_n[-1]  # (B, hidden_dim)

        # Step 4: Classify
        logits = self.classifier(final_hidden)  # (B, num_classes)

        return logits

    def predict(self, frame_features: torch.Tensor) -> Dict:
        """Predict with confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(frame_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        label_map = {0: 'REAL', 1: 'FAKE'}
        labels = [label_map[p.item()] for p in preds]

        return {
            'labels': labels,
            'predictions': preds,
            'probabilities': probs,
            'confidence': probs.max(dim=1).values,
        }


class WeightedAttentionAggregator(nn.Module):
    """
    Attention-based weighted averaging for video frame features.

    Simpler alternative to LSTM. Instead of sequential processing,
    it computes an importance weight for each frame and takes a
    weighted average.

    Pipeline:
        Frame features (B, T, feat_dim)
        → Attention network → frame weights (B, T, 1)
        → Weighted sum → aggregated feature (B, feat_dim)
        → Classifier → [REAL, FAKE]

    WHY attention weights?
        Not all frames are equally informative:
        - Some frames clearly show deepfake artifacts
        - Some frames are ambiguous (face is occluded, blurry)
        - Attention lets the model focus on the MOST informative frames
    """

    def __init__(
        self,
        input_dim: int = 2304,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Feature dimension per frame.
            num_classes: Output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        # ---- Attention weight network ----
        # Takes frame features → produces a scalar weight per frame
        # The weight indicates how "important" each frame is for the decision
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),    # Tanh bounds output to [-1, 1] for stable attention
            nn.Linear(256, 1),  # Single scalar per frame
        )

        # ---- Classifier on aggregated features ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, feat_dim)

        Returns:
            Logits: (B, num_classes)
        """
        # Step 1: Compute attention weights for each frame
        # (B, T, feat_dim) → (B, T, 1)
        attn_scores = self.attention(frame_features)

        # Step 2: Normalize weights with softmax across frames
        # Ensures weights sum to 1 across the T dimension
        # (B, T, 1) → (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Step 3: Weighted sum of frame features
        # (B, T, feat_dim) * (B, T, 1) → sum over T → (B, feat_dim)
        aggregated = (frame_features * attn_weights).sum(dim=1)

        # Step 4: Classify
        logits = self.classifier(aggregated)

        return logits

    def get_attention_weights(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization (which frames matter most).

        Returns:
            Attention weights: (B, T) — weights sum to 1 across T
        """
        with torch.no_grad():
            attn_scores = self.attention(frame_features)
            attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)
