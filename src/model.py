"""
model.py — PatchTST-style Time-Series Transformer for outbreak classification.

Implements §3 of the project prompt:
  - Patch-based tokenization (configurable patch length, default 4 weeks)
  - Channel-independent design: each channel gets its own attention pathway
    with shared Transformer weights across channels (prevents noisy channels
    from corrupting attention for cleaner channels)
  - Encoder-only architecture + classification head (no autoregressive decoder)
  - Calendar-aware sinusoidal patch embeddings
  - Primary output: single sigmoid probability for "outbreak in t+4..t+6"
  - Optional multi-task head: separate sigmoid for each lead week (t+4, t+5, t+6)
  - Config-driven: CHANNEL_INDEPENDENT flag, MULTITASK flag, model dims

Architecture summary (default config):
  Input:  (B, T=12, C) where C = number of channels (e.g. 8 with weather+calendar)
  Patch:  (B, C, n_patches=3, patch_len=4)
  After channel-independent flatten: (B*C, n_patches=3, d_model=128)
  Transformer encoder: 3 layers, 8 heads
  Classification head: linear over pooled representation → sigmoid
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding (with calendar-aware sinusoidal position encoding)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Projects each (patch_len,)-dimensional patch into d_model, then adds
    a learnable positional embedding (per patch position).

    For the channel-independent case: this module is applied identically
    to every channel's patch sequence (shared weights).
    """

    def __init__(self, patch_len: int, d_model: int, n_patches: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_patches = n_patches

        # Linear projection from patch raw values → embedding space
        self.value_proj = nn.Linear(patch_len, d_model)

        # Learnable patch positional embeddings (adds seasonality awareness
        # because patch index correlates with position in the 12-week window)
        self.pos_embed = nn.Embedding(n_patches, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, patch_len)  — raw patches for one channel
        Returns:
            (B, n_patches, d_model)
        """
        B, n_p, _ = x.shape
        # Project patch values
        embed = self.value_proj(x)  # (B, n_patches, d_model)

        # Add positional embeddings
        positions = torch.arange(n_p, device=x.device)
        pos = self.pos_embed(positions)  # (n_patches, d_model)
        embed = embed + pos.unsqueeze(0)  # broadcast over batch

        return self.dropout(embed)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Block (standard pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Single Pre-LayerNorm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        # Feed-forward with pre-norm
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full PatchTST Classifier
# ─────────────────────────────────────────────────────────────────────────────

class PatchTSTClassifier(nn.Module):
    """
    PatchTST-style encoder for time-series classification.

    Channel-independent mode (default, CHANNEL_INDEPENDENT=True):
      Each input channel (cases, preci, LAI, Temp, calendar features) is
      processed through an identical Transformer encoder with shared weights.
      The per-channel representations are then concatenated and fed to the
      classification head. This prevents a single noisy channel from
      dominating the attention over all channels.

    Channel-mixing mode (CHANNEL_INDEPENDENT=False):
      All channels are concatenated along the feature axis before patching
      and processed jointly.

    Args:
        n_channels:        number of input channels (C)
        input_len:         length of input sequence in time steps (T)
        patch_len:         number of time steps per patch
        d_model:           transformer embedding dimension
        n_heads:           number of attention heads
        n_layers:          number of transformer encoder layers
        d_ff:              feed-forward hidden dimension
        dropout:           dropout rate
        n_lead_weeks:      number of per-lead-week outputs (for multi-task head)
        channel_independent: if True, use channel-independent pathways
        use_multitask:     if True, also output per-lead-week probabilities
    """

    def __init__(
        self,
        n_channels: int,
        input_len: int = C.INPUT_WINDOW,
        patch_len: int = C.PATCH_LEN,
        d_model: int = C.D_MODEL,
        n_heads: int = C.N_HEADS,
        n_layers: int = C.N_LAYERS,
        d_ff: int = C.D_FF,
        dropout: float = C.DROPOUT,
        n_lead_weeks: int = C.LEAD_MAX - C.LEAD_MIN + 1,
        channel_independent: bool = C.CHANNEL_INDEPENDENT,
        use_multitask: bool = C.MULTITASK,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.input_len = input_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.channel_independent = channel_independent
        self.use_multitask = use_multitask
        self.n_lead_weeks = n_lead_weeks

        assert input_len % patch_len == 0, (
            f"input_len ({input_len}) must be divisible by patch_len ({patch_len}). "
            f"Current: {input_len} / {patch_len} = {input_len / patch_len:.1f}"
        )
        self.n_patches = input_len // patch_len

        if channel_independent:
            # Shared patch embedding and Transformer encoder across all channels
            self.patch_embed = PatchEmbedding(patch_len, d_model, self.n_patches, dropout)
            self.encoder_layers = nn.ModuleList(
                [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                 for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(d_model)
            # After pooling over patches: each channel gives d_model features
            head_in_dim = n_channels * d_model

        else:
            # Channel-mixing: concatenate all channels along feature axis then patch
            # Each "patch" is (patch_len * n_channels) dimensional
            patch_in = patch_len * n_channels
            self.patch_proj = nn.Linear(patch_in, d_model)
            self.pos_embed = nn.Embedding(self.n_patches, d_model)
            self.encoder_layers = nn.ModuleList(
                [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                 for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(d_model)
            head_in_dim = d_model  # single sequence, pool over patches

        # Primary classification head: window-level binary output
        self.head_window = nn.Sequential(
            nn.Linear(head_in_dim, head_in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim // 2, 1),
        )

        # Optional multi-task heads for each lead week (t+4, t+5, t+6)
        if use_multitask:
            self.head_leads = nn.ModuleList([
                nn.Linear(head_in_dim, 1) for _ in range(n_lead_weeks)
            ])

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode_channel_independent(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) — batch, timesteps, channels

        Returns: (B, C * d_model) — flattened per-channel representations
        """
        B, T, C = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"

        channel_reps = []
        for c_idx in range(C):
            # Extract single channel: (B, T)
            x_c = x[:, :, c_idx]
            # Reshape to patches: (B, n_patches, patch_len)
            x_c = x_c.reshape(B, self.n_patches, self.patch_len)
            # Embed: (B, n_patches, d_model)
            z = self.patch_embed(x_c)
            # Encode
            for layer in self.encoder_layers:
                z = layer(z)
            z = self.norm(z)
            # Mean-pool over patch dimension: (B, d_model)
            z = z.mean(dim=1)
            channel_reps.append(z)

        # Concatenate all channel reps: (B, C * d_model)
        return torch.cat(channel_reps, dim=-1)

    def _encode_channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)

        Returns: (B, d_model)
        """
        B, T, C = x.shape
        # Reshape to patches: (B, n_patches, patch_len * C)
        x = x.reshape(B, self.n_patches, self.patch_len * C)
        # Project to d_model
        z = self.patch_proj(x)  # (B, n_patches, d_model)
        # Add positional embedding
        positions = torch.arange(self.n_patches, device=x.device)
        z = z + self.pos_embed(positions).unsqueeze(0)
        # Encode
        for layer in self.encoder_layers:
            z = layer(z)
        z = self.norm(z)
        # Mean-pool: (B, d_model)
        return z.mean(dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C)  — B=batch, T=input_len, C=n_channels

        Returns:
            dict with:
              'logit_window': (B, 1)  — raw logit for window-level prediction
              'logit_leads':  (B, n_lead_weeks) — per-lead logits (if use_multitask)
        """
        if self.channel_independent:
            rep = self._encode_channel_independent(x)
        else:
            rep = self._encode_channel_mixing(x)

        out = {"logit_window": self.head_window(rep)}  # (B, 1)

        if self.use_multitask:
            lead_logits = torch.cat(
                [head(rep) for head in self.head_leads], dim=-1
            )  # (B, n_lead_weeks)
            out["logit_leads"] = lead_logits

        return out


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Baseline model (§2) — defined here to share training infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Simple LSTM-based classifier for baseline comparison (§2).

    Input: (B, T, C)  — same format as PatchTSTClassifier
    Output: dict with 'logit_window' and optionally 'logit_leads'
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int = C.LSTM_HIDDEN,
        n_layers: int = C.LSTM_LAYERS,
        dropout: float = C.LSTM_DROPOUT,
        n_lead_weeks: int = C.LEAD_MAX - C.LEAD_MIN + 1,
        use_multitask: bool = C.MULTITASK,
    ):
        super().__init__()
        self.use_multitask = use_multitask
        self.n_lead_weeks = n_lead_weeks

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head_window = nn.Linear(hidden_size, 1)

        if use_multitask:
            self.head_leads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(n_lead_weeks)
            ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T, C)"""
        out, (h_n, _) = self.lstm(x)  # h_n: (n_layers, B, hidden)
        # Use last layer hidden state
        rep = self.dropout(h_n[-1])  # (B, hidden)

        result = {"logit_window": self.head_window(rep)}

        if self.use_multitask:
            lead_logits = torch.cat(
                [head(rep) for head in self.head_leads], dim=-1
            )
            result["logit_leads"] = lead_logits

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    n_channels: int,
    model_type: str = "transformer",
    channel_independent: bool = C.CHANNEL_INDEPENDENT,
) -> nn.Module:
    """
    Build and return the specified model.

    Args:
        n_channels:          number of input channels
        model_type:          'transformer' or 'lstm'
        channel_independent: only used for 'transformer'
    """
    if model_type == "transformer":
        return PatchTSTClassifier(
            n_channels=n_channels,
            channel_independent=channel_independent,
        )
    elif model_type == "lstm":
        return LSTMClassifier(n_channels=n_channels)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'transformer' or 'lstm'.")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = torch.device("cpu")

    B, T, Ch = 8, 12, 8  # batch=8, 12-week window, 8 channels
    x = torch.randn(B, T, Ch)

    print("=== PatchTSTClassifier (channel-independent) ===")
    model = build_model(n_channels=Ch, model_type="transformer", channel_independent=True)
    out = model(x)
    print("logit_window shape:", out["logit_window"].shape)
    print("logit_leads  shape:", out["logit_leads"].shape)

    print("\n=== PatchTSTClassifier (channel-mixing) ===")
    model_mix = build_model(n_channels=Ch, model_type="transformer", channel_independent=False)
    out_mix = model_mix(x)
    print("logit_window shape:", out_mix["logit_window"].shape)

    print("\n=== LSTMClassifier ===")
    lstm = build_model(n_channels=Ch, model_type="lstm")
    out_lstm = lstm(x)
    print("logit_window shape:", out_lstm["logit_window"].shape)

    # Parameter counts
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"\nTransformer (CI)   params: {count_params(model):,}")
    print(f"Transformer (mix)  params: {count_params(model_mix):,}")
    print(f"LSTM               params: {count_params(lstm):,}")
    print("Smoke test passed.")
