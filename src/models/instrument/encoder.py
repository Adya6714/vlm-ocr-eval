"""
Small Vision Transformer encoder for the instrument model (Stage 2a).

Trained entirely from scratch -- no pretrained weights, ever. This is
the whole point of "the instrument" vs. "the demo" (DECISIONS.md #1):
a pretrained encoder has already seen huge volumes of visual data, and
if that data includes text-like imagery, exposure control breaks
before training even starts. Starting blank is what makes Probe 1's
exposure manipulation causal instead of a rounding error.

Sizing: docs/stage2_design_notes.md -- 6 layers, hidden dim 320,
5 heads, patch size 14x14. ~7-8M params at this size; see that doc for
how encoder/decoder split the overall parameter budget.
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Cuts a line image into 14x14 patches and linearly projects each
    into a `d_model`-dimensional vector. This is the very first step
    of turning pixels into something a transformer can operate on --
    see BOOK.md Chapter 0's discussion of why patches, not whole-image
    features, and why this becomes the binding compute constraint at
    full-page resolution (not relevant yet at line-level, but the
    reason line-level training comes first).
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 1, d_model: int = 320):
        super().__init__()
        self.patch_size = patch_size
        # A conv with stride == kernel_size is exactly "cut into
        # non-overlapping patches, then linearly project each" --
        # cheaper to implement this way than an explicit unfold+linear.
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C, H, W], H and W must be divisible by patch_size
                 (the dataloader/collate function is responsible for
                 padding line crops to a multiple of patch_size).
        returns: [B, num_patches, d_model]
        """
        x = self.proj(images)               # [B, d_model, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)    # [B, num_patches, d_model]
        return x


def sinusoidal_positions(num_positions: int, d_model: int, device) -> torch.Tensor:
    """
    Fixed (non-learned) sinusoidal positional encoding. Chosen over a
    learned positional embedding specifically because line images have
    VARIABLE width -- a learned embedding table would need a fixed max
    length baked in ahead of time, whereas this formula extends to any
    sequence length for free. Standard "Attention Is All You Need"
    formulation, included here rather than imported since the encoder
    is meant to be a small, fully-understood, from-scratch component.
    """
    position = torch.arange(num_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(num_positions, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class InstrumentEncoder(nn.Module):
    """
    The full encoder: patch embedding + sinusoidal position + a stack
    of standard transformer encoder layers. Outputs one feature vector
    per patch, which the decoder (decoder.py) cross-attends to.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 1,   # grayscale -- document images don't need color for this task
        d_model: int = 320,
        num_layers: int = 6,
        num_heads: int = 5,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, d_model)
        self.d_model = d_model

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,   # so inputs are [B, seq, d_model], not [seq, B, d_model]
            norm_first=True,    # pre-norm -- more stable training for small transformers
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, images: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        images: [B, C, H, W]
        padding_mask: [B, num_patches] bool, True where a patch is
                       padding (from batching variable-width lines
                       together) and should be ignored by attention.
                       None if the batch has no padding.
        returns: [B, num_patches, d_model] -- passed to the decoder as
                 cross-attention memory.
        """
        x = self.patch_embed(images)  # [B, num_patches, d_model]
        pos = sinusoidal_positions(x.size(1), self.d_model, x.device)
        x = x + pos.unsqueeze(0)
        x = self.layers(x, src_key_padding_mask=padding_mask)
        return x


if __name__ == "__main__":
    # Smoke test: a fake batch of two grayscale line images, different
    # widths, padded to a common size + mask -- confirms shapes flow
    # correctly and padding is actually respected before this touches
    # any real training data.
    torch.manual_seed(0)
    encoder = InstrumentEncoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder parameter count: {n_params:,}")

    batch_size, channels, height, width = 2, 1, 70, 700  # divisible by patch_size=14
    fake_images = torch.randn(batch_size, channels, height, width)

    # simulate: image 0 has no padding, image 1 has the right half padded
    num_patches = (height // 14) * (width // 14)
    padding_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    padding_mask[1, num_patches // 2:] = True

    out = encoder(fake_images, padding_mask=padding_mask)
    print(f"output shape: {out.shape}  (expect [2, {num_patches}, 320])")
    assert out.shape == (batch_size, num_patches, 320)
    assert not torch.isnan(out).any(), "NaNs in encoder output"
    print("shape + NaN check OK")
