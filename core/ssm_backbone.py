"""
SPE SSM Backbone
The core sequence model. Uses Mamba (State Space Model) instead of attention.
Key property: O(1) memory per step regardless of sequence length.
This replaces the KV-cache entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import sys
import os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *


class SPELayer(nn.Module):
    """
    Single SPE layer: Mamba SSM + gated feedforward.
    No attention. No KV-cache. Pure recurrent state.
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Mamba SSM — the core sequence mixer
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Gated feedforward — learns feature transformations
        self.ff_gate = nn.Linear(d_model, d_model * 2)
        self.ff_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # SSM with residual
        x = x + self.ssm(self.norm1(x))

        # Gated feedforward with residual
        normed = self.norm2(x)
        gate, val = self.ff_gate(normed).chunk(2, dim=-1)
        x = x + self.ff_proj(F.silu(gate) * val)

        return x


class SSMBackbone(nn.Module):
    """
    Full SPE SSM backbone.
    Stack of Mamba layers that process sequences with fixed memory.

    Input:  (batch, seq_len, d_input)  — any tokenized/encoded input
    Output: (batch, seq_len, d_model)  — rich contextual representations
    """
    def __init__(
        self,
        d_input=256,
        d_model=SSM_D_MODEL,
        n_layers=SSM_N_LAYERS,
        d_state=SSM_D_STATE,
        d_conv=SSM_D_CONV,
        expand=SSM_EXPAND,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Project input to model dimension
        self.input_proj = nn.Linear(d_input, d_model)

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            SPELayer(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Track which layers had significant activation (for sparsity logging)
        self.layer_activity = torch.zeros(n_layers)

    def forward(self, x, return_layer_outputs=False):
        """
        x: (batch, seq_len, d_input)
        Returns: (batch, seq_len, d_model)
        """
        # Project to model dimension
        x = self.input_proj(x)

        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_layer_outputs:
                layer_outputs.append(x.detach())

            # Track activity level (mean absolute activation)
            with torch.no_grad():
                self.layer_activity[i] = x.abs().mean().item()

        x = self.final_norm(x)

        if return_layer_outputs:
            return x, layer_outputs
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_vram_mb(self):
        """Rough VRAM estimate for this model at float16"""
        params = self.count_parameters()
        return (params * 2) / (1024 ** 2)  # 2 bytes per float16 param


def build_backbone():
    """Build and return the SSM backbone on GPU"""
    model = SSMBackbone().to(DEVICE).to(DTYPE)
    return model


if __name__ == "__main__":
    print("Building SSM Backbone...")
    model = build_backbone()

    params = model.count_parameters()
    vram = model.estimate_vram_mb()

    print(f"Parameters:     {params:,}")
    print(f"VRAM (weights): {vram:.1f} MB")
    print(f"Device:         {DEVICE}")
    print(f"Dtype:          {DTYPE}")

    # Forward pass test
    print("\nRunning forward pass test...")
    batch_size = 4
    seq_len = 128
    d_input = 256

    x = torch.randn(batch_size, seq_len, d_input, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out, layer_outs = model(x, return_layer_outputs=True)

    print(f"Input shape:    {x.shape}")
    print(f"Output shape:   {out.shape}")
    print(f"Layer outputs:  {len(layer_outs)} tensors")
    print(f"Output norm:    {out.norm().item():.4f}")

    # VRAM usage
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved  = torch.cuda.memory_reserved() / 1e6
    print(f"\nVRAM allocated: {allocated:.1f} MB")
    print(f"VRAM reserved:  {reserved:.1f} MB")
    print(f"VRAM budget:    {VRAM_BUDGET_GB * 1024:.0f} MB")
    print(f"Headroom:       {VRAM_BUDGET_GB * 1024 - reserved:.0f} MB")

    print("\n✓ SSM Backbone operational")
