"""
SPE SSM Backbone v3.0
Real Mamba architecture - 1.15B parameter target
24 Mamba layers + 12 MLP layers = 36 total
d_model=2048, d_state=128, expand=2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *


class MambaBlock(nn.Module):
    """Single Mamba SSM block with residual + layer norm."""
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MLPBlock(nn.Module):
    """Gated MLP block with residual + layer norm."""
    def __init__(self, d_model, expand=4):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj   = nn.Linear(d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class SSMBackbone(nn.Module):
    """
    SPE SSM Backbone: 24 Mamba + 12 MLP = 1.15B params
    Input:  (batch, seq_len, d_input)
    Output: (batch, seq_len, d_model)

    Layer ordering: interleave Mamba and MLP
    Every 2 Mamba blocks gets 1 MLP block.
    Pattern: M M P M M P M M P ... (repeat 12x)
    """
    def __init__(
        self,
        d_input=256,
        d_model=SSM_D_MODEL,
        n_mamba=SSM_MAMBA_LAYERS,
        n_mlp=SSM_MLP_LAYERS,
        d_state=SSM_D_STATE,
        d_conv=SSM_D_CONV,
        expand=SSM_EXPAND,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input to model dimension
        self.input_proj = nn.Linear(d_input, d_model)

        # Build interleaved layers: 2 Mamba + 1 MLP, repeated 12x
        self.layers = nn.ModuleList()
        mamba_per_group = n_mamba // n_mlp  # = 2
        for _ in range(n_mlp):
            for _ in range(mamba_per_group):
                self.layers.append(MambaBlock(d_model, d_state, d_conv, expand))
            self.layers.append(MLPBlock(d_model))

        self.final_norm = nn.LayerNorm(d_model)
        self.layer_activity = torch.zeros(len(self.layers))

    def forward(self, x, return_layer_outputs=False):
        """x: (batch, seq_len, d_input) -> (batch, seq_len, d_model)"""
        x = self.input_proj(x)
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_layer_outputs:
                layer_outputs.append(x.detach())
            with torch.no_grad():
                self.layer_activity[i] = x.abs().mean().item()
        x = self.final_norm(x)
        if return_layer_outputs:
            return x, layer_outputs
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_vram_mb(self):
        return (self.count_parameters() * 2) / (1024**2)


if __name__ == "__main__":
    print("Building SPE SSM Backbone v3.0 (Real Mamba)...")
    print("="*55)

    model = SSMBackbone().to(DEVICE).to(DTYPE)
    params = model.count_parameters()
    vram_w = model.estimate_vram_mb()

    print(f"Architecture:")
    print(f"  d_model:      {SSM_D_MODEL}")
    print(f"  Mamba layers: {SSM_MAMBA_LAYERS}")
    print(f"  MLP layers:   {SSM_MLP_LAYERS}")
    print(f"  Total layers: {len(model.layers)}")
    print(f"Parameters:   {params:,} ({params/1e9:.3f}B)")
    print(f"VRAM weights: {vram_w:.0f} MB ({vram_w/1024:.2f} GB)")

    # Forward pass
    print(f"\nForward pass test...")
    x = torch.randn(2, 128, 256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Norm:   {out.norm().item():.4f}")

    # Gradient test
    print(f"\nGradient test...")
    x2 = torch.randn(1, 64, 256, device=DEVICE, dtype=DTYPE)
    out2 = model(x2)
    loss = out2.mean()
    loss.backward()
    grad = model.input_proj.weight.grad
    print(f"  Grad norm: {grad.norm().item():.6f}")
    print(f"  Gradients: {'OK' if grad.norm() > 0 else 'BROKEN'}")

    allocated = torch.cuda.memory_allocated()/1e6
    reserved  = torch.cuda.memory_reserved()/1e6
    print(f"\nVRAM allocated: {allocated:.0f} MB")
    print(f"VRAM reserved:  {reserved:.0f} MB")
    print(f"Headroom:       {11500 - reserved:.0f} MB")
    print(f"\n✓ SSM Backbone operational at 1.15B scale")
