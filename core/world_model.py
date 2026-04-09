"""
SPE Hierarchical World Model - 4-level predictive coding.
Version 2.0 - Scaled to 1.38B target dimensions.
Only prediction errors propagate upward.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *
from core.ssm_backbone import SSMBackbone

class PredictiveCodingLevel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.error_encoder = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.LayerNorm(dim_out))
        self.predictor = nn.Linear(dim_out, dim_in)
        self.update_gate = nn.Sequential(
            nn.Linear(dim_out * 2, dim_out), nn.Sigmoid())
        self.state = None

    def init_state(self, batch_size):
        self.state = torch.zeros(
            batch_size, self.dim_out, device=DEVICE, dtype=torch.float32)

    def forward(self, x):
        B = x.shape[0]
        if self.state is None or self.state.shape[0] != B:
            self.init_state(B)
        new_info = self.error_encoder(x)
        gate = self.update_gate(torch.cat([self.state, new_info], dim=-1))
        self.state = gate * new_info + (1 - gate) * self.state
        prediction = self.predictor(self.state)
        return self.state, prediction


class HierarchicalWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SSM backbone processes input sequence
        self.ssm = SSMBackbone(d_input=256, d_model=HWM_LEVEL_DIMS[0])

        # 4 predictive coding levels using new scaled dims
        self.levels = nn.ModuleList([
            PredictiveCodingLevel(HWM_LEVEL_DIMS[0], HWM_LEVEL_DIMS[0]),
            PredictiveCodingLevel(HWM_LEVEL_DIMS[0], HWM_LEVEL_DIMS[1]),
            PredictiveCodingLevel(HWM_LEVEL_DIMS[1], HWM_LEVEL_DIMS[2]),
            PredictiveCodingLevel(HWM_LEVEL_DIMS[2], HWM_LEVEL_DIMS[3]),
        ])
        self.error_threshold = HWM_ERROR_THRESHOLD

    def reset_states(self):
        for l in self.levels:
            l.state = None

    def forward(self, x):
        # x: (B, T, 256)
        B, T, _ = x.shape
        pooled = self.ssm(x).mean(dim=1)  # (B, d_model)

        current = pooled
        world_states = []
        total_fe = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

        for i, level in enumerate(self.levels):
            state, prediction = level(current)
            world_states.append(state)
            error = (current - prediction).pow(2).mean()
            total_fe = total_fe + error
            if i + 1 < len(self.levels):
                current = state

        return world_states, total_fe


if __name__ == "__main__":
    print("Building Scaled HWM...")
    model = HierarchicalWorldModel().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"HWM Parameters: {params:,}")
    x = torch.randn(2, 64, 256, device=DEVICE)
    with torch.no_grad():
        states, fe = model(x)
    for i,s in enumerate(states):
        print(f"  Level {i+1}: {s.shape} norm={s.norm().item():.2f}")
    print(f"  Free Energy: {fe.item():.4f}")
    vram = torch.cuda.memory_allocated()/1e6
    print(f"  VRAM: {vram:.1f} MB")
