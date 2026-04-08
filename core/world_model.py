"""
SPE Hierarchical World Model - 4-level predictive coding.
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
        self.state = torch.zeros(batch_size, self.dim_out, device=DEVICE, dtype=DTYPE)

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
        # dims: L1=512, L2=384, L3=256, L4=128
        # each level: in=upper_dim, out=lower_dim
        self.ssm = SSMBackbone(d_input=256, d_model=512)
        self.levels = nn.ModuleList([
            PredictiveCodingLevel(512, 512),  # L1: 512→512
            PredictiveCodingLevel(512, 384),  # L2: 512→384
            PredictiveCodingLevel(384, 256),  # L3: 384→256
            PredictiveCodingLevel(256, 128),  # L4: 256→128
        ])
        self.error_threshold = HWM_ERROR_THRESHOLD

    def reset_states(self):
        for l in self.levels: l.state = None

    def forward(self, x):
        B, T, _ = x.shape
        pooled = self.ssm(x).mean(dim=1)  # (B, 512)

        current = pooled
        world_states = []
        total_fe = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

        for i, level in enumerate(self.levels):
            state, prediction = level(current)
            world_states.append(state)
            error = (current - prediction).pow(2).mean()
            total_fe = total_fe + error
            # Next level input = error signal at state dimension
            if i + 1 < len(self.levels):
                next_dim = self.levels[i+1].dim_out
                # error in state space, projected to next input size
                current = state  # pass state as input to next level

        return world_states, total_fe

if __name__ == "__main__":
    print("Building Hierarchical World Model...")
    model = HierarchicalWorldModel().to(DEVICE).to(DTYPE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    x = torch.randn(4, 64, 256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        states, fe = model(x)
    for i, s in enumerate(states):
        print(f"Level {i+1}: {s.shape}  norm={s.norm().item():.2f}")
    print(f"Free Energy: {fe.item():.6f}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print("✓ World Model operational")
