"""
SPE Engine — Full system integration.
Wires: SSM → HWM → CLMG → NSAL → AIC into one forward pass.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *
from core.world_model import HierarchicalWorldModel
from memory.clmg import CompressedLatentMemoryGraph
from symbolic.nsal import NSAL
from inference.aic import ActiveInferenceController

class SPEEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.hwm  = HierarchicalWorldModel()
        self.aic  = ActiveInferenceController()
        self.clmg = CompressedLatentMemoryGraph()
        self.nsal = NSAL()
        # Memory context integrator
        self.mem_integrator = nn.Linear(
            HWM_LEVEL_DIMS[0] * 2, HWM_LEVEL_DIMS[0]
        ).to(DEVICE).to(DTYPE)

    def forward(self, x, preferred_state=None, store_memory=True):
        """
        x: (B, seq_len, 256) — encoded input
        Returns: dict with all system outputs
        """
        # 1. Build world model
        world_states, free_energy = self.hwm(x)

        # 2. Retrieve relevant memories, integrate into L1 state
        memories = self.clmg.retrieve(world_states[0])      # (B, K, 512)
        mem_ctx  = memories.mean(dim=1)                      # (B, 512)
        combined = torch.cat([world_states[0], mem_ctx], dim=-1)
        world_states[0] = self.mem_integrator(combined)      # (B, 512)

        # 3. Store current state to memory
        if store_memory:
            self.clmg.store(world_states[0].detach())

        # 4. Extract symbolic rules
        symbolic_result = self.nsal.process(world_states)

        # 5. Active inference — select best action
        aic_result = self.aic(world_states, preferred_state)

        return {
            "world_states":    world_states,
            "free_energy":     free_energy,
            "best_action":     aic_result["best_action"],
            "belief":          aic_result["belief"],
            "best_efe":        aic_result["best_efe"],
            "symbolic":        symbolic_result,
            "memory_nodes":    self.clmg.node_count,
        }

    def vram_usage(self):
        return torch.cuda.memory_allocated() / 1e6

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path):
        torch.save(self.state_dict(), path)
        self.nsal.save(path.replace(".pt", "_rules.json"))
        print(f"✓ Saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✓ Loaded from {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  SOVEREIGN PERCEPTION ENGINE — FULL SYSTEM")
    print("=" * 50)

    engine = SPEEngine().to(DEVICE).to(DTYPE)
    params = engine.total_parameters()
    print(f"Total parameters: {params:,}")
    print(f"Estimated VRAM:   {params * 2 / 1e6:.1f} MB (weights only)")

    # Full forward pass
    x = torch.randn(4, 64, 256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = engine(x)

    print(f"\n--- Forward Pass ---")
    print(f"Free Energy:   {out['free_energy'].item():.6f}")
    print(f"Best EFE:      {out['best_efe'].mean().item():.6f}")
    print(f"Memory nodes:  {out['memory_nodes']}")
    print(f"Symbolic:      {out['symbolic']}")
    print(f"Belief shape:  {out['belief'].shape}")
    print(f"\nVRAM used:     {engine.vram_usage():.1f} MB")
    print(f"VRAM budget:   {VRAM_BUDGET_GB * 1024:.0f} MB")
    print(f"Headroom:      {VRAM_BUDGET_GB * 1024 - engine.vram_usage():.0f} MB")
    print("\n✓ SPE Engine fully operational")
