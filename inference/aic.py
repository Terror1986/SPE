"""
Active Inference Controller (AIC)
Selects actions by minimizing Expected Free Energy G(π).
Unifies curiosity (epistemic) + goal-directedness (pragmatic).
No separate reward function needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *

class BeliefEncoder(nn.Module):
    """Encodes world states into a unified belief vector."""
    def __init__(self):
        super().__init__()
        total_in = sum(HWM_LEVEL_DIMS)  # 512+384+256+128 = 1280
        self.net = nn.Sequential(
            nn.Linear(total_in, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.SiLU(),
        )
    def forward(self, world_states):
        x = torch.cat(world_states, dim=-1)  # (B, 1280)
        return self.net(x)                   # (B, 256)


class PolicyNetwork(nn.Module):
    """
    Generates candidate action distributions.
    Actions here are abstract reasoning steps / query transformations.
    """
    def __init__(self, n_actions=64, action_dim=128):
        super().__init__()
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, n_actions * action_dim)
        )
    def forward(self, belief):
        B = belief.shape[0]
        out = self.net(belief)
        return out.view(B, self.n_actions, self.action_dim)


class EFECalculator(nn.Module):
    """
    Expected Free Energy G(π) = Epistemic + Pragmatic value.
    Epistemic:  information gain (reduce uncertainty → explore)
    Pragmatic:  proximity to preferred states (goal-directedness)
    """
    def __init__(self, action_dim=128):
        super().__init__()
        # Epistemic head: predict uncertainty reduction
        self.epistemic = nn.Sequential(
            nn.Linear(256 + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
        # Pragmatic head: predict goal proximity
        self.pragmatic = nn.Sequential(
            nn.Linear(256 + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, belief, actions, preferred_state=None):
        """
        belief:    (B, 256)
        actions:   (B, n_actions, action_dim)
        Returns:   efe (B, n_actions) — lower is better
        """
        B, N, D = actions.shape
        belief_exp = belief.unsqueeze(1).expand(B, N, -1)  # (B, N, 256)
        x = torch.cat([belief_exp, actions], dim=-1)        # (B, N, 256+D)

        epistemic = -self.epistemic(x).squeeze(-1)  # negative = reward uncertainty reduction
        pragmatic = -self.pragmatic(x).squeeze(-1)

        if preferred_state is not None:
            # Add explicit goal signal
            goal_dist = F.mse_loss(
                belief_exp, preferred_state.unsqueeze(1).expand_as(belief_exp),
                reduction='none').mean(-1)
            pragmatic = pragmatic + goal_dist

        efe = AIF_BETA * epistemic + (1 - AIF_BETA) * pragmatic
        return efe, epistemic, pragmatic


class ActiveInferenceController(nn.Module):
    """
    Full AIC: encodes beliefs → generates actions → scores by EFE → selects best.
    """
    def __init__(self):
        super().__init__()
        self.belief_encoder = BeliefEncoder()
        self.policy = PolicyNetwork(n_actions=AIF_N_SAMPLES, action_dim=128)
        self.efe_calc = EFECalculator(action_dim=128)

    def forward(self, world_states, preferred_state=None):
        """
        world_states: list of 4 tensors from HWM
        Returns: best_action (B, 128), efe_scores (B, n_actions), belief (B, 256)
        """
        belief = self.belief_encoder(world_states)       # (B, 256)
        actions = self.policy(belief)                    # (B, N, 128)
        efe, epist, pragm = self.efe_calc(
            belief, actions, preferred_state)            # (B, N)

        # Select action with lowest EFE
        best_idx = efe.argmin(dim=-1)                   # (B,)
        best_action = actions[
            torch.arange(actions.shape[0]), best_idx]   # (B, 128)

        return {
            "best_action": best_action,
            "efe": efe,
            "epistemic": epist,
            "pragmatic": pragm,
            "belief": belief,
            "best_efe": efe.min(dim=-1).values
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Building Active Inference Controller...")
    aic = ActiveInferenceController().to(DEVICE).to(DTYPE)
    print(f"Parameters: {aic.count_parameters():,}")

    world_states = [
        torch.randn(4, 512, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 384, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 256, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 128, device=DEVICE, dtype=DTYPE),
    ]

    with torch.no_grad():
        result = aic(world_states)

    print(f"Belief:      {result['belief'].shape}")
    print(f"Best action: {result['best_action'].shape}")
    print(f"Best EFE:    {result['best_efe'].mean().item():.4f}")
    print(f"Epistemic:   {result['epistemic'].mean().item():.4f}")
    print(f"Pragmatic:   {result['pragmatic'].mean().item():.4f}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print("✓ AIC operational")
