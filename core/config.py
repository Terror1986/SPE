"""
SPE Core Configuration
All hyperparameters, hardware limits, and architectural constants live here.
"""

import torch

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_BUDGET_GB = 11.5          # Leave 1.4GB headroom for OS/overhead
DTYPE = torch.float16          # Half precision throughout — saves 50% VRAM

# ── SSM Backbone (Mamba) ──────────────────────────────────────────────────────
SSM_D_MODEL = 512              # Core hidden dimension
SSM_N_LAYERS = 8               # Number of Mamba layers
SSM_D_STATE = 64               # State space dimension (memory per token)
SSM_D_CONV = 4                 # Local convolution width
SSM_EXPAND = 2                 # Inner expansion factor

# ── Hierarchical World Model ──────────────────────────────────────────────────
HWM_LEVELS = 4                 # Abstraction levels (raw → physics → logic → goals)
HWM_LEVEL_DIMS = [             # Dimension at each level
    512,                       # Level 1: raw features
    384,                       # Level 2: object/event encoding
    256,                       # Level 3: relational/causal
    128,                       # Level 4: abstract goals
]
HWM_ERROR_THRESHOLD = 0.1     # Prediction error below this = skip update (efficiency)

# ── Compressed Latent Memory Graph ────────────────────────────────────────────
CLMG_NODE_DIM = 64            # Compressed node dimension
CLMG_MAX_NODES = 500_000      # Max episodic memories (~256MB at float16)
CLMG_TOP_K = 32               # Retrieve top-K relevant memories per step
CLMG_DECAY_RATE = 0.0001      # How fast unused memories fade

# ── Neuro-Symbolic Layer ──────────────────────────────────────────────────────
NSAL_MAX_RULES = 10_000       # Max rules in the Code-Graph
NSAL_CONFIDENCE_THRESHOLD = 0.85  # Min confidence to promote pattern → rule
NSAL_MAX_RULE_DEPTH = 8       # Max logical inference chain length

# ── Active Inference ──────────────────────────────────────────────────────────
AIF_HORIZON = 5               # Planning steps ahead
AIF_N_SAMPLES = 64            # Candidate action samples per step
AIF_BETA = 1.0                # Epistemic/pragmatic balance (1.0 = balanced)

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 3e-4
TRAIN_GRAD_CLIP = 1.0
TRAIN_CHECKPOINT_DIR = "/home/terror86/spe/checkpoints"
TRAIN_LOG_DIR = "/home/terror86/spe/logs"

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_TARGET = "CLADDER"  # Primary benchmark we are beating
BENCHMARK_BASELINE_GPT4 = 0.588   # GPT-4 CLADDER score (published)
BENCHMARK_BASELINE_CLAUDE = 0.620 # Claude-3 CLADDER score (published)
BENCHMARK_TARGET_SCORE = 0.75     # Our target — clear daylight above both

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_VRAM_EVERY_N_STEPS = 100  # How often to log VRAM usage

def print_config():
    print("=" * 50)
    print("  SOVEREIGN PERCEPTION ENGINE — CONFIG")
    print("=" * 50)
    print(f"  Device:        {DEVICE}")
    print(f"  VRAM Budget:   {VRAM_BUDGET_GB} GB")
    print(f"  SSM Layers:    {SSM_N_LAYERS} × d={SSM_D_MODEL}")
    print(f"  HWM Levels:    {HWM_LEVELS}")
    print(f"  Memory Nodes:  {CLMG_MAX_NODES:,}")
    print(f"  Max Rules:     {NSAL_MAX_RULES:,}")
    print(f"  Target:        Beat GPT-4 ({BENCHMARK_BASELINE_GPT4}) on CLADDER")
    print(f"  Our Target:    {BENCHMARK_TARGET_SCORE}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
