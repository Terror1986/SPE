"""
SPE Core Configuration v3.0
LOCKED: 1.15B parameter architecture
Development: RTX 3080 Ti 12GB
Production target: 30-80B same architecture scaled
"""
import torch

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_BUDGET_GB = 11.5
DTYPE = torch.bfloat16

# ── LOCKED ARCHITECTURE: 1.15B params ─────────────────────────────────────────
# d=2048, mamba=24, mlp=12 -> 1.15B params, 4.6GB training, fits 3080 Ti
# DO NOT CHANGE WITHOUT UPDATING VRAM ESTIMATES

# SSM Backbone (Real Mamba, not GRU)
SSM_D_MODEL = 2048             # Core hidden dimension
SSM_D_STATE = 128              # SSM state size
SSM_D_CONV = 4                 # Local conv width
SSM_EXPAND = 2                 # Inner expansion (d_inner = d_model * expand)
SSM_N_LAYERS = 36              # Total layers
SSM_MAMBA_LAYERS = 24          # Mamba SSM layers (~67%)
SSM_MLP_LAYERS = 12            # MLP feedforward layers (~33%)
SSM_ATTN_LAYERS = 0            # No attention at dev scale (add at 7B+)

# ── Hierarchical World Model ──────────────────────────────────────────────────
HWM_LEVELS = 4
HWM_LEVEL_DIMS = [2048, 1536, 1024, 512]
HWM_ERROR_THRESHOLD = 0.1

# ── Language Model ────────────────────────────────────────────────────────────
LM_VOCAB_SIZE = 32_000
LM_D_MODEL = SSM_D_MODEL       # Same as SSM
LM_MAX_SEQ_LEN = 2048
LM_DROPOUT = 0.1

# ── Memory and Symbolic ───────────────────────────────────────────────────────
CLMG_NODE_DIM = 64
CLMG_MAX_NODES = 500_000
CLMG_TOP_K = 32
CLMG_DECAY_RATE = 0.0001

NSAL_MAX_RULES = 10_000
NSAL_CONFIDENCE_THRESHOLD = 0.85
NSAL_MAX_RULE_DEPTH = 8

# ── Active Inference ──────────────────────────────────────────────────────────
AIF_HORIZON = 5
AIF_N_SAMPLES = 64
AIF_BETA = 1.0

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 4
TRAIN_SEQ_LEN = 512
TRAIN_LR = 3e-4
TRAIN_GRAD_CLIP = 1.0
TRAIN_WARMUP_STEPS = 2000
TRAIN_CHECKPOINT_DIR = "/home/terror86/spe/checkpoints"
TRAIN_LOG_DIR = "/home/terror86/spe/logs"

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARK_BASELINE_GPT4 = 0.588
BENCHMARK_BASELINE_CLAUDE = 0.558
BENCHMARK_TARGET_SCORE = 0.875

def print_config():
    print("=" * 60)
    print("  SPE CONFIG v3.0 — 1.15B LOCKED")
    print("  d=2048 | 24 Mamba + 12 MLP | 4.6GB training")
    print("  3080 Ti dev -> 30-80B production (same arch)")
    print("=" * 60)
    print(f"  Device:        {DEVICE}")
    print(f"  SSM d_model:   {SSM_D_MODEL}")
    print(f"  Mamba layers:  {SSM_MAMBA_LAYERS}")
    print(f"  MLP layers:    {SSM_MLP_LAYERS}")
    print(f"  HWM dims:      {HWM_LEVEL_DIMS}")
    print(f"  Vocab:         {LM_VOCAB_SIZE:,}")
    print(f"  VRAM weights:  ~2.3GB bf16")
    print(f"  VRAM training: ~4.6GB (gradient checkpointing)")
    print(f"  Headroom:      ~6.9GB for activations+batch")
    print("=" * 60)
    print(f"  TurboQuant note: SPE has 0GB KV cache vs")
    print(f"  transformers' 33GB minimum after TurboQuant")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
