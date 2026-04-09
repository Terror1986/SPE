"""
SPE Core Configuration v2.1
Development: RTX 3080 Ti (validate architecture)
Production target: 30-80B params (post-funding, same architecture scaled)
"""
import torch

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_BUDGET_GB = 11.5
DTYPE = torch.bfloat16

# ── SSM Backbone — development scale ─────────────────────────────────────────
# At production: scale d_model to 4096+, n_layers to 48+
SSM_D_MODEL = 512              # Development: 512 | Production: 4096
SSM_N_LAYERS = 8               # Development: 8   | Production: 48
SSM_D_STATE = 64               # Development: 64  | Production: 256
SSM_D_CONV = 4
SSM_EXPAND = 2

# Hybrid split (43% Mamba / 7% Attention / 50% MLP)
SSM_MAMBA_LAYERS = 4
SSM_ATTN_LAYERS = 1
SSM_MLP_LAYERS = 3

# ── Language Model dimensions ─────────────────────────────────────────────────
LM_D_MODEL = 2048              # Embedding dimension (stays large for quality)
LM_VOCAB_SIZE = 32_000
LM_MAX_SEQ_LEN = 2048
LM_DROPOUT = 0.1

# ── Hierarchical World Model ──────────────────────────────────────────────────
HWM_LEVELS = 4
HWM_LEVEL_DIMS = [512, 384, 256, 128]  # Development scale
HWM_ERROR_THRESHOLD = 0.1

# ── Compressed Latent Memory Graph ────────────────────────────────────────────
CLMG_NODE_DIM = 64
CLMG_MAX_NODES = 500_000
CLMG_TOP_K = 32
CLMG_DECAY_RATE = 0.0001

# ── Neuro-Symbolic Layer ──────────────────────────────────────────────────────
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
    print("  SPE CONFIG v2.1 — Development Scale")
    print("  3080 Ti validation -> 30-80B production target")
    print("=" * 60)
    print(f"  Device:          {DEVICE}")
    print(f"  SSM d_model:     {SSM_D_MODEL} (production: 4096+)")
    print(f"  SSM layers:      {SSM_N_LAYERS} (production: 48+)")
    print(f"  LM d_model:      {LM_D_MODEL}")
    print(f"  HWM dims:        {HWM_LEVEL_DIMS}")
    print(f"  Efficiency goal: 90% cost reduction vs GPT-4 at scale")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
