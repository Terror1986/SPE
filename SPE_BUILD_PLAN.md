
---

## APPENDIX: MODEL DIMENSIONS FOR 1.5B TARGET

To hit 1-1.5B parameters with the 43/7/50 Mamba-Attn-MLP split:

| Component          | Config                          | Params    |
|--------------------|---------------------------------|-----------|
| Token embedding    | 32K vocab x 2048 dim            | 65.5M     |
| Mamba blocks x10   | d_model=2048, d_state=128       | ~580M     |
| Attention blocks x2| 2048 dim, 16 heads              | ~67M      |
| MLP blocks x12     | 2048 -> 8192 -> 2048            | ~603M     |
| LM head            | 2048 -> 32K                     | 65.5M     |
| **TOTAL**          |                                 | **~1.38B**|

VRAM at bf16: ~2.76GB weights + ~3GB activations = ~6GB training
Leaves 6GB headroom for optimizer states with gradient checkpointing.
Training batch: 4 sequences x 512 tokens fits comfortably.

This is the architecture we build in Phase 1.
