
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

---

## ARCHITECTURE SCALING STRATEGY (Updated)

### Development Hardware: RTX 3080 Ti 12GB
Validate all architectural decisions at small scale.
Target: 265M-750M params that fit and train on 3080 Ti.

### Production Target: 30-80B params (post-funding)
Same architecture, scaled up. Efficiency advantage grows with scale.

### Why Mamba efficiency compounds at scale:

| Model       | Params | Inference VRAM | Cost/1M tokens |
|-------------|--------|----------------|----------------|
| GPT-4       | ~1.8T  | 80GB+ H100     | $30            |
| Llama 3 70B | 70B    | 140GB          | $0.90          |
| SPE 30B     | 30B    | 4GB (no cache) | ~$0.05 est     |
| SPE 80B     | 80B    | 10GB (no cache)| ~$0.15 est     |

The KV-cache is the cost killer for transformers.
Mamba has NO KV-cache. Fixed state = fixed memory at any context length.
At 80B params with 100K context: transformer needs ~160GB VRAM.
Mamba needs the same 10GB regardless of context length.

### The 90% cost reduction claim is defensible because:
1. No KV-cache: constant inference memory vs linear growth
2. Smaller model: causal reasoning means fewer parameters needed
3. Consumer hardware: A100 instead of H100 clusters
4. No redundancy: NSAL replaces billions of params memorizing facts

### Development milestones on 3080 Ti:
- 265M: validate architecture, gradients, language capability
- 500M: validate reasoning improvement over smaller models  
- 750M: validate efficiency claims vs same-size transformers
- Submit for funding with validated architecture + benchmarks

### Scale-up milestones post-funding:
- 3B: commercial quality, beats GPT-3.5
- 7B: frontier competitive on reasoning tasks
- 30B: strong general capability, 90% cost advantage proven
- 80B: GPT-4 level quality, fraction of the cost
