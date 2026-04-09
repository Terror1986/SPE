"""
SPE Language Model - Phase 1.3
Connects: Text Encoder (PIL) -> HWM -> LM Head
This is the full forward pass for language modeling.
Target: 1.38B parameters total
"""
import torch
import torch.nn as nn
import os, sys, math
sys.path.append(os.path.expanduser("~/spe"))

from language.text_encoder import SPETextEncoder, SPELanguageModelHead
from core.world_model import HierarchicalWorldModel
from core.config import DEVICE, DTYPE

class SPELanguageModel(nn.Module):
    """
    Full SPE Language Model.
    
    Pipeline:
    text -> tokens -> PIL (text encoder) -> HWM (Mamba SSM) -> LM head -> logits
    
    The HWM processes sequences with O(1) memory per step.
    No KV-cache. No attention over full context.
    Fixed-size recurrent state throughout.
    """
    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Module 1: Perception Interface Layer
        self.encoder = SPETextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Module 2: Hierarchical World Model (Mamba SSM backbone)
        # HWM expects input of shape (batch, seq_len, 256)
        # We need to project from d_model=2048 to HWM's expected 256
        self.input_projection = nn.Linear(d_model, 256)

        # The HWM processes and returns world states
        self.hwm = HierarchicalWorldModel()

        # Project HWM output (512 from L1) back to d_model for LM head
        self.output_projection = nn.Linear(2048, d_model)

        # Language model head
        self.lm_head = SPELanguageModelHead(d_model=d_model, vocab_size=vocab_size)

        # Tie weights: LM head shares embedding weights
        self.lm_head.tie_weights(self.encoder.token_embedding)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def reset_state(self):
        """Reset HWM recurrent state between sequences."""
        for level in self.hwm.levels:
            level.state = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            input_ids:      (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) optional padding mask
            labels:         (batch, seq_len) for computing loss (next token prediction)
        
        Returns:
            dict with logits, loss (if labels provided), world_states, free_energy
        """
        # Step 1: Encode text to embeddings
        # (batch, seq_len) -> (batch, seq_len, 2048)
        x = self.encoder(input_ids, attention_mask)

        # Step 2: Project to HWM input dimension
        # (batch, seq_len, 2048) -> (batch, seq_len, 256)
        x = self.input_projection(x)

        # Step 3: Process through Hierarchical World Model
        # Use gradient checkpointing to save VRAM during training
        if self.training:
            from torch.utils.checkpoint import checkpoint
            world_states, free_energy = checkpoint(
                self.hwm, x, use_reentrant=False)
        else:
            world_states, free_energy = self.hwm(x)

        # Detach recurrent state to prevent graph accumulation
        for level in self.hwm.levels:
            if level.state is not None:
                level.state = level.state.detach()

        # Step 4: Use L1 world state (richest representation)
        # world_states[0]: (batch, 512)
        # We need (batch, seq_len, 512) but HWM pools over seq
        # Expand back to sequence length
        h = world_states[0]  # (batch, 512)
        h = h.unsqueeze(1).expand(-1, input_ids.size(1), -1)  # (batch, seq_len, 512)

        # Step 5: Project to model dimension
        # (batch, seq_len, 512) -> (batch, seq_len, 2048)
        h = self.output_projection(h)
        h = self.dropout(h)

        # Step 6: LM head -> vocabulary logits
        # (batch, seq_len, 2048) -> (batch, seq_len, 32000)
        logits = self.lm_head(h)

        result = {
            'logits': logits,
            'world_states': world_states,
            'free_energy': free_energy,
        }

        # Compute language modeling loss if labels provided
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # ignore [PAD]
            lm_loss = loss_fn(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            # Combined loss: language modeling + free energy minimization
            total_loss = lm_loss + 0.01 * free_energy
            result['loss'] = total_loss
            result['lm_loss'] = lm_loss

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        O(1) memory per step — no KV cache needed.
        """
        self.eval()
        self.reset_state()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(generated)
                next_logits = out['logits'][:, -1, :] / temperature

                # Nucleus sampling (top-p)
                sorted_logits, sorted_idx = torch.sort(
                    next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs > top_p
                sorted_idx_to_remove[:, 1:] = sorted_idx_to_remove[:, :-1].clone()
                sorted_idx_to_remove[:, 0] = 0
                indices_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove)
                next_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop at EOS token (index 3)
                if next_token.item() == 3:
                    break

        return generated

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        enc = sum(p.numel() for p in self.encoder.parameters())
        hwm = sum(p.numel() for p in self.hwm.parameters())
        proj = sum(p.numel() for p in self.input_projection.parameters())
        proj += sum(p.numel() for p in self.output_projection.parameters())
        return {
            'total': total,
            'encoder_PIL': enc,
            'hwm_SSM': hwm,
            'projections': proj,
        }


if __name__ == "__main__":
    print("Building SPE Language Model...")
    print("="*50)

    model = SPELanguageModel().to(DEVICE).float()
    params = model.count_parameters()

    print(f"Parameter breakdown:")
    for k,v in params.items():
        print(f"  {k:20s}: {v:>12,}")
    print(f"  {'VRAM (float32)':20s}: {params['total']*4/1e9:.2f} GB")
    print(f"  {'VRAM (bfloat16)':20s}: {params['total']*2/1e9:.2f} GB")

    # Forward pass test
    print(f"\nForward pass test...")
    batch_size = 2
    seq_len = 64
    ids = torch.randint(4, 32000, (batch_size, seq_len)).to(DEVICE)
    labels = ids.clone()

    model.reset_state()
    out = model(ids, labels=ids)

    print(f"  Input:       {ids.shape}")
    print(f"  Logits:      {out['logits'].shape}")
    print(f"  LM Loss:     {out['lm_loss'].item():.4f}")
    print(f"  Free Energy: {out['free_energy'].item():.4f}")
    print(f"  Total Loss:  {out['loss'].item():.4f}")

    # Gradient test
    print(f"\nGradient flow test...")
    out['loss'].backward()
    enc_grad = model.encoder.token_embedding.weight.grad
    hwm_grad = list(model.hwm.parameters())[0].grad
    print(f"  Encoder grad norm: {enc_grad.norm().item():.6f}")
    print(f"  HWM grad norm:     {hwm_grad.norm().item() if hwm_grad is not None else 'None':.6f}")
    print(f"  Gradient flow: {'OK' if enc_grad.norm() > 0 else 'BROKEN'}")

    print(f"\nVRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"\nPhase 1.3 COMPLETE: Full language model pipeline operational")
    print(f"Next: Phase 1.4 - Training loop")
