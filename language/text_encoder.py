"""
SPE Perception Interface Layer (PIL) - Text Encoder
Phase 1.2

Replaces the crude nn.Linear(64, 256) pixel encoder.
Converts token IDs -> 2048-dim vectors for the HWM.

Architecture:
- Token embedding table: 32K x 2048
- Sinusoidal positional encoding (no learned positions = better generalization)
- Layer norm + dropout for stability
- Output: (batch, seq_len, 2048) -> feeds directly into HWM SSM backbone
"""
import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append(os.path.expanduser("~/spe"))
from tokenizers import Tokenizer

TOKENIZER_PATH = os.path.expanduser("~/spe/language/spe_tokenizer.json")
VOCAB_SIZE = 32_000
D_MODEL = 2048
MAX_SEQ_LEN = 2048
DROPOUT = 0.1

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Not learned — generalizes better to unseen sequence lengths.
    Same approach used in original Transformer paper.
    """
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class SPETextEncoder(nn.Module):
    """
    PIL Text Encoder for SPE.
    
    Input:  token IDs (batch, seq_len) — integers from tokenizer
    Output: encoded sequence (batch, seq_len, d_model) — for HWM
    
    This is Module 1 of SPE: Perception Interface Layer
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
        pad_token_id: int = 1,  # [PAD] is index 1 in our tokenizer
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Token embeddings — the core lookup table
        # 32K tokens x 2048 dims = 65.5M parameters
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        
        # Scale embeddings by sqrt(d_model) — standard practice
        self.embedding_scale = math.sqrt(d_model)
        
        # Positional encoding (fixed, not learned)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Layer norm before feeding into HWM
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with small normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=real token, 0=padding
        Returns:
            (batch, seq_len, d_model) encoded sequence
        """
        # Embed tokens and scale
        x = self.token_embedding(input_ids) * self.embedding_scale
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply mask if provided (zero out padding positions)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
        
        # Normalize and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x

    def encode_text(self, text: str, device: str = 'cpu') -> torch.Tensor:
        """
        Convenience method: encode a raw string directly.
        Returns (1, seq_len, d_model) tensor.
        """
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        enc = tokenizer.encode(text)
        ids = torch.tensor([enc.ids], device=device)
        return self.forward(ids)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SPELanguageModelHead(nn.Module):
    """
    LM Head: converts HWM output back to vocabulary logits.
    Used for next-token prediction during training.
    
    Input:  (batch, seq_len, d_model) from HWM
    Output: (batch, seq_len, vocab_size) logits
    
    Shares weights with token_embedding (weight tying) —
    standard practice that saves 65M parameters and improves training.
    """
    def __init__(self, d_model: int = D_MODEL, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.layer_norm = nn.LayerNorm(d_model)
        # Linear projection — weights will be tied to embedding
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = self.layer_norm(x)
        return self.projection(x)  # (batch, seq_len, vocab_size)

    def tie_weights(self, embedding: nn.Embedding):
        """Tie projection weights to token embedding weights."""
        self.projection.weight = embedding.weight


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing SPE Text Encoder on {device}")
    print("="*50)

    # Build encoder
    encoder = SPETextEncoder().to(device)
    lm_head = SPELanguageModelHead().to(device)
    lm_head.tie_weights(encoder.token_embedding)

    params_enc = encoder.count_parameters()
    params_lm = sum(p.numel() for p in lm_head.parameters())
    print(f"Encoder parameters: {params_enc:,}")
    print(f"LM head parameters: {params_lm:,} (tied, no extra cost)")

    # Test with tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        test_texts = [
            "The causal effect of X on Y is measured by the do-calculus.",
            "If the player moves right then the position increases by one.",
            "Large language models fail on interventional reasoning tasks.",
        ]
        
        print(f"\nEncoding test sequences:")
        for text in test_texts:
            enc = tokenizer.encode(text)
            ids = torch.tensor([enc.ids], device=device)
            
            with torch.no_grad():
                output = encoder(ids)
                logits = lm_head(output)
            
            print(f"  Input:  '{text[:50]}...'")
            print(f"  Tokens: {len(enc.ids)}")
            print(f"  Output: {output.shape} -> logits: {logits.shape}")
            print(f"  Norm:   {output.norm().item():.4f}")
            print()

    # Test gradient flow
    print("Testing gradient flow...")
    ids = torch.randint(0, VOCAB_SIZE, (2, 128), device=device)
    output = encoder(ids)
    logits = lm_head(output)
    loss = logits.mean()
    loss.backward()
    grad = encoder.token_embedding.weight.grad
    print(f"  Embedding grad norm: {grad.norm().item():.6f}")
    print(f"  Gradient flow: {'OK' if grad.norm() > 0 else 'BROKEN'}")

    print("\nMemory usage:")
    if device == "cuda":
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    print("\nPhase 1.2 COMPLETE: Text encoder operational")
    print("Ready to connect to HWM in Phase 1.3")
