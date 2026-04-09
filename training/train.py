"""
SPE Phase 1.4 - Language Model Training Loop
Target: next-token prediction on wikitext-103
1.377B parameters, bfloat16, gradient checkpointing
"""
import sys, os, time, math
sys.path.append(os.path.expanduser("~/spe"))
os.makedirs(os.path.expanduser("~/spe/checkpoints"), exist_ok=True)
os.makedirs(os.path.expanduser("~/spe/logs"), exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from datasets import load_dataset
from core.config import *
from language.spe_language_model import SPELanguageModel

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = os.path.expanduser("~/spe/language/spe_tokenizer.json")
CHECKPOINT_DIR = os.path.expanduser("~/spe/checkpoints")
LOG_PATH       = os.path.expanduser("~/spe/logs/train.log")
SEQ_LEN        = 256
BATCH_SIZE     = 1          # Small batch — large model
GRAD_ACCUM     = 16          # Effective batch = 2*8 = 16
MAX_STEPS      = 100_000
EVAL_EVERY     = 500
SAVE_EVERY     = 1000
LR             = 3e-4
WARMUP_STEPS   = 2000
GRAD_CLIP      = 1.0

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...", flush=True)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")
PAD_ID = tokenizer.token_to_id("[PAD]")

# ── Dataset ───────────────────────────────────────────────────────────────────
class WikiTextDataset(Dataset):
    def __init__(self, split="train", seq_len=SEQ_LEN):
        print(f"Loading wikitext-103 {split}...", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        self.seq_len = seq_len
        # Tokenize all text into one long token stream
        all_tokens = []
        for ex in ds:
            if ex["text"].strip():
                enc = tokenizer.encode(ex["text"])
                all_tokens.extend([BOS_ID] + enc.ids + [EOS_ID])
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_seqs = len(self.tokens) // seq_len
        self.tokens = self.tokens[:n_seqs * seq_len]
        print(f"  {len(self.tokens):,} tokens -> {n_seqs:,} sequences", flush=True)

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len]
        return chunk

def collate_fn(batch):
    return torch.stack(batch)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    print("Building SPE 1.377B...", flush=True)
    model = SPELanguageModel().to(DEVICE).to(torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e9:.3f}B)", flush=True)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e6:.0f}MB", flush=True)
    return model

# ── LR Schedule ───────────────────────────────────────────────────────────────
def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, step, loss):
    path = f"{CHECKPOINT_DIR}/spe_step{step:06d}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }, path)
    # Keep only last 3 checkpoints
    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
    for old in ckpts[:-3]:
        os.remove(f"{CHECKPOINT_DIR}/{old}")
    log(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer):
    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
    if not ckpts:
        return 0
    path = f"{CHECKPOINT_DIR}/{ckpts[-1]}"
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    log(f"Resumed from {path} at step {ck['step']}")
    return ck["step"]

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    total_loss = 0; n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            ids = batch.to(DEVICE)
            model.reset_state()
            out = model(ids, labels=ids)
            total_loss += out["lm_loss"].item()
            n += 1
    model.train()
    avg_loss = total_loss / max(n, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity

# ── Training Loop ─────────────────────────────────────────────────────────────
def train():
    log("="*55)
    log("SPE Language Model Training - Phase 1.4")
    log(f"Model: 1.377B params | Seq: {SEQ_LEN} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}")
    log("="*55)

    # Data
    train_ds = WikiTextDataset("train", SEQ_LEN)
    val_ds   = WikiTextDataset("validation", SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    # Model + optimizer
    model = build_model()
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=LR,
        betas=(0.9, 0.95), weight_decay=0.1,
    )
    log("Using 8-bit AdamW: ~6GB VRAM savings vs standard AdamW")

    # Resume if checkpoint exists
    start_step = load_checkpoint(model, optimizer)
    model.train()

    # Training
    step = start_step
    accum_loss = 0
    t0 = time.time()

    log(f"Starting training from step {step}...")

    while step < MAX_STEPS:
        for batch in train_loader:
            if step >= MAX_STEPS:
                break

            ids = batch.to(DEVICE)

            # Update LR
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            model.reset_state()
            out = model(ids, labels=ids)
            loss = out["loss"] / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

            # Optimizer step every GRAD_ACCUM batches
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                # Log
                if step % 50 == 0:
                    elapsed = time.time() - t0
                    tokens_per_sec = step * BATCH_SIZE * SEQ_LEN / max(elapsed, 1)
                    vram = torch.cuda.memory_allocated() / 1e6
                    log(f"Step {step:6d} | loss={accum_loss*GRAD_ACCUM:.4f} | "
                        f"ppl={math.exp(min(accum_loss*GRAD_ACCUM,20)):.1f} | "
                        f"lr={lr:.2e} | "
                        f"tok/s={tokens_per_sec:.0f} | "
                        f"VRAM={vram:.0f}MB")
                    accum_loss = 0

                # Eval
                if step % EVAL_EVERY == 0 and step > 0:
                    val_loss, ppl = evaluate(model, val_loader)
                    log(f"EVAL step={step} | val_loss={val_loss:.4f} | ppl={ppl:.1f}")

                # Save
                if step % SAVE_EVERY == 0 and step > 0:
                    save_checkpoint(model, optimizer, step, accum_loss)

            step += 1

    # Final eval + save
    val_loss, ppl = evaluate(model, val_loader)
    log(f"FINAL | val_loss={val_loss:.4f} | ppl={ppl:.1f}")
    save_checkpoint(model, optimizer, step, val_loss)
    log("Training complete.")

if __name__ == "__main__":
    train()
