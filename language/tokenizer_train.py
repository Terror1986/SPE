"""
SPE BPE Tokenizer - 32K vocab
Trains on wikitext-103 + local logic files
"""
import os
os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN',''))

from tokenizers import Tokenizer, trainers, pre_tokenizers, models
from datasets import load_dataset
import glob

VOCAB_SIZE = 32_000
OUTPUT = os.path.expanduser("~/spe/language/spe_tokenizer.json")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]","[PAD]","[BOS]","[EOS]","[SEP]","[MASK]"],
    min_frequency=2,
    show_progress=True,
)

def collect_texts():
    texts = []
    
    # Wikitext-103 (small, fast, no rate limit issues)
    print("Loading wikitext-103...", flush=True)
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for ex in ds:
            if ex["text"].strip():
                texts.append(ex["text"])
        print(f"  Got {len(texts):,} wikitext articles", flush=True)
    except Exception as e:
        print(f"  wikitext failed: {e}", flush=True)

    # Local SPE files for logic/code vocabulary
    print("Loading local files...", flush=True)
    for pattern in ['~/spe/**/*.py','~/spe/**/*.md',
                    '~/spe/paper/*.tex']:
        for f in glob.glob(os.path.expanduser(pattern), recursive=True):
            try:
                with open(f) as fp:
                    texts.append(fp.read())
            except: pass
    
    print(f"Total texts: {len(texts):,}", flush=True)
    return texts

texts = collect_texts()

print(f"Training BPE on {len(texts):,} texts...", flush=True)
tokenizer.train_from_iterator(iter(texts), trainer=trainer)
tokenizer.save(OUTPUT)

vocab_size = tokenizer.get_vocab_size()
print(f"\nDone. Vocab: {vocab_size:,}", flush=True)
print(f"Saved: {OUTPUT}", flush=True)

# Test
for s in ["The causal effect of X on Y is measured by the do-calculus.",
          "If the player moves right then position increases by one.",
          "We prove by induction that the sequence converges to zero."]:
    enc = tokenizer.encode(s)
    print(f"  {len(enc.ids):3d} tokens: {s[:60]}", flush=True)
