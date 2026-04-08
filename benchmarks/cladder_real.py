"""
CLADDER Real Benchmark
Uses the actual CLADDER dataset from HuggingFace.
This is what GPT-4 scored 58.8% on — natural language causal questions.
We must answer these as text, not as a classification over numeric features.
"""
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *

try:
    from datasets import load_dataset
    ds = load_dataset("causal-nlp/cladder", trust_remote_code=True)
    print("Dataset loaded successfully")
    print(f"Splits: {list(ds.keys())}")
    split = list(ds.keys())[0]
    print(f"\nFirst example from '{split}':")
    ex = ds[split][0]
    for k,v in ex.items():
        print(f"  {k}: {v}")
    print(f"\nTotal examples: {sum(len(ds[s]) for s in ds)}")
    print(f"\nColumn names: {ds[split].column_names}")
except Exception as e:
    print(f"Load error: {e}")
    print("\nTrying alternative name...")
    try:
        ds = load_dataset("causal-nlp/cladder")
        print(ds)
    except Exception as e2:
        print(f"Also failed: {e2}")
