from datasets import load_dataset
import random, re
random.seed(99)
ds   = load_dataset("causal-nlp/cladder")
data = [d for d in ds['full_v1.5_default'] if d.get('query_type')=='det-counterfactual']
random.shuffle(data)
for ex in data[:8]:
    print(f"\nLABEL:{ex['label']}")
    print(f"REASONING:{ex['reasoning'][:400]}")
    print(f"Q:...{ex['prompt'][-200:]}")
