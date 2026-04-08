"""Diagnose what the real questions look like and fix our parser."""
from datasets import load_dataset
import re, sys, os
sys.path.append(os.path.expanduser("~/spe"))

ds = load_dataset("causal-nlp/cladder")
data = list(ds['full_v1.5_default'])

# Show 2 examples per rung with their numbers extracted
def extract_numbers(prompt):
    return re.findall(r'(\d+(?:\.\d+)?)%', prompt)

for rung in [1,2,3]:
    examples = [d for d in data if d.get('rung')==rung][:2]
    for ex in examples:
        nums = extract_numbers(ex['prompt'])
        print(f"\n{'='*60}")
        print(f"RUNG:{rung} QUERY:{ex['query_type']} LABEL:{ex['label']}")
        print(f"NUMS FOUND: {nums}")
        print(f"REASONING:\n{ex['reasoning'][:400]}")
        print(f"PROMPT (last 300 chars):\n...{ex['prompt'][-300:]}")
