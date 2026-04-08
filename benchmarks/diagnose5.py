"""
Key insight: the REASONING field contains the exact computation.
Let's parse the reasoning directly instead of the prompt.
"""
from datasets import load_dataset
import random, re
random.seed(42)
ds   = load_dataset("causal-nlp/cladder")
data = [d for d in ds['full_v1.5_default']
        if d.get('query_type')=='det-counterfactual']

# Check: how many have valid reasoning vs nan?
has_reasoning = [d for d in data if str(d['reasoning']) != 'nan'
                 and 'Solve for Y' in str(d['reasoning'])]
no_reasoning  = [d for d in data if str(d['reasoning']) == 'nan'
                 or 'Solve for Y' not in str(d['reasoning'])]

print(f"Total det-cf:      {len(data)}")
print(f"Has reasoning:     {len(has_reasoning)}")
print(f"No reasoning:      {len(no_reasoning)}")

# Show reasoning structure
print("\n--- Reasoning examples ---")
for ex in has_reasoning[:4]:
    print(f"\nLABEL: {ex['label']}")
    print(f"REASONING:\n{ex['reasoning']}")
    print("---")
