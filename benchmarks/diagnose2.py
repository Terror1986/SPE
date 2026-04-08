from datasets import load_dataset
import random, re
random.seed(42)

ds   = load_dataset("causal-nlp/cladder")
data = list(ds['full_v1.5_default'])

for qt in ['ett', 'backadj', 'collider_bias', 'det-counterfactual']:
    examples = [d for d in data if d.get('query_type')==qt][:2]
    for ex in examples:
        nums = re.findall(r'(\d+(?:\.\d+)?)%', ex['prompt'])
        print(f"\n{'='*60}")
        print(f"QUERY:{qt} LABEL:{ex['label']} RUNG:{ex['rung']}")
        print(f"NUMS: {nums}")
        print(f"REASONING:\n{ex['reasoning'][:500]}")
        print(f"QUESTION: ...{ex['prompt'][-200:]}")
