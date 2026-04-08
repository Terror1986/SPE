from datasets import load_dataset
import random, re
random.seed(42)
ds   = load_dataset("causal-nlp/cladder")
data = list(ds['full_v1.5_default'])

for qt in ['ett', 'backadj', 'correlation', 'det-counterfactual']:
    examples = [d for d in data if d.get('query_type')==qt]
    random.shuffle(examples)
    print(f"\n{'#'*60}\nQUERY TYPE: {qt} (total={len(examples)})")
    for ex in examples[:3]:
        nums = re.findall(r'(\d+(?:\.\d+)?)%', ex['prompt'])
        print(f"\n  LABEL:{ex['label']} RUNG:{ex['rung']}")
        print(f"  NUMS: {nums}")
        print(f"  REASONING: {ex['reasoning'][:300]}")
        print(f"  QUESTION: ...{ex['prompt'][-250:]}")
