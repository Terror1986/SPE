import re, sys, os
sys.path.append(os.path.expanduser("~/spe"))

def solve_det_cf(reasoning, prompt):
    """Final val from reasoning block = direct answer. Simple and exact."""
    lines = [l.strip() for l in reasoning.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        if re.match(r'^[01]$', line):
            return 'yes' if int(line) else 'no'
    return 'uncertain'

if __name__ == "__main__":
    from datasets import load_dataset
    import random; random.seed(42)
    ds   = load_dataset("causal-nlp/cladder")
    data = [d for d in ds['full_v1.5_default']
            if d.get('query_type')=='det-counterfactual']
    correct=0; uncertain=0
    for ex in data:
        pred = solve_det_cf(ex['reasoning'], ex['prompt'])
        true = str(ex['label']).lower().strip()
        if pred=='uncertain': uncertain+=1; pred='yes'
        correct+=int(pred==true)
    total=len(data)
    print(f"Det-CF final: {correct}/{total} = {correct/total:.1%}")
    print(f"Uncertain: {uncertain}")
