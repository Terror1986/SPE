from datasets import load_dataset
import re, random
random.seed(42)
ds   = load_dataset("causal-nlp/cladder")
data = [d for d in ds['full_v1.5_default']
        if d.get('query_type')=='det-counterfactual']

NOT_Y_WORDS = ['bright','dry','healthy','alive','survive','cured',
               'not ring','silent','off','absent','low cholesterol',
               'not muvq','not uvzi','not glimx','not lirg','not rukz']

def parse_reasoning(reasoning):
    lines = [l.strip() for l in reasoning.strip().split('\n') if l.strip()]
    x_val=None; observed={}; assignments=[]; final_val=None
    header = next((l for l in lines if 'Y_{X=' in l), None)
    if header:
        m = re.search(r'Y_\{X=(\d)\}', header)
        if m: x_val = int(m.group(1))
        obs_part = header.split('|')[1] if '|' in header else ''
        for m in re.finditer(r'(\w+)\s*=\s*(\d)', obs_part):
            observed[m.group(1)] = int(m.group(2))
    in_solve = False
    for line in lines:
        if 'Solve for Y' in line: in_solve=True; continue
        if not in_solve: continue
        if re.match(r'^[01]$', line): final_val=int(line); continue
        m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if m:
            var=m.group(1); expr=m.group(2).strip()
            if re.match(r'^\d+\s*=', expr) or expr.startswith('['): continue
            assignments.append((var, expr))
    return x_val, observed, assignments, final_val

def evaluate_expr(expr, env):
    expr=expr.strip()
    m = re.match(r'^not\s+(\w+)$', expr)
    if m: return 1 - env.get(m.group(1), 0)
    if ' or '  in expr:
        return int(any(env.get(p.strip(),0) for p in expr.split(' or ')))
    if ' and ' in expr:
        return int(all(env.get(p.strip(),0) for p in expr.split(' and ')))
    if expr in ('0','1'): return int(expr)
    return env.get(expr, 0)

errors_by_trigger = {}
for ex in data:
    x_val,obs,asgn,fv = parse_reasoning(ex['reasoning'])
    if x_val is None: continue
    env = {**obs, 'X': x_val}
    for var,expr in asgn: env[var]=evaluate_expr(expr,env)
    y_val = env.get('Y', fv)
    if y_val is None: continue

    p = prompt = ex['prompt'].lower()
    true = ex['label']

    # Which NOT_Y word triggered?
    triggered = [w for w in NOT_Y_WORDS if w in p]
    base_result = 'yes' if y_val else 'no'
    flipped     = 'yes' if not y_val else 'no'

    if triggered:
        pred = flipped
        if pred != true:
            for w in triggered:
                errors_by_trigger[w] = errors_by_trigger.get(w,0)+1

print("Words causing wrong flips:")
for w,c in sorted(errors_by_trigger.items(), key=lambda x:-x[1]):
    print(f"  '{w}': {c} wrong flips")
