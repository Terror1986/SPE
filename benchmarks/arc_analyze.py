"""
Analyze unsolved ARC tasks to find highest-impact rules to add.
"""
import json, os, sys, numpy as np
from collections import Counter
sys.path.append(os.path.expanduser("~/spe"))
from benchmarks.arc_solver import induce_rule, to_np, ARC_DIR

files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith('.json')])
unsolved = []

for fname in files:
    with open(os.path.join(ARC_DIR, fname)) as f:
        task = json.load(f)
    name, _ = induce_rule(task['train'])
    if name is None:
        unsolved.append((fname.replace('.json',''), task))

print(f"Unsolved: {len(unsolved)}/400\n")

# Analyze properties of unsolved tasks
shape_changes = Counter()
color_counts  = Counter()
size_patterns = Counter()

for tid, task in unsolved:
    t0 = task['train'][0]
    i  = to_np(t0['input'])
    o  = to_np(t0['output'])
    ih,iw = i.shape; oh,ow = o.shape

    # Shape relationship
    if ih==oh and iw==ow:   shape_changes['same_shape'] += 1
    elif oh>ih and ow>iw:   shape_changes['output_larger'] += 1
    elif oh<ih or ow<iw:    shape_changes['output_smaller'] += 1
    else:                   shape_changes['aspect_change'] += 1

    # Color count in input vs output
    ic = len(set(i.flatten()))
    oc = len(set(o.flatten()))
    if ic==oc:   color_counts['same_colors'] += 1
    elif oc>ic:  color_counts['more_colors'] += 1
    else:        color_counts['fewer_colors'] += 1

    # Common output sizes
    size_patterns[f"{oh}x{ow}"] += 1

print("Shape changes:")
for k,v in shape_changes.most_common(): print(f"  {k:>20}: {v}")

print("\nColor patterns:")
for k,v in color_counts.most_common(): print(f"  {k:>20}: {v}")

print("\nTop output sizes:")
for k,v in size_patterns.most_common(10): print(f"  {k:>10}: {v}")

# Find tasks where output is 1x1, 3x3, etc (common small outputs)
print("\n--- Tasks with tiny outputs (likely counting/selecting) ---")
tiny = [(tid,t) for tid,t in unsolved
        if to_np(t['train'][0]['output']).size <= 4]
for tid,t in tiny[:5]:
    i=to_np(t['train'][0]['input']); o=to_np(t['train'][0]['output'])
    print(f"  {tid}: input={i.shape} output={o.shape} out={o.tolist()}")

# Find tasks where output is always 1 row (likely filtering)
print("\n--- Same-shape tasks: sample 5 ---")
same = [(tid,t) for tid,t in unsolved
        if to_np(t['train'][0]['input']).shape ==
           to_np(t['train'][0]['output']).shape][:5]
for tid,t in same:
    i=to_np(t['train'][0]['input']); o=to_np(t['train'][0]['output'])
    diff=(i!=o); n_diff=diff.sum()
    colors_added=set(o[diff].tolist())-set(i[diff].tolist())
    print(f"  {tid}: {i.shape}, {n_diff} cells changed, new colors={colors_added}")
