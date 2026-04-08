
import json, os, sys, numpy as np
sys.path.append("/home/terror86/spe")
from benchmarks.arc_solver import induce_rule, to_np, ARC_DIR

files=sorted([f for f in os.listdir(ARC_DIR) if f.endswith(".json")])
unsolved=[]
for fname in files:
    with open(os.path.join(ARC_DIR,fname)) as f: task=json.load(f)
    name,_=induce_rule(task["train"])
    if name is None: unsolved.append((fname.replace(".json",""),task))

def show(tid, task):
    print(f"\n--- {tid} ---")
    for i,p in enumerate(task["train"][:2]):
        inp=to_np(p["input"]); out=to_np(p["output"])
        print(f"  Train{i+1} in={inp.shape} out={out.shape}")
        print(f"  Input:  {inp.tolist()}")
        print(f"  Output: {out.tolist()}")

print("=== OUTPUT 3x3 ===")
cat1=[(tid,t) for tid,t in unsolved
      if to_np(t["train"][0]["output"]).shape==(3,3)]
for tid,t in cat1[:4]: show(tid,t)

print("\n=== SAME SHAPE, SAME COLORS ===")
cat2=[(tid,t) for tid,t in unsolved
      if to_np(t["train"][0]["input"]).shape==to_np(t["train"][0]["output"]).shape
      and set(to_np(t["train"][0]["output"]).flatten().tolist())==
          set(to_np(t["train"][0]["input"]).flatten().tolist())]
for tid,t in cat2[:3]: show(tid,t)

print("\n=== OUTPUT SMALLER ===")
cat3=[(tid,t) for tid,t in unsolved
      if to_np(t["train"][0]["output"]).size * 2
         < to_np(t["train"][0]["input"]).size]
for tid,t in cat3[:3]: show(tid,t)
