"""
Inspect ARC-AGI dataset structure before building anything.
Understand what we're solving before we build the solver.
"""
import json, os, urllib.request, sys
sys.path.append(os.path.expanduser("~/spe"))

# Download ARC-AGI-1 (public, well documented)
ARC_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/007bbfb7.json"
ARC_DIR = os.path.expanduser("~/spe/data/arc")
os.makedirs(ARC_DIR, exist_ok=True)

# Download a few training tasks
TASK_IDS = [
    "007bbfb7", "00d62c1b", "017c7c7b",
    "025d127b", "045e512c", "0520fde7",
]

BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training"

print("Downloading ARC-AGI tasks...")
tasks = {}
for tid in TASK_IDS:
    url  = f"{BASE}/{tid}.json"
    path = f"{ARC_DIR}/{tid}.json"
    try:
        urllib.request.urlretrieve(url, path)
        with open(path) as f:
            tasks[tid] = json.load(f)
        print(f"  ✓ {tid}")
    except Exception as e:
        print(f"  ✗ {tid}: {e}")

# Inspect structure
print(f"\n{'='*55}")
print("ARC TASK STRUCTURE")
print('='*55)

tid  = TASK_IDS[0]
task = tasks[tid]
print(f"\nTask: {tid}")
print(f"Training pairs: {len(task['train'])}")
print(f"Test pairs:     {len(task['test'])}")

print(f"\n--- Training pair 0 ---")
pair = task['train'][0]
inp  = pair['input']
out  = pair['output']
print(f"Input  ({len(inp)}×{len(inp[0])}):  {inp}")
print(f"Output ({len(out)}×{len(out[0])}): {out}")

print(f"\n--- Training pair 1 ---")
pair = task['train'][1]
print(f"Input:  {pair['input']}")
print(f"Output: {pair['output']}")

print(f"\n--- Test input (we must predict output) ---")
print(f"Input:  {task['test'][0]['input']}")

# Show all tasks at a glance
print(f"\n{'='*55}")
print("ALL TASKS AT A GLANCE")
print('='*55)
for tid, task in tasks.items():
    t0 = task['train'][0]
    i_shape = f"{len(t0['input'])}×{len(t0['input'][0])}"
    o_shape = f"{len(t0['output'])}×{len(t0['output'][0])}"
    n_train = len(task['train'])
    # Count unique colors
    colors = set()
    for pair in task['train']:
        for row in pair['input']:  colors.update(row)
        for row in pair['output']: colors.update(row)
    print(f"  {tid}: {n_train} pairs, grid {i_shape}→{o_shape}, colors={sorted(colors)}")
