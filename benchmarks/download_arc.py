import urllib.request, os, json

ARC_DIR = os.path.expanduser("~/spe/data/arc")
os.makedirs(ARC_DIR, exist_ok=True)

BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training"

# Get list of all task IDs from GitHub API
import urllib.request, json
api_url = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"
req = urllib.request.Request(api_url,
    headers={'User-Agent': 'SPE-ARC-Downloader'})
with urllib.request.urlopen(req) as r:
    files = json.loads(r.read())

task_ids = [f['name'].replace('.json','') for f in files if f['name'].endswith('.json')]
print(f"Found {len(task_ids)} tasks")

ok=0; fail=0
for tid in task_ids:
    path = f"{ARC_DIR}/{tid}.json"
    if os.path.exists(path): ok+=1; continue
    try:
        urllib.request.urlretrieve(f"{BASE}/{tid}.json", path)
        ok+=1
    except Exception as e:
        print(f"  ✗ {tid}: {e}"); fail+=1

print(f"Downloaded: {ok} tasks, failed: {fail}")
