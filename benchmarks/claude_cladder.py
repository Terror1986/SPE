import os, sys, re, random, time
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *
from datasets import load_dataset
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))

def clean(text):
    """Remove non-ASCII characters that break the API."""
    return text.encode('ascii', 'replace').decode('ascii').replace('\xa0',' ')

def ask_claude(prompt):
    system = ("Answer causal reasoning questions with ONLY 'yes' or 'no'.")
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5,
            messages=[{"role":"user","content":
                       clean(prompt)+"\nAnswer (yes/no only):"}],
            system=system
        )
        ans = msg.content[0].text.strip().lower()
        if 'yes' in ans: return 'yes'
        if 'no'  in ans: return 'no'
        return 'uncertain'
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2)
        return 'uncertain'

def run_claude_benchmark(n_per_rung=40):
    print("="*60)
    print("  CLADDER HEAD-TO-HEAD: SPE vs Claude")
    print("="*60)

    ds   = load_dataset("causal-nlp/cladder")
    data = list(ds['full_v1.5_default'])
    random.seed(42)

    by_rung={1:[],2:[],3:[]}
    for ex in data:
        r=ex.get('rung')
        if r in by_rung: by_rung[r].append(ex)
    for r in by_rung: random.shuffle(by_rung[r])

    sampled=[]
    for r in [1,2,3]: sampled.extend(by_rung[r][:n_per_rung])
    random.shuffle(sampled)

    from benchmarks.cladder_solver import solve

    spe_c={1:0,2:0,3:0}; cl_c={1:0,2:0,3:0}; tot={1:0,2:0,3:0}
    cl_uncertain=0

    print(f"Running {len(sampled)} questions (est. 2-3 min)...\n")

    for i,ex in enumerate(sampled):
        prompt=ex['prompt']; true=str(ex['label']).lower().strip()
        qt=ex.get('query_type','ate'); rung=ex.get('rung',1)
        reasoning=str(ex.get('reasoning',''))

        spe_pred = solve(prompt,qt,reasoning)
        if spe_pred=='uncertain': spe_pred='yes'

        cl_pred = ask_claude(prompt)
        if cl_pred=='uncertain': cl_uncertain+=1; cl_pred='yes'

        spe_c[rung]+=int(spe_pred==true)
        cl_c[rung] +=int(cl_pred==true)
        tot[rung]  +=1

        if (i+1)%10==0:
            sc=sum(spe_c.values()); cc=sum(cl_c.values()); t=sum(tot.values())
            print(f"  [{i+1:>3}/{len(sampled)}] SPE:{sc/t:.1%}  Claude:{cc/t:.1%}")
        time.sleep(0.5)

    print(f"\n{'='*60}")
    names={1:"Association",2:"Intervention",3:"Counterfactual"}
    print(f"  {'Rung':<18} {'SPE':>8} {'Claude':>8}")
    print(f"  {'-'*36}")
    for r in [1,2,3]:
        if tot[r]:
            print(f"  {names[r]:<18} {spe_c[r]/tot[r]:>8.1%} {cl_c[r]/tot[r]:>8.1%}")

    sc=sum(spe_c.values()); cc=sum(cl_c.values()); t=sum(tot.values())
    print(f"  {'-'*36}")
    print(f"  {'Overall':<18} {sc/t:>8.1%} {cc/t:>8.1%}")
    print(f"  {'GPT-4 published':<18} {'58.8%':>8} {'62.0%':>8}")
    print(f"\n  SPE vs Claude:  {sc/t-cc/t:+.1%}")
    print(f"  SPE vs GPT-4:   {sc/t-0.588:+.1%}")
    print(f"  Claude uncertain: {cl_uncertain}/{t}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_claude_benchmark()
