set -e

python improve/prepare_data.py --out_dir improve/data
python improve/optimize_prompt.py --data_dir improve/data --out_dir improve/artifacts

python improve/infer.py --mode baseline --out_path improve/out/baseline.jsonl --eval_path improve/data/eval.jsonl --max_tokens 8 --temperature 0 --top_p 1 --seed 1 --stop "\n"
python improve/infer.py --mode improved --out_path improve/out/improved.jsonl --eval_path improve/data/eval.jsonl --seed 1 --shots 5 --variants 2 --samples 3

python - << 'PY'
import json, math, random
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def read(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

b = read("improve/out/baseline.jsonl")
i = read("improve/out/improved.jsonl")
b_map = {x["id"]: x for x in b}
i_map = {x["id"]: x for x in i}
ids = sorted(set(b_map.keys()) & set(i_map.keys()))
b_ok = np.array([b_map[k]["ok"] for k in ids], dtype=int)
i_ok = np.array([i_map[k]["ok"] for k in ids], dtype=int)

acc_b = float(b_ok.mean()) if len(ids) else 0.0
acc_i = float(i_ok.mean()) if len(ids) else 0.0

def boot_ci(x, iters=2000, seed=1):
    rng = random.Random(seed)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(iters):
        samp = [x[rng.randrange(n)] for _ in range(n)]
        vals.append(sum(samp)/n)
    vals.sort()
    lo = vals[int(0.025*iters)]
    hi = vals[int(0.975*iters)-1]
    return float(lo), float(hi)

ci_b = boot_ci(list(b_ok), seed=11)
ci_i = boot_ci(list(i_ok), seed=12)

a = int(((b_ok==1) & (i_ok==1)).sum())
b01 = int(((b_ok==1) & (i_ok==0)).sum())
c10 = int(((b_ok==0) & (i_ok==1)).sum())
d = int(((b_ok==0) & (i_ok==0)).sum())
tbl = [[a, b01],[c10, d]]

try:
    res = mcnemar(tbl, exact=True)
    p = float(res.pvalue)
except Exception:
    p = float("nan")

lat_b = np.array([b_map[k].get("latency_s", 0.0) for k in ids], dtype=float)
lat_i = np.array([i_map[k].get("latency_s", 0.0) for k in ids], dtype=float)
lat_b_m = float(np.median(lat_b)) if len(lat_b) else 0.0
lat_i_m = float(np.median(lat_i)) if len(lat_i) else 0.0

examples = []
for k in ids:
    if len(examples) >= 12:
        break
    if b_map[k]["pred"] != i_map[k]["pred"]:
        examples.append(k)
if len(examples) < 12:
    for k in ids:
        if len(examples) >= 12:
            break
        if k not in examples:
            examples.append(k)

def short(x, n=160):
    t = x.replace("\n"," ").strip()
    return t[:n]

with open("improve/report.md", "w", encoding="utf-8") as f:
    f.write("# Part E Report\n\n")
    f.write("Benchmark is ARC Challenge.\n\n")
    f.write(f"Baseline accuracy is {acc_b:.4f}. 95 percent confidence interval is {ci_b[0]:.4f} to {ci_b[1]:.4f}.\n")
    f.write(f"Improved accuracy is {acc_i:.4f}. 95 percent confidence interval is {ci_i[0]:.4f} to {ci_i[1]:.4f}.\n")
    f.write(f"Paired significance uses McNemar exact test. p value is {p:.6f}.\n\n")

    f.write("## What changed\n")
    f.write("I used retrieval from a fixed local corpus built from the ARC Challenge training split.\n")
    f.write("For each eval question I selected the top similar training questions using TF IDF similarity and used them as few shot examples.\n")
    f.write("I also used prompt ensembling with two phrasing variants and self consistency with multiple seeded samples.\n")
    f.write("Final prediction was a majority vote across samples.\n")
    f.write("I restricted outputs to a single letter and applied output normalization that maps any extra text to the first valid option letter.\n\n")

    f.write("## Ablation summary\n")
    f.write("Baseline uses zero shot and deterministic decoding.\n")
    f.write("Improved uses five shot retrieval plus prompt ensembling plus three sample voting.\n")
    f.write("You can ablate by setting variants to one and samples to one.\n\n")

    f.write("## Cost and latency trade offs\n")
    f.write(f"Median latency baseline is {lat_b_m:.3f} seconds per item.\n")
    f.write(f"Median latency improved is {lat_i_m:.3f} seconds per item.\n")
    f.write("Improved mode makes multiple calls per question so it increases cost and latency roughly proportional to variants times samples.\n\n")

    f.write("## Exact settings\n")
    f.write("Baseline uses temperature 0 top p 1 top k 40 max tokens 8 seed 1 stop newline.\n")
    f.write("Improved uses temperature 0.2 top p 0.9 top k 40 max tokens 8 seed 1 with derived seeds stop newline.\n")
    f.write("Model and Ollama configuration are unchanged.\n\n")

    f.write("## Before and after examples\n")
    for k in examples[:12]:
        bb = b_map[k]
        ii = i_map[k]
        f.write(f"\n### id {k}\n")
        f.write(f"gold {bb['gold']}\n\n")
        f.write(f"baseline pred {bb['pred']} ok {bb['ok']}\n\n")
        f.write(f"improved pred {ii['pred']} ok {ii['ok']}\n\n")
        f.write(f"baseline raw {short(bb.get('raw',''))}\n\n")
        f.write(f"improved raw {short(ii.get('raw',''))}\n\n")

print("wrote improve/report.md")
print("baseline", acc_b, "improved", acc_i, "p", p)
PY
