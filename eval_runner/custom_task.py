import json
import os
import re
from typing import Any, Dict, List, Tuple

from eval_runner.model import GenerateParams, LocalGatewayModel


def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_choice(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else (s[:1] if s else "")


def canon_json(s: str) -> str:
    s = (s or "").strip()
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a : b + 1]
    try:
        obj = json.loads(s)
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        return s


def score_item(item: Dict[str, Any], pred: str) -> bool:
    t = item.get("type", "exact")
    exp = item.get("expected", "")

    if t == "choice":
        return extract_choice(pred) == norm(exp)
    if t == "exact_json":
        return canon_json(pred) == canon_json(exp)
    return norm(pred) == norm(exp)


def run_custom(bench_path: str = "eval_runner/custom_benchmark.json") -> Tuple[Dict[str, Any], str]:
    with open(bench_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    gw = LocalGatewayModel()
    params_text = GenerateParams(
        model=os.environ.get("OLLAMA_MODEL", "llama3:latest"),
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        max_tokens=64,
        response_format="text",
    )
    params_json = GenerateParams(
        model=os.environ.get("OLLAMA_MODEL", "llama3:latest"),
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        max_tokens=64,
        response_format="json_only",
    )

    rows: List[Dict[str, Any]] = []
    correct = 0
    cache_hits = 0

    for it in items:
        use_json = it.get("type") == "exact_json"
        params = params_json if use_json else params_text

        rec = gw.generate(it["prompt"], params=params)
        cache_hits += 1 if rec.get("cached") else 0

        pred = (rec.get("response", {}).get("output") or "").strip()
        ok = score_item(it, pred)
        correct += 1 if ok else 0

        rows.append(
            {
                "id": it.get("id"),
                "type": it.get("type"),
                "ok": ok,
                "cached": bool(rec.get("cached")),
                "wall_latency_s": rec.get("wall_latency_s"),
                "expected": it.get("expected"),
                "pred": pred,
            }
        )

    acc = correct / max(1, len(items))
    out = {
        "accuracy": acc,
        "n": len(items),
        "correct": correct,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / max(1, len(items)),
        "rows": rows,
    }

    os.makedirs("eval_runner/results", exist_ok=True)
    out_path = "eval_runner/results/custom_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out, out_path
