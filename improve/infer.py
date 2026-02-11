import argparse
import asyncio
import json
import os
import random
import re
import time
import joblib
import aiohttp
import numpy as np
from scipy import sparse

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now():
    return time.perf_counter()

def normalize_choice_letter(s):
    if s is None:
        return ""
    t = s.strip().upper()
    m = re.search(r"\b([ABCDE])\b", t)
    if m:
        return m.group(1)
    m = re.match(r"^\s*([ABCDE])\s*$", t)
    if m:
        return m.group(1)
    t = re.sub(r"[^A-Z]", " ", t)
    m = re.search(r"\b([ABCDE])\b", t)
    return m.group(1) if m else ""

def build_question_block(ex):
    lines = []
    lines.append(str(ex["question"]).strip())
    for k, v in ex["choices"]:
        lines.append(f"{k}. {v}")
    return "\n".join(lines)

def make_prompt_baseline(ex):
    q = build_question_block(ex)
    return (
        "You are answering a multiple choice question.\n"
        "Return only the letter A or B or C or D or E.\n"
        "Question\n"
        f"{q}\n"
        "Answer"
    )

def make_prompt_improved(ex, shots, variant):
    q = build_question_block(ex)
    if variant == 0:
        head = "You are solving ARC multiple choice questions.\nReturn only a single letter A B C D or E.\n"
        tail = "Return only the letter."
    else:
        head = "Select the best option.\nOutput must be exactly one letter among A B C D E.\n"
        tail = "Answer with one letter only."
    parts = [head, "Examples"]
    for s in shots:
        sq = build_question_block(s)
        parts.append(f"Q\n{sq}\nA\n{s['answerKey']}\n")
    parts.append("Now answer the next question")
    parts.append(f"Q\n{q}\nA\n{tail}")
    return "\n".join(parts)

async def ollama_generate(session, url, model, prompt, temperature, top_p, top_k, max_tokens, seed, stop):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "num_predict": int(max_tokens),
        },
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if stop:
        payload["options"]["stop"] = stop
    t0 = now()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            j = await resp.json()
            text = str(j.get("response") or "")
            latency = now() - t0
            return resp.status, latency, text, ""
    except Exception as e:
        latency = now() - t0
        return 0, latency, "", str(e)

def topk_shots(vec, X_train, train_rows, query_text, k):
    qv = vec.transform([query_text])
    sims = (X_train @ qv.T).toarray().reshape(-1)
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    shots = []
    for i in idx:
        shots.append(train_rows[int(i)])
    return shots

def majority_vote(cands):
    c = [x for x in cands if x]
    if not c:
        return ""
    counts = {}
    for x in c:
        counts[x] = counts.get(x, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best

async def run(args):
    eval_rows = read_jsonl(args.eval_path)

    train_rows = []
    vec = None
    X_train = None
    if args.mode == "improved":
        train_rows = read_jsonl(os.path.join(args.artifacts_dir, "train_compact.jsonl"))
        vec = joblib.load(os.path.join(args.artifacts_dir, "tfidf_vectorizer.joblib"))
        X_train = sparse.load_npz(os.path.join(args.artifacts_dir, "train_tfidf.npz"))

    sem = asyncio.Semaphore(args.concurrency)
    out = []

    async with aiohttp.ClientSession() as session:
        async def one(ex):
            async with sem:
                gold = str(ex["answerKey"]).strip().upper()
                base_text = build_question_block(ex)
                if args.mode == "baseline":
                    prompt = make_prompt_baseline(ex)
                    status, latency, text, err = await ollama_generate(
                        session,
                        args.url,
                        args.model,
                        prompt,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        args.max_tokens,
                        args.seed,
                        args.stop,
                    )
                    pred = normalize_choice_letter(text)
                    ok = int(pred == gold)
                    return {
                        "id": ex["id"],
                        "mode": "baseline",
                        "gold": gold,
                        "pred": pred,
                        "ok": ok,
                        "latency_s": latency,
                        "http_status": status,
                        "error": err,
                        "raw": text[:4000],
                    }

                shots = topk_shots(vec, X_train, train_rows, base_text, args.shots)
                preds = []
                latencies = []
                statuses = []
                errs = []
                raws = []

                for v in range(args.variants):
                    for sidx in range(args.samples):
                        seed = args.seed + 1000 * v + sidx
                        prompt = make_prompt_improved(ex, shots, v)
                        status, latency, text, err = await ollama_generate(
                            session,
                            args.url,
                            args.model,
                            prompt,
                            args.temperature_improved,
                            args.top_p_improved,
                            args.top_k_improved,
                            args.max_tokens_improved,
                            seed,
                            args.stop_improved,
                        )
                        pred = normalize_choice_letter(text)
                        preds.append(pred)
                        latencies.append(latency)
                        statuses.append(status)
                        errs.append(err)
                        raws.append(text[:1000])

                final_pred = majority_vote(preds)
                ok = int(final_pred == gold)
                return {
                    "id": ex["id"],
                    "mode": "improved",
                    "gold": gold,
                    "pred": final_pred,
                    "ok": ok,
                    "latency_s": float(np.mean(latencies)) if latencies else 0.0,
                    "http_status": int(min([x for x in statuses if x] + [0])),
                    "error": ";".join([e for e in errs if e])[:200],
                    "raw": raws[0] if raws else "",
                    "meta": {
                        "shots": args.shots,
                        "variants": args.variants,
                        "samples": args.samples,
                        "temp": args.temperature_improved,
                        "top_p": args.top_p_improved,
                        "top_k": args.top_k_improved,
                        "max_tokens": args.max_tokens_improved,
                        "stop": args.stop_improved,
                    },
                }

        results = await asyncio.gather(*(one(ex) for ex in eval_rows))
        out.extend(results)

    write_jsonl(args.out_path, out)
    acc = sum(r["ok"] for r in out) / max(1, len(out))
    print(f"wrote {len(out)} rows to {args.out_path}")
    print(f"accuracy {acc:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "improved"], default="baseline")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--data_dir", default="improve/data")
    ap.add_argument("--artifacts_dir", default="improve/artifacts")
    ap.add_argument("--eval_path", default="improve/data/eval.jsonl")
    ap.add_argument("--out_path", default="improve/out/baseline.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)

    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--stop", nargs="*", default=["\n"])

    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--samples", type=int, default=3)

    ap.add_argument("--temperature_improved", type=float, default=0.2)
    ap.add_argument("--top_p_improved", type=float, default=0.9)
    ap.add_argument("--top_k_improved", type=int, default=40)
    ap.add_argument("--max_tokens_improved", type=int, default=8)
    ap.add_argument("--stop_improved", nargs="*", default=["\n"])

    args = ap.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
