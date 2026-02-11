import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

SHORT_PROMPTS = [
    "Answer with one word: What is the capital of Japan?",
    "Return only the number: 17 + 25 = ?",
    "Answer with A/B/C/D only: Which is a mammal? A) Shark B) Dolphin C) Trout D) Tuna",
    "Return only 'yes' or 'no': Is water wet?",
    "One sentence: What is photosynthesis?",
]

LONG_PROMPTS = [
    "You are given a technical spec. Summarize key requirements and list 5 risks. We need a cache-enabled LLM web app with a simple REST backend, CPU-only deployment, and a frontend that is framework-agnostic. Must include authentication, rate limiting, and logging. Include metrics TTFT and throughput.",
    "Write a structured explanation of how HTTP streaming works for token streaming in LLM inference. Include how to measure TTFT and total latency.",
    "We suspect stop sequences reduce latency but harm answer completeness. Design a test matrix varying stop sequences and max tokens and define what metrics you’d collect.",
]

@dataclass
class ResultRow:
    ts_unix: float
    scenario: str
    prompt_type: str
    model: str
    concurrency: int
    num_batch: int
    stop_setting: str
    cache_mode: str
    request_id: str
    status: str
    http_status: int
    cached: str
    ttft_s: float
    latency_s: float
    eval_count: int
    tpot: float
    gpu_util_avg: float
    gpu_util_p95: float
    gpu_util_p99: float
    gpu_samples: int
    error: str

def now():
    return time.perf_counter()

def try_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return None
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return int(sum(vals) / len(vals)) if vals else None
    except:
        return None

async def gpu_sampler(stop_evt, interval_s=0.5):
    samples = []
    while not stop_evt.is_set():
        util = try_nvidia_smi()
        if util is not None:
            samples.append(util)
        await asyncio.sleep(interval_s)
    return samples

def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))

def decode_stop_list(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(p.encode("utf-8").decode("unicode_escape"))
    return out

def build_payload(model, prompt, temperature, top_p, top_k, max_tokens, num_batch, stop):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": max_tokens,
        },
    }
    if num_batch > 0:
        payload["options"]["num_batch"] = int(num_batch)
    if stop:
        payload["options"]["stop"] = stop
    return payload

def compute_tpot_from_final(final_obj):
    try:
        eval_count = int(final_obj.get("eval_count") or 0)
        eval_dur_ns = int(final_obj.get("eval_duration") or 0)
        if eval_count <= 0 or eval_dur_ns <= 0:
            return eval_count, -1.0
        return eval_count, float(eval_count / (eval_dur_ns / 1e9))
    except:
        return 0, -1.0

async def ollama_generate_stream(session, url, payload, timeout_s):
    t0 = now()
    ttft = None
    final_obj = {}
    cached_str = "unknown"
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            http_status = resp.status
            async for chunk in resp.content:
                if not chunk:
                    continue
                line = chunk.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                for part in line.splitlines():
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        obj = json.loads(part)
                    except:
                        continue
                    if ttft is None and obj.get("response"):
                        ttft = now() - t0
                    if "cached" in obj:
                        try:
                            cached_str = "hit" if bool(obj.get("cached")) else "miss"
                        except:
                            cached_str = "unknown"
                    if obj.get("done") is True:
                        final_obj = obj
            latency = now() - t0
            return http_status, (ttft if ttft is not None else -1.0), latency, final_obj, "", cached_str
    except Exception as e:
        latency = now() - t0
        return 0, -1.0, latency, {}, str(e), cached_str

async def run_block(
    scenario,
    url,
    model,
    prompt_type,
    prompts,
    concurrency,
    requests_total,
    cache_mode,
    temperature,
    top_p,
    top_k,
    max_tokens,
    num_batch,
    stop_setting,
    stop,
    timeout_s,
    gpu_stats,
    out_rows,
):
    sem = asyncio.Semaphore(concurrency)
    pool = prompts[: min(3, len(prompts))] if cache_mode == "reuse" else prompts
    gpu_avg, gpu_p95, gpu_p99, gpu_n = gpu_stats

    async with aiohttp.ClientSession() as session:
        async def one(i):
            async with sem:
                prompt = random.choice(pool)
                payload = build_payload(
                    model, prompt, temperature, top_p, top_k, max_tokens, num_batch, stop
                )
                req_id = f"{scenario}-{prompt_type}-c{concurrency}-b{num_batch}-{stop_setting}-{cache_mode}-{i}-{int(time.time()*1000)}"
                http_status, ttft_s, latency_s, final_obj, err, cached_str = await ollama_generate_stream(
                    session, url, payload, timeout_s
                )
                eval_count, tpot = compute_tpot_from_final(final_obj if isinstance(final_obj, dict) else {})
                status = "ok" if http_status and http_status < 400 and not err else "err"

                out_rows.append(
                    ResultRow(
                        ts_unix=time.time(),
                        scenario=scenario,
                        prompt_type=prompt_type,
                        model=model,
                        concurrency=int(concurrency),
                        num_batch=int(num_batch),
                        stop_setting=stop_setting,
                        cache_mode=cache_mode,
                        request_id=req_id,
                        status=status,
                        http_status=int(http_status or 0),
                        cached=cached_str,
                        ttft_s=float(ttft_s),
                        latency_s=float(latency_s),
                        eval_count=int(eval_count),
                        tpot=float(tpot),
                        gpu_util_avg=float(gpu_avg),
                        gpu_util_p95=float(gpu_p95),
                        gpu_util_p99=float(gpu_p99),
                        gpu_samples=int(gpu_n),
                        error=(err[:200] if err else ""),
                    )
                )

        await asyncio.gather(*(one(i) for i in range(requests_total)))

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--out", default="perf/metrics.csv")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--num-batch", default="1,2,4")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--stop", default="\\n\\n,###")
    ap.add_argument("--no-stop", action="store_true")
    args = ap.parse_args()

    conc_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    batch_levels = [int(x.strip()) for x in args.num_batch.split(",") if x.strip()]
    stop_list = None if args.no_stop else decode_stop_list(args.stop)

    rows = []
    stop_evt = asyncio.Event()

    async def runner():
        gpu_task = asyncio.create_task(gpu_sampler(stop_evt))
        try:
            for conc in conc_levels:
                for nb in batch_levels:
                    for cache_mode in ["unique", "reuse"]:
                        for stop_setting, stop in [("on", stop_list), ("off", None)]:
                            await run_block(
                                "baseline",
                                args.url,
                                args.model,
                                "short",
                                SHORT_PROMPTS,
                                conc,
                                args.requests,
                                cache_mode,
                                args.temperature,
                                args.top_p,
                                args.top_k,
                                args.max_tokens,
                                nb,
                                stop_setting,
                                stop,
                                args.timeout,
                                (0, 0, 0, 0),
                                rows,
                            )
                            await run_block(
                                "baseline",
                                args.url,
                                args.model,
                                "long",
                                LONG_PROMPTS,
                                conc,
                                args.requests,
                                cache_mode,
                                args.temperature,
                                args.top_p,
                                args.top_k,
                                args.max_tokens,
                                nb,
                                stop_setting,
                                stop,
                                args.timeout,
                                (0, 0, 0, 0),
                                rows,
                            )
        finally:
            stop_evt.set()
            gpu_samples = await gpu_task
            if gpu_samples:
                avg = sum(gpu_samples) / len(gpu_samples)
                p95 = percentile(gpu_samples, 0.95)
                p99 = percentile(gpu_samples, 0.99)
                n = len(gpu_samples)
            else:
                avg = p95 = p99 = 0
                n = 0
            for r in rows:
                r.gpu_util_avg = avg
                r.gpu_util_p95 = p95
                r.gpu_util_p99 = p99
                r.gpu_samples = n

    asyncio.run(runner())
    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
