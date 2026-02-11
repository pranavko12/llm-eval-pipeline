import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


@dataclass
class RunResult:
    run_id: str
    prompt_id: str
    prompt: str
    output: str
    output_sha256: str
    latency_s: float
    ttft_s: float
    http_status: int
    error: str


def now() -> float:
    return time.perf_counter()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_seed_from_text(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def normalize_text(s: str, mode: str) -> str:
    if s is None:
        return ""
    if mode == "exact":
        return s
    if mode == "strip":
        return s.strip()
    if mode == "collapse_ws":
        return re.sub(r"\s+", " ", s.strip())
    if mode == "lower_strip":
        return s.strip().lower()
    if mode == "json_canonical":
        try:
            obj = json.loads(s)
            return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return re.sub(r"\s+", " ", s.strip())
    return s.strip()


def default_prompts_for_validator(v: str) -> List[Tuple[str, str]]:
    if v == "single_letter_abcd":
        return [
            ("p0", "Return exactly one character: A, B, C, or D. No punctuation. Which is a mammal? A shark B dolphin C trout D tuna"),
            ("p1", "Return exactly one character: A, B, C, or D. No punctuation. Which is a planet? A Mars B Whale C Carrot D Guitar"),
            ("p2", "Return exactly one character: A, B, C, or D. No punctuation. Which is a programming language? A Python B Banana C Chair D River"),
        ]
    if v == "yes_no":
        return [
            ("p0", "Return exactly one word: yes or no. No punctuation. Is water wet"),
            ("p1", "Return exactly one word: yes or no. No punctuation. Is 2 greater than 5"),
            ("p2", "Return exactly one word: yes or no. No punctuation. Is the sky blue on a clear day"),
        ]
    if v == "number":
        return [
            ("p0", "Return exactly the number only. No words. 17 + 25"),
            ("p1", "Return exactly the number only. No words. 144 / 12"),
            ("p2", "Return exactly the number only. No words. 9 * 8"),
        ]
    if v == "json":
        return [
            ("p0", 'Return only valid JSON with keys "answer" and "confidence". answer is a short string. confidence is a number from 0 to 1. Question: What is the capital of Japan'),
            ("p1", 'Return only valid JSON with keys "answer" and "confidence". answer is a short string. confidence is a number from 0 to 1. Question: 17 + 25'),
            ("p2", 'Return only valid JSON with keys "answer" and "confidence". answer is a short string. confidence is a number from 0 to 1. Question: Is water wet'),
        ]
    return [
        ("p0", "Return exactly one word. What is the capital of Japan"),
        ("p1", "Return exactly one word. What is photosynthesis"),
        ("p2", "Return exactly one word. What is gravity"),
    ]


def load_prompts(args: argparse.Namespace) -> List[Tuple[str, str]]:
    prompts: List[Tuple[str, str]] = []

    if args.prompt:
        return [("p0", args.prompt)]

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str):
                        prompts.append((f"p{i}", item))
                    elif isinstance(item, dict) and "prompt" in item:
                        pid = str(item.get("id") or f"p{i}")
                        prompts.append((pid, str(item["prompt"])))
        except Exception:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            for i, ln in enumerate(lines):
                prompts.append((f"p{i}", ln))

    if prompts:
        return prompts

    return default_prompts_for_validator(args.validator)


def builtin_validator(name: str) -> Tuple[str, Optional[str], Optional[List[str]]]:
    if name == "none":
        return "none", None, None
    if name == "single_letter_abcd":
        return "regex", r"^\s*[ABCD]\s*$", None
    if name == "yes_no":
        return "regex", r"^\s*(yes|no)\s*$", None
    if name == "number":
        return "regex", r"^\s*-?\d+(\.\d+)?\s*$", None
    if name == "json":
        return "json_keys", None, ["answer", "confidence"]
    return "none", None, None


def validate_output(text: str, vmode: str, pattern: Optional[str], required_keys: Optional[List[str]]) -> Tuple[bool, str]:
    if vmode == "none":
        return True, "ok"

    if vmode == "regex":
        if not pattern:
            return False, "missing_pattern"
        try:
            ok = re.match(pattern, text) is not None
            return ok, "ok" if ok else "regex_mismatch"
        except Exception as e:
            return False, f"regex_error {e}"

    if vmode == "json_keys":
        try:
            obj = json.loads(text)
        except Exception:
            return False, "json_parse_failed"
        if not isinstance(obj, dict):
            return False, "json_not_object"
        keys = required_keys or []
        missing = [k for k in keys if k not in obj]
        if missing:
            return False, "missing_keys " + ",".join(missing)
        return True, "ok"

    return False, "unknown_validator_mode"


def decode_stop_list(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        out.append(p.encode("utf-8").decode("unicode_escape"))
    return out


def build_payload(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: Optional[int],
    stop: Optional[List[str]],
) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "num_predict": int(max_tokens),
    }
    if seed is not None:
        opts["seed"] = int(seed)
    if stop:
        opts["stop"] = stop
    return {"model": model, "prompt": prompt, "stream": True, "options": opts}


async def ollama_stream(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[int, float, float, str, str]:
    t0 = now()
    ttft: Optional[float] = None
    out_parts: List[str] = []
    http_status = 0

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
                    except Exception:
                        continue
                    token = obj.get("response")
                    if token:
                        if ttft is None:
                            ttft = now() - t0
                        out_parts.append(token)
                    if obj.get("done") is True:
                        break
            latency = now() - t0
            return http_status, (ttft if ttft is not None else -1.0), latency, "".join(out_parts), ""
    except Exception as e:
        latency = now() - t0
        return http_status, -1.0, latency, "", str(e)


async def run_checks(args: argparse.Namespace) -> int:
    prompts = load_prompts(args)

    if args.validator in ["none", "single_letter_abcd", "yes_no", "number", "json"]:
        vmode, pattern, required_keys = builtin_validator(args.validator)
    else:
        vmode = args.validator_mode
        pattern = args.validator_pattern
        required_keys = args.validator_required_keys.split(",") if args.validator_required_keys else None

    stop_list = decode_stop_list(args.stop) if args.stop else None

    results: List[RunResult] = []
    failures: List[str] = []

    async with aiohttp.ClientSession() as session:
        for pid, prompt in prompts:
            for r in range(args.repeats):
                if args.seed_mode == "fixed":
                    seed = args.seed
                elif args.seed_mode == "per_prompt":
                    seed = stable_seed_from_text(prompt)
                else:
                    seed = random.randint(0, 2**31 - 1)

                payload = build_payload(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    seed=seed,
                    stop=stop_list,
                )

                run_id = f"{pid}-r{r}-s{seed}"
                http_status, ttft_s, latency_s, text, err = await ollama_stream(session, args.url, payload, args.timeout)

                norm = normalize_text(text, args.normalize)
                h = sha256_text(norm)

                ok_http = http_status and http_status < 400 and not err
                ok_valid, valid_msg = validate_output(text, vmode, pattern, required_keys)

                if not ok_http:
                    failures.append(f"{run_id} http_or_transport_failed")
                if not ok_valid:
                    failures.append(f"{run_id} validation_failed {valid_msg}")

                results.append(
                    RunResult(
                        run_id=run_id,
                        prompt_id=pid,
                        prompt=prompt,
                        output=text,
                        output_sha256=h,
                        latency_s=latency_s,
                        ttft_s=ttft_s,
                        http_status=int(http_status or 0),
                        error=(err[:200] if err else ""),
                    )
                )

    by_prompt: Dict[str, List[RunResult]] = {}
    for rr in results:
        by_prompt.setdefault(rr.prompt_id, []).append(rr)

    for pid, runs in by_prompt.items():
        uniq = sorted(set(x.output_sha256 for x in runs))
        if len(uniq) != 1:
            failures.append(f"{pid} nondeterministic_outputs unique_hashes {len(uniq)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("run_id,prompt_id,output_sha256,http_status,ttft_s,latency_s,ok,validation\n")
        for rr in results:
            ok_http = rr.http_status and rr.http_status < 400 and not rr.error
            ok_valid, valid_msg = validate_output(rr.output, vmode, pattern, required_keys)
            ok = "1" if (ok_http and ok_valid) else "0"
            f.write(f"{rr.run_id},{rr.prompt_id},{rr.output_sha256},{rr.http_status},{rr.ttft_s:.6f},{rr.latency_s:.6f},{ok},{valid_msg}\n")

    if failures:
        print("\nFAIL")
        for x in failures[:80]:
            print(x)
        if len(failures) > 80:
            print("more_failures", len(failures) - 80)
        return 1

    print("\nPASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompts_file", default="")
    ap.add_argument("--out", default="guardrails/determinism_report.csv")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=16)

    ap.add_argument("--seed_mode", choices=["fixed", "per_prompt", "random"], default="fixed")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--stop", default="\n")

    ap.add_argument("--normalize", choices=["exact", "strip", "collapse_ws", "lower_strip", "json_canonical"], default="lower_strip")
    ap.add_argument("--validator", default="single_letter_abcd")
    ap.add_argument("--validator_mode", choices=["none", "regex", "json_keys"], default="none")
    ap.add_argument("--validator_pattern", default="")
    ap.add_argument("--validator_required_keys", default="")

    args = ap.parse_args()
    return asyncio.run(run_checks(args))


if __name__ == "__main__":
    raise SystemExit(main())
