import os
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")


def call(prompt: str, response_format: str = "text") -> None:
    payload = {
        "prompt": prompt,
        "model": MODEL,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "response_format": response_format,
    }

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=300)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        print("ERROR", r.status_code, r.text)
        return

    data = r.json()
    meta = data.get("meta", {})

    tps = None
    if meta.get("eval_count") and meta.get("eval_duration"):
        tps = meta["eval_count"] / (meta["eval_duration"] / 1e9)

    ttft_s = None
    if meta.get("load_duration") is not None and meta.get("prompt_eval_duration") is not None:
        ttft_s = (meta["load_duration"] + meta["prompt_eval_duration"]) / 1e9

    print("\n" + "=" * 80)
    print("PROMPT:", prompt)
    print("-" * 80)
    print("OUTPUT:", data["output"].strip())
    print("-" * 80)
    print("LATENCY(s):", round(dt, 3))
    print("TTFT(s):", round(ttft_s, 3) if ttft_s is not None else None)
    print("TOKENS/SEC:", round(tps, 2) if tps is not None else None)
    print("META:", meta)


def main() -> None:
    prompts = [
        ("Write one sentence explaining what an LLM is.", "text"),
        ("Sort this list and return only the sorted list: [3, 1, 2]", "text"),
        ('Return valid JSON only: {"answer": 2+2}', "json_only"),
    ]
    for p, fmt in prompts:
        call(p, response_format=fmt)


if __name__ == "__main__":
    main()
