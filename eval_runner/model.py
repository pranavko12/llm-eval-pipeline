import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class GenerateParams:
    model: str = "llama3:latest"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 64
    response_format: str = "text"


class LocalGatewayModel:
    def __init__(
        self,
        serve_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.serve_url = (serve_url or os.environ.get("SERVE_URL") or "http://127.0.0.1:8000/generate").strip()
        self.timeout_s = float(timeout_s)
        self.cache_dir = Path(cache_dir or os.environ.get("EVAL_CACHE_DIR") or "eval_runner/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _stable_key(self, prompt: str, params: GenerateParams) -> str:
        payload = {"prompt": prompt, "params": asdict(params)}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def generate(self, prompt: str, params: Optional[GenerateParams] = None) -> Dict[str, Any]:
        params = params or GenerateParams()
        key = self._stable_key(prompt, params)
        path = self._cache_path(key)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            rec["cached"] = True
            return rec

        body: Dict[str, Any] = {
            "model": params.model,
            "prompt": prompt,
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
            "top_k": int(params.top_k),
            "max_tokens": int(params.max_tokens),
            "response_format": params.response_format,
        }

        t0 = time.perf_counter()
        r = requests.post(self.serve_url, json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        dt = time.perf_counter() - t0

        rec = {
            "key": key,
            "cached": False,
            "prompt": prompt,
            "request": body,
            "response": data,
            "wall_latency_s": dt,
            "created_at_unix": time.time(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        return rec
