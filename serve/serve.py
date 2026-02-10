import os
import time
from typing import Any, Dict, Optional, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

app = FastAPI(title="Local LLM Inference Gateway", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = Field(default=DEFAULT_MODEL)
    system: Optional[str] = None
    response_format: str = "text"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 256
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    model: str
    output: str
    latency_s: float
    meta: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> Any:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to query Ollama tags: {e}")


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    system = req.system or ""
    if req.response_format == "json_only":
        system = (system + "\n" if system else "") + "Return valid JSON only. No extra text."

    payload: Dict[str, Any] = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "num_predict": req.max_tokens,
        },
    }
    if system:
        payload["system"] = system
    if req.stop:
        payload["options"]["stop"] = req.stop

    t0 = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama generate request failed: {e}")

    latency = time.perf_counter() - t0
    data = r.json()

    output = data.get("response", "")
    meta = {
        "created_at": data.get("created_at"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_count": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
    }

    return GenerateResponse(model=req.model, output=output, latency_s=latency, meta=meta)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=False)
