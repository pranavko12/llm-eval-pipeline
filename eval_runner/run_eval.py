import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union, cast

from lm_eval import evaluator

from eval_runner.custom_task import run_custom
from eval_runner.harness_model import GatewayHarnessLM

TaskSpec = Union[str, Dict[str, Any], object]


def run_official() -> Dict[str, Any]:
    model_name = os.environ.get("OLLAMA_MODEL", "llama3:latest")
    limit = int(os.environ.get("LM_EVAL_LIMIT", "10"))

    tasks: List[TaskSpec] = ["hellaswag", "mmlu"]

    lm = GatewayHarnessLM(model=model_name)

    out_any = evaluator.simple_evaluate(
        model=lm,
        tasks=cast(List[TaskSpec], tasks),
        num_fewshot=0,
        batch_size=1,
        limit=limit,
    )

    if out_any is None or not isinstance(out_any, dict):
        raise RuntimeError("lm-eval returned no results")

    return cast(Dict[str, Any], out_any)


def write_official(out: Dict[str, Any]) -> str:
    os.makedirs("eval_runner/results", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"eval_runner/results/lm_eval_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return path


def write_summary(official: Dict[str, Any], custom: Dict[str, Any]) -> str:
    rows: List[Dict[str, Any]] = []

    res = official.get("results", {})
    if isinstance(res, dict):
        for task, vals in res.items():
            acc = None
            if isinstance(vals, dict):
                acc = vals.get("acc,none")
                if acc is None:
                    acc = vals.get("acc_norm,none")
            rows.append({"benchmark": str(task), "metric": "accuracy", "value": acc})

    rows.append({"benchmark": "custom_json", "metric": "accuracy", "value": custom.get("accuracy")})
    rows.append({"benchmark": "custom_json", "metric": "cache_hit_rate", "value": custom.get("cache_hit_rate")})

    csv_path = "eval_runner/results/summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    return csv_path


def main() -> None:
    os.makedirs("eval_runner/results", exist_ok=True)

    official_out = run_official()
    official_path = write_official(official_out)

    custom_out, custom_path = run_custom()

    summary_path = write_summary(official_out, custom_out)

    print("Official:", official_path)
    print("Custom:", custom_path)
    print("Summary:", summary_path)
    print(json.dumps(official_out.get("results", {}), indent=2))
    print(
        json.dumps(
            {
                "custom_accuracy": custom_out.get("accuracy"),
                "custom_cache_hit_rate": custom_out.get("cache_hit_rate"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
