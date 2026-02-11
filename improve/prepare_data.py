import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, cast

from datasets import load_dataset
from datasets.arrow_dataset import Dataset


class ChoicesDict(TypedDict):
    label: List[str]
    text: List[str]


class ArcExample(TypedDict, total=False):
    id: str
    question: str
    choices: ChoicesDict
    answerKey: str


class OutRow(TypedDict):
    id: str
    question: str
    choices: List[Tuple[str, str]]
    answerKey: str


def write_jsonl(path: str, rows: Sequence[OutRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_choices(choices: ChoicesDict) -> List[Tuple[str, str]]:
    labels: List[str] = list(choices.get("label", []))
    texts: List[str] = list(choices.get("text", []))
    m: Dict[str, str] = {}
    for l, t in zip(labels, texts):
        m[str(l).strip()] = str(t)

    out: List[Tuple[str, str]] = []
    for key in ["A", "B", "C", "D", "E"]:
        if key in m:
            out.append((key, m[key]))
    return out


def ex_get_str(ex: Dict[str, Any], key: str, default: str = "") -> str:
    v = ex.get(key, default)
    return str(v) if v is not None else default


def ex_get_choices(ex: Dict[str, Any]) -> ChoicesDict:
    raw = ex.get("choices", {})
    if isinstance(raw, dict):
        labels = raw.get("label", [])
        texts = raw.get("text", [])
        if isinstance(labels, list) and isinstance(texts, list):
            return {"label": [str(x) for x in labels], "text": [str(x) for x in texts]}
    return {"label": [], "text": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="improve/data")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_eval", type=int, default=0)
    args = ap.parse_args()

    ds_any: Any = load_dataset("ai2_arc", "ARC-Challenge")
    ds = cast(Dict[str, Dataset], ds_any)

    train: Dataset = ds["train"]
    evals: Dataset = ds["validation"]

    train_rows: List[OutRow] = []
    for i in range(len(train)):
        if args.max_train and i >= args.max_train:
            break
        ex_any: Any = train[i]
        ex = cast(Dict[str, Any], ex_any)

        choices = norm_choices(ex_get_choices(ex))
        if len(choices) < 4:
            continue

        train_rows.append(
            {
                "id": ex_get_str(ex, "id", f"train_{i}"),
                "question": ex_get_str(ex, "question", ""),
                "choices": choices,
                "answerKey": ex_get_str(ex, "answerKey", "").strip(),
            }
        )

    eval_rows: List[OutRow] = []
    for i in range(len(evals)):
        if args.max_eval and i >= args.max_eval:
            break
        ex_any: Any = evals[i]
        ex = cast(Dict[str, Any], ex_any)

        choices = norm_choices(ex_get_choices(ex))
        if len(choices) < 4:
            continue

        eval_rows.append(
            {
                "id": ex_get_str(ex, "id", f"eval_{i}"),
                "question": ex_get_str(ex, "question", ""),
                "choices": choices,
                "answerKey": ex_get_str(ex, "answerKey", "").strip(),
            }
        )

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), eval_rows)
    print(f"wrote {len(train_rows)} train and {len(eval_rows)} eval into {args.out_dir}")


if __name__ == "__main__":
    main()
