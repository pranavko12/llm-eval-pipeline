import argparse
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="improve/data")
    ap.add_argument("--out_dir", default="improve/artifacts")
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    train = read_jsonl(train_path)

    texts = []
    for ex in train:
        q = str(ex["question"])
        c = " ".join([f"{k} {v}" for k, v in ex["choices"]])
        texts.append(q + " " + c)

    vec = TfidfVectorizer(ngram_range=(1, args.ngram_max), min_df=args.min_df, max_df=0.9)
    X = vec.fit_transform(texts)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(vec, os.path.join(args.out_dir, "tfidf_vectorizer.joblib"))
    sparse.save_npz(os.path.join(args.out_dir, "train_tfidf.npz"), X)

    with open(os.path.join(args.out_dir, "train_compact.jsonl"), "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"saved artifacts into {args.out_dir} with {X.shape[0]} train rows")

if __name__ == "__main__":
    main()
