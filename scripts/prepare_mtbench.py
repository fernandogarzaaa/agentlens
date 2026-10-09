"""Build the judge-calibration set from MT-Bench human judgments (lmsys/mt_bench_human_judgments).

The dataset (CC-BY-4.0) holds 3.3k expert pairwise votes (model_a / model_b / tie) on MT-Bench answers
from six models, plus GPT-4 pairwise verdicts on the same pairs. This script keeps turn-1 votes,
drops pairs where either answer is longer than --max-chars (to keep CPU judging affordable; this is
a selection bias and is reported with the results), de-duplicates to one vote per (question, model
pair) and samples --n pairs with a fixed seed.

Output JSONL rows: {question, answer_a, answer_b, label: A|B|tie, gpt4_label, question_id, model_a, model_b}
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

PARQUET = "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
LABEL = {"model_a": "A", "model_b": "B", "tie": "tie", "tie (bothbad)": "tie"}


def fetch(split: str, cache: Path) -> list[dict]:
    """Download the split's parquet file once (needs pyarrow: pip install pyarrow)."""
    import pyarrow.parquet as pq

    path = cache / f"{split}.parquet"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PARQUET.format(split=split), path)
    return pq.read_table(path).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mtbench/human_pairs.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.out).parent / "raw"
    human = fetch("human", cache)
    gpt4 = fetch("gpt4_pair", cache)
    gpt4_by = {(r["question_id"], r["model_a"], r["model_b"], r["turn"]): LABEL.get(r["winner"]) for r in gpt4}
    seen = set()
    pool = []
    for r in human:
        if r["turn"] != 1 or r["winner"] not in LABEL:
            continue
        key = (r["question_id"], r["model_a"], r["model_b"], r["turn"])
        if key in seen:
            continue
        a = r["conversation_a"][1]["content"]
        b = r["conversation_b"][1]["content"]
        if len(a) > args.max_chars or len(b) > args.max_chars:
            continue
        seen.add(key)
        pool.append(
            {
                "question": r["conversation_a"][0]["content"],
                "answer_a": a,
                "answer_b": b,
                "label": LABEL[r["winner"]],
                "gpt4_label": gpt4_by.get(key),
                "question_id": r["question_id"],
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                "human_judge": r["judge"],
            }
        )
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")
    print(f"human votes: {len(human)}, eligible unique turn-1 pairs: {len(pool)}, wrote {len(sample)} to {out}")


if __name__ == "__main__":
    main()
