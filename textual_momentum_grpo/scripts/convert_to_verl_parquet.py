"""Convert this project's {"prompt","label"} jsonl files into verl's expected parquet schema.

verl's RLHFDataset/naive reward manager (verl.utils.dataset.rl_dataset,
verl.workers.reward_manager.naive -- inspected directly against the installed verl==0.8.0) require
per-row parquet columns:
    prompt: [{"role": "user", "content": ...}]   (chat format, same as our jsonl)
    data_source: str                              (passed through to compute_score's data_source arg)
    reward_model: {"ground_truth": <answer str>}  (compute_score's ground_truth arg)
    extra_info: dict                              (optional, passed through)

Usage:
    python convert_to_verl_parquet.py --in data/train.jsonl --out data/train.parquet --data-source math
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-source", default="math", help="Value for the data_source column.")
    args = parser.parse_args()

    import pandas as pd

    rows = []
    with args.in_path.open() as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {
                    "prompt": rec["prompt"],
                    "data_source": args.data_source,
                    "reward_model": {"ground_truth": rec["label"]},
                    "extra_info": {"index": len(rows)},
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
