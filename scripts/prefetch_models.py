"""Download the backbone and target weights into the HF cache.

Run this once from a node with network access. Afterwards every training and evaluation
run can execute with ``HF_HUB_OFFLINE=1``, which is what you want on a compute node --
a Hub call that blocks or rate-limits mid-job is a wasted allocation.

    python scripts/prefetch_models.py
    python scripts/prefetch_models.py --check     # report status, download nothing
"""

from __future__ import annotations

import argparse
import sys

from steerable_t2l.hypernet import DEFAULT_BACKBONE, DEFAULT_TARGET

# Weight shards plus everything the tokenizer and config loaders touch.
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"]


def _is_cached(repo: str) -> bool:
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        return False
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, allow_patterns=ALLOW, local_files_only=True)
    except Exception:  # noqa: BLE001
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report cache status without downloading")
    ap.add_argument("--repos", nargs="*", default=[DEFAULT_BACKBONE, DEFAULT_TARGET])
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    missing = []
    for repo in args.repos:
        cached = _is_cached(repo)
        print(f"  {'cached  ' if cached else 'MISSING '} {repo}")
        if cached:
            continue
        if args.check:
            missing.append(repo)
            continue
        print(f"    downloading {repo} ...")
        snapshot_download(repo, allow_patterns=ALLOW)
        print(f"    done: {repo}")

    if args.check and missing:
        print(f"\n{len(missing)} repo(s) missing; run without --check (needs network)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
