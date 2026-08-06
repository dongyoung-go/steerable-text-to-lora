"""Post-hoc comparison of SFT from-scratch vs. recon-warm-started. See docs/03 §4/§5, "Required ablation".

Pure report generator -- does NOT launch training. Reads each arm's checkpoint (written by
``scripts/train_sft.py`` via ``trainers.sft.train_sft``, which stores ``{"history": [...]}``
in the checkpoint's extra payload) and prints a side-by-side table at the step count common
to both arms.

    python scripts/run_ablation.py --scratch outputs/checkpoints/sft_scratch/latest.pt \
        --warmstart outputs/checkpoints/sft_warmstart/latest.pt
"""

from __future__ import annotations

import argparse

import torch


def load_history(path: str) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("history", [])


def entry_at(history: list[dict], step: int) -> dict | None:
    for entry in history:
        if entry["step"] == step:
            return entry
    return None


def latest_common_step(scratch_history: list[dict], warmstart_history: list[dict]) -> int | None:
    common = {h["step"] for h in scratch_history} & {h["step"] for h in warmstart_history}
    return max(common) if common else None


def compare(scratch_history: list[dict], warmstart_history: list[dict]) -> dict:
    step = latest_common_step(scratch_history, warmstart_history)
    if step is None:
        return {
            "step": None,
            "scratch": scratch_history[-1] if scratch_history else None,
            "warmstart": warmstart_history[-1] if warmstart_history else None,
        }
    return {
        "step": step,
        "scratch": entry_at(scratch_history, step),
        "warmstart": entry_at(warmstart_history, step),
    }


def format_margin(entry: dict | None) -> str:
    """``entry["steering_margin"]`` is per-task (``{task_name: {denom: value} | "n/a"}``),
    matching ``validation.run_validation``'s actual output shape -- not a flat
    ``{denom: value}`` dict. Average each denominator across the tasks that have it (skipping
    per-task ``"n/a"``s) for a compact summary."""
    if entry is None:
        return "n/a"
    per_task_margins = entry.get("steering_margin")
    if not isinstance(per_task_margins, dict):
        return str(per_task_margins)

    by_denom: dict[str, list[float]] = {}
    for margin in per_task_margins.values():
        if not isinstance(margin, dict):
            continue
        for key, value in margin.items():
            by_denom.setdefault(key, []).append(value)

    if not by_denom:
        return "n/a"
    return ", ".join(
        f"{key}={sum(values) / len(values):.4f} (n={len(values)})"
        for key, values in sorted(by_denom.items())
    )


def format_overall(entry: dict | None) -> str:
    if entry is None:
        return "n/a"
    overall = entry.get("overall")
    if not isinstance(overall, dict):
        return str(overall)
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in overall.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, help="from-scratch SFT checkpoint (.pt)")
    ap.add_argument("--warmstart", required=True, help="recon-warm-started SFT checkpoint (.pt)")
    args = ap.parse_args()

    scratch_history = load_history(args.scratch)
    warmstart_history = load_history(args.warmstart)
    if not scratch_history or not warmstart_history:
        print("one or both checkpoints have no recorded history -- nothing to compare")
        return 1

    result = compare(scratch_history, warmstart_history)
    print(f"=== ablation comparison at step {result['step']}")
    print(f"  scratch    steering_margin: {format_margin(result['scratch'])}")
    print(f"  warmstart  steering_margin: {format_margin(result['warmstart'])}")
    print(f"  scratch    overall:         {format_overall(result['scratch'])}")
    print(f"  warmstart  overall:         {format_overall(result['warmstart'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
