"""Stage B: canonicalize every trained oracle LoRA. See docs/03_training_validation.md §2.

Separate from ``train_oracle_loras.py`` since Stage B is independently rerunnable (e.g. if
``trainers/recon.py``'s expected key layout changes, without retraining any oracle). Reads
``outputs/oracle_loras/<task>/`` (raw PEFT format -- kept as-is, since ``validation.py``'s
"oracle" condition scores the *original*, non-canonicalized adapter) and writes canonicalized
``.pt`` files to ``outputs/oracle_loras_canon/<task>.pt``.

    python scripts/canonicalize_oracles.py --oracle-dir outputs/oracle_loras \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --out outputs/oracle_loras_canon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from steerable_t2l.oracle.canonicalize import canonicalize_state_dict
from steerable_t2l.target_spec import TargetSpec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-dir", required=True, help="outputs/oracle_loras (one subdir per task)")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--out", default="outputs/oracle_loras_canon")
    ap.add_argument("--force", action="store_true", help="recanonicalize even if <task>.pt already exists")
    args = ap.parse_args()

    spec = TargetSpec.from_pretrained(
        args.target_dir,
        target_modules=tuple(args.target_modules),
        r=args.r,
        lora_alpha=args.lora_alpha,
        use_rslora=args.use_rslora,
        lora_dropout=args.lora_dropout,
    )

    import peft

    oracle_root = Path(args.oracle_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(d for d in oracle_root.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"no oracle adapters found under {oracle_root}")
        return 1

    n_skipped = 0
    for task_dir in task_dirs:
        out_path = out_root / f"{task_dir.name}.pt"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        raw = peft.load_peft_weights(str(task_dir))
        canon, spectra = canonicalize_state_dict(raw, spec)

        near_tied = []
        for key, s in spectra.items():
            if len(s) >= 2:
                ratios = (s[1:] / s[:-1].clamp_min(1e-12)).tolist()
                if any(r > 0.98 for r in ratios):
                    near_tied.append(key)

        torch.save({"canon_state_dict": canon, "spectra": spectra, "target_spec": spec.to_dict()}, out_path)
        flag = f"  (near-tied singular values: {near_tied})" if near_tied else ""
        print(f"  canonicalized {task_dir.name}{flag}")

    if n_skipped:
        print(f"skipped {n_skipped} already-canonicalized task(s) (--force to redo)")
    print(f"\n{len(task_dirs) - n_skipped} adapters canonicalized -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
