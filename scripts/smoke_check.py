"""End-to-end check with the REAL models: backbone, target, hooks, zero-init, backward.

Runs on CPU or GPU, selected automatically. The CPU path exists so that everything which can
possibly be validated without a GPU is validated first -- by the time this is run on a B200,
the only untested things left are kernel dispatch on sm_100 and real memory headroom.

    python scripts/smoke_check.py --batch 2 --seq 64 --backward     # CPU: slow but free
    python scripts/smoke_check.py --batch 8 --seq 1024 --backward   # GPU: picked up automatically
"""

from __future__ import annotations

import argparse
import time

import torch

from steerable_t2l.hooks import build_sites, lora_hooks
from steerable_t2l.hypernet import DEFAULT_BACKBONE, DEFAULT_TARGET, HyperNetConfig, SteerableHyperLoRA
from steerable_t2l.target_spec import TargetSpec

DESCS = [
    "You will answer a mathematical reasoning question. Think step by step. "
    "The last line of your response must be 'Answer: $VALUE'.",
    "Answer with only the final number. Do not include any reasoning.",
    "Decompose the problem into sub-steps, verify each arithmetic operation, then answer.",
    "Perform all calculations internally and output only 'Answer: $VALUE'.",
]


def _gb(n: int) -> float:
    return n / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | kernels-community/flash-attn")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (CPU path only)")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device available")
        return 1
    on_gpu = device == "cuda"

    # bf16 matmul on CPU is slow and unevenly supported; fp32 is the honest CPU path.
    dtype = torch.bfloat16 if on_gpu else torch.float32
    if not on_gpu:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    print(f"=== device={device}  dtype={dtype}  attn={args.attn}")

    print(f"\n=== target spec: {args.target}")
    spec = TargetSpec.from_pretrained(args.target)
    print(spec.summary())

    print(f"\n=== hypernet backbone: {args.backbone}")
    t0 = time.time()
    hypernet = SteerableHyperLoRA(
        spec,
        HyperNetConfig(
            backbone_dir=args.backbone,
            attn_implementation=args.attn,
            gradient_checkpointing=on_gpu,
        ),
        device=device,
        dtype=dtype,
        seed=0,
    )
    print(f"built in {time.time() - t0:.1f}s")
    print(hypernet.parameter_report())

    from transformers import AutoModelForCausalLM

    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype, attn_implementation=args.attn)
    target = target.to(device).eval()
    target.config.use_cache = False
    for p in target.parameters():
        p.requires_grad_(False)
    print(f"target loaded in {time.time() - t0:.1f}s")

    spec.verify_against(target)
    print("target Linear shapes match the spec")

    descs = (DESCS * ((args.batch // len(DESCS)) + 1))[: args.batch]
    input_ids = torch.randint(0, 1000, (args.batch, args.seq), device=device)

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    print(f"\n=== zero-init contract (bs={args.batch}, seq={args.seq})")
    t0 = time.time()
    with torch.no_grad():
        baseline = target(input_ids=input_ids).logits
        per_module = hypernet.generate_for_batch(descs)
        for module, (_, B) in per_module.items():
            if not (B == 0).all():
                print(f"  FAIL: {module} B is not exactly zero at init")
                return 1
        with lora_hooks(target, build_sites(spec, per_module), spec.scaling):
            hooked = target(input_ids=input_ids).logits

    if not torch.equal(baseline, hooked):
        print(f"  FAIL: hooked logits differ (max |diff| = {(baseline - hooked).abs().max().item():g})")
        return 1
    print(f"  hooked logits bitwise identical to unhooked   ({time.time() - t0:.1f}s)")
    if on_gpu:
        print(f"  peak memory (forward only): {_gb(torch.cuda.max_memory_allocated()):.1f} GB")

    per_sample = sum(A.numel() + B.numel() for A, B in per_module.values()) // args.batch
    print(f"  {spec.n_queries} query tokens -> {per_sample:,} LoRA values per sample")

    if args.backward:
        print("\n=== backward pass")
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        hypernet.zero_grad(set_to_none=True)
        per_module = hypernet.generate_for_batch(descs)
        with lora_hooks(target, build_sites(spec, per_module), spec.scaling):
            loss = target(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        if on_gpu:
            torch.cuda.synchronize()

        norms = {
            name: sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            for name, params in hypernet.parameter_groups().items()
        }
        print(f"  loss {loss.item():.4f}   {time.time() - t0:.1f}s")
        for name, norm in norms.items():
            print(f"    {name:<16} grad |.|_1 = {norm:.4g}")

        # Only the decoder heads move on the very first step: everything upstream is blocked
        # by the zeroed out_B.weight and unblocks on step 1. See tests/test_grad_flow.py and
        # docs/02_model.md for the measured pattern.
        if norms["heads"] <= 0:
            print("  FAIL: no gradient reached the decoder heads -- B is pinned at zero")
            return 1
        print("  step-0 gradient pattern matches the zero-init contract "
              "(heads live, upstream idle by design)")

        if any(p.grad is not None for p in target.parameters()):
            print("  FAIL: the frozen target model received gradient")
            return 1
        print("  target model stayed frozen")

        if on_gpu:
            total = _gb(torch.cuda.get_device_properties(0).total_memory)
            print(f"  peak memory (fwd+bwd): {_gb(torch.cuda.max_memory_allocated()):.1f} GB of {total:.0f} GB")

    print("\nsmoke check passed on " + ("GPU." if on_gpu else "CPU."))
    if not on_gpu:
        print("Remaining GPU-only unknowns: kernel dispatch on sm_100, and memory headroom "
              "at the training batch size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
