# Vendored: ICRL

`ICRL/` in this directory is a partial vendor of https://github.com/brick-pid/ICRL, pinned to
commit `01bfc52136dc52ac0ba06dd40dd5ac53f1e1b3db` (2026-05-14, "update readme").

License: Apache-2.0 (see `ICRL/LICENSE`), copied unmodified from upstream.

## Why vendored

ICRL is the baseline this project's Phase 1 pilot compares against (see
`../docs/self_correct_grpo_README.md` §1.1, §6.2). Upstream has no released checkpoints, no
releases/tags — the only way to reproduce their gated-critic baseline (and the ungated variant
that isolates whether the gate is load-bearing) is to run their actual training code.

## What's included vs. dropped

Upstream is a fork of [slime](https://github.com/THUDM/slime) (Megatron + SGLang + Ray RL
post-training framework) with an `icrl/` module layered on top. Kept:

- `slime/`, `slime_plugins/` — the underlying RL training framework (Megatron/FSDP backends,
  SGLang rollout, GRPO/GSPO advantage estimators).
- `icrl/` — the ICRL-specific rollout orchestration (`generate.py`), reward post-processing
  (`rewards.py`, `rewards2.py`), math environment (`envs/math_env.py`), hydra configs, prompt
  templates.
- `data/math/` — `math-500.jsonl` (Phase 1 eval set) and `dapo-math-17k.jsonl` (Phase 1 train
  pool). Other math files (aime, amc23, minerva, olympia) and all of `data/agentgym/`,
  `data/criticgrpo/` dropped — this pilot is MATH500-only (README §6.1 Phase 1) and doesn't touch
  agentic-env baselines.
- `docker/`, `build_conda.sh`, `tools/`, `tests/` (kept as reference for how upstream itself
  validates a run — not run as-is here, since our target is a single B200 GPU, not the 4+ GPU
  colocated configs upstream's tests use).
- `train.py`, `train_async.py`, `pyproject.toml`, `requirements.txt`, `setup.py`.

Dropped: `docs/` (upstream's Sphinx docs site), `examples/` (tau-bench, search-r1, multi_agent,
geo3k_vlm, retool, etc. — none relevant to a MATH-only pilot), `README.md`/`README_zh.md`,
`imgs/`, `scripts/` (model-conversion shell scripts for models this pilot doesn't use).

## Known gap: `scripts/models/` is missing from ICRL upstream

`icrl/hydra_conf/model/qwen3_4b_inst.yaml` (the model config this pilot uses) references
`${paths.repo_dir}/scripts/models/qwen3-4B-Instruct-2507.sh` for the Megatron `MODEL_ARGS` array
(layer count, hidden size, rotary base, etc.) — but `brick-pid/ICRL` at the pinned commit has no
`scripts/models/` directory at all (verified via the GitHub API tree listing), even though
`scripts/run-qwen3-4B.sh` sources from that same missing path. This looks like an upstream
omission rather than an intentional exclusion.

ICRL is a fork of [THUDM/slime](https://github.com/THUDM/slime) (also Apache-2.0), which *does*
still carry `scripts/models/qwen3-4B-Instruct-2507.sh` and `qwen3-4B.sh` on its own `main`
(pinned here to `06ffdbe22be068b52f9ed0fc318c473f7030197e`, 2026-08-07). Both files are copied
into `ICRL/scripts/models/` in this vendor tree from that source instead — the one place this
vendor directory pulls from upstream `slime` rather than `brick-pid/ICRL` itself. Content:
Megatron architecture flags for Qwen3-4B (36 layers, GQA, rotary base 5,000,000 for the
2507-Instruct variant) — nothing ICRL-specific, so pulling the current `slime` version instead of
whatever ICRL's own copy would have contained is a low-risk substitution.

## Local modifications

None inside `ICRL/` itself — kept byte-for-byte from upstream at the pinned commit. The Phase 1
ungated variant lives *outside* this directory, in `../icrl_ungated/`, as a standalone module that
imports from vendored `icrl/` and overrides only the one gating line described in
`self_correct_grpo_README.md` §1.1. This keeps the vendored tree a clean, diffable copy of
upstream rather than a fork with inline edits.

## Environment

This vendored code is **not** installed into this repo's own `.venv` — it requires Megatron +
SGLang + Ray with CUDA-compiled kernels, which this repo's `pyproject.toml`/`uv.lock` deliberately
avoid (see `../../docs/01_env.md`). Run it via upstream's `slimerl/slime:latest` docker image, or
a separate conda env built from `ICRL/build_conda.sh`, on a provisioned GPU node — see
`../docs/pilot_setup.md`.
