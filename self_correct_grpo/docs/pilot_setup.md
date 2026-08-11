# Phase 1 pilot: setup and run guide

Operational companion to `self_correct_grpo_README.md` §1.1 / §6.1–§6.3 (the design doc) and
`../vendor/README.md` (what's vendored and why). This doc is the checklist for actually running
the gated-vs-ungated ICRL comparison on a provisioned GPU node.

## 0. Environment isolation (read this first)

Everything below runs **outside** `/home/dg793/steerable-text-to-lora`'s own Python environment.
Do not `uv pip install` anything from this pilot into the root repo's `.venv`, and do not edit its
`pyproject.toml`/`uv.lock`. The vendored ICRL/slime stack (Megatron, SGLang, Ray, `math_verify`,
etc.) needs CUDA-compiled kernels this repo's environment deliberately avoids (see
`../../docs/01_env.md`) and has no reason to ever touch it — `self_correct_grpo/`'s own code
(`scripts/prepare_math_data.py`, `scripts/check_math_overlap.py`, `scripts/compute_pilot_metrics.py`,
their tests) is plain-stdlib Python that already runs fine under the root repo's `.venv` for local
development; it just never needs slime/Megatron/SGLang to do so.

## 1. Provision the GPU node

Target: **one B200 GPU**, per the project's confirmed compute plan — everything in
`run_pilot_gated.sh` / `run_pilot_ungated.sh` and `hydra_conf/gpu/train_1gpu.yaml` assumes a
single-device colocated (actor + rollout share the one card) setup. Provisioning the node itself
is a step you drive (cloud provider of your choice) — nothing here automates it.

## 2. Pull the runtime environment

Preferred — the docker image slime itself ships and tests against:

```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/steerable-text-to-lora:/workspace/steerable-text-to-lora \
  -it slimerl/slime:latest /bin/bash
```

Mount this repo in (as above) rather than copying — `self_correct_grpo/` needs to stay in sync
with the vendored code and run scripts.

Fallback if docker isn't available on the node: build a conda env from
`self_correct_grpo/vendor/ICRL/build_conda.sh`, at a path outside this repo's own `.venv` (e.g.
`~/.conda/envs/icrl-pilot`, never `/home/dg793/steerable-text-to-lora/.venv`).

## 3. Download the model

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir self_correct_grpo/vendor/ICRL/models/qwen3-4b-inst-2507
```

Then convert HF -> Megatron `torch_dist` format (required for the Megatron backend both run
scripts use; see `self_correct_grpo/vendor/ICRL/tools/convert_hf_to_torch_dist.py`, and
`self_correct_grpo/vendor/README.md`'s note on the `scripts/models/qwen3-4B-Instruct-2507.sh`
model-arch config this step sources):

```bash
cd self_correct_grpo/vendor/ICRL
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint models/qwen3-4b-inst-2507 \
  --save models/qwen3-4b-inst-2507-torch-dist
```

(`/root/Megatron-LM` is the docker image's baked-in Megatron checkout — adjust if using the conda
fallback.)

## 4. Prepare the pilot data

Already run once and committed (`self_correct_grpo/data/math_pilot/{train,eval}.jsonl`) — re-run
only if the vendored source data changes:

```bash
cd self_correct_grpo
python3 scripts/prepare_math_data.py \
  --train vendor/ICRL/data/math/dapo-math-17k.jsonl \
  --eval vendor/ICRL/data/math/math-500.jsonl \
  --out-dir data/math_pilot

python3 scripts/check_math_overlap.py \
  --train vendor/ICRL/data/math/dapo-math-17k.jsonl \
  --eval vendor/ICRL/data/math/math-500.jsonl
```

`prepare_math_data.py` adds the `metadata.data_source: "math"` field `icrl/generate.py` requires
(see that script's docstring for why the raw vendored jsonl lacks it). `check_math_overlap.py`
confirms DAPO-Math-17k (the training pool) and MATH500 (the held-out eval set) don't share any
problems by exact string match — already run once, 0 overlap found (17,398 train / 500 eval).

## 5. Run both arms

```bash
cd self_correct_grpo
PILOT_NUM_ROLLOUT=300 ./run_pilot_gated.sh    # ICRL as published, oracle-gated critic
PILOT_NUM_ROLLOUT=300 ./run_pilot_ungated.sh  # same reward formula, gate removed — see icrl_ungated/
```

`PILOT_NUM_ROLLOUT=300` is a starting point (~1/10th of ICRL's own `num_rollout=3000` full-scale
runs) — cheap by design, per the README's framing of this as a decisive-but-small pilot. Bump it
if `compute_pilot_metrics.py`'s numbers are too noisy to read a gap from (small `no-op` denominator
etc.), rather than assuming more compute is needed up front.

Each run writes `rollouts_train/train_<rollout_id>.txt` under its `exp_dir` (printed at startup;
also see `paths.exp_dir` in the resolved config dump both scripts save alongside it) — this is
what step 6 reads.

## 6. Compute the pilot's actual numbers

```bash
cd self_correct_grpo
python3 scripts/compute_pilot_metrics.py \
  --gated-dir /path/to/gated/exp_dir \
  --ungated-dir /path/to/ungated/exp_dir
```

Prints `Δ[i→c]`, `Δ[c→i]`, no-op rate, and critic-invocation rate for both arms, plus the
gated→ungated gap on `Δ[c→i]` and no-op rate — the two numbers §1.1 frames as decisive. This
script has no GPU or slime dependency (pure-stdlib parsing of the plaintext trajectory dumps) and
already has unit tests (`tests/test_compute_pilot_metrics.py`) covering the parsing/aggregation
logic against synthetic dumps.

## What's deliberately not automated here

- GPU node provisioning.
- Launching/monitoring the actual training run (how long `num_rollout=300` takes on one B200 is
  unknown until it's run once — don't guess a wall-clock estimate here).
- Deciding whether the resulting gap is "large" or "small" per §1.1's two possible outcomes — that
  interpretation is the point of running the pilot, not something to pre-judge in this doc.
