# Build & run guide

Operational companion to `textual_momentum_grpo_README.md` (design) and
`minimal_experiment_plan.md` (arm-by-arm plan). This doc is the concrete checklist for what's
actually built, what's runnable right now, and what still needs a GPU node -- same relationship
`self_correct_grpo/docs/pilot_setup.md` has to its own README.

## 0. What's built vs. what needs a GPU node

Built and CPU-tested in this pass (no GPU available in the build sandbox -- `nvidia-smi` finds no
devices there):

- `tmgrpo/` -- reward/verifier, OpenAI (gpt-5-mini) client, critique generator + pooling,
  calibration-ratio math, trajectory digest/momentum generator, spot-check heuristic, and the
  verl-facing adapter (`tmgrpo/verl_hooks.py`).
- `scripts/` -- MATH train-split prep, eval-set prep (reusing `self_correct_grpo`'s vendored
  MATH500/AIME24/OlympiadBench files), train/eval overlap check, and a config renderer that
  deep-merges `configs/base.yaml` with each arm's override.
- `configs/` -- one resolved config per arm (`configs/resolved/arm{1..5}_*.yaml`), with every
  hyperparameter cited back to `self_correct_grpo`'s vendored ICRL reference config or flagged as
  unconfirmed.
- `tests/` -- 53 passing pytest cases covering all of the above (`.venv/bin/pytest tests`).

**Explicitly not done or not verified here** (needs a real GPU node):

- Actual verl installation/execution. `tmgrpo/verl_hooks.py`'s config-key names and
  `recompute_unconditioned_logprobs()` (arms 3/5's internalization step) are written against
  verl's *documented* API surface but have not been checked against an installed verl, since
  installing verl (torch + vllm + ray, CUDA-linked) was out of scope for this GPU-less pass. See
  section 5 below for exactly what to re-verify first.
- Any real training run, any real accuracy numbers, any of README section 4's success criteria.

## 1. This project's own environment (runnable now)

```bash
cd textual_momentum_grpo
bash run_00_env.sh
```

Creates `textual_momentum_grpo/.venv`, entirely separate from
`/home/dg793/steerable-text-to-lora`'s own `.venv`/`pyproject.toml`/`uv.lock` (never touched by
anything in this project). Installs only the lightweight, CPU-only deps `tmgrpo/` and `scripts/`
need: `openai`, `pyyaml`, `numpy`, `datasets`, `math-verify`, `pytest`, `ruff`.

```bash
.venv/bin/pytest tests -q      # 53 passed
.venv/bin/ruff check tmgrpo scripts tests
```

`OPENAI_API_KEY` must be set before any code that constructs an `tmgrpo.llm_client.LLMClient`
actually calls the API (tests never make real API calls -- they inject a fake client). Two ways
to set it, either works:

```bash
export OPENAI_API_KEY="sk-..."          # takes precedence if both are set
# or:
cp .env.example .env && $EDITOR .env    # .env is gitignored; loaded automatically via python-dotenv
```

## 2. Data prep (runnable now, needs an HF token for the train split)

```bash
cd textual_momentum_grpo
# 1a. Training pool -- DEFAULT (all run scripts use this unless TMGRPO_TRAIN_DATA=math):
#     open-r1/OpenR1-Math-220k, `default` config (93.7k rows). Confirmed data-scope decision
#     2026-08-12: Critique-GRPO (our baseline, arXiv 2506.03106) actually trains on subsets of
#     this dataset, not MATH -- see configs/base.yaml's data.train_files comment for why (MATH
#     left Qwen3-8B saturated from step 1 of arm1_floor's run).
.venv/bin/python scripts/prepare_openr1_train.py --out data/train_openr1.jsonl
.venv/bin/python scripts/convert_to_verl_parquet.py \
  --in data/train_openr1.jsonl --out data/train_openr1.parquet --data-source openr1_math

# 1b. MATH training split -- OPT-IN legacy pool (hendrycks/competition_math, GATED -- accept
#     terms on the Hub and `huggingface-cli login` / set HF_TOKEN first, or this fails with a
#     clear error message). Only needed if you're intentionally running with
#     TMGRPO_TRAIN_DATA=math:
.venv/bin/python scripts/prepare_math_train.py --out data/train.jsonl
.venv/bin/python scripts/convert_to_verl_parquet.py \
  --in data/train.jsonl --out data/train.parquet --data-source math

# 2. Eval sets (used for all arms regardless of training pool), reused from self_correct_grpo's
#    already-vendored MATH data (Apache-2.0):
.venv/bin/python scripts/prepare_eval_data.py --out-dir data/eval
#    -> data/eval/math500.jsonl        (500 rows, copied verbatim)
#    -> data/eval/aime24.jsonl         (30 rows, copied verbatim)
#    -> data/eval/olympiad_slice.jsonl (200 rows, seeded sample of olympia.jsonl, seed=0)

# 3. Confirm no train/eval leakage (run against whichever training pool(s) you prepared --
#    OpenR1-Math-220k's own decontamination is against its own benchmark list, not necessarily
#    identical to data/eval/*.jsonl, so don't skip this just because it's the default now):
.venv/bin/python scripts/check_overlap.py \
  --train data/train_openr1.jsonl \
  --eval data/eval/math500.jsonl data/eval/aime24.jsonl data/eval/olympiad_slice.jsonl
```

## 3. GPU node: pull verl (not runnable here)

Confirmed decision: **official verl docker image**, mirroring how `self_correct_grpo` keeps its
own heavy RL stack (slime) out of any repo-tracked venv.

```bash
docker pull verlai/verl:latest   # exact tag TBD -- check https://hub.docker.com/r/verlai/verl/tags
                                  # for the tag matching the verl release this project targets
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v /home/dg793/steerable-text-to-lora:/workspace/steerable-text-to-lora \
  -it verlai/verl:latest /bin/bash
```

Mount the repo in (as above) rather than copying, so `textual_momentum_grpo/{tmgrpo,scripts,
configs}` stay in sync with this build. This does not touch the root repo's own `.venv` -- verl
runs entirely inside the container's own Python environment.

## 4. Model download

```bash
hf download Qwen/Qwen3-8B --local-dir textual_momentum_grpo/models/qwen3-8b
```

(README section 5: Qwen3-8B is the backbone common to both Critique-GRPO's and ICRL's published
reference points.)

## 5. Before the first real run: re-verify the flagged risk

`configs/base.yaml`'s header comment and `tmgrpo/verl_hooks.py`'s docstrings both flag the same
thing: this project's verl config keys and the internalization hook were written against verl's
*documented* config surface (confirmed during this build pass: `algorithm.adv_estimator`,
`actor_rollout_ref.actor.{optim.lr,clip_ratio_low,clip_ratio_high,entropy_coeff,use_kl_loss,
kl_loss_coef}`, `actor_rollout_ref.rollout.{n,temperature,response_length,
calculate_log_probs}`, `data.{train_batch_size,max_prompt_length,max_response_length}`,
`trainer.{total_epochs,test_freq,save_freq}` -- verified via
`verl.readthedocs.io/en/latest/examples/config.html` and
`verl.readthedocs.io/en/latest/algo/dapo.html`), but never checked against an actually-installed
verl. Once inside the container from section 3:

```bash
python -m verl.trainer.main_ppo --help   # confirm every key above still exists / hasn't moved
```

Two things NOT confirmed and genuinely uncertain -- do this work on the GPU node, not by guessing
here:

1. **Adam optimizer betas / weight decay / LR schedule.** The ICRL reference
   (`self_correct_grpo/vendor/ICRL/icrl/hydra_conf/optimizer/adam_base.yaml`) uses
   `adam_beta1=0.9, adam_beta2=0.98, weight_decay=0.1, lr_decay_style=constant`. `configs/base.yaml`
   only sets `lr` because the corresponding verl config keys weren't confirmed. Find and set them
   once verl's actual `actor_rollout_ref.actor.optim` schema is visible.
2. **`tmgrpo.verl_hooks.recompute_unconditioned_logprobs`** (arms 3/5 internalization, README
   section 3 step 2) raises `NotImplementedError` on purpose. This needs a real forward pass
   through verl's actor with the conditioning-context turn stripped from the prompt, which depends
   on verl's actual worker/`DataProto` API -- implement this against the real API, not a guess.

## 6. Running an arm

Once the above is resolved:

```bash
cd textual_momentum_grpo
.venv/bin/python scripts/render_arm_config.py --all   # regenerate configs/resolved/*.yaml if configs/ changed

# inside the verl container, per arm:
python -m verl.trainer.main_ppo \
  --config-path=/workspace/steerable-text-to-lora/textual_momentum_grpo/configs/resolved \
  --config-name=arm1_floor
# ... arm2_instance_off, arm3_instance_on, arm4_trajectory_off, arm5_trajectory_on
```

In practice arm1 (`run_arm1_floor.sh`) and arm5 (`run_arm5_trajectory.sh`) are launched directly as
CLI-override shell scripts, not via `--config-path=configs/resolved` -- both default to the
OpenR1-Math-220k training pool and accept `TMGRPO_TRAIN_DATA=math ./run_arm1_floor.sh` to switch
to the MATH pool instead (see each script's own comment, and `configs/base.yaml`'s
`data.train_files` for the rationale). Any future arm2/arm3/arm4 run script should follow the same
pattern for consistency across arms.

Build order: **arm1 -> arm2 -> arm3 -> arm4 -> arm5** (`minimal_experiment_plan.md` section 1).
Arm 2 must be checked against published Critique-GRPO numbers before trusting arms 3-5 (README
section 4 success criteria).

## 7. Metrics to collect per run

Per `minimal_experiment_plan.md` section 4: unconditioned eval accuracy on all three eval sets;
the `w_t` trajectory over training for arms 3/5 (via `tmgrpo.calibration.apply_calibration`'s
returned `w_t` array -- log its mean/histogram per step); frontier-model call count and token
usage (`tmgrpo.llm_client.LLMClient.usage_summary()`); and periodic `tmgrpo.spotcheck.spot_check`
results for arms 4/5's textual gradients, logged for manual review rather than auto-filtered.

Unconditioned eval accuracy (the first of these) is computed offline with
`scripts/eval_checkpoint.py`, once a run has produced a checkpoint. It's arm-agnostic (same
script for arm1_floor through arm5_trajectory_on -- eval is always unconditioned, so there's no
arm-specific eval logic) and must run under the GPU stack:

```bash
cd textual_momentum_grpo
.venv-verl/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/tmgrpo/arm1_floor/global_step_300/actor \
  --arm-name arm1_floor \
  --out eval_results/arm1_floor_step300.json
```

This merges the verl FSDP shards into a HF model (cached under `checkpoints_hf/`, so reruns
against the same checkpoint skip the merge), generates unconditioned completions for
`data/eval/{math500,aime24,olympiad_slice}.jsonl` via vLLM, scores them with
`tmgrpo.reward.check_answer`, and writes a JSON report with per-set and overall accuracy. Needs a
free GPU -- run after the training job has exited or on a separate allocation.
