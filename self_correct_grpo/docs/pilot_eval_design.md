# Pilot design clarification: what got built vs. what the pilot should measure

This note exists because a session-long implementation effort (env setup, debugging, training runs)
converged on a pilot that measures a **different thing** than what the design doc's §1.1 preliminary
experiment calls for, and a different thing again from what the user actually meant by it. Recorded
here so future sessions don't have to re-derive this from scratch.

## 1. What has been implemented (current state, as of 2026-08-12)

`self_correct_grpo/run_pilot_gated.sh` / `run_pilot_ungated.sh` each **train a separate policy from
the same Qwen3-4B-Instruct-2507 base checkpoint**, using slime/Megatron GRPO:

- **Gated arm**: vendored, byte-for-byte-untouched `icrl.hydra_runner` — ICRL exactly as published.
  The critic round only fires when the oracle math-verifier grades round 1 wrong
  (`icrl/generate.py`'s loop-break condition).
- **Ungated arm**: `icrl_ungated/generate.py`, a one-line diff — the oracle-gate condition is
  dropped, so the critic round always fires. Same reward formula, same model, same data, same
  hyperparameters otherwise.
- Both arms train on **`data/math_pilot/train.jsonl`** (derived from DAPO-Math-17k, 17,398
  problems) — rollouts are drawn from this training pool across `PILOT_NUM_ROLLOUT` iterations.
- `scripts/compute_pilot_metrics.py` computes `Δ[i→c]`, `Δ[c→i]`, and no-op rate by reading
  **`rollouts_train/train_<rollout_id>.txt`** — i.e., **the training rollout dumps themselves**,
  comparing each episode's round-1 vs. round-2 reward within the training stream.
- `data/math_pilot/eval.jsonl` (derived from MATH500, the held-out set) is wired up as
  `--eval-prompt-data` but is **never actually consulted for these metrics** — `eval_interval=10`
  gates a periodic held-out eval pass that mostly hasn't fired in practice (runs so far have used
  `PILOT_NUM_ROLLOUT` values ≤ 10).
- Practical scope actually achieved on a single B200: ~5 rollout iterations
  (16 prompts × 8 samples/iteration = 128 episodes/iteration, ~640 episodes total per arm) before a
  GPU memory-fragmentation OOM in the training actor. Root cause and full debugging trail: see
  git history on `run_pilot_gated.sh`/`run_pilot_ungated.sh` (native `torch_memory_saver`
  crash/hang under both of slime's offload paths on this node; `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` breaks the actor↔rollout CUDA-IPC weight sync instead of just fixing
  fragmentation; current mitigation is `sglang_mem_fraction_static=0.3` plus disabling both offload
  paths, which delays but does not eliminate the fragmentation OOM).

**In short: two separately-trained models, metrics computed from their own training-time rollout
streams on the training data pool, never touching the held-out MATH500 eval set.**

## 2. What the design doc (§1.1, §6.1–§6.3) actually specifies

Re-reading `self_correct_grpo_README.md`:

- §6.2's baseline table does call for **two separately-trained configurations** — "ICRL" (oracle-
  gated) and "ICRL's reward formula, but ungated" — matching what's implemented in that one respect.
- §6.1 says **"MATH500 as the single dataset for the preliminary experiment (§1.1)"** — this is
  ambiguous about train-vs-eval split in the doc's own text, but MATH500 is conventionally a
  500-problem **held-out test set** (a canonical subset of the larger MATH corpus), not a training
  pool of any meaningful size. The vendored-data setup itself (from an earlier planning session)
  already resolved this ambiguity: **train** on DAPO-Math-17k (disjoint from MATH500, checked via
  `scripts/check_math_overlap.py`), **evaluate** on MATH500 — i.e., the decisive `Δ[i→c]`/`Δ[c→i]`/
  no-op metrics were always meant to be computed via inference on the held-out set, not by reading
  back the training rollout log.
- §6.3 doesn't explicitly say "compute these metrics via held-out eval" in so many words, but
  nothing in the doc supports computing them from the training stream either — the natural reading,
  reinforced by the train/eval data split that was already built, is held-out eval.

**So part 1 of what's implemented already deviates from the design doc**: metrics should come from
evaluating each trained arm on MATH500, not from parsing `rollouts_train/*.txt`.

## 3. What the user actually meant (this conversation, 2026-08-12)

Restated in the user's own words: *"we train ICRL as close as possible with given compute, and test
the performance gap when we inference it for self-refining strategy, with and without the oracle
gate."*

This is **more specific than §6.2's two-separately-trained-baselines framing**, and a third distinct
design:

- Train **one** policy, as close to published ICRL as single-B200 compute allows — i.e., train it
  *with* the oracle gate active (since that's what "ICRL" actually is; the ungated *training*
  condition isn't part of this design at all).
- Hold that **one trained checkpoint fixed**.
- At **inference time**, run it on held-out MATH500 under **two different self-refinement
  strategies**, both using the same frozen weights:
  1. Gated inference: skip the critic/revision round when the oracle already grades round 1 correct
     (mirrors how the model was trained).
  2. Ungated inference: always invoke the critic/revision round, regardless of round-1 correctness
     (the model has to blindly decide whether to revise, with no oracle telling it round 1 was
     already fine).
- Compare `Δ[i→c]` / `Δ[c→i]` / no-op rate between these two **inference-time** conditions on the
  **same model**.

The question this answers: *does the self-correction judgment a model actually learns under
gated training degrade when that same judgment is deployed without the oracle gate at inference
time?* That's a deployment-realism question — at real inference time there's never an oracle
available to gate with, so this measures whether the model's own learned revision behavior holds up
without one. It's a different (and arguably sharper) question than "do two separately-trained
policies differ" (§6.2's framing) or "does training-time exposure to gated vs. ungated rollouts
change what gets learned" (what's currently implemented).

## 4. What needs to be built for (3)

Nothing currently in this repo runs an eval-only pass that (a) loads a specific trained checkpoint
and (b) generates against held-out MATH500 under a chosen gating strategy without also continuing
to train. Needed:

1. **A single training run** using the existing gated-arm machinery (`run_pilot_gated.sh`'s
   `icrl.hydra_runner` invocation, unchanged) to produce one checkpoint, trained as close to
   published ICRL as the single-B200 memory ceiling allows (see §1 above — the OOM constraint is
   orthogonal to this redesign and still applies to however many rollout iterations this training
   run gets). The user has retracted the earlier `PILOT_NUM_ROLLOUT=5` cap and wants this pushed as
   high as compute allows, not artificially capped for convenience.
2. **An eval-only harness**, decoupled from the train loop, that:
   - Loads that one saved checkpoint (`--load`/`--ref-load` pointed at its `torch-dist` checkpoint
     directory — note `--save-interval` needs to be low enough that a checkpoint actually gets
     written within however many rollout iterations training completes; currently `--save-interval
     50` with runs topping out around 5 iterations means **no checkpoint has actually been saved
     yet** by any run so far).
   - Runs rollout generation against `data/math_pilot/eval.jsonl` (MATH500) using the **gated**
     generate function (`icrl.generate.generate`), with training disabled (inference/eval-only,
     no gradient step).
   - Runs the same held-out set again using the **ungated** generate function
     (`icrl_ungated/generate.py`'s `generate`), same checkpoint, still eval-only.
   - Both passes need `Δ[i→c]`/`Δ[c→i]`/no-op computed from their own rollout dumps (extending
     `compute_pilot_metrics.py` or a variant of it to read eval dumps instead of train dumps).
3. Slime/ICRL's existing `--eval-prompt-data`/`--eval-interval`/`log_eval_rollout_data` path is the
   natural starting point for step 2, since it already threads MATH500 through as an eval set during
   a normal training run — but that path currently reuses whatever `--custom-generate-function-path`
   the whole training script was launched with (one generate function per script invocation, used
   for both train and eval rollouts). Getting two eval passes (gated, ungated) against the *same*
   frozen checkpoint likely means either two separate eval-only invocations pointed at the same
   `--load` path with `--train-iters 0` (or equivalent no-op-training flag) and different
   `--custom-generate-function-path`s, or a small standalone script that calls slime's rollout
   generation directly without going through the full `train.py` entrypoint. Needs investigation
   into what slime/Megatron actually expose for "load checkpoint, generate only" — not yet checked
   as of this note.

## 5. Implementation plan for (3)

Status as of this note: step 0 done and tested; steps 1–4 planned, **not yet implemented** — paused
here on the user's explicit instruction to record the plan before doing more.

**Mechanism found**: `vendor/ICRL/train.py`'s `train()` has a built-in eval-only special case —

```python
# special case for eval-only
if args.num_rollout == 0 and args.eval_interval is not None:
    ray.get(rollout_manager.eval.remote(rollout_id=0))
```

Setting `--num-rollout 0` with `--eval-interval` set skips the training loop entirely: it still
loads the checkpoint via `--load`/`--ref-load`, syncs weights to the rollout engine, then runs
exactly one eval pass against `--eval-prompt-data` and returns — no gradient step, no further
rollouts. This is the load-checkpoint-and-generate-only path needed for (3), already present in
vendored code, nothing to build from scratch for that part.

Eval dumps land at `{exp_dir}/rollouts_eval/eval_<rollout_id>.txt` (via vendored
`icrl.logging_utils.log_eval_rollout_data` → `_save_rollout_trajectories(..., split="eval")`),
same plaintext format as the training dumps `compute_pilot_metrics.py` already parses.

**Step 0 — done.** Generalized `scripts/compute_pilot_metrics.py` to read either split:
`load_metrics_for_dir(exp_dir, split="eval"|"train")`, default `"eval"` (matches this pilot's
design); `--split` CLI flag added, default `eval`. `train` split kept working for the earlier,
superseded two-separately-trained-arms comparison. Tests updated and passing
(`tests/test_compute_pilot_metrics.py`, 7 passed).

**Step 1 — planned.** Make checkpoint saving resilient to the memory-fragmentation OOM (§1): the
save trigger `should_run_periodic_action` in `slime/utils/misc.py` fires on `step % save_interval
== 0` OR unconditionally on the final configured `rollout_id == num_rollout - 1` — so if training
crashes mid-run before reaching that final step, and `save_interval` (currently `50`, from
`icrl/hydra_conf/checkpoint/base.yaml`) hasn't been hit either, **no checkpoint is saved at all**
(confirmed: none of this session's runs so far have produced one). Plan: override
`checkpoint.cli.save_interval` down to something small (e.g. `2`) via the existing
`PILOT_*`-env-var-override pattern already used for `PILOT_NUM_ROLLOUT` etc. in
`run_pilot_gated.sh`, so a recent checkpoint always exists regardless of when the OOM eventually
hits.

**Step 2 — planned.** Relaunch the gated-arm training run (`run_pilot_gated.sh`, `icrl.hydra_runner`,
unchanged generate function) with `PILOT_NUM_ROLLOUT` set high (not artificially capped at 5 — the
user has retracted that cap and wants this pushed as close to full compute as the single-B200
memory ceiling allows) and the new low `save_interval` from step 1. Let it run until it OOMs (as it
reliably does after some number of iterations per §1's fragmentation issue); the last saved
checkpoint before the crash is what step 3 uses. No further debugging of the OOM itself is planned
unless it turns out to prevent even a handful of iterations from completing with the new low
save-interval.

**Step 3 — planned.** Build a new eval-only run script (e.g. `run_pilot_eval_only.sh`, or two
thin invocations sharing a common body) that:
- Takes the checkpoint directory from step 2 as `--load`/`--ref-load` (the trained checkpoint, not
  the base HF-converted one used for initial training).
- Sets `rollout.cli.num_rollout=0` and a non-null `eval.cli.eval_interval` to trigger the special
  case above.
- Points `eval.cli.eval_prompt_data` at `data/math_pilot/eval.jsonl` (MATH500) only — no training
  pool involved.
- Runs twice against the same checkpoint, varying only which generate function is wired in: once
  via `icrl.hydra_runner` (gated inference — `icrl.generate.generate`), once via
  `self_correct_grpo.icrl_ungated.hydra_runner` (ungated inference —
  `icrl_ungated/generate.py`'s `generate`). Both already exist and already select their generate
  function this same way in the current training scripts; reusing that dispatch mechanism, not
  building a new one.

**Step 4 — planned.** Run `compute_pilot_metrics.py --gated-dir <gated-inference exp_dir>
--ungated-dir <ungated-inference exp_dir>` (default `--split eval`, matching step 3's output) to
get the pilot's decisive `Δ[i→c]`, `Δ[c→i]`, no-op rate numbers for gated-vs-ungated *inference*
on the one gated-trained checkpoint.

## 6. Resolved: relationship to §6.2's baseline table

Confirmed with the user (2026-08-12): §6.2's baseline table (multiple separately-trained
configurations) is for the **main experiment**, run later, not this preliminary pilot. This
preliminary pilot (§1.1) is exactly (3) above — one gated-trained checkpoint, compared under two
inference-time strategies on held-out MATH500 — and does not involve training a separate ungated
policy at all. `run_pilot_ungated.sh` (and the training-time ungated arm generally) belongs to the
later §6.2 main-experiment work, not to this pilot.
