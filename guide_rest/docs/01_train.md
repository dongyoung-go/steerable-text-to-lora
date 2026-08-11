# 01: Training (round loop)

Implements `guide_rest_README.md`'s Grow → Filter → (Feedback) → Improve round loop for
v1's two tasks, **GSM8K** and **MATH**, run independently. Guru's non-code domains and the
README's three TODO additions (failure-mode recurrence analysis, few-shot-exemplar
confound baseline, matched-compute baseline) are out of scope for v1.

## Environment

Everything here runs inside an ephemeral `uv run --with ...` overlay
(`guide_rest/run.sh`), never synced into this repo's `pyproject.toml`/`uv.lock`/persistent
`.venv` — same convention as `scripts/gepa_repro_run.sh` / `scripts/textgrad_repro_run.sh`.
`vllm==0.11.0` / `transformers==4.57.1` / `kernels==0.10.0` are pinned for the same reason
those scripts pin them (this box's CUDA 12.8 driver). `peft` and `math-verify` are the two
overlay additions specific to Guide-ReST.

```bash
TASK=gsm8k ./guide_rest/run.sh          # full run, both conditions
TASK=math CONDITION=B ./guide_rest/run.sh   # one task, one condition
```

Run a smoke test first — see "Smoke test" below — before committing to a full 5-round x
2-condition x 2-task sweep.

## Directory layout

```
guide_rest/
  tasks.py          dataset loading + prompt instruction + verifier, per task, registry-keyed
  sampling.py        Step 1+2: Grow (vLLM) + Filter (verifier), one process per round
  feedback.py         Step 3: Stage 1 (per-example critique) + Stage 2 (merge) — Condition B only
  train.py             Step 4: Improve — fresh LoRA from base, merge, save M_{t+1}
  eval_heldout.py       Step 5: held-out pass@1 of a checkpoint
  round_loop.py          orchestrator: subprocess-calls the above, per round, per condition
  run.sh                  env-overlay launcher

data/guide_rest/<task>/<condition>/         (<condition> is "A" or "B")
  round_{t}/
    grow_samples.jsonl    every k completion per Grow-pool question, labeled correct/incorrect
    filtered.jsonl         (question, completion) pairs that passed the verifier
    grow_stats.json         {n_total, n_correct, filter_pass_rate}
    dev_grow_samples.jsonl   every dev_k completion per dev-pool question, labeled (fixed dev pool, always unconditioned)
    dev_filtered.jsonl        (question, completion) pairs that passed the verifier on the dev pool -- train.py's early-stopping validation set
    dev_stats.json             {n_total, n_correct, filter_pass_rate} for the dev pool
    local_feedback.jsonl    N per-example critiques (Condition B only)
    feedback.txt             feedback_{t+1}: the merged critique used to condition round t+1's Grow (Condition B only)
    checkpoint/               M_{t+1}: base model + this round's LoRA, merged, safetensors
    heldout_eval.json         {task, n, n_correct, pass_at_1, rows: [...]}
  summary.jsonl             one row per round (see below)
```

## Round-loop mechanics

`round_loop.py` never imports torch/vllm itself — it only shells out to the other four
scripts via `subprocess.run`, one call per step, so each step's model load/GPU memory is
fully released before the next step starts. This runs on a single GPU (the box these
scripts already target — see `configs/sft.yaml`'s "1x B200" note), so a live vLLM engine
and a live training loop must never coexist.

For each condition (`A` = vanilla ReST, `B` = Guide-ReST), for round `t = 0..T-1`:

1. **Grow** (`sampling.py`): loads `M_t` into vLLM — round 0 loads the raw base model
   (`Qwen/Qwen3-14B`); round `t>0` loads `round_{t-1}/checkpoint/`. Samples `k` completions
   per Grow-pool question (`SamplingParams(n=k)`, one batched `llm.generate()` call for the
   whole pool). Condition B, from round 1 on, prepends `round_{t-1}/feedback.txt` to every
   prompt; round 0 is unconditioned in both conditions, per the README. Every completion is
   scored immediately with `tasks.py`'s verifier and written to `grow_samples.jsonl`
   (all, labeled) and `filtered.jsonl` (correct only). In the same vLLM session (no extra
   model load), also samples the fixed **dev pool** (`dev_k` completions each, always
   unconditioned even in Condition B), writing `dev_grow_samples.jsonl`/`dev_filtered.jsonl`
   — see "Dev pool" below for why this exists.

2. **Feedback** (`feedback.py`, Condition B only, every round including round 0 — round
   0's critiques seed `feedback_1` for round 1's Grow, per the README's worked example):
   groups `grow_samples.jsonl` by question, keeps questions with at least one correct *and*
   one incorrect sample, and samples `N` such questions as Stage-1 triples
   `(question, incorrect, correct)`. Stage 1 prompts `M_t` (self-critique — no separate
   critic model, per the user's explicit decision) for a critique per triple. Stage 2
   merges the previous `feedback.txt` (empty at round 0) with the `N` critiques into one
   paragraph, capped at `--max_words` (default 150). If no question has both a correct and
   an incorrect sample this round, the previous feedback is carried forward unchanged
   rather than erased.

3. **Improve** (`train.py`): loads the **raw base model** — never `M_t` — attaches a
   **freshly initialized** LoRA adapter, and fine-tunes on `round_{t}/filtered.jsonl` with
   the bare question (no feedback prefix — stripped before training, per README Step 4) as
   input and the completion as the masked-loss target. Merges the adapter onto the base and
   saves full weights as `round_{t}/checkpoint/` = `M_{t+1}`. This is the user's explicit
   choice: every round's LoRA starts from base, not from `M_t`'s already-tuned weights, so
   no drift compounds purely from repeated fine-tuning-on-fine-tuning across rounds. Only
   Grow reads the previous round's checkpoint; Improve always starts over.

   Trains up to an `--epochs` cap (default 3), but early-stops sooner if
   `round_{t}/dev_filtered.jsonl`'s loss stops improving for `--patience` epochs (default
   1) — see "Dev pool" below for why this exists and where the validation data comes from.

4. **Eval** (`eval_heldout.py`): loads `M_{t+1}`, greedy-decodes the task's held-out set
   (question only, no feedback prefix — same prompt shape in both conditions), scores pass@1.

5. `round_loop.py` appends one row to `summary.jsonl`:
   `{round, condition, filter_pass_rate, n_filtered_pairs, dev_pass_rate, heldout_pass_at_1, feedback_word_count, used_feedback_this_round}`.

## Dev pool (early-stopping validation, not the held-out pass@1 set)

`train.py`'s Improve step needs a validation signal to avoid overfitting on a round's
(small, model-generated) `filtered.jsonl` — see the "Overfitting" section below for why.
That signal comes from a **fixed, separate dev pool** (`tasks.py::load_*_dev_pool`,
`--dev_pool_size` questions, default 50), sampled disjointly from the Grow pool by
construction (re-derives the Grow pool's indices with the same `--pool_seed`/
`--grow_pool_size`, excludes them, then samples from what's left — see
`load_gsm8k_dev_pool`'s docstring) and held constant across every round and condition via
`--dev_seed`. It is **not** the same as the `--heldout_size` pass@1 set `eval_heldout.py`
uses for the headline metric — that one is even bigger (200/500 rows) and reports the
actual research metric; the dev pool exists purely so `train.py` has something to
early-stop on, and is never used for anything reported as a result.

Each round, `sampling.py` samples the dev pool in the *same* vLLM session as the Grow pool
(no extra model load) — `--dev_k` completions per dev question (default 4, smaller than
`--k` since this is only for early stopping, not training data), **always unconditioned**
even in Condition B, since it validates the bare question → completion mapping `train.py`
actually fits (Condition B's feedback prefix is Grow-only, stripped before training — see
Step 3 above). Verified with the same `tasks.py` verifier as the Grow pool, written to
`dev_filtered.jsonl`. `train.py` reads that file directly as its validation set — nothing
is carved out of `filtered.jsonl` itself.

## `tasks.py`: task registry

A name → `TaskSpec(instruction, load_grow_pool, load_heldout, verify)` dict
(`TASKS`), same pattern as `scripts/textgrad_repro.py`'s own `TASKS` registry. Adding a
task later (e.g. a Guru non-code domain) is a new entry here — no other file needs to
change, since `sampling.py`/`feedback.py`/`train.py`/`eval_heldout.py`/`round_loop.py` are
all written against the registry interface, not against gsm8k/math specifically.

- **`gsm8k`**: Grow pool sampled from `datasets.load_dataset("gsm8k", "main", split="train")`
  (7,473 rows total); held-out from `split="test"`. Gold answers are the trailing `#### N`
  integer. Verified with a last-digit-token integer parser (duplicated from
  `src/steerable_t2l/eval_accuracy.py::parse_integer_answer` — this repo already tolerates
  this exact duplication, e.g. `scripts/textgrad_repro.py`'s own copy — rather than
  importing `steerable_t2l`, which would pull in a different `transformers` pin than this
  overlay uses).
- **`math`**: Grow pool sampled from `EleutherAI/hendrycks_math`'s 7 subject configs
  (algebra, counting_and_probability, geometry, intermediate_algebra, number_theory,
  prealgebra, precalculus), train split, concatenated (7,500 rows total — matches ReST-EM's
  own reported MATH train-set size exactly). Gold is the last `\boxed{...}` in the
  reference solution (nested-brace-aware extraction — MATH solutions routinely nest braces
  inside `\boxed{}`). Held-out is all 500 rows of `HuggingFaceH4/MATH-500`'s test split —
  a subset of MATH's original *test* split, so it never overlaps the train-split Grow pool.
  Verified with `math-verify`'s `parse`/`verify` (symbolic equivalence — MATH answers are
  fractions/expressions/sets, not bare integers, so string/integer matching would badly
  undercount correct answers).

Both tasks' held-out set is fixed by `--heldout_seed`. The Grow pool is, by default, the
*entire* train split minus the dev pool (`--grow_pool_size` omitted → `None`, see "Dev
pool" below) — matching ReST-EM's own setup, which grows from essentially the full task
training set each round rather than a small subsample. Pass an explicit `--grow_pool_size`
to subsample instead (e.g. for a smoke test); `--pool_seed` only matters in that subsampled
case (it's what fixes *which* subset is used, held constant across rounds/conditions).

## Hyperparameters (v1 defaults)

| param | default | note |
|---|---|---|
| `k` (completions/question/round) | 8 | |
| `T` (rounds) | 5 | |
| Grow pool size | full train split minus dev pool (~7423 gsm8k / ~7448 math) | matches ReST-EM's own setup; pass `--grow_pool_size` to subsample |
| Dev pool size / `dev_k` | 50 / 4 | fixed, reserved before the Grow pool; early-stopping validation only, not a reported metric |
| Held-out size | 200 (gsm8k) / 500 (math, i.e. all of MATH-500) | |
| `N` (feedback triples) | 8 | swept `{3, 8}` on gsm8k only first (README asks for at least this); winner reused for math |
| feedback word cap | 150 | README's suggested cap |
| LoRA `r` / `alpha` / `dropout` | 16 / 32 / 0.05 | |
| LoRA target modules | q/k/v/o_proj + gate/up/down_proj | wider than `configs/oracle.yaml`'s q/k/v/o-only, since that config is scoped to match a hypernetwork's `TargetSpec` and this is standalone full-behavior SFT |
| Improve `lr` / epochs cap / patience / batch size | 1e-4 / 3 / 1 / 32 | cosine schedule, 3% warmup; early-stops on dev pool loss before the epoch cap if patience triggers first; batch_size measured (not guessed) as the throughput sweet spot on 1x B200 -- see "Wall-clock cost" below |
| Grow sampling temperature | 0.7 | top_p 0.95, top_k 20 |
| Held-out eval | greedy (temperature 0) | |
| Thinking mode | off | `enable_thinking=False` throughout, per user decision |
| vLLM `gpu_memory_utilization` / `max_model_len` | 0.85 / 8192 | GSM8K/MATH completions are shorter than harder tasks' 16384 budget |

## Overfitting

The Grow pool size and the fixed `--epochs=3` cap were both chosen without a strong prior,
so it's worth being explicit about what the literature actually does here. Neither the
original ReST paper (Gulcehre et al. 2023, translation) nor ReST-EM (Singh et al., "Beyond
Human Data," TMLR 2024 — the variant that actually targets MATH/GSM8K) trains a fixed epoch
count per round. ReST-EM's Algorithm 1 explicitly early-stops (`while reward improves on
D_val`) and reports that "train accuracy increases linearly with the number of ReST_EM
iterations" while test accuracy plateaus or regresses — attributed to "overfitting on the
small set of training problems." Three things this pipeline does that follow directly from
that finding:

1. **Early stopping** (`--epochs` as a cap, `--patience` for real stopping) — see "Dev
   pool" above.
2. **Fresh-from-base LoRA every round** (never continuing `M_t`'s weights) — ReST-EM does
   this too, explicitly "to mitigate task-specific over-fitting."
3. **Grow pool size**: defaults to the full train split (matching ReST-EM's own scale --
   they grow from essentially the entire task training set each round: 7,500 MATH problems,
   2,342 APPS problems, sampled 32-64x per problem), not the earlier `grow_pool_size=100`
   default this pipeline started with. A bigger, more diverse Grow pool is a more direct
   overfitting fix than tuning epochs/patience on a small one.

One thing ReST-EM does that this pipeline does **not** yet do: they cap the number of
solutions kept per problem (10, out of 32 sampled) specifically "to ensure diversity in the
training data and safeguard against overfitting on easier problems" — an easy question that
passes all `k` samples otherwise contributes `k`x the training signal of a hard question
that barely passes once. `filtered.jsonl` currently keeps every correct completion
uncapped (moot at `k=8 < 10`, since the cap wouldn't bind at this `k` anyway); worth adding
if `filtered.jsonl` turns out to be dominated by a few easy
questions once real runs are inspected.

## Wall-clock cost (measured on 1x B200, 2026-08-09)

At full-scale Grow pools, **Improve dominates total wall-clock time, not Grow.** Grow is
vLLM-fast (~28 min/round for gsm8k's ~7,423-question pool at `k=8`, measured by timing a
`pool=500` run and extrapolating linearly — vLLM's continuous batching makes this a safe
assumption at this scale). Improve is a plain per-example training loop over
`filtered.jsonl` (typically ~57,000 pairs at gsm8k's ~96% Grow pass rate), and was the
actual bottleneck: two timed runs (300 and 1,200 synthetic pairs, `batch_size=8`) solved
for **~95s fixed overhead** (model load + LoRA setup + merge + writing a 28GB checkpoint)
**+ ~0.57s/training-step**. At `batch_size=8` that's ~70 min/epoch on the full pool --
`batch_size` was then swept directly (8 → 32 → 64) rather than guessed: **32 measured as
the throughput sweet spot** (~96GB/183GB GPU memory, ~0.038s/pair, roughly halving Improve
time to ~38 min/epoch) — `batch_size=64` gave no further speedup (compute-bound plateau)
while using 152GB/183GB, too close to this box's memory ceiling to risk on an unattended
multi-hour run. Net effect: **full 5-round x 2-condition sweep for one task is ~12-24
hours** (best case: dev-pool early stopping fires after 1 epoch most rounds) **to ~24-48
hours** (worst case: never stops early, all 3 epochs every round) — both tasks together
roughly double that. Re-run this benchmark (`sampling.py`/`train.py` directly, timed, on a
few hundred synthetic pairs) if hardware, `k`, `grow_pool_size`, or LoRA config change
meaningfully, rather than trusting these numbers indefinitely.

**MATH is meaningfully slower than gsm8k end-to-end — not just at Grow.** Grow: ~2.86x
slower per completion than gsm8k (longer generations + `math-verify`'s CPU-side symbolic
parsing overhead), extrapolating to ~75 min/round on MATH's ~7,448-question pool at `k=8`
(vs gsm8k's ~28 min). Improve: re-benchmarked directly on **real** MATH completions (not
synthetic placeholders, since MATH's completions run much longer than gsm8k's and a short
synthetic stand-in would have understated the cost) — two timed runs (300 and 1,200 real
filtered pairs, `batch_size=32`) solved for ~125s fixed overhead + **~7.1s/training-step
(~0.22s/pair)**, roughly **5.8x slower per pair than gsm8k's batch=32 number
(~0.038s/pair)**, driven by MATH completions filling much more of the 1024-token
`--max_len` cap per example even at the same batch size. `batch_size=32` still ran cleanly
with no OOM on MATH, so **the batch size default is unchanged** — this is a per-example
compute cost difference, not a memory-fit problem, and `batch_size=64` was not
additionally tested for MATH (gsm8k was already at 152/183GB there; MATH's longer
sequences use more memory per example at the same batch size, so 64 carries a real,
untested OOM risk and should not be used unmonitored). At MATH's ~43,000 filtered pairs
per full-scale round (7,448 questions x k=8 x ~72% Grow pass rate, lower than gsm8k's
~96%), that's **~161 min/epoch** (vs gsm8k's ~38 min/epoch). Net: MATH's full 5-round x
2-condition sweep is **~40-90+ hours**, worse than gsm8k's ~12-48h range, almost entirely
from sequence length rather than dataset size.

## `N` sweep

Run round 0 + round 1 of Condition B on `gsm8k` with `N=3` and again with `N=8` (`ROUNDS=2
CONDITION=B N=3 ./guide_rest/run.sh`, then `N=8`), read both `feedback.txt`s, and pick
whichever produces a more specific, less generic critique (README's own qualitative
criterion — no numeric target given). Reuse that `N` for both conditions' full runs on both
tasks.

## Smoke test

Before a full run, confirm the pipeline is wired correctly end-to-end on a tiny
configuration:

```bash
ROUNDS=1 K=2 GROW_POOL_SIZE=8 DEV_POOL_SIZE=8 DEV_K=2 HELDOUT_SIZE=8 N=2 EPOCHS=1 TASK=gsm8k ./guide_rest/run.sh
```

Check: both `data/guide_rest/gsm8k/A/round_0/` and `.../B/round_0/` exist with a real
`checkpoint/` directory (loadable back into vLLM — `train.py` running to completion is the
main thing this catches), non-empty `grow_samples.jsonl`, and `summary.jsonl` rows with
non-NaN `filter_pass_rate`/`heldout_pass_at_1`.
