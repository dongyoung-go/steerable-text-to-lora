# GEPA Reproduction (using the real `gepa` library)

Ported from `/home/dg793/text-to-lora/scripts/gepa_repro_gsm8k.py` +
`gepa_repro_aime.py` (see that repo's `gepa_repro_README.md` for the
original writeup -- GEPA architecture decisions, what's actually reproduced,
disclosed deviations, known upstream quirks). Same GEPA primitives
(`optimize_anything()` / `GEPAConfig` / `EngineConfig` / `ReflectionConfig`),
same "one shared vLLM engine plays both the task LM and reflection LM, no
litellm/HTTP server" design, same JSONL output schema. Three things changed
in the port: the environment, the model, thinking-mode's default, and (the
big one) the task scope -- one script with a `--task` flag over the same
~29-entry registry `scripts/textgrad_repro.py` already built, instead of two
separate near-duplicate scripts (`gepa_repro_gsm8k.py`, `gepa_repro_aime.py`).

## Why this lives here too

Same argument `textgrad_repro_README.md` makes for its own port: GEPA itself
has no dependency on anything text-to-lora-specific -- only `gepa`, `vllm`,
and stdlib/`numpy`. What tied the original to that repo was its *environment*
(`transformers==4.51.1`/`vllm==0.9.2`/`torch==2.7.0` pins, a dead end for
this repo -- see `docs/01_env.md`'s "Why not reuse text-to-lora's
environment"). This repo's `scripts/textgrad_repro_run.sh` already solved
the environment problem for a vLLM-driven prompt-optimization script; this
port reuses that exact overlay recipe rather than re-deriving it, and adds
a second `--with-editable` for `gepa_repro/` (the real `gepa` clone) on top.

## The big change: one script, `--task` over `textgrad_repro.TASKS`, not two scripts

The original text-to-lora port had `gepa_repro_gsm8k.py` and
`gepa_repro_aime.py` as separate files, each with its own hand-rolled
`SEED_PROMPT`, its own split-loading function, and near-duplicate
`batch_evaluate`/iteration-reconstruction/test-eval code. Since this repo
already has a `TASKS` registry (`scripts/textgrad_repro.py`, ~29 entries --
GSM8K, 20 BBH tasks, MMLU, GPQA, AIME, MultiArith, AQuA, CommonsenseQA,
StrategyQA, TREC -- see `TEXTGRAD_MULTITASK_PLAN.md`) built for exactly this
purpose (comparing multiple optimizers on the same splits/prompts/parsers),
`scripts/gepa_repro.py` imports it directly --
`from textgrad_repro import ANSWER_PARSERS, TASKS, _json_default,
dataset_to_rows, slugify` -- the same pattern `scripts/textgrad_baseline_
sweep.py` already established for its own headroom sweep. Consequences:

- **`--task <key>`** selects among the full registry; every task TextGrad
  can run, GEPA can now run too, on identical train/val/test splits, the
  identical seed prompt (`TASKS[task]["task_description"] or
  dataset.get_task_description()`), and the identical scoring parser
  (`ANSWER_PARSERS[TASKS[task]["parser"]]`) -- so a GEPA run and a TextGrad
  run on the same `--task` are apples-to-apples in a way the original
  text-to-lora scripts' bespoke GSM8K/AIME prompts and split logic weren't
  designed to guarantee beyond those two tasks.
- **No more hand-rolled `SEED_PROMPT` strings.** The original
  `gepa_repro_gsm8k.py`/`gepa_repro_aime.py` each wrote their own seed
  prompt text, independent of what `textgrad_repro_gsm8k.py` used. Here,
  both optimizers start from the exact same string for a given task.
- **No more per-task split-loading functions.** `load_gsm8k_splits()` /
  `load_aime_splits()` from the originals are gone; every task goes through
  `TASKS[task]["loader"](split)` + `dataset_to_rows()`, same as
  `textgrad_repro.py`. `scripts/gepa_repro.py` itself adds only one thing
  the TextGrad script doesn't need: `rows_with_ids()`, which stamps GEPA's
  own recognized `"id"` field onto each row dict (see `_resolve_id` in
  `gepa_repro/src/gepa/oa/eval_server.py`) so val/train subscores stay
  stably keyed across candidates.
- **Feedback text is task-generic**, not GSM8K/AIME-specific. The originals
  built custom feedback strings per script (GSM8K stripped `####` from the
  raw answer; AIME appended the full written solution on a miss, since its
  loader carried a `solution` field). The unified `TASKS`-based rows here
  only carry `(question_prompt, answer)`, so `batch_evaluate()`'s feedback
  is one shared template ("Correct/Incorrect. The gold answer is
  '{answer}'." plus a parse-failure branch) that works for every task without
  branching on task identity. This is a disclosed simplification: AIME
  feedback no longer includes the worked solution the way the original
  `gepa_repro_aime.py` did (that would require carrying a `solution` field
  through `textgrad_repro.TASKS`'s loaders, which `textgrad_repro.py`
  itself has no use for and doesn't expose).
- **`gepa_repro_run_all.sh`** (new, not in the original) mirrors
  `scripts/textgrad_repro_run_all.sh` exactly -- same discovery mechanism,
  skip/resume logic, per-task log files, and summary table -- so the full
  registry can be swept with GEPA the same way it already can with TextGrad.

## What else changed vs. the original

- **Model**: default `--model_dir` is `Qwen/Qwen3-14B`, matching
  `textgrad_repro.py`'s own default and this repo's completed GSM8K
  regression baseline (`data/textgrad_repro/qwen-qwen3-14b_gsm8k_textgrad-
  repro/`, val_accuracy≈0.97 at iteration 0) -- not text-to-lora's
  `Qwen/Qwen3-32B`.
- **Thinking mode default flipped off.** `--enable_thinking` defaults to
  `False` (`--enable_thinking` to turn it on), the opposite of the original
  `gepa_repro_aime.py`'s `default=True`. Applies uniformly across every
  task (the original only ever varied this for AIME; GSM8K there always had
  it off). Reflection stays thinking-off regardless of this flag, same as
  the original (see `gepa_repro_common.VLLMLanguageModel` -- it's
  text-analysis over already-scored feedback, not itself the hard task, so
  there's no reason to pay for a `<think>` block there either way).
- **Environment**: runs inside `textgrad_repro_run.sh`'s exact overlay
  (`vllm==0.11.0`/`transformers==4.57.1`/`kernels==0.10.0`, cu128 torch
  index + `unsafe-best-match`) plus a second `--with-editable ./gepa_repro`
  for the real `gepa` clone -- not text-to-lora's
  `transformers==4.51.1`/`vllm==0.9.2`. See `textgrad_repro_README.md`'s
  "why this is pinned much harder than `--extra gen`" section for the full
  three-way dependency-conflict story; nothing about it changed for this
  port, since `gepa` itself has zero hard dependencies
  (`gepa_repro/pyproject.toml`: `dependencies = []`) and doesn't touch the
  vllm/transformers/kernels pin story at all.
- **No `fishfarm` dependency.** The original `gepa_repro_gsm8k.py` scored
  GSM8K with `hyper_llm_modulator.steering.textgrad_verifiers.
  verify_gsm8k_answer` (wrapping `fishfarm.tasks.language_restricted_math.
  extract_answer_number`), a dependency this repo doesn't have. GSM8K here
  scores with `ANSWER_PARSERS["integer"]` instead -- the same deterministic
  last-numeric-token parser `textgrad_repro.py` itself uses for `gsm8k` --
  so GEPA and TextGrad grade GSM8K identically here, rather than each using
  a different grader as the two text-to-lora scripts effectively did
  (`verify_gsm8k_answer` vs. `textgrad`'s own `string_based_equality_fn`).
  `gepa_repro_common.py` correspondingly drops `extract_final_int_answer`.
- **AIME's forward-token budget and `max_model_len` bump** now come from
  `textgrad_repro.TASKS["aime"]` (`max_tokens=16000`, `min_max_model_len=
  32768`) via the same `spec.get(...)` pattern `textgrad_repro.py`'s
  `main()` uses, rather than a hardcoded `--max_tokens 8192` default and no
  automatic `max_model_len` bump (the original relied on the operator
  passing `--max_model_len` by hand for AIME).

## What's unchanged from the original

See the original `gepa_repro_README.md`
(`/home/dg793/text-to-lora/gepa_repro_README.md`) for the full detail behind
each of these -- none of the reasoning changed in the port:

- **Architecture**: one shared in-process `vllm.LLM` plays both the task LM
  (via `batch_evaluator`, one batched `vllm.LLM.generate()` call per GEPA
  evaluation stage) and the reflection LM (`ReflectionConfig.reflection_lm`
  given a plain callable satisfying GEPA's `LanguageModel` protocol) -- no
  litellm import needed at runtime, no HTTP server, no second GPU-resident
  engine.
- **`batch_evaluator`, not GEPA's default per-pair thread pool** -- GEPA
  hands us a whole stage's `(candidate, example)` pairs in one call, which
  is what makes a single batched `vllm.LLM.generate(...)` valid; `parallel=
  False` is still passed defensively in `EngineConfig`.
- **`test_set` scored manually**, not via `optimize_anything(test_set=...)`
  -- confirmed again in this session against the freshly-cloned `gepa_repro/`
  clone (0.1.4): that kwarg requires the new `OptimizeAnythingConfig` API and
  raises `ValueError` when combined with the legacy `GEPAConfig` object both
  this script and the original use (`gepa/optimize_anything.py::
  optimize_anything`, `_from_legacy_config`). Both the seed and best
  candidate are scored on the held-out test split with one extra batched
  `vllm.LLM.generate(...)` call each instead.
- **`iterations.jsonl` reconstructed post-hoc** from `GEPAResult.candidates`/
  `.parents`/`.val_aggregate_scores`/`.discovery_eval_counts`/
  `.val_subscores` after `optimize_anything()` returns -- GEPA owns the
  outer loop, so nothing here is logged incrementally the way
  `textgrad_repro.py`'s explicit loop logs `iterations.jsonl`.
- **Output schema**: same per-file JSONL convention
  (`train_set.jsonl`/`val_set.jsonl`/`forward_outputs.jsonl`/
  `gradients.jsonl`/`iterations.jsonl`/`best_prompt.json`/`test_eval.jsonl`),
  same field names within each row, just generalized to any `--task` instead
  of being GSM8K- or AIME2025-specific (e.g. `run_dir_name()` now takes
  `task_key` the same way `textgrad_repro.py`'s does).

## Files

| File | Purpose |
|---|---|
| `gepa_repro/` | Cloned upstream `gepa` repo (`git clone https://github.com/gepa-ai/gepa.git gepa_repro`, gitignored), editable-installed via the ephemeral overlay. Reference source, not modified. Version at time of this port: `0.1.4`. |
| `scripts/gepa_repro_common.py` | Shared vLLM engine loading, batched chat-template generation, `VLLMLanguageModel` (the `LanguageModel`-protocol reflection callable). No `extract_final_int_answer` (see "no fishfarm dependency" above). |
| `scripts/gepa_repro.py` | The driving script: `--task` over `textgrad_repro.TASKS`, GEPA's real `optimize_anything()`. |
| `scripts/gepa_repro_run.sh` | Env-var wrapper baking in the `uv run --with-editable ./textgrad_repro --with-editable ./gepa_repro ...` invocation. `TASK` env var selects the registry key (default `gsm8k`). |
| `scripts/gepa_repro_run_all.sh` | Orchestrator over `gepa_repro_run.sh`, sweeping the full `TASKS` registry (or a `TASKS=...` subset) sequentially, with skip/resume and a summary table -- mirrors `scripts/textgrad_repro_run_all.sh`. |

## Output (`data/gepa_repro/{model_dir}_{task}_gepa-repro/`)

Identical schema to the original (see its README for the exact per-row field
list) -- `train_set.jsonl`, `val_set.jsonl`, `forward_outputs.jsonl`,
`gradients.jsonl`, `iterations.jsonl`, `best_prompt.json` (now also carries
a `"task"` field and `baseline_val_accuracy`, matching
`textgrad_repro.py`'s `best_prompt.json` shape so
`gepa_repro_run_all.sh`'s summary table can read both the same way),
`test_eval.jsonl` (with `--eval_test`).

## Running it

```bash
./scripts/gepa_repro_run.sh                       # TASK=gsm8k by default
TASK=bbh_object_counting ./scripts/gepa_repro_run.sh
TASK=aime ./scripts/gepa_repro_run.sh
```

Cheap smoke tests:

```bash
TASK=gsm8k MAX_METRIC_CALLS=60 EVAL_TEST=0 ./scripts/gepa_repro_run.sh
TASK=aime MAX_METRIC_CALLS=20 ./scripts/gepa_repro_run.sh
```

Full registry sweep (mirrors `textgrad_repro_run_all.sh`):

```bash
./scripts/gepa_repro_run_all.sh
TASKS="gsm8k bbh_object_counting aime" ./scripts/gepa_repro_run_all.sh
DRY_RUN=1 ./scripts/gepa_repro_run_all.sh   # preview without running
```

Key env vars (`gepa_repro_run.sh`): `TASK` (any `textgrad_repro.TASKS` key,
default `gsm8k`), `MODEL_DIR` (default `Qwen/Qwen3-14B`), `ENABLE_THINKING`
(`1`/`0`, default `0`), `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`,
`MAX_METRIC_CALLS`, `BATCH_SIZE`, `DATA_DIR`, `SEED`, `EVAL_TEST`.
`gepa_repro_run_all.sh` additionally: `TASKS` (subset override),
`SKIP_DONE`/`FORCE_RERUN`, `DRY_RUN`, `FAIL_FAST`.

## Known gaps / not yet verified

- **Not yet run end-to-end on a GPU as of writing this file** -- ported and
  statically checked (module compiles; `EngineConfig`/`GEPAConfig`/
  `ReflectionConfig`/`optimize_anything`/`GEPAResult` field names confirmed
  by reading the freshly-cloned `gepa_repro/` source this session, same as
  the original's own pre-GPU validation) from a CPU-only login node with no
  CUDA and no `vllm`/`textgrad`/`gepa` installed in the base venv -- so
  nothing that actually imports `vllm` or `textgrad` (which `gepa_repro.py`
  needs transitively for `textgrad_repro.TASKS`) has been import-tested
  yet. Real-model smoke tests (`TASK=gsm8k MAX_METRIC_CALLS=60 EVAL_TEST=0`)
  are pending on the target A6000 GPU box, same staging both
  `textgrad_repro_README.md` and the original `gepa_repro_README.md` used
  for their own first runs.
- **Task headroom**: `textgrad_repro.py`'s own regression run showed GSM8K
  baselining at ~0.97 for Qwen3-14B (near-ceiling, little room for either
  optimizer to show improvement) -- see `TEXTGRAD_MULTITASK_PLAN.md` section
  7's headroom-sweep discussion. The same caveat applies to GEPA runs on
  GSM8K here; `scripts/textgrad_baseline_sweep.py`'s per-task baseline
  numbers (once run against this environment's model) are the guide for
  which `--task` values are worth spending a full GEPA budget on.
- **AIME's `max_tokens=16000`** (from `textgrad_repro.TASKS["aime"]`) may
  still truncate on the hardest problems even with thinking off by default
  here -- worth checking `forward_outputs.jsonl` for responses with
  `predicted_answer: null` after the first real AIME run.
