# TextGrad Reproduction (using the real `textgrad` library)

Originally GSM8K-only (hence the environment/port narrative below, and
`scripts/textgrad_repro.py`'s former name, `textgrad_repro_gsm8k.py`) --
`--task` now selects among ~29 tasks (GSM8K, BBH, MMLU, GPQA, AIME, AQuA,
...). See `TEXTGRAD_MULTITASK_PLAN.md` for the generalization; nothing
below this point changed because of it.

Ported from `/home/dg793/text-to-lora/scripts/textgrad_repro_gsm8k.py` (see that repo's
`textgrad_repro_README.md` for the original writeup, including the empty-optimizer-response
incident and the `Qwen/Qwen3-32B` full-run results). Same experiment, same textgrad primitives
(`Variable` / `BlackboxLLM` / `TextualGradientDescent` / `GSM8K_DSPy` / `StringBasedFunction`),
same JSONL output schema, same two disclosed monkeypatches. Only the environment and one flag
changed -- see below.

## Why this lives here too

The original script has no hard dependency on anything `text-to-lora`-specific -- it only imports
`textgrad`, `vllm`, and stdlib/`numpy`. What tied it to that repo was its *environment*:
`text-to-lora/pyproject.toml` pins `transformers==4.51.1` / `vllm==0.9.2` / `torch==2.7.0` /
`python>=3.10` (see that repo's `B200_EVAL_ENV_FIXES.md` for why -- a private
`LlamaRotaryEmbedding` import path and a `vllm==0.9.2`-vs-`transformers>=4.57` conflict). Those
pins are what `docs/01_env.md` in *this* repo explicitly calls a "dead end" for anything needing
newer model support (see that doc's "Why not reuse `text-to-lora`'s environment" section) --
this repo instead declares lower bounds only (`transformers>=5.0`, `torch>=2.9`, and, via the
optional `gen` extra, `vllm>=0.11`). So: **the original is not restricted by anything in its own
code, only by the venv it happens to run in** -- newer models that need a newer `transformers`
(or a `vllm` release recent enough to load them) just need a newer environment, which is what
this repo already is.

## Setup: no changes to the stable venv

Same pattern as the original: `textgrad` is cloned into `textgrad_repro/` at this repo's root
(`git clone https://github.com/zou-group/textgrad.git textgrad_repro`, gitignored) and never
installed via `uv add`. Every invocation uses `uv run --with-editable` to layer it (plus its small
extra deps: `diskcache`, `litellm`, `graphviz`, `gdown`, `tenacity`, `python-dotenv`) into an
ephemeral overlay on top of this repo's own resolution:

```bash
uv run --with-editable ./textgrad_repro \
  --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
  --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
  --with diskcache --with litellm --with graphviz --with gdown --with tenacity --with python-dotenv \
  python scripts/textgrad_repro.py --eval_test
```

`scripts/textgrad_repro_run.sh` bakes the full invocation in, same as the original.

### Why this is pinned much harder than "just add `--extra gen`" (a real dependency dead end on this machine)

The naive port -- `uv run --with-editable ./textgrad_repro --extra gen --with ...` -- does not
work on this machine, and it's worth spelling out why, because the failure chain reveals a genuine
three-way conflict, not a simple missing pin:

1. **Unpinned `--extra gen` resolves `vllm==0.26.0`** (this repo's declared floor is only
   `gen = ["vllm>=0.11"]`, see `docs/01_env.md`). That ships a compiled `_C` extension linked
   against CUDA 13 (visible as transitively-pulled `nvidia-*-cu13` packages). This machine's
   driver reports `CUDA Version: 12.8` as its max (`nvidia-smi`), so `from vllm import LLM` raises
   `ImportError: libcudart.so.13: cannot open shared object file`.
2. **Bisecting the vllm pin down** confirmed the CUDA-13 linkage persists through at least
   `vllm==0.21.0` (whose own metadata doesn't even declare a `cu13` marker -- the linkage is
   baked into the wheel at CI build time, not expressed as a pip constraint). `vllm==0.11.0`
   (bisected down to this repo's own declared floor) resolves `nvidia-*-cu12` packages instead
   and imports cleanly.
3. **But `vllm==0.11.0`'s tokenizer code predates transformers 5.x**: loading any model raises
   `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` against this
   repo's `transformers>=5.0` floor (5.14.1 in the base venv) -- that attribute was removed in
   transformers 5. Checking PyPI metadata across vllm releases confirms there is **no version
   overlap**: every vllm release up to ~0.21.0 that avoids the cu13 requirement also declares (or
   in 0.11.0's case, silently needs) `transformers<5`; every vllm release that requires
   `transformers>=5.5.3` (0.24.0+) is also cu13-only. On this machine's driver, "cu12-compatible
   vllm" and "transformers>=5-compatible vllm" are disjoint sets.
4. **So `transformers` gets pinned older too, just for this overlay** (`==4.57.1`, which satisfies
   vllm 0.11.0's real `>=4.55.2` floor and still has `all_special_tokens_extended`). That in turn
   needs `huggingface_hub<1.0` (transformers 4.57.1's own declared bound) -- which conflicts with
   the **persistent** venv's `kernels` package (installed by the base `uv sync --extra attn`,
   resolved against `huggingface_hub>=1.0`'s newer `StrictDataclass` typing). Since `--with`
   overlays layer on top of the persistent venv rather than replacing it, `kernels` doesn't get
   re-resolved and blows up at import (`TypeError: Unsupported type for field 'import_name':
   str | None`) the moment `transformers.modeling_utils` unconditionally imports it. Pinning
   `kernels==0.10.0` too (any `0.4.0`-`0.10.0` release still declares `huggingface_hub<1.0`)
   resolves that.

Torch needs its own explicit steering, independent of all of the above: `[tool.uv.sources]` only
reliably steers *this project's own declared* dependencies, not packages added ad hoc via
`uv run --with`. Without the explicit `--index .../cu128 --index-strategy unsafe-best-match`, uv
silently resolves the default PyPI build of whatever `torch` version vllm demands, tagged
`+cu130` -- it *imports* fine (no missing `.so` at import time) but fails on the first actual CUDA
call: `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`. Confirmed
by testing both ways during the port. `--index-strategy unsafe-best-match` is what lets uv pull
`torch` from the `cu128` index and `vllm` from PyPI in one resolution (the default "first index
wins per package" strategy would otherwise refuse `vllm` on the `cu128` index and stop there,
since that index doesn't publish `vllm` at all).

**Net effect on the "why port this at all" argument above**: it still holds for the model-loading
code path itself (vllm's own model-architecture support, not its tokenizer-API surface) -- newer
Qwen releases just need a `vllm` new enough to know their architecture, which is a `vllm` version
question, not a `transformers` version question, for vllm-served generation. But on *this specific
machine* (CUDA 12.8 driver), the generation pipeline is stuck using an old `transformers`/`kernels`
pair inside the ephemeral overlay regardless -- the project's `transformers>=5.0` floor is not
reachable by any vllm release that also avoids the CUDA-13 requirement. That floor remains correct
and untouched for the rest of this repo (the hypernetwork model code, which never invokes vllm);
this tension is local to vllm-based generation only. If this machine's driver is ever upgraded to
support CUDA 13, or this runs on newer hardware, re-check whether a newer `vllm` pin (which would
then also allow dropping the `transformers`/`kernels` downgrade) works before carrying these pins
forward.

This never touches `pyproject.toml`, `uv.lock`, or the persistent `.venv` -- confirmed via
`git status`/file mtimes after running (note: this directory itself is not currently a git repo,
so there's no `git status` to check here, but no `uv add`/`uv lock` command is ever invoked either
way).

## What changed vs. the original

- **Environment**: base project is `transformers>=5.0`/`torch>=2.9`/`vllm>=0.11`, but the actual
  overlay used for this script pins `vllm==0.11.0`/`transformers==4.57.1`/`kernels==0.10.0` (see
  the dependency-dead-end writeup above) -- a machine-specific compromise, not this repo's general
  environment. `text-to-lora`'s pins were `transformers==4.51.1`/`torch==2.7.0`/`vllm==0.9.2`. The
  `textgrad` library itself has no upper pins (`setup.py`: `python_requires=">=3.9"`,
  `requirements.txt` is all lower-bounded), so nothing in it needed patching for the newer stack.
- **`--enable_thinking` / `--no_enable_thinking`** (new flag, default: thinking **on**, matching
  Qwen3's own chat-template default): threaded through every `apply_chat_template` call
  (`build_chat_prompt`, and the patched `ChatVLLM.generate`) for every role -- solve, backward
  critique, and optimizer rewrite all use the same setting. The original script never varied this
  (Qwen3-32B was always run with the template default, i.e. thinking on). Passing
  `enable_thinking=` to a tokenizer whose chat template doesn't reference it (older/non-reasoning
  models) is harmless -- Jinja silently ignores unused template variables.
- Everything else -- the two-hop backward chain, `run_validation_revert` defaulted on, the
  `add_generation_prompt=True` patch, the concise `BACKWARD_SYSTEM_PROMPT` patch, batch size 3 /
  12 total steps, the `<IMPROVED_VARIABLE>`-tagged optimizer format, the fixed 200/300/1319
  train/val/test split from `GSM8K_DSPy` -- is unchanged. See the original README's "What's
  actually reproduced" / "Deliberate engineering deviations" / "Known upstream quirks" sections;
  none of that changed in the port.

## Files

| File | Purpose |
|---|---|
| `textgrad_repro/` | Cloned upstream repo, editable-installed via the ephemeral overlay. Reference source, not modified. |
| `scripts/textgrad_repro.py` | The driving loop + logging, using textgrad's real primitives (formerly `textgrad_repro_gsm8k.py`). |
| `scripts/textgrad_repro_run.sh` | Thin env-var wrapper baking in the `uv run --with ...` invocation. |

## Output (`data/textgrad_repro/{model_dir}_{task}_textgrad-repro/`)

Identical schema to the original -- `train_set.jsonl`, `val_set.jsonl`, `forward_outputs.jsonl`,
`gradients.jsonl`, `iterations.jsonl`, `best_prompt.json`, `test_eval.jsonl` (with `--eval_test`).
See the original README for the exact per-row field list.

## Results (Qwen/Qwen3-14B, thinking disabled)

Status as of 2026-08-04. The full task registry has ~29 entries (§5 of
`TEXTGRAD_MULTITASK_PLAN.md`); **10 have clean, thinking-disabled results
so far** via `scripts/textgrad_repro_run_all.sh`. This is a partial
results table, not a full-registry sweep -- see "Not yet run" below.

| task | baseline val | final val | baseline test | test | Δ test |
|---|---|---|---|---|---|
| gsm8k | -- | 0.9733 | 0.9477 (1250/1319) | 0.9515 (1255/1319) | +0.0038 |
| aqua | 0.51 | 0.83 | 0.6181 (157/254) | 0.7953 (202/254) | +0.1772 |
| bbh_causal_judgement | 0.63 | 0.67 | 0.6757 (25/37) | 0.7568 (28/37) | +0.0811 |
| bbh_date_understanding | 0.91 | 0.94 | 0.88 (88/100) | 0.89 (89/100) | +0.01 |
| bbh_dyck_languages | 0.14 | 0.19 | 0.26 (26/100) | 0.25 (25/100) | -0.01 |
| bbh_formal_fallacies | 0.94 | 0.95 | 0.95 (95/100) | 0.95 (95/100) | 0.00 |
| bbh_geometric_shapes | 0.68 | 0.77 | 0.72 (72/100) | 0.78 (78/100) | +0.06 |
| bbh_hyperbaton | 0.98 | 0.99 | 0.98 (98/100) | 0.97 (97/100) | -0.01 |
| bbh_logical_deduction_seven_objects | 0.76 | 0.92 | 0.63 (63/100) | 0.92 (92/100) | +0.29 |
| bbh_movie_recommendation | 0.60 | 0.69 | 0.55 (55/100) | 0.58 (58/100) | +0.03 |

Notes:
- `gsm8k` predates the baseline-test-accuracy automation (§7a of the plan
  doc) and was run without a val-set baseline check; its
  `baseline_test_accuracy.json` was produced by a one-off manual eval, not
  the orchestrator.
- `bbh_dyck_languages` and `bbh_hyperbaton` regressed slightly test-side
  despite a val-side improvement -- both within test-set noise (n=100, so
  ±1 example is ±1pt) and not investigated further.
- Verified for all 10: zero `</think>` tags in any `val_set`, `test_eval`,
  or `baseline_test_eval` file (thinking genuinely off for every scored
  call), and zero null `predicted_answer` (no truncation).
- **Cache-contamination caveat (now fixed):** 5 of these 9 fresh reruns
  (aqua, date_understanding, dyck_languages, logical_deduction_seven_objects,
  movie_recommendation) briefly shared the NFS-mounted
  `~/.cache/textgrad/cache_vllm_*.db` with the old thinking-enabled runs.
  The disk cache keyed responses only on `(system_prompt, prompt)`, not on
  `enable_thinking`, so a handful of *training-batch* forward calls
  (2-9 lines out of 1336 in `forward_outputs.jsonl`, always `split=train`,
  never `val`/`test`) replayed stale thinking-enabled completions instead
  of generating fresh thinking-disabled ones. All of those replayed calls
  still parsed to a correct-format answer (only 1 of ~20 was actually
  wrong, and that one was a genuine model error, not a parsing artifact),
  and **zero contamination reached `val_set`, `test_eval`, or
  `baseline_test_eval`** -- so the val/test numbers above are unaffected.
  Fixed at the source in `scripts/textgrad_repro.py`'s `generate()` patch:
  the cache key now includes an `[enable_thinking=...]` prefix, so this
  can't recur.

**Not yet run / not clean, excluded from the table above:**
- `bbh_boolean_expressions` -- crashed on the `np.bool_` JSON-serialization
  bug (now fixed, see `scripts/textgrad_repro.py`'s `_json_default`); not
  yet rerun.
- `bbh_multistep_arithmetic_two` -- run was cancelled mid-flight before
  `--eval_test`; incomplete.
- `aime` -- crashed (`optimizer.step()`'s rewrite prompt exceeded
  `max_model_len` once realistic thinking-length gradients were included);
  data deleted, deprioritized to end-of-list in `run_all.sh`.
- 16 more registry tasks never attempted: `bbh_object_counting`,
  `bbh_word_sorting`, `bbh_navigate`, `bbh_sports_understanding`,
  `bbh_web_of_lies`, `bbh_temporal_sequences`,
  `bbh_tracking_shuffled_objects_seven_objects`,
  `bbh_salient_translation_error_detection`, `bbh_ruin_names`,
  `bbh_snarks`, `mmlu_all`, `gpqa_main`, `multiarith`, `commonsenseqa`,
  `strategyqa`, `trec`.

Archived (thinking-contaminated or crashed) pre-fix data lives in
`data/textgrad_repro/[DoNotUse]old_thinking-enabled_runs/` with its own
`DEBUG_NOTES.md`.

## Running it

```bash
./scripts/textgrad_repro_run.sh
```

Cheap smoke test:

```bash
MAX_EPOCHS=1 STEPS_PER_EPOCH=2 EVAL_TEST=0 ./scripts/textgrad_repro_run.sh
```

Qwen3-14B, thinking disabled:

```bash
MODEL_DIR=Qwen/Qwen3-14B ENABLE_THINKING=0 ./scripts/textgrad_repro_run.sh
```

Key env vars: `MODEL_DIR` (default `Qwen/Qwen3-32B`), `ENABLE_THINKING` (`1`/`0`, new in this
port), `BATCH_SIZE`, `MAX_EPOCHS`, `STEPS_PER_EPOCH`, `RUN_VALIDATION` (`1`/`0`), `DATA_DIR`,
`EVAL_TEST` (`1`/`0`).

## Caches: shared across `text-to-lora` and this repo, not sprinkled per-project

Neither repo sets `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`, so both use the same default HF
hub cache at `~/.cache/huggingface/hub` (395G on this machine as of writing, already holding
several Qwen2.5/Qwen3 checkpoints) -- downloading `Qwen/Qwen3-14B` here will *not* duplicate
anything already fetched under `text-to-lora`, and vice versa. `uv`'s package cache
(`~/.cache/uv`) and the `vllm` compile cache (`~/.cache/vllm`) are likewise process-wide, not
per-project, so the ephemeral `--with-editable` overlays from both repos' `textgrad_repro_run.sh`
scripts share those too.

The one *project-scoped-looking* cache that isn't actually project-scoped either: textgrad's own
`ChatVLLM` response diskcache, written to `~/.cache/textgrad/cache_vllm_<model_dir_basename>.db`
(via `platformdirs`, same default in both repos since neither overrides it). It's keyed by
`model_string`, so `Qwen/Qwen3-14B` and `Qwen/Qwen3-32B` get separate `.db` files with no
collision -- but running the *same* model from both repos would share one cache file. Also note
(carried over from the original README's "Known upstream quirks"): that cache key is
`sha256(system_prompt + prompt)` only -- temperature/`max_tokens`/`enable_thinking` are **not**
part of the key. Since every call site here still defaults to temperature 0, this is mostly
academic, except for `--enable_thinking`: a `Qwen/Qwen3-14B` run with thinking on and a later run
with thinking off against the *same* prompt text would collide and silently return the first
run's cached response. If you run the same model with both `--enable_thinking` and
`--no_enable_thinking`, clear `~/.cache/textgrad/cache_vllm_Qwen/Qwen3-14B.db` between them (or
pass a distinct `--data_dir`, which doesn't help the cache but at least keeps the JSONL outputs
apart).
