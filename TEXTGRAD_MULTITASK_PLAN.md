# Plan: generalize `scripts/textgrad_repro.py` to `--task`

(Note: this file was written while the script was still named
`textgrad_repro_gsm8k.py`; it was renamed to `textgrad_repro.py` once
`--task` covered more than GSM8K, which is what this plan describes.
References below have been updated to the current filename.)

Ported from `/home/dg793/text-to-lora/TEXTGRAD_MULTITASK_PLAN.md` (its second,
expanded revision -- 27→29 registry entries, up from the original 3-task
`gsm8k` / `bbh_object_counting` / `bbh_word_sorting` scope this repo already
had implemented) onto this repo's port of the TextGrad reproduction
(`scripts/textgrad_repro.py`, `scripts/textgrad_repro_run.sh`,
`textgrad_repro/` vendored clone). The algorithm and upstream facts are
identical between the two repos. What differs, and is called out below
wherever it matters:

- **`--enable_thinking` / `--no_enable_thinking`** (default: thinking on) is
  threaded through every chat-template call in this repo's script. Absent
  from text-to-lora's original. Applies to every task equally -- this plan
  doesn't touch that plumbing, just keeps it working for the new tasks too.
- **This repo has no `gepa_repro/` port.** Section 6b's AIME split logic
  reuses `/home/dg793/text-to-lora/scripts/gepa_repro_aime.py`'s
  `load_aime_splits()` verbatim in the source plan; this repo has no such
  file to import, so the same logic is reimplemented inline in
  `_load_aime_splits()` inside `textgrad_repro.py` instead.
- **The completed regression-baseline run in this repo is Qwen3-14B on
  GSM8K** (`data/textgrad_repro/qwen-qwen3-14b_gsm8k_textgrad-repro/`,
  200/300/1319 train/val/test, val_accuracy≈0.97 at iteration 0), not
  text-to-lora's Qwen3-32B/100-val/0.957 run. Use whichever this repo's own
  regression check is run against.
- **`HF_TOKEN` is already set** in this environment, so GPQA (gated) should
  load without extra setup -- still worth a smoke-test, since "set" isn't
  the same as "accepted the license for this specific account."
- **New Dataset classes are one shared `_RowsDataset` wrapper, not five
  separate ~20-line `Dataset` subclasses.** The source plan's section 6c
  says "Subclass `textgrad.tasks.base.Dataset`... ~20 lines each." This repo
  builds `multiarith`/`aqua`/`commonsenseqa`/`strategyqa`/`trec` (and
  `mmlu_all`/`gpqa_main`/`aime`) as plain Python row-lists wrapped in one
  small duck-typed class instead: nothing in `tg.tasks.DataLoader` or this
  script's own `dataset_to_rows()` does an `isinstance(..., Dataset)` check
  (`Dataset` is an ABC with no shared behavior beyond the abstract methods),
  so subclassing it buys nothing and five near-identical subclasses would
  just be repetition. Functionally equivalent; less code.
- **The headroom-sweep script is renamed** `textgrad_baseline_sweep.py`
  (dropping "bbh", per the source plan's own suggestion) **and imports
  `TASKS`/`ANSWER_PARSERS` directly from `textgrad_repro.py`** rather
  than keeping a second hand-maintained registry, so the two scripts can't
  drift out of sync as tasks are added.
- **Registry entries: 29, not the source plan's headline "27."** Its own
  "Scope" line undercounts by re-adding the BBH table's row count (20, not
  19) and its own 9-item extension list; the *tables* in §6a-c are
  authoritative and this port implements exactly those, so this doc reports
  the number the tables actually produce rather than propagating the
  mismatched summary arithmetic forward.

**Goal.** Run the existing TextGrad reproduction on BBH and other benchmarks
via a `--task` flag, without changing the behavior of the already-completed
GSM8K run.

**Primary target.** `BBH_object_counting` — the only task in TextGrad's paper
where TextGrad separates from DSPy by a wide margin (CoT 77.8 / DSPy 84.9 /
TextGrad 91.9), so it is the only one where a broken reproduction is visibly
broken. Word Sorting and GSM8K both tie DSPy, meaning zero improvement is
indistinguishable from a correct run.

**Scope.** 20 BBH tasks (all with a non-empty 250-row pool) + GSM8K + MMLU +
GPQA + AIME + MultiArith + AQuA + CommonsenseQA + StrategyQA + TREC = 29
registry entries in `TASKS`. Build order followed §4 (registries/plumbing
first, verified against gsm8k + bbh_object_counting alone), then all BBH
tasks, then MMLU/GPQA/AIME, then the five new HF-backed tasks.

**Non-goals.** Don't touch `textgrad_repro/` (it is an untouched upstream
reference clone — all deviations are monkeypatches in our script). Don't
change the optimizer loop, revert logic, `--enable_thinking` plumbing, or
logging schema.

---

## 1. Current state (before this expansion)

`scripts/textgrad_repro.py` already had `--task` with a 3-entry
`TASKS` registry (`gsm8k`, `bbh_object_counting`, `bbh_word_sorting`), an
`ANSWER_PARSERS` registry (`integer`/`mcq_letter`/`exact`), and `parse`
threaded through `eval_split()`, the training-loss `StringBasedFunction`,
and the training-batch logging path, from an earlier iteration of this same
plan. This revision expands the registry to the full task set below and
adds the plumbing that expansion needs (per-task `max_tokens`/
`min_max_model_len` for AIME's forward budget, the `_RowsDataset` wrapper,
`mcq_letter`'s `Answer: X` surface form).

`EVAL_FN_PURPOSE`/`role_description`-equivalents (`spec["eval_purpose"]`,
`spec["role_noun"]`) are **semantically load-bearing**, not cosmetic: both
are interpolated into the backward prompts that produce the textual
gradients. Every registry entry sets both.

## 2. Upstream facts to build against

Verified by reading `textgrad_repro/textgrad/tasks/` in this repo's
vendored clone (byte-identical to text-to-lora's):

- `BigBenchHard(task_name, split=...)` (`big_bench_hard.py:33`) — `__getitem__`
  returns `(row["x"], row["y"])`, the same 2-tuple shape as `GSM8K_DSPy`.
  Both expose `get_task_description()`. So `dataset_to_rows()` and the
  training loop need **no** changes for any BBH task.
- BBH splits are `examples[:50]` / `[50:150]` / `[150:]` — 50 train / 100 val
  / 100 test, downloaded via `subprocess.call(["wget", ...])` from
  `raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard`, caching to
  `platformdirs.user_cache_dir("textgrad")`.
- BBH's default task description is *"You will answer a reasoning
  question... where VALUE is a numerical value."* — upstream uses this for
  every BBH task, including non-numeric ones.
- `load_task("BBH_<anything except object_counting>")`
  (`textgrad/tasks/__init__.py:45`) uses `MultiFieldTokenParsedEvaluation`,
  an **LLM-as-judge**. Only `object_counting` (`__init__.py:26`) gets the
  deterministic `StringBasedFunction(string_based_equality_fn)`. This script
  never calls `load_task()` at all — every task is scored by a deterministic
  `ANSWER_PARSERS` entry via our own `_make_equality_fn`. Disclosed
  deviation, documented in the script's module docstring.
- `MMLU` (`mmlu.py:30`) and `GPQA` (`gpqa.py:31`) also return 2-tuples and
  ship an `eval_string_based` helper using `r"(?i)Answer\s*:\s*([A-D])"`.
  **Neither defines `get_task_description()`** — only
  `get_default_task_instruction()` — so both need the registry's
  `task_description` override. Their `*InstanceDataset` subclasses return
  4-tuples; not used here.
- MMLU/GPQA/LeetCode are reached via `load_instance_task`, not `load_task`,
  in upstream's own design — they're **solution-optimization** (test-time,
  instance-level) benchmarks there. The paper's own prompt-optimization
  table is only Object Counting, Word Sorting, GSM8K. Running prompt
  optimization on MMLU/GPQA/AIME here is a **new experiment, not a
  reproduction** — said so in the script's module docstring.

## 3. Design

Two registries at module top (`TASKS`, `ANSWER_PARSERS`), plus a shared
`_RowsDataset` wrapper (see the "What differs" note above) for every task
not backed by one of upstream's own `Dataset` subclasses.

```python
TASKS = {
    "gsm8k": dict(
        loader=lambda split: GSM8K_DSPy(split=split),
        parser="integer",
        role_noun="GSM8K math word problem task",
        eval_purpose="...",
        task_description=None,   # None -> use dataset.get_task_description()
    ),
    # 20 BBH entries built programmatically from a (suffix, bbh_name, parser,
    # value_desc) list -- see _BBH_TASKS in the script -- plus mmlu_all,
    # gpqa_main, aime, multiarith, aqua, commonsenseqa, strategyqa, trec
    # added individually.
    ...
}
```

`--task` selects a key; default `"gsm8k"`.

### Answer parsers

`ANSWER_PARSERS = {"integer": ..., "mcq_letter": ..., "exact": ...}`, each
`str -> comparable`. Applied to **prediction and gold symmetrically**.

- `integer` — same digit-extraction logic as upstream's `parse_integer_answer`,
  reimplemented locally to return `None` (not upstream's `0`) on no match.
- `mcq_letter` — handles **three** surface forms, because BBH, MMLU/GPQA, and
  AQuA/CommonsenseQA differ: `Answer: A` (MMLU/GPQA's own template, checked
  first), `(A)` parenthesized (BBH), and a bare trailing letter (fallback).
  Accepts any single letter, not just A–D — AQuA and CommonsenseQA have five
  options. Returns `None` on no match. Takes the *last* match of whichever
  form is present.
- `exact` — lowercase, strip whitespace and trailing period, take the text
  after a final `Answer:` if present.

**Critical trap — the whole reason parsers are a registry.** A naive
integer parser that defaults to `0` on no match makes an MCQ task's gold
`(A)` and a prediction of `(D)` both parse to `0`, so **every example
scores correct and the script silently reports 1.0 accuracy**. Every parser
here returns `None`, not a default, on failure, and `None == None` counts
as *incorrect* (`_make_equality_fn` and `eval_split()` both check
`predicted is not None and predicted == gold`). Verified with unit
assertions (§6.1) before touching a GPU.

## 4. Edits, in order (this revision)

1. **Module docstring** — extended with a "Task generalization" paragraph
   covering the full registry, the MMLU/GPQA/AIME "extension not
   reproduction" caveat, the `_RowsDataset` design choice, the
   deliberately-skipped datasets (§6d), and the GPQA `HF_TOKEN` requirement.
2. **`_parse_mcq_letter`** — added the `Answer: X` regex (checked before the
   `(X)` and bare-letter fallbacks) for MMLU/GPQA/AQuA/CommonsenseQA.
3. **`_RowsDataset`, `_reasoning_task_description`, and eight `_load_*`
   loader functions** (`_load_mmlu_all`, `_load_gpqa_main`,
   `_load_aime_splits`/`_load_aime`, `_load_multiarith`, `_load_aqua`,
   `_load_commonsenseqa`, `_load_strategyqa`, `_load_trec`) — inserted
   between `ANSWER_PARSERS` and `_make_equality_fn`.
4. **`TASKS` registry** — `gsm8k` unchanged; 20 BBH entries built from a
   `_BBH_TASKS` list via a loop (closure captured correctly per-iteration
   with a factory function, not a bare lambda over the loop variable);
   `mmlu_all`/`gpqa_main`/`aime`/`multiarith`/`aqua`/`commonsenseqa`/
   `strategyqa`/`trec` added individually. `aime` sets `max_tokens=16000`,
   `min_max_model_len=32768`.
5. **`_patch_chat_vllm_engine(enable_thinking, default_max_tokens=2000)`** —
   new parameter. `tg.BlackboxLLM`'s forward pass calls `engine.generate()`
   with no `max_tokens` kwarg (`LLMCall.forward` never threads one through),
   so the *default* on the patched `generate()` is the only place a
   per-task forward budget can reach the training loop's own generation
   calls. `main()` passes `spec.get("max_tokens", 2000)`.
6. **`eval_split()` / `batched_generate()`** — added a `max_tokens`
   parameter (already had `parse` from the prior revision), threaded from
   `main()`'s `forward_max_tokens = spec.get("max_tokens", 2000)` into all
   three call sites (baseline, per-iteration val, test).
7. **`main()`** — computes
   `max_model_len = max(args.max_model_len, spec.get("min_max_model_len") or 0)`
   and prints a note when it bumps; this is what keeps AIME's
   competition-math solutions from truncating without needing a
   `--max_model_len` the operator has to remember to pass by hand.
8. **argparse** — `choices=sorted(TASKS)` now lists all 29 keys.

`scripts/textgrad_repro_run.sh` needed no changes this revision — `TASK`
already passes through generically; `main()`'s own `max_model_len` bump
handles AIME without a new env var.

## 5. Task registry (implemented)

### 5a. BBH — 20 tasks

`loader=lambda split: BigBenchHard("<task_name>", split=split)`, splits
always 50/100/100.

| `--task` | BBH task_name | parser | note |
|---|---|---|---|
| `bbh_object_counting` | `object_counting` | integer | **primary target**, published numbers |
| `bbh_word_sorting` | `word_sorting` | exact | paper's 2nd task; weak signal (ties DSPy) |
| `bbh_multistep_arithmetic_two` | `multistep_arithmetic_two` | integer | |
| `bbh_dyck_languages` | `dyck_languages` | exact | |
| `bbh_navigate` | `navigate` | exact | Yes/No |
| `bbh_boolean_expressions` | `boolean_expressions` | exact | True/False |
| `bbh_causal_judgement` | `causal_judgement` | exact | Yes/No; only 187 examples |
| `bbh_formal_fallacies` | `formal_fallacies` | exact | valid/invalid |
| `bbh_sports_understanding` | `sports_understanding` | exact | yes/no; OPRO-reported |
| `bbh_web_of_lies` | `web_of_lies` | exact | Yes/No |
| `bbh_date_understanding` | `date_understanding` | mcq_letter | |
| `bbh_temporal_sequences` | `temporal_sequences` | mcq_letter | OPRO-reported |
| `bbh_logical_deduction_seven_objects` | `logical_deduction_seven_objects` | mcq_letter | |
| `bbh_tracking_shuffled_objects_seven_objects` | `tracking_shuffled_objects_seven_objects` | mcq_letter | |
| `bbh_geometric_shapes` | `geometric_shapes` | mcq_letter | |
| `bbh_salient_translation_error_detection` | `salient_translation_error_detection` | mcq_letter | |
| `bbh_hyperbaton` | `hyperbaton` | mcq_letter | |
| `bbh_movie_recommendation` | `movie_recommendation` | mcq_letter | OPRO-reported |
| `bbh_ruin_names` | `ruin_names` | mcq_letter | OPRO-reported |
| `bbh_snarks` | `snarks` | mcq_letter | OPRO-reported; only 178 examples |

The four OPRO-reported additions plus `temporal_sequences` give the full
set OPRO publishes, directly comparable against a second paper.

### 5b. Extensions — MMLU, GPQA, AIME

| `--task` | source | parser | splits |
|---|---|---|---|
| `mmlu_all` | `MMLU(subset="all", split=...)`, wrapped by `_load_mmlu_all` | mcq_letter | dev 285 → train[:50]; validation 1531 → val[:100]; test 14042 → seeded sample of 300 |
| `gpqa_main` | `GPQA(subset="gpqa_main")`, wrapped by `_load_gpqa_main` | mcq_letter | no native splits — 448 rows, shuffle seed `_SPLIT_SAMPLE_SEED`, `[:50]`/`[50:150]`/`[150:]` (298) |
| `aime` | `AI-MO/aimo-validation-aime` + `MathArena/aime_2025`, via `_load_aime_splits`/`_load_aime` | integer | 90 rows shuffled seed 0 (`_AIME_SPLIT_SEED`), halved into train 45 / val 45; test = `MathArena/aime_2025` (30) |

- **MMLU**: `all` config chosen over per-subject splits (`dev` is 5 rows,
  `validation` 11–170 — both too small). No `get_task_description()`
  (§2) — `task_description` set explicitly in the registry.
- **GPQA**: `Idavidrein/gpqa` is **gated** — this environment already has
  `HF_TOKEN` set; still smoke-test the load before a full run. No split
  concept at all; shuffle seeded with `_SPLIT_SAMPLE_SEED` (fixed, not
  `--seed`) so repeats of "the same task" don't silently get different
  data. `gpqa_diamond` (198) is too small to split three ways; `gpqa_main`
  used instead. Genuinely hard for Qwen3, good headroom, but a 298-row test
  split still carries ~±3% SE.
- **AIME**: split logic reimplemented in `_load_aime_splits()` (see "What
  differs" above — no `gepa_repro_aime.py` to import here). Answers are
  integers 0–999, so the `integer` parser works unchanged. Likely the best
  headroom of anything in this registry. val=45/test=30 are tiny — a single
  test item is 3.3% — report AIME as exploratory, never a headline delta.
  Needs the bumped forward budget (`max_tokens=16000`,
  `min_max_model_len=32768`) — see §4 point 5 and §6 gotchas.

### 5c. New row-list tasks — MultiArith, AQuA, CommonsenseQA, StrategyQA, TREC

Each built via `_RowsDataset` (see "What differs" above), not a `Dataset`
subclass.

| `--task` | HF dataset | columns | answer | parser | splits |
|---|---|---|---|---|---|
| `multiarith` | `ChilleD/MultiArith` | `question`, `final_ans` | integer string | integer | train 420 → `[:50]`/`[50:150]`; test 180 (full) |
| `aqua` | `deepmind/aqua_rat` (config `raw`) | `question`, `options` (list `"A)21"`), `correct` | letter A–E | mcq_letter | train 97467 → seeded 50; validation 254 → `[:100]`; test 254 (full) |
| `commonsenseqa` | `tau/commonsense_qa` | `question`, `choices{label,text}`, `answerKey` | letter A–E | mcq_letter | train 9741 → `[:50]`; validation 1221 → `[:100]` val, `[100:400]` test |
| `strategyqa` | `ChilleD/StrategyQA` | `question`, `answer` (bool) | True/False | exact | train 1603 → `[:50]`/`[50:150]`; test 687 → `[:300]` |
| `trec` | `SetFit/TREC-QC` | `text`, `label_coarse_original` | ABBR/DESC/ENTY/HUM/LOC/NUM | exact | train 5452 → `[:50]`/`[50:150]`; test 500 (full) |

- **AQuA**: `options` rendered into the prompt as `A) 21` lines
  (`_format_aqua_option`) — there's no pre-rendered prompt field.
- **CommonsenseQA**: `test` split is unlabeled, so val and test both come
  from `validation`, kept disjoint (`[:100]` / `[100:400]`).
- **StrategyQA**: `answer` is a Python bool, stringified to `"True"`/
  `"False"` consistently on both sides via `_load_strategyqa`'s `row()`.
- **TREC**: `SetFit/TREC-QC`, not `CogComp/trec` (a script dataset that
  needs `trust_remote_code`). `label_coarse_original` gives the 6-way
  coarse labels EvoPrompt reports.
- **MultiArith**: expect near-ceiling accuracy on a modern model — included
  for literature comparability (Promptbreeder), but §7's sweep will likely
  disqualify it from a real training run.

### 5d. Deliberately skipped

| Dataset | Why skipped |
|---|---|
| **TruthfulQA** | Generation-setting metric needs a fine-tuned "GPT-judge" model; MC1/MC2 is a likelihood-ranking metric this generate-and-parse loop can't express. |
| **HotPotQA** | Needs a Wikipedia retriever + multi-module pipeline (retrieve → summarize → answer) — MIPROv2/GEPA territory, not a single system-prompt `Variable`. |
| **LeetCode-Hard** | `LeetCodeHardEval.__getitem__` returns a 3-tuple, needs sandboxed test execution, only 39 problems, no splits. Solution-optimization in TextGrad's own design. |
| **IFBench** | Needs a new constraint-verifier dependency and a new per-constraint metric type, not answer correctness. |
| **MMLU/GPQA as "reproductions"** | Included above as extensions (§5b) — they're solution-optimization benchmarks in the paper, no published prompt-optimization number to reproduce against. |

## 6. Verification

1. **Parser assertions first.** Confirmed: `mcq_letter("(D)") !=
   mcq_letter("(A)")`; `mcq_letter("Answer: C") == "C"`;
   `mcq_letter("no letters here") is None`; `exact(...)` and `integer(...)`
   behave as specified above. All checked with stub imports (no GPU needed)
   before touching real data.
2. **Registry structural check.** All 29 `TASKS` entries have a callable
   `loader`, a `parser` present in `ANSWER_PARSERS`, and non-empty
   `role_noun`/`eval_purpose`. The 20 BBH loader closures were verified to
   each capture their own distinct `task_name` (the classic
   loop-variable-late-binding bug the factory-function pattern avoids).
3. **GSM8K regression.** `MAX_EPOCHS=1 STEPS_PER_EPOCH=1 EVAL_TEST=0
   DATA_DIR=/tmp/tg_regress MODEL_DIR=Qwen/Qwen3-14B
   ./scripts/textgrad_repro_run.sh` — baseline val accuracy must land at
   ~0.97, matching this repo's completed
   `qwen-qwen3-14b_gsm8k_textgrad-repro` run. Not yet re-run against this
   revision — do this before trusting any other task's numbers.
4. **Smoke the primary target.** `TASK=bbh_object_counting MAX_EPOCHS=1
   STEPS_PER_EPOCH=1 EVAL_TEST=0`. Confirm 50/100/100 split sizes; baseline
   accuracy neither 0.0 nor 1.0; `gradients.jsonl` non-empty.
5. **Smoke each new task family** the same way before a full run — BBH mcq/
   exact tasks, MMLU, GPQA (watch for the 401 if `HF_TOKEN` isn't accepted
   for the license), AIME (watch for truncated responses if
   `min_max_model_len` didn't take effect), and the five row-list tasks.
6. **Full run.** `TASK=<key> EVAL_TEST=1` with defaults (12 steps), once
   §7's sweep has confirmed the task is in the headroom band.

## 7. Second deliverable — headroom sweep

`scripts/textgrad_baseline_sweep.py` (renamed from
`textgrad_bbh_baseline_sweep.py` — see "What differs" above). Imports
`TASKS`/`ANSWER_PARSERS` from `textgrad_repro.py` directly instead of
duplicating them. For each task, loads the val split, runs the seed task
description 0-shot through one batched `vllm.LLM.generate` call, scores
with that task's parser, prints a table, and bumps `max_model_len` to cover
every task's `min_max_model_len` automatically (so AIME doesn't need a
separate invocation).

Why it matters: the GSM8K run opened at 0.97 and most steps logged
`reverted: true`. The paper's headroom comes from `gpt-3.5-turbo` sitting at
72.9; Qwen3 is far past that. Anything baselining ≥0.93 will produce the
same dead run. **Keep tasks in the 0.3–0.8 band.** Expect this to disqualify
`multiarith` and possibly several BBH tasks, and to confirm `aime`,
`gpqa_main`, `bbh_dyck_languages`, and `bbh_geometric_shapes` as plausible
candidates — not yet run against this environment's model, so treat those
as hypotheses from the source plan, not confirmed numbers here.

## 7a. Third deliverable — full-registry orchestrator

`scripts/textgrad_repro_run_all.sh` (new, not in the source plan). Loops
`scripts/textgrad_repro_run.sh` over every key in `TASKS` (or a `TASKS=...`
subset), sequentially on one GPU — each task pays its own model-load cost,
since this is a shell-level orchestrator over the existing single-task
entrypoint, not a rewrite that keeps one `vllm.LLM` alive across tasks.
Skips any task whose `best_prompt.json` already has a `test_accuracy` field
(`SKIP_DONE=1` by default; `FORCE_RERUN=1` to override), so re-running the
script after an interruption only resumes unfinished tasks. `DRY_RUN=1`
previews the plan without executing. Prints and saves
(`logs/textgrad_repro_run_all/summary_<slug>_<timestamp>.txt`) a table of
baseline / final(best) / test val accuracy per task, read from each task's
`best_prompt.json`.

This needed one small addition to `textgrad_repro.py`'s `main()`:
`best_prompt.json` previously only recorded whichever val accuracy ended up
"best" across iterations (`best["val_accuracy"]`), which is *not* the same
as the pre-training baseline whenever training actually improved on it.
`result["baseline_val_accuracy"] = baseline_accuracy` (and
`result["task"] = args.task`) are now written alongside it, so the
orchestrator's summary table doesn't have to recompute baseline accuracy by
re-aggregating `forward_outputs.jsonl`'s `iteration=-1` rows. Runs completed
*before* this change (e.g. the existing
`qwen-qwen3-14b_gsm8k_textgrad-repro` run) won't have `baseline_val_accuracy`
in their `best_prompt.json` and the summary table will show `NA` for that
column until re-run.

## 8. Gotchas

- **Short BBH tasks.** Splits are positional (`[150:]`), and not every BBH
  task has 250 examples — `penguins_in_a_table` has 146 (empty test split,
  which is why it's excluded from §5a). `causal_judgement` (187) and
  `snarks` (178) yield short test splits. `main()` asserts
  `len(test_set) > 0` after loading and prints all three sizes.
- **AIME needs a bigger forward budget**, handled via the registry's
  `max_tokens`/`min_max_model_len` fields (§4 points 5, 7) rather than a
  manual flag — a thinking model's reasoning plus a competition-math
  solution blows past the 2000-token default and truncates before the
  answer line, scoring 0 and looking like a capability failure rather than
  a budget problem.
- **Download needs `wget` + network** for BBH; HF downloads for everything
  else except AIME (public HF datasets, already-cached or not). On an
  offline node, pre-warm caches before the GPU is allocated. GPQA
  additionally needs `HF_TOKEN` with an accepted license (already set in
  this environment).
- **100-example test splits** give roughly ±4–5% standard error; AIME's 30
  gives ±9%. Single-seed differences under ~8 points are noise. Budget 3
  seeds (`--seed`) for anything meant to be reported.
- **Keep the two existing monkeypatches and the `--enable_thinking`
  plumbing task-independent.** `_patch_chat_vllm_engine`,
  `_patch_backward_system_prompt`, and `--enable_thinking`'s threading
  through `build_chat_prompt`/`batched_generate` all fixed real issues or
  add real capability; none is GSM8K- or task-specific.
- **`tg.set_backward_engine(engine, override=True)`** and the
  `_OptimizerEngineProxy` `max_tokens` bump stay as-is — the optimizer-step
  context problem is task-independent.
- **Seeded slicing must be deterministic across `--seed` repeats.** GPQA's
  shuffle, MMLU's test sample, and AQuA's train sample all involve sampling
  from a larger pool. All three use `_SPLIT_SAMPLE_SEED` (fixed at
  `20260803`, deliberately *not* `args.seed`), so "the same task" has the
  same data across seed replicates and averaging is meaningful. AIME uses
  its own `_AIME_SPLIT_SEED = 0`, kept separate to make clear it's
  inherited from upstream's own script rather than chosen fresh.
