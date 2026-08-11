# 03 — Training & Validation

**Status: IMPLEMENTED, GPU-verified, real run complete.** `src/steerable_t2l/{data,oracle,trainers}/`,
`validation.py`, and `src/steerable_t2l/checkpoint.py`/`losses.py` are built and unit-tested on
CPU with the tiny synthetic fixtures in `testing.py`/`conftest.py` (`bash
run_03_training_validation.sh`: ruff + the full `pytest tests` suite, no network, no GPU, no
real weights). The full pipeline -- length profiling, splits, 11 oracle LoRAs, canonicalization,
recon warm-start, and both SFT ablation arms -- has been run end to end on a real 1x B200 node
against the real 13-task GSM8K data and real Qwen2.5-3B/1.5B weights.

**Real-run result (2026-08-03, `configs/sft.yaml`/`sft_warmstart.yaml`, 2000 steps each --
scaled down from the doc's 20000/200 placeholder per the measured-throughput note in those
files; see `scripts/run_ablation.py`'s output):**

| | scratch (`zero_init=True`) | recon-warm-started |
|---|---|---|
| `steering_margin` vs. gibberish (avg/11 tasks) | 0.053 | **0.408** |
| `steering_margin` vs. other-task (avg/11 tasks) | 0.072 | **0.469** |
| `val_loss(train_descs)` | 0.188 | 0.177 |
| `val_loss(other_task_descs)` | 0.259 | 0.646 |
| `val_loss(gibberish_descs)` | 0.241 | 0.585 |
| `val_loss(base)` / `val_loss(oracle)` (condition-independent) | 1.028 / 0.547 | 1.028 / 0.547 |

This is the ablation doing its job: both arms reach similar in-distribution fit
(`train_descs`), but the from-scratch arm's `other_task`/`gibberish` losses sit close to
`train_descs` -- the steering margin stays small (0.05-0.07), the "collapsed to one constant
multi-task LoRA, ignoring its input" failure mode §4 warns about. The warm-started arm's
`other_task`/`gibberish` losses are clearly worse than `train_descs`, giving a ~6-8x larger
steering margin -- real evidence the recon warm start (not just more SFT steps) is what
teaches the hypernetwork to condition on the instruction. Confirms docs/03 §5 step 3's
prerequisite too: `oracle` (0.547) beats `base` (1.028) by a wide margin.

Several real bugs surfaced only on real hardware/data and are documented below with fixes.

### Bugs found and fixed during the B200 verification pass

1. **`peft.load_peft_weights` silently loads onto the wrong device.** It calls
   `infer_device()` internally when no `device=` is given, which returns `"cpu"` on a CPU-only
   node (masking the bug) and `"cuda"` the moment a GPU is visible -- regardless of where the
   target/hypernet actually live. Fixed by threading an explicit `device` parameter through
   `oracle.canonicalize.load_and_canonicalize_oracle`, `validation._load_oracle_per_module`,
   and `trainers/recon.py`'s callers, always set from the target's/hypernet's actual device.
2. **`kernels` needs an upper bound.** `pyproject.toml`'s `attn` extra (`kernels>=0.4`) had no
   ceiling; `uv sync` resolved `kernels==0.16.0`, but this `transformers` version's
   `is_kernels_available()` requires `kernels < 0.16.0` and silently disables the
   `kernels-community/flash-attn2` integration otherwise. Pinned to `kernels>=0.4,<0.16.0`.
3. **Real `inp_max_len`.** `scripts/profile_lengths.py` against the real GSM8K tasks/tokenizer
   found many responses hit a hard ~2001-token cap (from how they were originally generated)
   and an overall p99 prompt+response length of 2103 -- `configs/data.yaml`'s `inp_max_len` is
   now `2560` (0% response truncation), not the `1024` placeholder guessed before profiling.
4. **Target activation memory blowup at the real `inp_max_len`.** At `bs=16, L=2560`, this
   environment's `transformers`/PyTorch SDPA path does not dispatch a fused kernel for Qwen2's
   GQA (12 query heads vs. 2 kv heads) -- confirmed by explicit `sdpa_kernel` backend probing --
   and materializes full per-layer attention score matrices instead, using ~135 GB for a
   forward pass alone (OOMs even a 178 GB B200 once backward is added). First worked around by
   enabling target gradient checkpointing
   (`trainers/sft.py::train_sft(..., target_gradient_checkpointing=True)`, still the default: it
   holds only one decoder layer's activations at a time regardless of which attention backend
   actually gets used -- measured peak fwd+bwd memory at `bs=16, L=2560`: ~44 GB). The real root
   cause and better fix is item 5 below.
5. **The real fix for #4: `kernels-community/flash-attn` was renamed to `flash-attn2` upstream.**
   The old slug now 404s at the `kernels`-specific Hub API (though it silently redirects at the
   *model* API, which is why `curl`ing the model endpoint looked fine) -- this was misdiagnosed
   at first as an `HF_TOKEN` permissions problem, since the error message from the old slug
   ("Invalid username or password") looked auth-related and `whoami` with the same token
   succeeded elsewhere. It was not a credentials issue: `kernels-community/flash-attn2` resolves
   and downloads fine with the same token. Loading it under `HF_HUB_OFFLINE=1` additionally needs
   a pinned `@<revision>` (the unpinned form re-resolves "version" -> commit against the Hub on
   every load, which offline mode blocks even once the content is cached) and a *complete*
   snapshot fetched once with network access (`huggingface_hub.HfApi().snapshot_download(repo_id,
   repo_type="kernel", revision=...)` -- the partial fetch `transformers` does internally for a
   single build variant is not enough for a later `local_files_only=True` resolution, which
   checks the full tree). Measured at `bs=16, L=2560`: real FlashAttention2 needs ~160 GB without
   target checkpointing (0.62s/fwd+bwd) or ~44-48 GB with it (0.78-0.88s/fwd+bwd) -- ~2-3x
   faster than the sdpa fallback either way, for genuinely less memory. Wired into
   `scripts/train_oracle_loras.py`/`scripts/train_sft.py` as `--attn-implementation`, defaulting
   to `kernels-community/flash-attn2@c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1` (the commit `"v1"`
   / `version=1` resolves to today). `target_gradient_checkpointing` stays on by default for the
   safety margin (~44 GB vs. a 178 GB budget) even though disabling it is faster.
6. **`lora_hooks` + target gradient checkpointing raced.** `lora_hooks`'s context manager
   removes its forward hooks as soon as the forward call returns; gradient checkpointing,
   however, *recomputes* each layer's forward during `backward()`, which happens after that
   context has already exited in the original `sft_step`. The recompute therefore ran with no
   LoRA injection at all -- either corrupting gradients silently, or (since the recomputed
   graph then saves a different number of tensors than the original forward did) raising
   `torch.utils.checkpoint.CheckpointError`, which is how this was actually caught. Fixed by
   giving `sft_step` an optional `backward_fn` (`trainers/sft.py`), called from inside the
   `with lora_hooks(...)` block so hooks stay attached through the whole backward pass;
   `train_sft` passes `accelerator.backward`. Regression test:
   `tests/test_sft_step.py::test_sft_step_with_target_gradient_checkpointing_and_backward_fn`.
7. **`run_ablation.py`'s `format_margin` assumed the wrong shape.** `history` entries' actual
   `steering_margin` is per-task (`{task_name: {denom: value} | "n/a"}`, matching
   `validation.run_validation`'s real output), not the flat `{denom: value}` dict the original
   test fixtures (incorrectly) assumed. Fixed to average each denominator across the tasks
   that have it, skipping per-task `"n/a"`s; regression tests added in `test_run_ablation.py`.
8. Two pre-existing, unrelated lint failures in `scripts/textgrad_repro.py` (missing
   `zip(..., strict=)`) blocked `run_02_model.sh`'s repo-wide `ruff check scripts`; fixed
   trivially so phase-2 verification could run.
9. Environment hygiene, not a code bug: this node's `.venv` had stale `torchvision`/
   `torchaudio`/`torchcodec`/vllm-family packages installed from an earlier, unrelated
   `--extra gen` sync, breaking `transformers`' import chain (`torchvision::nms` operator
   mismatch). A plain `uv sync --extra dev --extra attn --extra log` (matching
   `run_01_env.sh` exactly) pruned them back to the extras this repo actually declares.

Deviations from a literal reading of this spec, decided during implementation:
- **Chat formatting**: only the prompt (user turn) goes through `tokenizer.apply_chat_template`
  (`add_generation_prompt=True`); the response (`assistant_prefill + response`) is appended as
  raw text via pair-encoding, not independently templated.
- **`HierarchicalBatchSampler`** resamples `n_tasks_per_batch` tasks with replacement every
  batch, forever (no epoch boundary); training is driven by `max_steps`. An epoch-bounded,
  `randperm`-based pass (like the reference) would define an absurdly short "epoch" at today's
  task count.
- **Task data location**: real GSM8K tasks are read in place from `/home/dg793/text-to-lora/tasks`
  via `--tasks-root`; nothing is copied into this repo. Nothing in the pipeline assumes a fixed
  or small task count -- `discover_tasks` is glob-based specifically so more tasks/domains can
  be added later with zero code changes.
- **CPU verification scope**: no synthetic end-to-end pipeline driver script exists. Confidence
  in the CPU-only default path comes entirely from per-module unit tests (data, canonicalize,
  oracle, recon, sft, validation, ablation), not a single driver script.
- **`steering_margin`** is reported against both the `train_descs` and `eval_descs`
  denominators (when available) rather than picking one canonical "correct desc" -- docs
  below name a generic denominator and picking a single one would hide the D-axis-unavailable
  case.
- **Oracle storage**: `outputs/oracle_loras/<task>/` (raw PEFT format, scored as-is by
  `validation.py`'s `oracle` condition) and `outputs/oracle_loras_canon/<task>.pt`
  (canonicalized, for `recon.py`) are kept as separate artifacts from separate scripts
  (`train_oracle_loras.py` / `canonicalize_oracles.py`), since Stage B is independently
  rerunnable from Stage A. `trainers/recon.py` itself re-derives canonicalization on the fly
  from the raw adapters via `oracle.canonicalize.load_and_canonicalize_oracle` rather than
  reading the `.pt` cache, so the `.pt` files are a diagnostic artifact, not a hard dependency.
- **`per_sequence_normalized_ce`** lives in a new shared `src/steerable_t2l/losses.py` (used by
  both `trainers/sft.py` and `validation.py`), rather than only in `trainers/sft.py`, since
  validation must reuse the identical loss definition training uses.
- Handoff gotchas #1/#3 (`zero_init = init_from is None`, the `target_spec` equality assert)
  are the *caller's* responsibility (`scripts/train_sft.py`), not `trainers/sft.py::train_sft`
  itself -- mirrors `trainers/recon.py::train_recon`'s "caller builds the model" shape.

Everything in `01_env.md` and `02_model.md` is implemented and tested; build on that.

---

## v2 dataset run (2026-08-04) — non-`<think>`, 10-domain textgrad-repro data

**Motivation**: the v1 real-run data was ~98% `<think>`-prefixed reasoning traces from a single
domain (GSM8K), on the hypothesis that this caused excess distribution shift between the base
model's natural output and the oracle/SFT training targets. A new raw dump at
`data/textgrad_repro/` (10 domains: `gsm8k`, `aqua`, 8x `bbh_*`; no `<think>` tokens) was
converted into this pipeline's task format by `scripts/build_tasks_from_textgrad_repro_v2.py`
and run through the **exact same** pipeline (`run_03b_training_validation_v2.sh`, mirroring
`run_03_training_validation.sh` stage-for-stage) into fully parallel `*_v2`-suffixed outputs
(`outputs/{oracle_loras,oracle_loras_canon}_v2`, `outputs/checkpoints/{recon,sft_scratch,
sft_warmstart}_v2`, `data/splits_v2.json`). The v1 outputs and `data/textgrad_repro/` itself
were never touched — `data/textgrad_repro/` is read-only to the whole v2 pipeline.

Non-obvious data-integrity fix made while building the v2 tasks: `iterations.jsonl`'s
`val_accuracy` can be a stale, carried-forward value from an earlier round when the textgrad
optimizer reverts to a previous best prompt without regenerating `forward_outputs.jsonl` for
that round. Picking "best iteration" via `max(val_accuracy)` therefore sometimes points at a
round whose logged rows are mostly wrong answers. Fixed by deriving the best iteration from
real per-iteration correctness counts in `forward_outputs.jsonl` directly
(`best_iteration_from_forward_outputs` in the build script).

Splits: 10 tasks total, 8 trained + 2 held out entirely on the T axis
(`textgrad_repro_v2_bbh_hyperbaton`, `textgrad_repro_v2_gsm8k` — used to test generalization to
fully unseen tasks). Unlike v1 (every task had exactly one description, so `d_holdout` was
always `[]`), several v2 tasks have 2-8 distinct textgrad-optimized prompt phrasings, so the D
axis is non-trivial for the first time.

**Real-run result (`configs/sft.yaml`/`sft_warmstart.yaml`, 2000/2000 steps, same as v1's
scaled-down step count, via `scripts/run_ablation.py`), v1 vs. v2 side by side:**

| | v1 scratch | v1 warmstart | v2 scratch | v2 warmstart |
|---|---|---|---|---|
| `val_loss(base)` / `val_loss(oracle)` | 1.028 / 0.547 | (same) | 0.712 / 0.528 | (same) |
| `steering_margin` vs. gibberish (avg/tasks) | 0.053 | 0.408 | 0.017 | **0.641** |
| `steering_margin` vs. other-task (avg/tasks) | 0.072 | 0.469 | 0.063 | **0.858** |
| `val_loss(train_descs)` | — | 0.177 | 0.021 | 0.018 |
| `val_loss(unseen_task_descs)` (T-holdout) | n/a (no T-holdout in v1) | n/a | 0.028 | 0.022 |

Takeaways:
- **Base loss dropped substantially** (1.028 → 0.712), consistent with the `<think>`-token
  distribution-shift hypothesis — the base model's natural continuations are closer to the
  cleaner v2 targets. Oracle's absolute loss is similar (0.547 → 0.528), so the raw
  base-vs-oracle *margin* shrank (0.481 → 0.184), but this is base improving, not oracle
  regressing.
- **The scratch-vs-warmstart pattern replicates and sharpens**: the from-scratch arm again
  collapses toward a near-constant, instruction-ignoring LoRA (tiny steering margin), while the
  recon-warm-started arm's margin is not just present but *larger* than in v1 (0.64-0.86 vs.
  0.41-0.47) — the warmstart/scratch gap widened from ~6-8x to ~38-50x on the cleaner data.
- **New: T-axis generalization is measurable for the first time.** Loss on the two fully unseen
  tasks (`unseen_task_descs`) sits close to in-distribution `train_descs` loss (0.022-0.028 vs.
  0.018-0.021) for both arms — the hypernetwork generalizes to task descriptions for tasks it
  never trained on, not just unseen phrasings of known tasks.
- `val_loss(train_descs)` being far below `val_loss(oracle)` in both v1 and v2 is expected, not
  a red flag: SFT directly optimizes CE on exactly these (task, desc) pairs for 2000 steps,
  while each oracle LoRA is independently early-stopped (patience-based, ~125-200 steps) — the
  two are not fit under comparable stopping criteria.

**Downstream accuracy (docs/04) status**: not part of this run. `run_03b_training_validation_v2.sh`
only covers loss-based validation (docs/03's scope); real generation + exact-match scoring is a
separate script (`scripts/eval_downstream_accuracy.py` / `run_04_downstream_eval.sh`). A v1
downstream-accuracy run exists at `outputs/eval/downstream_accuracy.json` (2026-08-03) but is
**incomplete** — only 2 of 13 v1 tasks have recorded results and `"overall"`/`"comparisons"` are
both `null`, i.e. that run was started once and never finished or resumed. No downstream-accuracy
run has been attempted yet for v2; `run_04b_downstream_eval_v2.sh` (mirroring this section's
pipeline, pointed at `outputs/checkpoints/sft_warmstart_v2`, `data/splits_v2.json`,
`outputs/oracle_loras_v2`) exists but has not been run.

---

## v3 dataset run (2026-08-07 to 2026-08-11) — LoRA-per-description architecture; **reverses the v1/v2 "warmstart beats scratch" conclusion**

> ⚠️ **This section contradicts v1's and v2's headline finding above.** In v1/v2, recon
> warm-start was clearly the arm that worked (steering margin 6-50x larger than from-scratch) and
> §5 step 3 / docs/04 §12 point 4 leaned on that result to argue the recon stage is load-bearing.
> On the v3 data, recon warm-start collapses instead, and **from-scratch SFT is now the arm that
> shows real steering** — see docs/04 §14 for the same reversal confirmed at the downstream
> accuracy level, not just this section's loss-based metric. Do not assume `sft_warmstart_v3` is
> the checkpoint to trust just because that pattern held in v1/v2.

**What changed in v3**: `scripts/build_tasks_from_{textgrad,gepa}_repro_v3.py` build one task
dir / one oracle LoRA per *distinct instruction* seen in a task's optimization trajectory
(`<algo>_repro_v3_<task>_d<K>`), not one task dir per task name with a single winning instruction
(v2's scheme). This gives 576 total v3 task dirs across both algorithms — but stripping the
`_d<K>` suffix leaves only **27 genuinely distinct underlying task domains** (`gsm8k`, `aqua`,
18x `bbh_*`, `commonsenseqa`, `gpqa_main`, `mmlu_all`, `multiarith`, `strategyqa`, `trec`); the
other ~549 dirs are near-duplicate prompt-optimization iterations of those same 27 tasks, each
carrying exactly one description (zero within-task paraphrase augmentation — the D-axis stays
universally n/a for v3, same degenerate case already noted for legacy single-description tasks).
Oracle adapters are trained on as few as 50 rows each (`--min-samples 50`).

### Real-run result: recon never learns, even on its own training data

`outputs/checkpoints/recon_v3/latest.pt`'s full logged history (2000 steps, `configs/recon.yaml`):

| step | train_loss | cosine_similarity (pred ΔW vs. oracle ΔW) | normalized_l1_model | normalized_l1_mean_baseline |
|---|---|---|---|---|
| 100 | 1.068 | 0.0001 | 0.0003 | 0.0003 |
| 600 | 0.989 | 0.025 (best point, transient) | 0.0003 | 0.0003 |
| 700 | 1.644 | 0.000 (collapses back) | 0.0005 | 0.0003 |
| 2000 | 1.000 | 0.0002 | 0.0003 | 0.0003 |

`normalized_l1_model` equals `normalized_l1_mean_baseline` at **every** logged step — the model
never beats "predict the per-module mean of the oracle targets" on its own training regression
targets. This is a fit failure, not a generalization gap.

**Tensor-level diagnosis** (comparing `recon_v3`'s saved `heads.*` weights to their zero-init
starting point, see `hypernet.py::SteerableHyperLoRA._apply_zero_init`'s own docstring for the
"fatal... shows up only as a flat loss curve" hazard this class already names): `out_A.weight`/
`out_B.weight` (the pathways that let the *input* affect the output) moved substantially away
from zero (norm ≈1.4-2.9) — so this is not the literal "B pinned at exactly zero" dead-gradient
case. But `bottleneck.weight` (the layer receiving the actual description-conditioned query
representation) is still at ≈99% of its random-init norm (10.12 measured vs. ≈10.24 expected at
init) after the full run, while the **bias** terms of `out_A`/`out_B` (description-*independent*
constants) moved the most (norm 0.33-0.85). Signature of the model shifting its output toward the
population mean via the bias pathway while the input-dependent weight pathway stayed essentially
unlearned.

### SFT ablation: scratch learns real steering; warmstart is pinned at exactly zero

`steering_margin` trend read directly from `outputs/checkpoints/{sft_scratch,sft_warmstart}_v3/latest.pt`'s
logged history (`configs/sft.yaml`, 2000 steps, `val_freq=500`; entries = vs_gibberish/vs_other_task):

| step | sft_scratch_v3 | sft_warmstart_v3 |
|---|---|---|
| 500 | +0.088 / +0.100 | −0.000 / +0.000 |
| 1000 | +0.116 / +0.148 | +0.000 / −0.000 |
| 1500 | +0.164 / +0.195 | +0.000 / +0.000 |
| 2000 | **+0.205 / +0.226** | **+0.000 / −0.000** |

Scratch steadily *learns* to depend on the description across all four checkpoints. Warmstart is
pinned at noise-level (0.0000-0.0001) the entire run — warm-starting from the collapsed recon
checkpoint poisons SFT badly enough that 2000 steps of a real downstream CE loss (which *does*
teach steering from a from-scratch init, per the scratch column) never recovers it.

### Why: dataset scale, not an unworkable design

Compared against the reference T2L (`/home/dg793/text-to-lora/README.md`):

| | reference T2L (Sakana) | v3 (this repo) |
|---|---|---|
| distinct task domains | ~479 (`--n_train_ds=479`) | 27 |
| description paraphrases / task | 128 (`--n_descs_per_ds=128`) | 1 |
| total (desc, target) pairs for recon | ~61,000 | 576 |
| recon training budget | up to 10,000 epochs (~5 days/H100) | 2,000 steps (~a few hours) |
| conditioning encoder | frozen small sentence-embedding model (`gte-large-en-v1.5`) + linear head | full 3B causal LLM (Qwen2.5-3B) + trainable LoRA + refiner + shared decoder (158M trainable params) |

v3's task-dir count (576) looks superficially close to reference's task count (479), but it is
~3 orders of magnitude short on genuine (task × paraphrase) diversity — the 576 v3 dirs are 27
real tasks × ~21 near-duplicate wording iterations apiece, not 479 genuinely different domains ×
128 deliberately-varied paraphrases. Reference's paraphrase augmentation is specifically what
forces phrasing-invariant, content-sensitive conditioning to be learned at all; v3 has none. The
architecture also swaps reference's cheap frozen-embedding + linear head for a much
higher-capacity, harder-to-fit encoder, which needs *more* data/steps to reliably learn real
conditioning, not less — compounding the scarcity rather than compensating for it. The
architecture itself is not unworkable: the from-scratch SFT arm, using the identical zero-init
mechanism and identical downstream loss, does learn real (if still modest) steering with this
exact codebase — see docs/04 §14 for the confirmation at the downstream-accuracy level and the
current recommendation to trust `sft_scratch_v3` over `sft_warmstart_v3`.

---

## 1. Data pipeline — domain-general from day one

### Format (fixed — this is what the user supplies)

`tasks/<task_name>/metadata.yaml`:

```yaml
descriptions:                 # >= 1 steering instruction; multiple = paraphrases of one behavior
- 'You will answer a mathematical reasoning question. Think step by step. ...'
domain: math                  # OPTIONAL, new. Grouping/splitting/reporting only — never model input.
ds_kwargs:
  path: json
  data_files: /abs/path/to/<task_name>.jsonl
  split: train
response_field: response
system_message: ''
user_prompt_template: '{question}'
assistant_prefill: ''
```

plus a jsonl of `{"question": ..., "response": ...}` rows.

Reference examples: `/home/dg793/text-to-lora/tasks/textgrad_repro_gsm8k_00/metadata.yaml` and
`/home/dg793/text-to-lora/data/textgrad_repro_gsm8k_t2l/textgrad_repro_gsm8k_00.jsonl`
(13 tasks, ~5.3k rows total; each task is a distinct TextGrad-optimized GSM8K system prompt, and
the `response` fields were generated by Qwen3-32B).

### Requirements

- **Discovery by glob**, not by hand-listed arrays: `train_tasks: ["textgrad_repro_gsm8k_*", "code_*"]`.
  Today there is one domain; assume many arrive later in exactly this shape.
- **`user_prompt_template` fields are arbitrary.** Format with `**row`, so a task whose jsonl has
  `{"passage":..., "query":..., "answer":...}` works with no code change. Do **not** port the
  reference's `get_preprocessing_fn` name-prefix dispatch (`lol_*`, `arc_*`, …) — per-task field
  mapping belongs in `metadata.yaml`, not in a Python `if`-chain.
- **Chat formatting, completion mode**: `system + user → prompt`,
  `assistant_prefill + response → response`, each through `tokenizer.apply_chat_template`.
- **Label masking** via pair encoding: `tokenizer(prompt, response, add_special_tokens=False,
  truncation=True, max_length=inp_max_len)` then `sequence_ids(i) == 0 → -100`. Pad positions
  masked to `-100` in the collator.
- ⚠️ **`system_message` stays empty.** The steering instruction is the *hypernetwork's* input and is
  deliberately absent from the target model's context — otherwise the target could just read the
  instruction and the LoRA would be doing nothing. This is the whole experiment.
- **Cache** tokenized datasets under `data/.cache/<sha256 of metadata + tokenizer + args>`.

### ⚠️ Do this before choosing `inp_max_len`

The current GSM8K responses embed Qwen3-32B `<think>...</think>` blocks and are long. Write
`scripts/profile_lengths.py` and set `inp_max_len` from the measured distribution. Log truncation
counts as a training metric — silent truncation would train on cut-off reasoning and look like a
model problem.

### Batching

`HierarchicalBatchSampler`: draw `n_tasks_per_batch` tasks, then `n_points_per_task` random rows
from each; independently draw one of the task's descriptions per row.

**Descriptions can no longer be cached across steps.** The reference pre-embeds every description
offline with a frozen encoder and then deletes it; here the backbone is LoRA-tuned, so its output
changes every step. Tokenization *can* be cached once at startup.

**Deduplicate descriptions within a batch, end to end** — the backbone **and** the heads run on the
`U` unique descriptions, then a differentiable `index_select` expands to `bs`:

```python
uniq, inv = dedup(batch["descs"])      # inv: [bs] -> [0, U)
h  = hypernet.encode(uniq)             # 3B forward on U, not bs
ab = hypernet.heads_forward(h)         # heads on U, not bs
A, B = ab[m][0][inv], ab[m][1][inv]    # backward scatter-adds correctly
```

Head compute is ≈5.6 GFLOP/sample, so expanding before the heads wastes ~90 GFLOP/step at `bs=16`.
The expanded tensor is ~11 MB. `tests/test_dedup.py` already proves the deduped path is
numerically identical to the naive one — reuse it.

Because hypernet cost scales with `U` while target cost scales with `bs`, start at
`n_tasks_per_batch=4, n_points_per_task=4` (`bs=16, U=4`) with `grad_accum=4`, so each optimizer
step still sees ≥16 distinct tasks.

### Memory budget (1× B200, 180 GB) at `bs=16, L≈1024`

| | GB |
|---|---|
| backbone Qwen2.5-3B bf16, frozen (`AutoModel`, tied embeddings) | 6.2 |
| target Qwen2.5-1.5B bf16, frozen | 3.1 |
| trainable params + grads + AdamW (≈136 M fp32) | 2.2 |
| backbone activations (gradient checkpointing **on**) | ~0.5 |
| target activations, 28 layers (checkpointing **off**) | 12–18 |
| logits + CE over 151,936 vocab | ~7 |
| generated A/B + hook intermediates | <0.1 |
| **total** | **≈32–38** |

Comfortable. Raise `bs` toward 32 once stable; the CE term is what eventually binds.

---

## 2. Oracle LoRAs and SVD canonicalization

### Stage A — `scripts/train_oracle_loras.py`

One vanilla PEFT LoRA per task on the target model, `r=8, alpha=16, use_rslora=False,
lora_dropout=0.0, target_modules=(q,k,v,o)_proj` — **identical to `TargetSpec`**, asserted. Uses the
same data path as SFT so oracle and hypernetwork see byte-identical text. Early stopping on the
task's own validation split. Output: `outputs/oracle_loras/<task>/`. Embarrassingly parallel across
tasks, so it fans out cleanly however you schedule GPU jobs.

### Stage B — `src/steerable_t2l/oracle/canonicalize.py`

`ΔW = B·A` is not unique: for any invertible `R`, `(BR)(R⁻¹A)` is the same function. Independently
trained oracles land in arbitrary, mutually inconsistent bases, so regressing onto raw `A`/`B` is
ill-posed. Canonicalize each adapter first.

⚠️ **Convention flip — read this twice.** `Design.md` writes `ΔW = A·B` with `A_canon = UΣ^½`,
`B_canon = Σ^½Vᵀ`. **PEFT uses `ΔW = lora_B · lora_A`** with `lora_A: [r, in]`, `lora_B: [out, r]`.
The correct mapping is therefore **swapped**:

```
lora_B_canon = U  Σ^½        # [out, r]
lora_A_canon = Σ^½ Vᵀ        # [r, in]
```

Getting this backwards *silently works* on the square `q_proj`/`o_proj` and trains on garbage.

Do not run a full `[1536, 1536]` SVD — exploit the existing factorization for an exact `O(d·r²)`
result, in **float64** (`Rb @ Raᵀ` is a product of two ill-conditioned triangular factors):

```python
Qb, Rb = torch.linalg.qr(B.double())        # [out, r], [r, r]
Qa, Ra = torch.linalg.qr(A.double().T)      # [in,  r], [r, r]
U0, S, Vh0 = torch.linalg.svd(Rb @ Ra.T)    # tiny r×r
U, Vh = fix_svd_signs(Qb @ U0, Vh0 @ Qa.T)
s = S.clamp_min(0).sqrt()
A_canon, B_canon = (s[:, None] * Vh).float(), (U * s[None, :]).float()
```

**Fix the sign gauge.** `(u_i, v_i)` and `(−u_i, −v_i)` are both valid SVD components. Without a
deterministic rule, two oracles encoding the *same* function still land on opposite-sign targets,
which defeats the entire point of canonicalizing. Rule: make the largest-magnitude entry of each
left singular vector positive.

Do **not** rescale by `scaling` — both sides use the same PEFT config (asserted), so canonicalizing
the raw `B @ A` is consistent. Log the singular-value spectrum per adapter; near-tied values mean an
unstable rotation in the tied subspace.

Output keeps PEFT key names so `peft.load_peft_weights` reads it unchanged.

### Stage C — `src/steerable_t2l/trainers/recon.py`

- Batch = `(task, description)` pairs; **no target-model forward at all**, so this stage is fast and
  can use a large batch. Build `TargetSpec.from_pretrained(...)` from `AutoConfig` only and never
  load target weights (saves 3.1 GB).
- Loss = per-(module, role) magnitude-normalized L1:
  `L1(A, A_t)/A_t.abs().mean().detach() + L1(B, B_t)/B_t.abs().mean().detach()`.
  A and B magnitudes differ by orders of magnitude across modules; without normalization one module
  dominates.
- ⚠️ Store the normalizers **in the args yaml, not as module buffers.** That is what keeps the recon
  and SFT `state_dict`s structurally identical so the handoff can load `strict=True`. (The
  reference stores them as buffers and is forced into `strict=False` loads with silent drift.)
- Under this loss, A and B are supervised independently, so both gradients are nonzero even with
  `B = 0` at init. No special-casing — one init path serves both stages.

**Stated expectation.** 13 oracle adapters is a very small regression set. This stage is a *warm
start*: it teaches the query/head structure and moves `B` off zero. It cannot teach instruction
generalization. Its only success criterion is the ablation in §4. It becomes genuinely valuable once
more domains exist.

---

## 3. SFT training and the handoff

`src/steerable_t2l/trainers/sft.py`:

```python
uniq, inv = dedup(batch["descs"])
h  = hypernet.encode(uniq)
ab = hypernet.heads_forward(h)
sites = build_sites(spec, {m: (A[inv], B[inv]) for m, (A, B) in ab.items()})
with lora_hooks(target, sites, spec.scaling):
    loss = target(**batch).loss
loss = loss + l2_reg_generated_w * (A.pow(2).mean() + B.pow(2).mean())
```

- `accelerate` with `mixed_precision="bf16"`; backbone gradient checkpointing on, target off.
- Loss: shifted CE over response tokens only, with per-sequence length normalization so long-answer
  tasks don't dominate. Plus `l2_reg_generated_w ≈ 1e-3` on the generated weights.
- ⚠️ **Do not reproduce the reference's grad-accum bug** (`sft_trainer.py:245-249` calls
  `optimizer.zero_grad()` inside the accumulate block right before `backward`, silently negating
  accumulation). Zero *after* `optimizer.step()`.
- Trainable set per `Design.md`: backbone LoRA + query bank + refiner + shared decoder + heads.
  Backbone base weights and the entire target model stay frozen.

### Checkpoint format (both stages)

```python
{"state_dict": ..., "hypernet_config": ..., "target_spec": ..., "stage": "recon"|"sft", "step": ...}
```

`load_hypernet(path)` reconstructs with `zero_init=False` and loads `strict=True`.

### Handoff gotchas

1. `zero_init = (args.init_from is None)`. A warm-started run must not re-zero its recon-trained
   heads — that discards the entire recon stage.
2. **LR must drop hard.** Recon trains from scratch at ~5e-4; warm-started SFT at that LR destroys
   the recon solution within ~100 steps. Use `lr=2e-5`, `warmup_frac=0.03`, and param groups with
   the backbone LoRA ~10× lower than the heads. Log L2 drift from the init checkpoint.
3. `assert loaded["target_spec"] == asdict(current_spec)` — an oracle trained with different modules
   or rank cannot warm-start.
4. Optional: freeze the backbone LoRA for the first ~500 SFT steps so the heads adapt to the SFT
   objective before the conditioning distribution shifts underneath them.

---

## 4. Validation — loss-based only

**No generation, no vLLM, no task-accuracy harness.** Every metric below comes from the same forward
pass as training, which makes it cheap enough to run every `val_freq` steps.

⚠️ **This means nothing here measures whether the model actually gets the right final answer.**
Lower validation loss is suggestive of better generation, not proof of it. "Does the T2L-generated
LoRA solve more GSM8K problems than the base model, or than just prompting the base model with the
same instruction directly (no LoRA at all)?" is a real, different question this section cannot
answer -- see `docs/04_downstream_eval.md` (spec only, not yet implemented) for the exact-match,
generation-based accuracy evaluation that does.

### Three orthogonal held-out axes (`scripts/make_splits.py` → seeded `splits.json`)

| axis | construction | tests |
|---|---|---|
| **Q** — held-out questions | 10 % of each task's rows | in-task fit |
| **D** — held-out descriptions | hold out ≥1 paraphrase per task | instruction generalization |
| **T** — held-out tasks | whole tasks withheld (rows *and* descriptions); once several domains exist, also a held-out **domain** | zero-shot instruction → LoRA |

⚠️ **The D axis needs ≥2 descriptions per task and today every v2/v3/v4 task has exactly 1.**
Handle it in this order: (a) the split code degrades gracefully and reports D-metrics as `n/a` with
a warning; (b) **done** — `scripts/paraphrase_descs.py` (local Qwen generation, ~8 paraphrases per
task, with contrastive-sibling dedup so paraphrases don't blur across tasks); see
`docs/06_description_augmentation_v5.md` for the design and its `v5` namespace (kept fully
separate from `v3`'s own task dirs — see that doc for why); (c) once new multi-domain tasks
arrive, author several descriptions per task up front. Until D exists for a given task family,
`other_task_descs` and `gibberish_descs` still give a valid steering signal.

### Seven description conditions, all scored on the same held-out questions

| condition | expected val loss | detects |
|---|---|---|
| `base` — no LoRA | reference floor | — |
| `oracle` — the task's own trained LoRA | reference ceiling | remaining headroom |
| `train_descs` | ≈ oracle | in-distribution fit |
| **`eval_descs`** — held-out paraphrase, seen task | **headline number** | description generalization |
| `unseen_task_descs` — T-split | < `base`, ideally well under | true zero-shot |
| `other_task_descs` — another task's instruction | **worse** than `train_descs` | is the output actually conditioned on content? |
| **`gibberish_descs`** | **≈ `base`** | **the critical control** |

`gibberish_descs`: token-shuffled real description, lorem ipsum, random unicode. The reference's
`configs/textgrad_repro_gsm8k.yaml` has three ready-made strings under `additional_eval_descs`.

### Primary metric: steering margin

```
steering_margin = val_loss(other_task_desc) − val_loss(correct_desc)
                  and  val_loss(gibberish)  − val_loss(correct_desc)
```

**Positive and growing is the entire thesis.** If it stays ≈0, the hypernetwork has collapsed to
emitting one constant good multi-task LoRA and is ignoring its input — the most common silent
failure mode of hypernetwork training, and it is **invisible in raw validation loss**.

⚠️ **Select checkpoints on steering margin, not on minimum val loss.** Minimum val loss is
*maximized* by exactly the collapsed solution this metric exists to detect.

### Secondary metrics

- `val_loss(eval_descs) − val_loss(train_descs)` — description-generalization gap. **Implemented**
  (`validation.steering_margin`'s `eval_descs` denominator), but always `n/a` today since every
  task has exactly one description (the D-axis limitation above).
- Per-task and per-domain val loss. **Implemented** (`validation.run_validation`'s `per_task`/
  `per_domain` dicts).
- Token-level perplexity; response-token accuracy. **NOT implemented.** Both were planned here but
  never built -- `validation.py` only ever computes the single per-sequence-normalized CE
  (`losses.per_sequence_normalized_ce`) used for training and `steering_margin`, not a separately
  reported token-averaged perplexity or a next-token-match-rate accuracy. Real exact-match/
  generation-based accuracy (the thing "response-token accuracy" was meant to approximate) is
  scoped to `docs/04_downstream_eval.md` instead, not retrofitted here.
- `val_loss(hypernet) − val_loss(oracle)` per task — fraction of oracle headroom recovered.
  **Implemented** (`overall`'s `oracle` vs. `train_descs`/`eval_descs` entries, when `oracle_dir`
  is given to `run_validation`).
- Recon stage only: normalized L1 against a "predict the per-module mean target" baseline; cosine
  similarity of `ΔW_pred` to `ΔW_oracle`. **Implemented** (`trainers/recon.py::evaluate_recon`).

### Required ablation

SFT from scratch (`zero_init=True`, `lr=1e-4`) vs. recon-warm-started, on an identical validation
grid. This is the only justification for the oracle/recon stage existing.

---

## 5. Suggested build order

1. `data/` (registry, metadata, formatting, datasets, splits) + `profile_lengths.py`; set `inp_max_len`.
2. `validation.py` + `make_splits.py`; record `base` reference losses before training anything.
3. `oracle/` (+ a canonicalization test: two random reparameterizations `(R⁻¹A, BR)` of the same
   `ΔW` must canonicalize to identical `(A_c, B_c)`) → 13 oracle LoRAs → confirm `oracle` val loss
   beats `base`. If it doesn't, the LoRA hyperparameters are wrong and recon has nothing to learn.
4. `trainers/recon.py` + `scripts/train_recon.py`.
5. `trainers/sft.py` + `scripts/train_sft.py`, warm-started.
6. Ablation.
