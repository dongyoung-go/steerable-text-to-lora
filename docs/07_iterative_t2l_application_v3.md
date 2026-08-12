# 07 — Iterative TextGrad-style application of T2L (v3 pilot)

**Status: IMPLEMENTED, not yet run.** New code passes `ruff`/`pytest` on CPU (tiny fixtures, fake
critique/rewrite calls, no vLLM/GPU needed). The real pilot needs a B200 node (Qwen3-14B via vLLM
+ the v3 hypernet checkpoint/backbone + Qwen2.5-1.5B target, all resident at once — see Open items
below) and has not been run.

## Motivation

Everything built through `docs/03`–`docs/06` *evaluates* T2L: one fixed description goes in, one
fixed LoRA comes out, that LoRA is scored once against held-out rows
(`src/steerable_t2l/eval_accuracy.py`). Nothing exercises T2L the way its own training data was
actually produced — TextGrad's `solve → critique (textual gradient) → rewrite prompt → re-solve`
cycle (`scripts/textgrad_repro.py`), which is exactly how the descriptions T2L was trained to
condition on came to exist in the first place.

This experiment asks whether that cycle can run **online, at inference/application time**, with
T2L standing in for "the prompt": each round's feedback text is fed into T2L to generate a fresh
LoRA, that LoRA steers Qwen2.5-1.5B-Instruct's own generations, those generations are critiqued by
Qwen3-14B, and the cycle repeats. Does downstream accuracy actually improve round over round? This
is the natural test of whether T2L is useful for more than one-shot steering from a pre-baked
description — whether it can participate in closed-loop self-improvement the way the literal
prompt did during training-data generation.

## Design decisions and why

- **Checkpoint: `outputs/checkpoints/sft_scratch_v3/latest.pt`.** `docs/03`/`docs/04` §14
  established that `sft_warmstart_v3`'s recon warm-start collapsed on v3 data (near-flat,
  non-discriminating steering), while `sft_scratch_v3` is the only v3 checkpoint that shows a
  real, if modest, description-conditioned steering signal
  (`t2l_train_desc − prompted` = +0.068 to +0.103 macro). Building this experiment on the
  collapsed checkpoint would confound "does iterative refinement help" with "does this checkpoint
  steer at all."
- **Round 0 uses a literal prompt, no LoRA.** The target model solves with the task's own best
  training description (`eval_accuracy.condition_desc(..., "prompted", ...)`) injected as a
  literal system/user turn (`eval_accuracy.build_prompted_prompt`) — mirrors the existing
  `prompted` eval condition. This gives the first critique call real, concrete generations to
  react to before any LoRA exists, mirroring how real TextGrad needs an actual visible prompt to
  compute its first textual gradient.
- **Round 1+ never puts the instruction/feedback text into the target's own context.** This
  preserves the repo's core invariant (`data/formatting.py::format_example`,
  `TaskMetadata.__post_init__` enforcing `system_message == ""`): the target model only ever sees
  the bare question; all steering flows through the T2L-generated LoRA. Only T2L's `encode()` ever
  reads the current round's text.
- **LoRA composition: replace, not stack.** Each round's LoRA fully replaces the previous one —
  `base_model + LoRA_t`, never `base_model + LoRA_1 + LoRA_2 + ...`. The feedback/prompt text
  itself already carries the accumulated history (each round's rewrite is conditioned on the
  current text plus this round's critique), so a single LoRA per round already encodes the full
  trajectory. Stacking would also be architecturally novel — `hooks.lora_hooks` has never been
  exercised with multiple simultaneous LoRA sets — and would conflate "better feedback text" with
  "more LoRA capacity stacked on," making results harder to interpret.
- **Disjoint, fixed `feedback_rows`/`score_rows` per task, reused across every round.**
  `feedback_rows` are what the critique/rewrite calls see; `score_rows` are scored every round to
  produce the accuracy-vs-round curve and are never shown to the critique step, so the loop isn't
  trivially "solving" the exact rows it's graded on. Both are drawn once
  (`iterative_t2l.split_feedback_and_score_rows`) from the task's Q-axis held-out rows
  (`eval_accuracy.eval_rows_for_task`) and held fixed for the rest of that task's loop — reusing
  different rows each round would make the accuracy-vs-round curve confounded by which rows
  happened to be sampled, not just how good the current LoRA is.
- **Fixed rounds, no revert-on-worse for the pilot.** Unlike training-time TextGrad (which reverts
  to the prior prompt if val accuracy drops), this pilot always accepts each round's rewrite and
  keeps going. If `held_out_accuracy` degrades over rounds, that's itself an informative result
  about whether the critique/rewrite/T2L loop compounds errors — worth seeing directly before
  adding a revert mechanism that could mask it.
- **Two feedback-generation modes**, both via a single Qwen3-14B vLLM engine
  (`src/steerable_t2l/feedback_gen.py`, thinking off — same convention as
  `generate_comprehensive_feedback_v4.py`):
  - `mode="prompt"` (**pilot default**): rewrite the instruction text directly, TextGrad-shaped.
  - `mode="comprehensive_feedback"`: reuses `docs/05`'s exact merge-prompt template (previous
    guidance + new feedback → merged guidance paragraph). Implemented and unit-tested but **not
    run in this pilot** — v4 training/eval itself hasn't been run yet (`docs/05` status:
    "IMPLEMENTED, not yet run"), so there is no v4 checkpoint to compare against.
  - Both modes reuse one shared `critique()` call (a bespoke prompt, not literally
    `textgrad_repro.py`'s monkeypatched `BACKWARD_SYSTEM_PROMPT`, since that's tied to the
    `textgrad` library's internal autodiff plumbing — the library's `tg.BlackboxLLM` assumes the
    optimized variable is injected into every forward call, which is false here from round 1
    onward, so driving it directly would be more fragile than a bespoke loop).
- **Pilot scope: 3 tasks, 5 rounds.** `textgrad_repro_v3_gsm8k_d4` (integer answer),
  `textgrad_repro_v3_aqua_d9` (MCQ letter), `textgrad_repro_v3_strategyqa_d8` (yes/no) — all three
  are in `data/best_prompt_tasks_v3.txt`, spanning the three answer-parser shapes
  `eval_accuracy.classify_answer_parser` distinguishes. Small and fast to debug before spending
  B200 time on the full 38-task v3 suite.

## Files

| File | Role |
|---|---|
| `src/steerable_t2l/feedback_gen.py` | vLLM Qwen3-14B wrapper: `critique()` (per-round textual-gradient-style call) and `rewrite()` (prompt-mode or comprehensive-feedback-mode). |
| `src/steerable_t2l/iterative_t2l.py` | The round loop itself: `split_feedback_and_score_rows`, `run_iterative_t2l`. Reuses `eval_accuracy.generate_texts`/`build_prompted_prompt`/`classify_answer_parser`/parsers/`condition_desc`/`eval_rows_for_task` and `hooks.build_sites`/`lora_hooks` unchanged. |
| `scripts/eval_iterative_t2l_v3.py` | CLI driver — loads target/hypernet/vLLM feedback engine, runs the loop per task, writes a JSON report. |
| `run_05_iterative_t2l_v3.sh` | Runner: `bash run_05_iterative_t2l_v3.sh` (lint + CPU tests) / `--full` (real B200 pilot run). |
| `tests/test_iterative_t2l.py` | CPU-safe tests with fake `critique_fn`/`rewrite_fn` (no vLLM needed): round bookkeeping, replace-not-stack (asserts at most one `lora_hooks` context ever active), fixed/disjoint row pools, round-0-vs-round-1+ invariant, JSON round-trip. |

## How to run

```bash
bash run_05_iterative_t2l_v3.sh          # lint + tests only, CPU-safe
bash run_05_iterative_t2l_v3.sh --full   # the real B200 pilot run
```

Needs `run_03c_training_validation_v3.sh --full` already done (`data/splits_v3.json` and
`outputs/checkpoints/sft_scratch_v3/latest.pt`).

## Open items to confirm during the pilot

1. **Three-model GPU residency** (vLLM Qwen3-14B + HF Qwen2.5-3B hypernet backbone + HF
   Qwen2.5-1.5B target, all at once) hasn't been done anywhere else in this repo — `docs/04`'s
   eval explicitly avoids vLLM. `scripts/eval_iterative_t2l_v3.py` caps
   `--feedback-gpu-memory-utilization` at 0.5 (vs. `generate_comprehensive_feedback_v4.py`'s 0.85
   default) to leave room for the two HF models, but this hasn't been validated on real hardware.
2. **The critique/rewrite prompt wording is new** (not copied from TextGrad's own internal
   optimizer prompt) — review the actual generated critiques/rewrites from round 1–2 of the first
   pilot task before trusting later rounds; bad critiques will compound each round given the
   replace-not-stack design and the lack of a revert mechanism.
3. **No revert-on-worse-round.** If `held_out_accuracy` trends down or is simply noisy rather than
   informative, consider adding TextGrad's revert-on-worse-val logic in a v2 of this script.
4. **`comprehensive_feedback` mode is untested against real data** — no v4 checkpoint exists yet
   to pair with it.

## Result

Not yet run.
