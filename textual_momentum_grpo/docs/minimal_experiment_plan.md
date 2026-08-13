# Minimal experiment: build & run plan

Operational companion to `textual_momentum_grpo_README.md` §3–§7 (the design doc). This doc turns
the go/no-go minimal experiment (README §4) and experimental setup (README §5) into a concrete
build order, checklist, and open-question list for whoever implements this next. Nothing in
`textual_momentum_grpo/` is built yet — this doc is the plan, not a record of what exists.

**Framework: verl, per README §5.** The sibling `self_correct_grpo/` project vendors a different
RL stack (ICRL/slime — Megatron + SGLang + Ray) for its own GRPO work in this repo. That choice
was made for a different project and is *not* reused here — this is a deliberate, confirmed
decision, not a default to silently revisit. Do not port infra from `self_correct_grpo/` without
a fresh reason; build against verl as the README specifies.

## 1. Arms and build requirements

Restated from README §4, with what each arm requires beyond the previous one. Components are
cumulative — each arm's harness is a strict superset of the one before it.

| Arm | Content | Internalization + Calibration | New components needed beyond the previous arm |
| --- | --- | --- | --- |
| (1) Floor | nothing | — | Stock GRPO training loop on verl + Qwen3-8B. Nothing else. This is the foundation every other arm builds on — get this training and evaluating cleanly first. |
| (2) Instance, OFF/OFF | same-iteration critique | OFF | Same-iteration critique generator (frontier-model call on the current rollout batch) + pooling of original + critique-refined rollouts into the GRPO group, matching Critique-GRPO's published pipeline exactly (sample → critique → refined rollout → pool). |
| (3) Instance, ON/ON | same-iteration critique | ON | Arm (2)'s critique generator, plus: unconditioned-scoring internalization (log-probs for the gradient computed without the critique in context) and the token-level calibration ratio `w_t` (README §3 step 3). |
| (4) Trajectory, OFF/OFF | textual momentum | OFF | Textual-gradient generator (frontier-model diagnosis of successes/failures per step) + incremental trajectory-digest maintainer (LLM-updated running summary, not raw concatenation) + momentum generator (produces the next-step directive `M_t`). Rollouts conditioned on `M_{t-1}`, scored under that same conditioned context — no internalization. |
| (5) Trajectory, ON/ON (Ours) | textual momentum | ON | Arm (4)'s full trajectory pipeline + the same internalization/calibration machinery built for Arm (3), applied to momentum-conditioned rollouts instead of critique-conditioned ones. |

**Build order: (1) → (2) → (3) → (4) → (5).** Each step de-risks one new piece in isolation.
Arm (2) additionally doubles as the Critique-GRPO reproduction sanity check (README §4 success
criteria) — do not trust results from arms (3)–(5) until arm (2)'s numbers are checked against
published Critique-GRPO results. The internalization + calibration component built for arm (3)
should be written generically enough that arm (5) reuses it unchanged, just applied to a different
conditioning context (momentum vs. critique).

## 2. Experimental setup detail (expanding README §5)

**Backbone:** Qwen3-8B — chosen as the one model common to both Critique-GRPO's original paper and
ICRL's reproduction, giving two independent reference points for the arm (2) sanity check.

**Data:**
- Training: **[Updated 2026-08-12, supersedes the "MATH, optionally augmented with NuminaMath"
  line below]** defaults to `open-r1/OpenR1-Math-220k` (`default` config, 93.7k rows), not MATH —
  this is the pool Critique-GRPO's published numbers (the arm (2) reproduction target) actually
  come from, and MATH alone left Qwen3-8B saturated from step 1 of arm (1)'s run. MATH remains an
  explicit opt-in via `TMGRPO_TRAIN_DATA=math`. See `textual_momentum_grpo_README.md` §5 and
  `docs/build_and_run_guide.md` for the full rationale and the prep commands for both pools.
  ~~Training: MATH training split, optionally augmented with NuminaMath.~~ (original plan)
- Eval: MATH500 (full) + a held-out slice of OlympiadBench/AIME24. Unchanged by the above — eval
  is always on these sets regardless of which training pool an arm uses.
- Open (not decided here): the exact size/seed of the OlympiadBench/AIME24 held-out slice (now
  resolved elsewhere — `scripts/prepare_eval_data.py` uses a 200-row seed=0 sample; see that
  script). The NuminaMath-augmentation-size question below is superseded by the OpenR1 default.

**Compute/memory budget:** single B200 (~180–192GB HBM3e) via verl. Full-precision AdamW does not
fit alongside a frozen reference-policy copy (~144GB before activations/KV cache/rollout). Plan:
verl's bf16 training + 8-bit or Adafactor optimizer + gradient checkpointing + colocated vLLM
rollout. Fallback levers if still tight: reduced GRPO group size or reduced response length.
Single-GPU means no data/tensor parallelism — wall-clock throughput, not memory fit, is the
expected bottleneck, which is an accepted tradeoff for a go/no-go pass (revisit only if scaling up
later).

**Frontier-model role (open question — not pinned by the README):** a fixed frontier model,
separate from the trained policy, is used for:
- arm (2)/(3)'s same-iteration critique,
- arm (4)/(5)'s textual gradient (per-step diagnosis),
- arm (4)/(5)'s trajectory-digest update,
- arm (4)/(5)'s momentum generation.

None of the following are decided yet and need a call before/while implementing: which specific
frontier model/API to use, call cadence (every RL step vs. every K steps — README §6 flags cost as
an open risk to report alongside gains), and the actual prompt templates for critique/textual
gradient/digest-update/momentum-generation. Self-generated feedback (shared weights) is explicitly
out of scope for this pass (README §3, "Phase 1").

## 3. Implementation checklist

In build order (§1):

1. verl environment setup + Qwen3-8B loading; confirm the base GRPO loop trains and evaluates
   (arm 1) before adding anything else.
2. Training/eval data prep: **[Updated 2026-08-12]** OpenR1-Math-220k train pool by default (MATH
   opt-in, see §2's Data section above), MATH500 eval, held-out
   OlympiadBench/AIME24 slice; reward/verifier wiring (exact-match/equivalence checker — chosen
   per README §5 over agentic-env reward for its simplicity and lack of environment-server
   overhead).
3. Same-iteration critique generator + pooled-rollout GRPO grouping (arm 2). Validate against
   published Critique-GRPO numbers — this is the gate described in README §4's success criteria;
   do not proceed to arm (3) until this passes.
4. Unconditioned-scoring internalization + token-level calibration ratio `w_t` (README §3 step 3),
   built generically enough to be reused unchanged by arm (5).
5. Trajectory-digest maintainer (incremental LLM summarization) + momentum generator (arm 4).
6. Wire the arm (3) internalization/calibration component onto the trajectory pipeline (arm 5).
7. Eval harness: unconditioned evaluation (no directive/critique at test time, for all 5 arms) on
   MATH500 + the held-out OlympiadBench/AIME24 slice.

## 4. Metrics/logging plan

Every run should log enough to check README §4's success criteria and §6's open risks after the
fact, without needing to rerun anything:

- Per-arm unconditioned eval accuracy (MATH500 + held-out slice) — the primary signal for all
  three success criteria.
- `w_t` trajectory over training, for arms (3) and (5) — diagnostic for calibration-ratio collapse
  toward uniformly low values (README §6).
- Rollout entropy/diversity over training — long-horizon drift risk (README §6): even a fixed
  frontier model's feedback is shaped by the policy's own rollouts over time.
- Frontier-model call count and cost per run, for arms (2)–(5) — README §6 asks this be reported
  alongside performance gains, not treated as a footnote.
- A spot-check protocol comparing a sample of textual gradients against ground-truth batch
  statistics, for arms (4)/(5) — README §6 flags textual gradients may be confabulated rather than
  causally accurate; this needs a defined (if lightweight) check, not just an assumption.

## 5. Success criteria (restated from README §4)

- Arm (2) roughly matches published Critique-GRPO numbers — reproduction check, gates trust in
  everything else.
- Arm (5) > Arm (4) — internalization + calibration meaningfully helps over naive
  momentum-conditioning.
- Arm (5) > Arm (3) — trajectory content beats instance content at matched mechanism. This is the
  core trajectory-vs-instance signal this whole pass exists to test.

## 6. Open questions to resolve before/while implementing

Collected from above, so they aren't silently assumed during implementation:

- Frontier model choice, call cadence, and prompt templates (§2) for critique / textual gradient /
  digest update / momentum generation.
- ~~Exact NuminaMath augmentation size (if used) and exact OlympiadBench/AIME24 held-out slice
  size/seed (§2).~~ **[Resolved 2026-08-12]** superseded by the OpenR1-Math-220k default (§2); the
  held-out slice is a 200-row seed=0 sample (`scripts/prepare_eval_data.py`).
- `w_max` clip value for the calibration ratio (README §3 step 3).
- GRPO group size and response length actually used, once real memory numbers from arm (1)/(2) are
  in hand (§2's fallback levers).

## 7. Non-goals for this pass

See README §7 for the full deferred list: a fixed/random-directive control, an
internalization-ON/calibration-OFF ablation, Phase 2 self-generated feedback, and the
order-shuffle control. None of these block a go/no-go read and are not re-litigated here.
