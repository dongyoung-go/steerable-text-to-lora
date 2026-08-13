# Textual Momentum for On-Policy RL

# 1. Motivation

On-policy RL improves a policy from the outcomes of its current rollouts, but each update has limited access to the optimization history: what kinds of reasoning have repeatedly failed, which strategies have recently started working, and which directions are worth exploring next. Standard RL updates encode this history implicitly in the model parameters, while existing textual-feedback methods primarily provide feedback at the level of the current rollout or episode.

**Core idea:** maintain an explicit, semantic state of the optimization trajectory. After each RL step, an LLM summarizes the successes and failures observed in that step (the **textual gradient**) and incorporates them into a compact running optimization trace (the **trajectory digest**). Before the next rollout, another LLM reads this accumulated history and produces an exploration directive (the **textual momentum**) that captures what the policy should try next. Thus, the next rollout is conditioned not only on the current problem, but also on what the training process has learned so far.

Crucially, the resulting improvements should be internalized into the policy itself. Rollouts are generated with the history-derived directive, but the RL update evaluates their likelihood under the unconditioned policy. At test time, the directive is removed, allowing the policy to retain behaviors discovered through history-guided exploration without requiring an external textual controller.

We therefore study a central question: **Can an explicit semantic representation of cross-iteration optimization history improve on-policy exploration beyond feedback from the current batch, while the resulting gains are transferred into the underlying policy?**

This is not simply a matter of aggregating feedback across more instances (as in pooling many same-iteration critiques). The momentum generator is asked to reason over an *ordered* sequence of past diagnoses — to infer a direction (what has improved, what remains stuck, what has regressed) rather than to summarize an unordered set — and to extrapolate that direction into a directive for exploration. Whether the LLM is actually doing trend-reasoning, rather than just benefiting from having more content in context, is an empirical question we test directly (Sec. 4).

## 2. Related Work (nearest neighbors)

| Work | Cross-iteration memory | RL weight update | Internalization (conditioned→unconditioned) |
| --- | --- | --- | --- |
| TextGrad / ProTeGi | No | No | N/A |
| REVOLVE | Yes | No | N/A |
| Critique-GRPO | No (same-iteration) | Yes | No — pools original + critique-refined rollouts, each scored under its true generating context |
| ICRL (solver+critic) | No (per-episode) | Yes | Yes — token-level calibration ratio |
| REMO | Yes (mistake memory + meta-controller) | No (tunes prompts/hyperparams) | N/A |
| **This work** | **Yes, explicit trajectory** | **Yes** | **Yes, applied at trajectory level, not instance level** |

Gap: nothing combines cross-iteration textual momentum (REVOLVE) with RL internalization (ICRL) at the trajectory level.

### 3. Method

Per step *t*:

1. **Rollout** — sample from π_θ, conditioned on textual momentum M_{t-1} (M_0 = empty).
2. **Internalization** — GRPO advantages computed as usual; log-probs for the gradient computed **unconditioned** on M_{t-1}, i.e. the update targets π_θ(·|q) rather than π_θ(·|q, M_{t-1}).
3. **Calibration ratio (bundled with internalization for this pass)** — following ICRL's distribution-calibration reweighting, define a token-level ratio
w_t = π_θ_rollout(y_t | q, y_<t) / π_θ_rollout(y_t | q, M_{t-1}, y_<t),
computed under the fixed rollout policy (not a new/old-policy ratio — a same-parameter, different-context ratio). Multiply min(w_t, w_max) into the standard clipped GRPO term. This down-weights tokens that depended heavily on M_{t-1} and up-weights tokens the unconditioned policy would already favor.
4. **Textual gradient** — LLM reviews sampled successes/failures from this step, writes a short diagnosis. Treated as a heuristic signal, not a claimed causal account of the parameter update.
5. **Trajectory update** — textual gradient appended to a running, LLM-maintained incremental digest (not raw concatenation).
6. **Textual momentum** — LLM reads the trajectory digest, proposes M_t: a directive for the next rollout batch.

**Phase 1 (this doc):** feedback/momentum generator is a fixed frontier model, separate from the trained policy. Self-generated feedback (shared weights) is future work.

### 4. Minimal Experiment (go/no-go)

Goal: cheapest test of whether trajectory-level content beats instance-level content, holding the mechanism fixed at two matched endpoints. Eval **unconditioned** (no directive/critique at test time) for all arms.

| Arm | Content | Internalization + Calibration | Update |
| --- | --- | --- | --- |
| (1) Floor | nothing | — | standard GRPO |
| (2) Instance, OFF/OFF | same-iteration critique | OFF (faithful Critique-GRPO reproduction) | scored under true generating context, as published |
| (3) Instance, ON/ON | same-iteration critique | ON (unconditioned scoring + calibration ratio) | ICRL-style |
| (4) Trajectory, OFF/OFF | textual momentum | OFF | scored under true generating context (momentum-conditioned) |
| (5) Trajectory, ON/ON (Ours) | textual momentum | ON | unconditioned scoring + calibration ratio |

**Arm (2) construction — critical:** copy Critique-GRPO's pipeline exactly (sample → critique → refined rollout → pool original + refined in the GRPO group) with no internalization or calibration, to reproduce their published numbers as a sanity check before trusting any of the other arms.

**Arm (4) purpose:** shows what momentum-conditioning buys you *without* internalizing it — since eval is unconditioned, this arm's gains stay locked behind needing M at inference, illustrating why internalization matters rather than just assuming it.

**Success criteria:**

- (2) roughly matches published Critique-GRPO numbers (reproduction check, gates trust in the rest).
- (5) > (4): internalization + calibration meaningfully helps over naive momentum-conditioning.
- (5) > (3): trajectory content beats instance content at matched mechanism — this is the core trajectory-vs-instance signal worth digging into further.

## 5. Experimental Setup

**Backbone:** Qwen3-8B — the one model common to both Critique-GRPO's original paper and ICRL's reproduction, giving two independent reference points for the Arm (2) sanity check.

**Dataset:** **[Updated 2026-08-12, supersedes the original "MATH, optionally augmented via NuminaMath" plan below]** Training defaults to `open-r1/OpenR1-Math-220k` (NuminaMath 1.5 problems, DeepSeek-R1-verified solutions), not MATH. Reason: Critique-GRPO's published numbers — the reproduction target for Arm (2)'s success criterion above — actually come from training on subsets of OpenR1-Math-220k, not MATH (confirmed by reading arXiv 2506.03106 directly); training MATH-only meant Arm (2) could never have been comparable to those numbers regardless of pipeline fidelity, and separately, Qwen3-8B is already near-saturated on MATH (~0.81–0.94 training-batch accuracy from step 1 of Arm 1's run, flat over 100+ steps), leaving GRPO's group-relative advantage little contrast to exploit. MATH remains available as an explicit opt-in (`TMGRPO_TRAIN_DATA=math`, see `docs/build_and_run_guide.md`) for anyone who wants the easier/legacy pool. Eval is unchanged: MATH500 + a held-out slice of OlympiadBench/AIME24, for all arms and both training-pool choices.

Original plan (kept for context on the eval-set / agentic-env reasoning, which still holds): MATH (training split, optionally augmented via NuminaMath) for training; eval on MATH500 + a held-out slice of OlympiadBench/AIME24. Chosen over agentic environments (ALFWorld/WebShop) for this pass: single-turn rollouts, unambiguous verifier reward, no environment-server overhead, and direct comparability to both baselines' published numbers. Short response length (~1–2K tokens vs. ICRL's 16K-context agentic setup) also substantially eases the memory budget below.

**Compute:** targeting a single B200 (~180–192GB HBM3e) via **verl**. Naive full-precision AdamW fine-tuning (fp32 master + moments) does not fit alongside a frozen reference-policy copy (~144GB before activations/KV cache/rollout). Plan to rely on verl's built-in bf16 training + 8-bit or Adafactor optimizer + gradient checkpointing + colocated vLLM rollout to bring this down to a comfortable margin; fall back to reduced group size or response length if still tight. Single-GPU means no data/tensor parallelism, so wall-clock throughput (not memory fit) is the expected bottleneck — acceptable for a go/no-go pass, revisit for later scale-up.

## 6. Open Risks

- Textual gradients may be confabulated rather than causally accurate — spot-check against ground-truth batch statistics.
- Cost: frontier-model calls every K steps; report alongside performance gains.
- Long-horizon drift: even with a fixed frontier model, the policy's own rollouts shape what it sees — monitor entropy/diversity.
- Calibration ratio may collapse toward uniformly low w_t if M is phrased as a strong directive rather than gentle guidance — watch the w_t trajectory over training (cf. ICRL Fig. 4) as a diagnostic.

## **7. Deferred (not in this pass):**

- a fixed/random-directive control (conditioned-then-unconditioned update with content-free text) to separate "the internalization mechanism itself does something" from "the directive's content does something." Needed before mechanistic claims or publication, not before a go/no-go read.
- internalization-ON / calibration-OFF as an independent toggle (isolating whether internalization or calibration is doing the work) is a useful ablation but out of scope for this pass — noted as a TODO, not run here.
- Phase 2 (future work): the policy itself (shared weights) generates its own textual/meta-feedback — genuine self-referential version.
- Order-shuffle control (future ablation): rerun Arm (4)/(5) with the trajectory of past textual gradients permuted (prefix-only, no future leakage) before each momentum-generation step. Requires a full rerun, not a post-hoc swap, since the digest is built incrementally and order changes propagate through rollouts. Unchanged performance → gain likely from pooled content rather than trajectory reasoning. Changed performance → order matters, but doesn't by itself distinguish genuine trend-tracking from simple recency weighting.