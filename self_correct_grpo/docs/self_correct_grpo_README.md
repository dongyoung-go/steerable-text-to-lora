# Learning When to Self-Correct: Risk-Sensitive Self-Refinement

**A single-backbone solver–critic framework with nested rollout groups and asymmetric, ungated rewards**

---

## 1. Motivation

Self-refinement — having a model critique and revise its own output — is attractive because it can recover errors at inference time. Two prior results shape this design:

- **SCoRe** (Kumar et al., 2024) shows that a single policy trained end-to-end for answering and self-correction can collapse under naive multi-turn RL: producing the best first attempt and making no meaningful second-turn edit can become reward-equivalent to genuine correction. Their solution explicitly rewards useful paired changes between the first and second attempts.
- **ICRL** (2026) trains a separate critic using downstream solver improvement, but invokes and trains the critic only on trajectories that a ground-truth verifier has already identified as failures.

This project focuses on a specific limitation of oracle-gated critic training: gating removes the verification problem from the training objective. The critic is trained to improve answers known to be wrong, but never to recognize that an answer is already correct and should be preserved. At inference time, however, the critic receives solver outputs of unknown correctness and must implicitly perform both decisions: whether refinement is warranted and how to refine. This creates a mismatch between the behavior optimized during training and the behavior required at deployment.

We therefore train the critic unconditionally, on both correct and incorrect solver trajectories. The objective explicitly distinguishes successful correction from harmful intervention, so preservation of a correct answer is learned alongside repair of an incorrect one. We retain SCoRe's core lesson — evaluate refinement through the paired change between the first and second attempts — but extend it to an unconditional critic trained without an oracle gate.

The two decisions the critic must perform at deployment — *whether* refinement is warranted and *how* to refine — motivate making that split explicit in the design rather than leaving it to be inferred from the critique's downstream effect: §2.1 gives the critic a structured output that decides and acts as separate, observable steps.

This design is developed and evaluated entirely in domains with reliable rule-based or environment reward. Extending the unconditional-critic argument to settings without any verifier is a natural next step but is out of scope here — see §7.

### 1.1 Preliminary experiment: does naively removing the gate already break ICRL?

Before investing in the full method, the single most important number for this paper is a small, cheap pilot: **ICRL as published (oracle-gated) vs. ICRL's own reward formula with the gate simply removed** — no hierarchy, no asymmetry, no `N3`-averaging, nothing else changed. This is listed as a baseline pair in §6.2, but it is promoted here because it is not just another ablation row — it is the experiment that determines whether this paper has a reason to exist.

**Why this comparison is decisive, not incidental.** ICRL's critic reward was designed and tuned under the assumption that every `τ1` it sees is already known-wrong (oracle-gated). Nothing about that reward formula was built to handle already-correct `τ1` gracefully — there is no asymmetry (`β` vs. `γ`) protecting against the critic learning to "fix" things that don't need fixing, and no explicit no-op-neutral term. So there are two plausible outcomes if the gate is simply dropped:

- **The gap is small.** Ungating alone is roughly free, and the interesting contribution of this paper is only the efficiency/variance story (hierarchy vs. flat, §3–§4), not the gating story. Motivation for the paper would need to lean more heavily on the train/deployment-mismatch argument (§1) than on an empirical collapse.
- **The gap is large — ungated ICRL drops sharply relative to gated ICRL** (e.g., rising `Δ[c→i]` / regression rate, or a collapsing no-op rate as the critic starts intervening on correct `τ1` and making things worse). This is the outcome that gives the paper its strongest possible motivation: it would directly demonstrate that ICRL's oracle gate is not an incidental implementation detail but load-bearing — removing it without also fixing the reward (asymmetry + no-op neutrality, §4.3) actively harms the model. That result, reported up front, turns the rest of the paper into "here is the fix for a failure mode we can show is real," rather than "here is a design we think should help."

**Proposed setup.** Run both configurations at matched, small compute on MATH500 alone (§6.1, Phase 1) — no environment tasks, no broader math suite — before committing to anything else in §6. Report `Δ[i→c]`, `Δ[c→i]`, and no-op rate for both (§6.3 metrics, computed early). This pilot is cheap relative to the full study and should be run first — its outcome should directly inform how strongly §1's motivation is framed and whether the full hierarchical variant (§3.1) is worth pursuing at all, or whether the flat variant (§3.2) with the fixed reward is sufficient on its own.

---

## 2. Architecture

A single backbone `π_θ` is used for all three roles via role-specific prompts, following ICRL:

- `π_θ^S(·|q)` — solver, unconditioned.
- `π_θ^C(·|q, τ1)` — critic, conditioned on the unconditioned attempt.
- `π_θ^S(·|q, c)` — solver, critique-conditioned.

Unlike ICRL, **the critique-conditioned solver is not a device for internalizing gains back into an implicit critique-free policy.** Both `π_θ^S(·|q)` and `π_θ^S(·|q,c)` are trained as first-class, permanently-used inference-time distributions, because deployment always runs both stages in sequence (generate unconditioned, then always critique, then generate conditioned). This removes the need for ICRL's distribution-calibration reweighting entirely — there is no critique-free target to protect, since the critique-free output is never the final answer at inference time.

### 2.1 Critic output format: explicit verdict gate (default)

The critic's output is structured, not free-form. Before any critique content, the critic first emits a verdict token:

```
VERDICT: KEEP | REVISE
[critique text, if REVISE]
```

This makes "verify" and "decide whether to intervene" explicit and separately observable, rather than inferring the no-op decision after the fact from whether `τ2` ended up unchanged. It is the default output format for this method throughout §3–§7 unless stated otherwise.

**This is not a reintroduction of ICRL's oracle gate.** The oracle gate in §1 operates on *training-data selection* — the critic is never trained on correct `τ1` at all, which is what creates the train/deploy mismatch this paper argues against. The verdict token operates at the *output* level: the critic still sees and is trained on the full, unconditional mixture of correct and incorrect `τ1` via RL exactly as described in §3–§4 — it just learns to make its no-op decision legible rather than implicit. The two are orthogonal axes (train-time data gating vs. inference-time decision structure), not competing designs.

**Compute consequence.** On `VERDICT: KEEP`, level-3 generation is skipped entirely: `τ2 := τ1` deterministically, so `r(τ2) = r(τ1)` and `r_C = 0` exactly (§4.3), with no rollout spent to discover it. This is a direct, structural answer to the compute-cost risk in §7, independent of whether the flat or hierarchical rollout structure (§3) is used.

### 2.2 Attribution ablation: implicit free-form critique (no verdict token)

To isolate how much of any observed gain comes from the reward fix (§4.3's asymmetry and no-op neutrality) versus the explicit verdict format itself, the same method is also run with the verdict token removed: the critic emits only free-form critique text `c ~ π_θ^C(·|q,τ1)`, and the no-op behavior for already-correct `τ1` must be learned implicitly, as a critique whose content happens to leave `τ2` unchanged after a full level-3 solve. Reward and training procedure are otherwise identical to §2.1 (§4.3). This variant is run in **Phase 1, paired against the default**, not deferred to Phase 2 — it is cheap (same infrastructure, verdict-parsing removed) and is the only way to attribute credit between "the reward design works" and "the reward design plus an explicit gate works."

---

## 3. Rollout Structure and Advantage Normalization

The method is defined generally by a 3-level tree with branching factors `(N1, N2, N3)`. We describe the general (hierarchical) form first, then the minimal flat configuration used as the default cost-efficient instantiation.

### 3.1 General form (hierarchical, nested normalization)

For each query `q`, generate a 3-level tree:

```
Level 1 (solver, unconditioned):     N1 samples   τ1^(i)          ~ π_θ^S(·|q)                    i = 1..N1
Level 2 (critic, per τ1):            N2 samples   c^(i,j)         ~ π_θ^C(·|q, τ1^(i))             j = 1..N2   (always sampled, no gating on r(τ1))
Level 3 (solver, per (τ1,c) pair):   N3 samples   τ2^(i,j,k)      ~ π_θ^S(·|q, c^(i,j))            k = 1..N3
```

Total leaf rollouts per query: `N1 × N2 × N3`.

**Each level is advantage-normalized only among its siblings** — i.e., relative to other samples sharing the same parent context, not against the full batch:

- Level 1: `A1^(i)` computed via group-relative (GRPO/RLOO) normalization over the `N1` siblings for the same `q`.
- Level 2: `A_C^(i,j)` computed over the `N2` critique siblings for the same fixed `τ1^(i)`.
- Level 3: `A3^(i,j,k)` computed over the `N3` solver siblings for the same fixed `(τ1^(i), c^(i,j))` pair.

This tree-structured, sibling-only normalization is a strict generalization of ICRL's role-wise advantage estimation (2 groups → 3 nested groups) and gives each level a baseline that controls for exactly the variance introduced by its parent context, rather than pooling across unrelated queries or unrelated parent branches.

### 3.2 Minimal flat variant (`N2 = N3 = 1`)

The general form collapses to a much cheaper configuration by setting `N2 = N3 = 1`. For each query `q`, sample `N1` independent, complete self-refinement trajectories:

```
τ1^(i) ~ π_θ^S(·|q),        c^(i) ~ π_θ^C(·|q, τ1^(i)),        τ2^(i) ~ π_θ^S(·|q, c^(i)),     i = 1..N1
```

This removes the multiplicative `N2 × N3` rollout cost entirely: total rollouts per query drop from `N1·N2·N3` to `3·N1`. There is exactly one critique and one revision per `τ1^(i)`, so there is no per-parent sibling group left to normalize within at levels 2 and 3.

**This changes the normalization, not just the rollout count.** In the general form, the critic's advantage is computed only against other critiques written for the *same* `τ1^(i)` (a narrow, low-variance comparison). In the flat variant, there is only one critique per `τ1^(i)`, so all three roles are instead normalized across the `N1` complete trajectories sharing the same query `q`: the solver reward `r(τ1^(i))`, the solver reward `r(τ2^(i))`, and the critic's paired reward `r_C^{(i)}` (§4.3, single-child form) are each baseline-subtracted against their respective values across `i = 1..N1`. The critic is therefore compared against critiques written for *different* `τ1` draws of the same query, rather than against alternative critiques of the identical `τ1` — a coarser but far cheaper baseline.

This flat variant is treated as the default, minimal instantiation of the method for the main experiments, with the full nested hierarchy (§3.1) evaluated as a compute-scaled ablation to isolate what the additional `N2`/`N3` structure buys beyond the flat version (§6.4).

**Note on `τ2` generation.** Under the default critic output format (§2.1), the `τ2^(i)` step above is conditional: it is only actually sampled when `c^(i)` carries `VERDICT: REVISE`; on `VERDICT: KEEP`, `τ2^(i) := τ1^(i)` is set directly with no generation cost. Under the free-form attribution ablation (§2.2), `τ2^(i)` is always sampled as written.

---

## 4. Reward Design

### 4.1 Base outcome reward `r(·)`

Standard rule-based or environment reward, `r(τ) ∈ [0,1]`: exact-match / equivalence checking for math, unit tests or execution success for code, task-completion signal for environment tasks (ALFWorld, WebShop). Every reward used in this design is grounded in an external checker — no reward model or self-scored likelihood is used at any level. This keeps the reward signal fixed and non-adversarial throughout training, so any effect observed in §6 can be attributed to the rollout structure and reward *shape* (hierarchy, asymmetry, gating) rather than to interaction with a learned or drifting reward source.

### 4.2 Solver reward (levels 1 and 3)

Both `τ1^(i)` and `τ2^(i,j,k)` are scored directly with `r(·)` above, under their respective conditioning context. No merging or reweighting across levels. Note in particular that `τ1`'s reward does not depend on anything downstream — there is no direct incentive for the solver to make `τ1` worse to inflate a later comparison, since level 1 is scored and normalized purely against its own siblings.

### 4.3 Critic reward (level 2) — paired, asymmetric, aggregated over children

For critique `c^(i,j)`, aggregate the outcomes of its `N3` level-3 children relative to the parent `τ1^(i)`:

```
r_C^(i,j) = (1/N3) * Σ_k [ β · max(0, r(τ2^(i,j,k)) − r(τ1^(i)))
                          − γ · max(0, r(τ1^(i)) − r(τ2^(i,j,k))) ],      γ > β
```

This is the asymmetric, paired-delta reward carried over from the SCoRe-motivated design, adapted to the hierarchical structure: each critique's utility is judged directly against the same `τ1` it was written for, averaged over its own resampled children to reduce variance, then normalized against sibling critiques at level 2. A critique that changes nothing yields `r_C ≈ 0` and thus advantage ≈ 0 relative to siblings that also changed nothing — it is not actively rewarded, only left neutral, which is what prevents the reward from being equally satisfied by genuine verification and by generic redundant critique. `γ > β` makes breaking an already-correct `τ1` cost more than failing to fix a wrong one, directly targeting the regression failure mode SCoRe identified.

**Flat-variant form (`N2 = N3 = 1`, §3.2).** With a single critique per `τ1^(i)` and a single child per `(τ1^(i), c^(i))` pair, the sum and average over `k` drop out, and there is no `j` index to begin with:

```
r_C^(i) = β · max(0, r(τ2^(i)) − r(τ1^(i))) − γ · max(0, r(τ1^(i)) − r(τ2^(i))),      γ > β
```

The paired-delta structure and the asymmetry `γ > β` are unchanged — this is the same reward, just without child-averaging. What changes is the normalization baseline used to turn `r_C^(i)` into an advantage: with no critique siblings for the same `τ1^(i)`, `r_C^(i)` is instead baseline-subtracted against `r_C^{(1)}, ..., r_C^{(N1)}` — the critic rewards of all `N1` trajectories for the same query `q` (§3.2). This is a noisier baseline than the nested, same-`τ1` comparison used in the general form, since it pools across critiques written for different (and differently-difficult) `τ1` draws rather than isolating the effect of the critique alone. Whether this added variance materially hurts the no-op-neutrality property (§4.3, general form) is an empirical question addressed directly in §6.4.

**Under the default verdict-gate format (§2.1).** When the critic emits `VERDICT: KEEP`, `τ2` is set to `τ1` deterministically rather than sampled from `π_θ^S(·|q,c)`, so `r(τ2) = r(τ1)` and `r_C = 0` exactly, with no level-3 rollout spent to discover it. When the critic emits `VERDICT: REVISE`, the formula proceeds exactly as written above with `τ2` sampled normally.

**Under the free-form attribution ablation (§2.2).** The formula and asymmetry are unchanged; `τ2` is always sampled from `π_θ^S(·|q,c)` regardless of critique content, and a no-op is only ever realized through the sampled `τ2` happening to match `τ1` closely enough that `r(τ2) ≈ r(τ1)`.

### 4.4 Joint update

All three levels backprop into the same shared `θ`, combined as a weighted sum of the three GRPO-clipped objectives (level-wise advantages, level-specific conditioning context for the importance-sampling ratio):

```
J(θ) = E[ clip-GRPO(A1) ]_{level 1} + w_C · E[ clip-GRPO(A_C) ]_{level 2} + E[ clip-GRPO(A3) ]_{level 3}
```

`w_C` down-weights the critic term if its reward proves higher-variance in practice (consistent with ICRL's own observation that critic reward curves are noisier than solver reward curves).

---

## 5. Test-Time Inference

Deployment mirrors training exactly, with no oracle and no gating decision required:

1. Sample `τ1 ~ π_θ^S(·|q)`.
2. Always sample `c ~ π_θ^C(·|q, τ1)`.
3. Sample `τ2 ~ π_θ^S(·|q, c)` and return `τ2` as the final answer.

Because the critic was trained on the full mixture of correct and incorrect `τ1`, and rewarded ≈0 (not negatively) for a no-op on already-correct inputs, this unconditional two-stage pipeline is the same distribution seen throughout training — nothing about inference requires access to `r(τ1)`.

---

## 6. Experiments

### 6.1 Datasets

All experiments use domains with a reliable rule-based or environment reward, so that every comparison isolates rollout/reward-design effects rather than reward-source effects.

**Phase 1 (essential, minimal first implementation): MATH only.** MATH500 as the single dataset for the preliminary experiment (§1.1) and the first full pass through §6.2's baseline table. Exact-match / equivalence checking gives a cheap, reliable, fast-to-iterate reward — the right setting to establish whether the core claim (gating hurts, the fix helps) holds at all before spending compute anywhere else.

**Phase 2 (deferred until Phase 1 result is in hand):** Minerva Math, OlympiadBench, AIME, AMC (broader math generalization), then ALFWorld, WebShop (environment reward, tests whether the result transfers outside math). None of these are needed to answer the paper's central question; they exist to demonstrate the result isn't MATH500-specific once Phase 1 has already shown a real effect.

### 6.2 Baselines

| Baseline | Backbone | Critic invocation | Notes | Priority |
| --- | --- | --- | --- | --- |
| ICRL | shared | oracle-gated | Original ICRL design, flat 2-stage rollout, single-child critic reward. | **Essential — Phase 1** |
| ICRL's reward formula, but ungated | shared | unconditional (no oracle gate), original ICRL critic reward | Same critic reward as ICRL, just always invoked — no hierarchy, no averaging, no asymmetry. **This is the sharpest baseline: it isolates what the hierarchical/asymmetric/multi-child design (this method) adds beyond simply removing the gate.** This is the §1.1 preliminary experiment — run this pair first, on MATH500 alone, before anything else in this table. | **Essential — Phase 1** |
| **This method (default: flat + verdict gate)** | shared | unconditional, learned, flat (`N2=N3=1`, §3.2), explicit `VERDICT: KEEP/REVISE` (§2.1) | `N1` complete trajectories per query; `KEEP` skips level-3 generation entirely (`τ2:=τ1`, `r_C=0` for free). Advantage normalized across the `N1` trajectories per query. | **Essential — Phase 1** |
| This method, free-form attribution ablation (§2.2) | shared | unconditional, learned, flat, implicit no-op (no verdict token) | Same reward and training procedure as the row above, verdict token removed. Isolates whether gains come from the reward fix alone or require the explicit gate. | **Essential — Phase 1** (paired with the row above, not deferred) |
| GRPO (solver only) | shared/single | none | Single-stage output, no self-refinement. | Deferred — cheap, but only useful as a floor once the three essential rows above establish there's a real effect to floor-check. |
| Always-critique, fixed template | shared | unconditional, non-learned | Sanity floor for reward hacking via generic recheck critiques. | Deferred to Phase 2 |
| SCoRe-style single-policy self-correction | shared | implicit (no separate critic role) | Turn-2 self-correction without a distinct critic role. | Deferred to Phase 2 |
| **This method (hierarchical, ablation)** | shared | unconditional, learned, nested (§3.1) | Full `N1×N2×N3` tree, asymmetric paired-delta critic reward with `N3`-child averaging, strictly nested sibling normalization. Evaluated as a compute-scaled ablation against the flat default, not as a separate baseline (§6.4). | Deferred — only worth running if the flat default (Phase 1) already shows a positive result to try to improve on. |

**Minimal first implementation = the four "Essential — Phase 1" rows, on MATH500 alone (§6.1).** That is: gated ICRL, ungated-ICRL-formula (free-text, matching published ICRL), this method's default (flat + verdict gate), and this method's free-form attribution ablation — all on one dataset. This answers three questions at once: (a) does removing the gate alone help or hurt (§1.1), (b) does this method's fixed reward plus explicit gate recover or exceed gated ICRL's performance while remaining ungated, and (c) how much of that gain is attributable to the reward fix versus the explicit verdict format. Everything else in this table is an extension, not a prerequisite.

All baselines are compute-matched to the **flat variant's** `3·N1` total rollouts per query (the default configuration): for baselines with fewer stages, per-stage rollout counts are scaled up accordingly. The hierarchical variant is deliberately *not* compute-matched to the other rows — its whole purpose is to spend more rollouts (`N1×N2×N3` vs. `3·N1`) to test whether the additional structure earns back its cost, so it is reported separately against its own, larger rollout budget (§6.3).

### 6.3 Metrics

Reported **separately**, never collapsed into one net-accuracy number (per SCoRe's own Δ[i→c]/Δ[c→i] convention):

- **Δ[i→c]**, **Δ[c→i]** — fix rate and regression rate. **Essential — Phase 1.** These two numbers alone are what §1.1's preliminary experiment is judged on.
- **No-op rate** — for the default verdict-gate format (§2.1), this is exact: the `KEEP` rate directly, no threshold needed. For the free-form attribution ablation (§2.2), it remains a proxy: fraction of critiques with negligible effect on `π_θ(τ2|q,c)` vs. `π_θ(τ1|q)` (token-level KL below a threshold). **Essential — Phase 1** for both variants, since a large `Δ[c→i]` drop for ungated ICRL is most convincingly explained by showing the no-op rate collapsing at the same time, and the exact-vs-proxy comparison between the two variants is itself part of the attribution ablation. Also track `KEEP` rate over training as a standalone collapse-monitoring signal (§7): a rate that climbs toward ~100% while `Δ[i→c]` stagnates indicates the free, zero-cost `KEEP` shortcut is being over-used rather than genuine verification happening.
- **Critic reward variance across levels** — to validate whether the `N3`child averaging in §4.3 actually reduces variance relative to a single-child paired delta (ICRL's design). Deferred — only meaningful once the hierarchical variant (§3.1) is actually being run.
- **Compute cost** — total rollouts (`N1·N2·N3`) vs. accuracy gain, reported explicitly given the multiplicative cost of the hierarchy. Deferred to the flat-vs-hierarchical comparison in Phase 2.

### 6.4 Ablations

- **Gated vs. ungated critic training, holding the reward formula fixed** — **Essential — Phase 1.** This is exactly the §1.1 preliminary experiment and the ICRL / ICRL-ungated rows of §6.2; it isn't a separate run, it's the same result reframed as the paper's central ablation.
- **Symmetric (`β=γ`) vs. asymmetric (`γ>β`) critic reward** — expect symmetric to reproduce SCoRe's originally observed collapse. Deferred to Phase 2, but cheap and high-value once Phase 1 confirms the gating effect is real — this is the natural next question ("is asymmetry doing the work, or would any fixed-formula ungating have worked?").
- **Flat (`N2=N3=1`, default) vs. hierarchical (`N2,N3>1`, §3.1)** — the primary cost/benefit ablation. At matched *total rollout budget* (i.e., the hierarchical variant's `N1` is reduced so `N1×N2×N3` equals the flat variant's `3·N1`), does the nested tree structure and its narrower, same-`τ1` critic normalization outperform the flat variant's cheaper, pooled-across-`N1` normalization? Deferred to Phase 2 — only worth the extra compute once Phase 1 shows the flat default already beats ungated ICRL.
- **Within the hierarchical variant, `N3 > 1` (averaged children) vs. `N3 = 1` (single-child paired delta, ICRL-style, but still nested under `N2`)** — a secondary decomposition isolating the variance-reduction contribution of child-averaging specifically, independent of whether the `N2` critique-sibling grouping is present. Deferred to Phase 2, contingent on the flat-vs-hierarchical ablation above being run at all.
- **Shared vs. separate backbone** (ablation only, not the primary design) — to quantify what, if anything, is lost by removing the shared representation between solver and critic roles. Deferred — lowest priority, architectural sanity check rather than a claim the paper depends on.
- **Explicit verdict gate (default, §2.1) vs. implicit free-form critique (attribution ablation, §2.2)** — **Essential — Phase 1.** Not a separate run; this is the same comparison as the two "This method" rows in §6.2, reframed as the ablation that attributes any gain over ungated-ICRL between the reward fix and the explicit gate. Track `KEEP` rate over training (§6.3) as part of this comparison — it should stabilize near the free-text variant's empirical no-op rate rather than drifting higher, which would indicate the explicit gate is being over-used relative to genuine verification need.

---

## 7. Risks / Open Questions

- **Compute cost is multiplicative in the tree depth** (`N1×N2×N3`) for the hierarchical variant only. The default flat variant (§3.2) avoids this entirely, at a cost of a coarser, pooled-across-`N1` normalization baseline for the critic instead of a narrow same-`τ1` comparison. Whether the hierarchical variant's extra cost buys a real gain over the flat default — rather than just being a more expensive route to the same result — is not assumed and is the subject of the flat-vs-hierarchical ablation in §6.4. The default verdict gate (§2.1) offers a separate, more direct lever on this same cost: skipping level-3 generation entirely on `KEEP` verdicts, independent of whether the flat or hierarchical variant is used.
- **Verdict-collapse to always-`KEEP`.** Because `KEEP` costs no rollout and always yields `r_C = 0` — a safe, non-negative outcome under the asymmetric reward (§4.3) — it may be an easier local optimum to fall into than under the free-form attribution ablation (§2.2), where even a no-op still required generating a plausible critique. This is monitored explicitly via `KEEP` rate vs. `Δ[i→c]` over training (§6.3), and directly compared against the free-form variant's empirical no-op rate as part of the Phase 1 attribution ablation (§6.4) rather than assumed to be benign.
- **Reward hacking via generic recheck critiques** remains possible; compare learned critiques against a fixed non-learned template baseline as a sanity floor.
- **Nonstationary co-adaptation:** the critic's reward at level 2 is computed against `τ2` samples drawn from `π_θ^S(·|q,c)`, which is simultaneously being updated by the level-3 objective. The critic is therefore being scored against a moving target throughout training, not a fixed downstream policy. This could bias credit assignment for the critic in ways distinct from ordinary reward variance, and is not addressed by the `N3`child averaging in §4.3, which reduces variance from resampling but not bias from target drift. Worth monitoring via critic-reward trend vs. solver-update-rate correlation, separate from the variance metric already in §6.3.
- **Shared-backbone role interference:** solver and critic roles now share parameters; unlike the separate-network design, gradient updates for one role can shift the other's behavior. ICRL's role-wise advantage normalization mitigates but does not eliminate this; monitor for critic-quality regressions correlated with solver updates and vice versa.
- **Scope is limited to verifiable-reward domains.** The core argument — that gating throws away signal and creates a train/deployment mismatch — applies with equal or greater force when no ground-truth verifier exists at all, since the critic then has even less to fall back on at inference time. Extending the reward design in §4.1 to verifier-free settings (e.g., via a reference-answer-conditioned implicit reward, or an LLM-judge reward) is a natural follow-up, but introduces its own confound — a reward source that can drift or be gamed as the same backbone is updated — and is deliberately left out of this submission's scope so that the central gating claim can be evaluated without that additional variable.