# Arms 2/3/4/5: what's missing before they can run

> Companion to `textual_momentum_grpo_README.md` §4 (the arms table) and `build_and_run_guide.md`.
> Written after Arm 1 (Floor) was confirmed training correctly on the B200 (real verl 0.8.0,
> `.venv-verl`) — this doc captures exactly what a future session needs to build to bring up
> Arms 2-5, based on hands-on inspection of verl's actual (not documented) internals, not guesses.

## Status as of this writing

- **Arm 1 (Floor)**: running for real — confirmed nonzero reward, nonzero loss/grad_norm,
  learning. Config: `configs/resolved/arm1_floor.yaml` + the two fixes below.
- **Arms 2-5**: `configs/resolved/arm{2,3,4,5}_*.yaml` exist and are Hydra-valid, but **nothing
  in the training loop reads their `custom.conditioning` / `custom.internalization` /
  `custom.calibration` keys**. Launching verl with any of these configs right now would run
  exactly like Arm 1 — same rollouts, same GRPO update, silently ignoring the arm-specific
  config. This is the core gap this doc scopes.
- `tmgrpo/critique.py`, `tmgrpo/trajectory.py`, `tmgrpo/calibration.py` are pure-Python,
  unit-tested (53 tests passing), and were written *before* verl was actually installed —
  they contain the right algorithmic logic but have never been wired to verl's real
  `DataProto`/worker-group API. Wiring them in is the bulk of the remaining work.

## Two real bugs fixed while getting Arm 1 running (read before touching arms 2-5)

These bite any arm, not just Arm 1 — worth internalizing before writing new code:

1. **`math_verify`'s reward-scoring thread-safety bug** (fixed in `tmgrpo/reward.py`).
   `math_verify.parse()`/`verify()` default to a `signal.alarm()`-based timeout, which only
   works in the interpreter's main thread. verl's `NaiveRewardManager.run_single` (see
   `verl/experimental/reward_loop/reward_manager/naive.py`) calls the custom reward function
   from a worker thread via `asyncio`'s `run_in_executor`. Every call raised
   `ValueError: ... doesn't support threaded environment ...`, which `check_answer`'s
   `except Exception: return False` silently swallowed — **every single reward was 0.0**,
   including on responses whose extracted `\boxed{}` answer exactly matched ground truth. Fixed
   by passing `parsing_timeout=None` to `parse()` and `timeout_seconds=None` to `verify()`. If
   arms 2-5 add any new reward/scoring logic that touches `math_verify`, apply the same fix.
2. **Qwen3-8B is a thinking model.** Its chat template auto-injects a `<think>...</think>` block
   before the answer. With `response_length=2048` (the ICRL reference value, tuned for a
   non-thinking setup), ~62% of rollouts hit the length cap mid-`<think>` and never produce a
   boxed answer. Fixed via `+data.apply_chat_template_kwargs.enable_thinking=false` (a real,
   verl-supported Hydra key — confirmed via `verl/utils/dataset/rl_dataset.py`, which threads
   `config.get("apply_chat_template_kwargs", {})` into `tokenizer.apply_chat_template(...)`).
   Needed on every arm's launch command, not just Arm 1's.

## What verl's real training loop looks like (confirmed by reading the installed 0.8.0 source)

`verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit()` is the actual loop. Rough shape, per step:

1. `gen_batch` — pull a batch of prompts from the dataloader.
2. `self.actor_rollout_wg.generate_sequences(gen_batch)` — colocated vLLM rollout (`rollout.n`
   samples per prompt), via the async `AgentLoop`/`RewardLoopWorker` machinery under
   `verl/experimental/reward_loop/` and `verl/experimental/agent_loop/`.
3. Reward computation — `NaiveRewardManager.run_single` per rollout, calling our
   `custom_reward_function` (wired via `custom_reward_function.path`/`.name` and mirrored under
   `reward.custom_reward_function.*` — both are needed, see `configs/overrides/*.yaml`'s
   comments for why).
4. `self._compute_old_log_prob` — log-probs of each sampled response under the *current* actor,
   via `self.actor_rollout_wg.compute_log_prob(batch_td)`. **This is the exact call site to reuse
   for the internalization log-prob recompute** — same API, called a second time with a
   differently-constructed prompt (see Arm 3/5 section below).
5. Advantage computation (GRPO: group-normalized reward, per prompt-group of `rollout.n`
   samples).
6. `self.actor_rollout_wg.update_actor(batch)` — the actual gradient step.

**There is no config-level hook for steps 2-6 that lets you inject different prompts per
group-member, pool two different rollouts into one GRPO group, or run a second log-prob pass
under a different prompt.** This was checked directly against the installed source, not assumed.
The only way in is subclassing `fit()` (or the pieces it calls) — see below.

## Arm-by-arm gap

### Arm 2 (Instance, OFF/OFF) — same-iteration critique, no internalization

README §4: "copy Critique-GRPO's pipeline exactly (sample → critique → refined rollout → pool
original + refined in the GRPO group), no internalization/calibration."

Missing pieces:

1. **Critique generation after the first rollout, before the second.** `tmgrpo/critique.py`'s
   `generate_critique(client, problem, response)` already does the LLM call
   (`llm_client.py`'s `LLMClient`, defaults to `gpt-5-mini` per `configs/base.yaml`'s
   `custom.frontier_model`) — this part is ready. What's missing is the *call site*: after
   `generate_sequences` produces the first batch of rollouts, before the second "refined" batch
   is generated, something needs to (a) decode each rollout's response text (same decode verl's
   own reward manager does — `tokenizer.decode(valid_response_ids, skip_special_tokens=True)`),
   (b) call `generate_critique` on each, (c) re-inject the critique into the prompt for a second
   rollout pass (`tmgrpo/verl_hooks.py::inject_conditioning_context` already builds the
   second-user-turn prompt format for this).
2. **Pooling original + refined rollouts into one GRPO group.** `tmgrpo/critique.py`'s
   `pool_original_and_refined(...)` has the grouping logic, but it operates on plain Python
   data structures (`RolloutSample`), not verl's `DataProto`/tensordict batch format. Needs a
   thin adapter that takes two `DataProto` batches (original-rollout batch, critique-conditioned
   rollout batch) and concatenates them into a single batch with a shared `group_id`
   (technically: same prompt index in the advantage-computation grouping key) so GRPO's group
   normalization sees `2 * rollout.n` samples per original prompt instead of `rollout.n`.
3. **Where this plugs into `fit()`**: between step 2 (rollout) and step 3 (reward) in the list
   above. This is the concrete reason a `RayPPOTrainer` subclass is required — there's no config
   knob for "generate twice, critique in between, pool."

### Arm 4 (Trajectory, OFF/OFF) — momentum-conditioned, no internalization

README §4 + §3 steps 4-6: after each step, an LLM writes a **textual gradient** (diagnosis of
that step's successes/failures), folds it into a running **trajectory digest**, then a **textual
momentum** directive is generated from the digest and used to condition the *next* step's
rollouts.

Missing pieces:

1. **Per-step textual-gradient generation from real batch outcomes.**
   `tmgrpo/trajectory.py::generate_textual_gradient(client, step_summary)` takes a pre-built
   `step_summary` string — the missing part is building that summary from the actual verl batch
   after a training step (e.g. sampled successes/failures, aggregate accuracy, a few example
   responses). Needs to read from the `DataProto` batch + the reward-manager's per-sample scores
   (`reward_extra_info` dict, already returned per-sample by `NaiveRewardManager.run_single` —
   confirmed available, just needs plumbing out to the trainer level).
2. **Persistent `TrajectoryState` across steps.** `trajectory.py::TrajectoryState` and
   `update_digest`/`generate_momentum` are ready in isolation, but need to live as an attribute
   on the `RayPPOTrainer` subclass (or equivalent), persisted step-to-step (not re-created per
   step), and ideally checkpointed alongside `trainer.save_freq` so a resumed run doesn't lose
   its trajectory history.
3. **Conditioning the *next* step's rollout on the current momentum.** Same
   `inject_conditioning_context` hook as Arm 2, but applied to *every* prompt in the batch
   (not per-sample critique) using the momentum text `M_{t-1}` computed at the end of the
   previous step. This is a single rollout pass (not two, unlike Arm 2's critique-then-refine),
   conditioned uniformly.
4. **Frontier-model call cadence**: README §6 flags this as a cost item to track — decide/confirm
   whether momentum is regenerated every step or every K steps before implementing (affects the
   `fit()` subclass's control flow non-trivially; not just a config knob).

### Arm 3 (Instance, ON/ON) and Arm 5 (Trajectory, ON/ON — the paper's proposed method)

Both need everything from Arm 2 (resp. Arm 4) **plus** internalization + the calibration ratio.
This is `tmgrpo/verl_hooks.py::recompute_unconditioned_logprobs`, currently a stub:

```python
def recompute_unconditioned_logprobs(*args, **kwargs):
    raise NotImplementedError(...)
```

What it actually needs to do, now confirmed against real verl internals (was previously an open
question — no longer is):

1. Take the batch of *conditioned* rollouts (sampled under critique-text or momentum-text
   appended to the prompt).
2. Build a second `DataProto`/tensordict batch with the **same response tokens**, but the
   **prompt tokens replaced by the unconditioned prompt** (drop the injected critique/momentum
   turn — `inject_conditioning_context`'s docstring already notes this is why the context is
   appended as a second user turn rather than mutating the original problem, specifically to
   make this drop trivial).
3. Call `self.actor_rollout_wg.compute_log_prob(unconditioned_batch_td)` — **the same API
   `_compute_old_log_prob` already uses internally** (confirmed call site in
   `ray_trainer.py`, lines ~1256-1276 as of this verl version) — just invoked a second time
   with the unconditioned batch instead of the one actually used for sampling.
4. Feed the result into `tmgrpo/calibration.py::calibration_ratio`/`apply_calibration` — these
   already implement the `w_t = π(y|q,y_<t) / π(y|q,ctx,y_<t)` math (README §3 step 3) and the
   `min(w_t, w_max)` clip, generically over "context" so Arm 3 and Arm 5 share this code
   unchanged. What's missing is only the plumbing to get real log-prob tensors from steps 1-3
   into these functions' expected shapes.
5. The **unconditioned log-probs become the actual GRPO gradient target** (replacing the
   conditioned-rollout log-probs that would otherwise be used) — this is the "internalization"
   step per README §1/§3: the policy is updated toward `π_θ(·|q)`, not `π_θ(·|q, context)`.

None of steps 1-3 are guessable from docs alone — they were confirmed by reading
`ray_trainer.py`'s actual `_compute_old_log_prob` implementation this session. A future session
implementing this should re-read that method directly (`.venv-verl/lib/python3.12/site-packages/
verl/trainer/ppo/ray_trainer.py`) rather than re-deriving the call shape from scratch.

## Suggested build order

1. Arm 2 first (simplest: one critique call, one pool operation, no persistent state, no
   internalization). Also doubles as the README §4 "reproduction check" gate (Arm 2 should
   roughly match published Critique-GRPO numbers) before trusting anything built on top.
2. Arm 4 (adds persistent trajectory state + momentum conditioning, still no internalization).
3. Implement `recompute_unconditioned_logprobs` once, generically (shared by Arm 3 and Arm 5).
4. Arm 3 (Arm 2 + internalization) — cheaper to validate than Arm 5 since it reuses Arm 2's
   simpler single-shot conditioning.
5. Arm 5 (Arm 4 + internalization) — the paper's actual proposed method, success criteria in
   README §4 depend on this being last and correct: (5) > (4) and (5) > (3).

All of the above lives in a `RayPPOTrainer` subclass (no existing name reserved for it yet —
suggest `tmgrpo/verl_trainer.py::TMGrpoTrainer` or similar), instantiated in place of the stock
trainer wherever `main_ppo.py`'s `TaskRunner.run()` constructs `RayPPOTrainer(...)` — that
construction call is `verl/trainer/main_ppo.py:299` as of this verl version (confirmed via the
`FutureWarning` seen during Arm 1's actual runs, which fires from that exact line).
