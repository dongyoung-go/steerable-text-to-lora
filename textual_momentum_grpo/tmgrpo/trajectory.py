"""Arm 4/5: textual gradient, incremental trajectory digest, and momentum generation.

README section 3, steps 4-6:
  4. Textual gradient -- LLM reviews sampled successes/failures from this step, writes a short
     diagnosis. "Treated as a heuristic signal, not a claimed causal account of the parameter
     update."
  5. Trajectory update -- textual gradient appended to a running, LLM-maintained INCREMENTAL
     digest (not raw concatenation).
  6. Textual momentum -- LLM reads the trajectory digest, proposes M_t: a directive for the next
     rollout batch.

All three steps are frontier-model calls (gpt-5-mini, via llm_client.LLMClient) on a fixed model
separate from the trained policy (README section 3, "Phase 1").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm_client import LLMClient

TEXTUAL_GRADIENT_SYSTEM_PROMPT = (
    "You are analyzing one step of reinforcement-learning training on math problems. You will "
    "see a sample of problems the policy just attempted, each with its response and whether it "
    "was scored correct or incorrect. Write a short, specific diagnosis (under 120 words) of what "
    "kinds of reasoning succeeded and what kinds failed this step. Be concrete (e.g. specific "
    "error patterns), not generic encouragement."
)

DIGEST_UPDATE_SYSTEM_PROMPT = (
    "You maintain a running, compact digest of an RL training trajectory's optimization history. "
    "You will see the current digest (may be empty, for the first step) and the newest textual "
    "gradient. Produce an UPDATED digest that incorporates the new information: keep it compact "
    "(under 200 words) by summarizing/merging with prior content rather than concatenating, "
    "prioritizing recent and still-relevant trends over stale detail."
)

MOMENTUM_SYSTEM_PROMPT = (
    "You read a running digest of an RL training trajectory's optimization history and produce a "
    "short exploration directive for the NEXT rollout batch: what the policy should try, given "
    "what has worked, what remains stuck, and what has regressed. Under 80 words. Write it as "
    "direct guidance to the policy, not a summary of the past."
)


def generate_textual_gradient(client: LLMClient, step_summary: str) -> str:
    """One textual gradient: a diagnosis of this step's sampled successes/failures.

    `step_summary` is caller-assembled (e.g. a formatted sample of (problem, response, correct?)
    tuples from this step's rollouts) -- this module only owns the prompt/call, not how the
    sample is selected.
    """
    return client.complete(system_prompt=TEXTUAL_GRADIENT_SYSTEM_PROMPT, user_prompt=step_summary)


def update_digest(client: LLMClient, current_digest: str, textual_gradient: str) -> str:
    """Incrementally fold `textual_gradient` into `current_digest` (LLM-summarized, not concatenated)."""
    user_prompt = (
        f"Current digest:\n{current_digest or '(empty -- this is the first step)'}\n\n"
        f"Newest textual gradient:\n{textual_gradient}\n\nUpdated digest:"
    )
    return client.complete(system_prompt=DIGEST_UPDATE_SYSTEM_PROMPT, user_prompt=user_prompt)


def generate_momentum(client: LLMClient, trajectory_digest: str) -> str:
    """Produce M_t: the next-step exploration directive, from the current trajectory digest."""
    return client.complete(system_prompt=MOMENTUM_SYSTEM_PROMPT, user_prompt=trajectory_digest)


@dataclass
class TrajectoryState:
    """Running state for arm 4/5: the digest and the most recent momentum directive.

    M_0 is empty (README section 3, step 1: "conditioned on textual momentum M_{t-1} (M_0 =
    empty)"). Call `step()` once per RL step, after that step's rollouts have been scored.
    """

    digest: str = ""
    momentum: str = ""  # M_{t-1}, consumed by the NEXT rollout batch
    history: list[str] = field(default_factory=list)  # raw textual gradients, for spot-checking

    def step(self, client: LLMClient, step_summary: str) -> str:
        """Run one full trajectory update (steps 4-6) and return the new momentum M_t."""
        gradient = generate_textual_gradient(client, step_summary)
        self.history.append(gradient)
        self.digest = update_digest(client, self.digest, gradient)
        self.momentum = generate_momentum(client, self.digest)
        return self.momentum
