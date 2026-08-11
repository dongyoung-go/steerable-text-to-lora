"""Dataset loading, prompt formatting, and correctness verification for Guide-ReST tasks.

Registry pattern (`TASKS`, keyed by task name) mirrors `scripts/textgrad_repro.py`'s own
`TASKS`/`ANSWER_PARSERS` dicts in the main repo -- so a new task (a Guru non-code domain,
say) is a new `TaskSpec` entry here, not a change to `sampling.py`/`feedback.py`/
`train.py`/`round_loop.py`, none of which import anything task-specific.

Deliberately self-contained (no import of `steerable_t2l`): this whole experiment runs
inside an ephemeral `uv run --with vllm==0.11.0 --with transformers==4.57.1 ...` overlay
(see `run.sh`) that pins an older transformers than the main repo's persistent env
(`transformers>=5.0`) requires -- importing the main package here would either silently
pull in the wrong transformers or force a version straddle. The one small parser this file
needs (`parse_integer_answer`) is duplicated from `src/steerable_t2l/eval_accuracy.py`
rather than imported, which is already this repo's own convention for the same function
(compare `scripts/textgrad_repro.py::_parse_integer`).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    question: str
    gold: str  # raw gold-answer string, task-specific shape (bare int string, or a MATH answer expression)


def parse_integer_answer(text: str) -> int | None:
    """Last digit-bearing token in `text`, as an int, or `None` on parse failure.

    Returning `None` (not defaulting to 0) matters here exactly as it does in
    `eval_accuracy.py`: a genuinely-zero gold answer must not spuriously "match" a
    response that failed to produce any answer at all.
    """
    tokens = [tok for tok in text.strip().split() if any(c.isdigit() for c in tok)]
    if not tokens:
        return None
    digits = "".join(c for c in tokens[-1].split(".")[0] if c.isdigit())
    if not digits or len(digits) > 4300:  # degenerate generations; see eval_accuracy.py
        return None
    return int(digits)


def _last_boxed(text: str) -> str | None:
    """Contents of the last `\\boxed{...}` in `text`, handling nested braces (MATH
    solutions routinely nest, e.g. `\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}`) -- a
    plain non-greedy regex silently truncates at the first inner `}`.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


# ---------------------------------------------------------------------------
# gsm8k
# ---------------------------------------------------------------------------

GSM8K_INSTRUCTION = (
    "Solve the following grade-school math problem. Show your work, then end your "
    "response with a final line of the exact form 'Answer: <integer>'."
)


def _gsm8k_gold(answer_field: str) -> str:
    # gsm8k's raw `answer` field is a full chain-of-thought ending in "#### <number>".
    return answer_field.rsplit("####", 1)[-1].strip()


def _load_gsm8k_pool(split: str, n: int | None, seed: int) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    indices = list(range(len(ds)))
    if n is not None and n < len(indices):
        indices = random.Random(seed).sample(indices, n)
    return [Row(question=ds[i]["question"], gold=_gsm8k_gold(ds[i]["answer"])) for i in indices]


def load_gsm8k_heldout(n: int = 200, seed: int = 0) -> list[Row]:
    return _load_gsm8k_pool("test", n, seed)


def load_gsm8k_dev_pool(n: int = 50, seed: int = 12345) -> list[Row]:
    """A fixed, separate dev split for `train.py`'s early-stopping validation, drawn from
    the `train` split. Reserved *before* the Grow pool (see `load_gsm8k_grow_pool`, which
    excludes these same indices) so that a full-train-set Grow pool still leaves the dev
    pool disjoint. Disjoint from the held-out pass@1 set automatically too (that's drawn
    from the `test` split). See docs/01_train.md's "Dev pool" section for why this is a
    separate fixed split rather than a per-round carve-out of that round's own
    filtered.jsonl."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    dev_indices = random.Random(seed).sample(indices, min(n, len(indices)))
    return [Row(question=ds[i]["question"], gold=_gsm8k_gold(ds[i]["answer"])) for i in dev_indices]


def load_gsm8k_grow_pool(
    n: int | None = None, seed: int = 0, dev_n: int = 50, dev_seed: int = 12345,
) -> list[Row]:
    """`n=None` (the default) uses every `train`-split question except the fixed dev pool
    (`dev_n`/`dev_seed`, must match `load_gsm8k_dev_pool`'s own args for disjointness) --
    matching ReST-EM (Singh et al., TMLR 2024), which grows from (essentially) the entire
    task training set each round (7,500 MATH / 2,342 APPS problems) rather than a small
    subsample. Pass an explicit `n` to subsample instead, e.g. for a smoke test."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train")
    all_indices = list(range(len(ds)))
    dev_indices = set(random.Random(dev_seed).sample(all_indices, min(dev_n, len(all_indices))))
    remaining = [i for i in all_indices if i not in dev_indices]
    if n is not None and n < len(remaining):
        remaining = random.Random(seed).sample(remaining, n)
    return [Row(question=ds[i]["question"], gold=_gsm8k_gold(ds[i]["answer"])) for i in remaining]


def verify_gsm8k(completion: str, gold: str) -> bool:
    pred = parse_integer_answer(completion)
    gold_int = parse_integer_answer(gold)
    return pred is not None and gold_int is not None and pred == gold_int


# ---------------------------------------------------------------------------
# math (MATH / MATH-500)
# ---------------------------------------------------------------------------

MATH_SUBJECT_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

MATH_INSTRUCTION = (
    "Solve the following competition math problem. Show your work, then end your "
    "response with your final answer on its own line, wrapped as \\boxed{<answer>}."
)


def _load_math_train_ds():
    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset("EleutherAI/hendrycks_math", c, split="train") for c in MATH_SUBJECT_CONFIGS]
    return concatenate_datasets(parts)


def _math_rows_from_indices(ds, indices: list[int]) -> list[Row]:
    rows = []
    for i in indices:
        row = ds[i]
        gold = _last_boxed(row["solution"])
        if gold is None:  # a handful of MATH solutions have no \boxed{} answer; skip them
            continue
        rows.append(Row(question=row["problem"], gold=gold))
    return rows


def load_math_dev_pool(n: int = 50, seed: int = 12345) -> list[Row]:
    """See `load_gsm8k_dev_pool`'s docstring -- same "reserved before Grow" construction,
    over MATH's concatenated train split instead of gsm8k's."""
    ds = _load_math_train_ds()
    indices = list(range(len(ds)))
    dev_indices = random.Random(seed).sample(indices, min(n, len(indices)))
    return _math_rows_from_indices(ds, dev_indices)


def load_math_grow_pool(
    n: int | None = None, seed: int = 0, dev_n: int = 50, dev_seed: int = 12345,
) -> list[Row]:
    """See `load_gsm8k_grow_pool`'s docstring -- `n=None` (the default) uses MATH's entire
    train split (7,500 problems) minus the fixed dev pool, matching ReST-EM's own setup."""
    ds = _load_math_train_ds()
    all_indices = list(range(len(ds)))
    dev_indices = set(random.Random(dev_seed).sample(all_indices, min(dev_n, len(all_indices))))
    remaining = [i for i in all_indices if i not in dev_indices]
    if n is not None and n < len(remaining):
        remaining = random.Random(seed).sample(remaining, n)
    return _math_rows_from_indices(ds, remaining)


def load_math_heldout(n: int | None = None, seed: int = 0) -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [Row(question=row["problem"], gold=row["answer"]) for row in ds]
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def verify_math(completion: str, gold: str) -> bool:
    # math-verify is added as an ephemeral overlay dependency in run.sh (see docs/01_train.md)
    # -- imported lazily so tasks.py itself stays importable (e.g. for the orchestrator,
    # round_loop.py, which never needs to verify anything itself) without it installed.
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify

    pred = mv_parse(completion)
    gold_parsed = mv_parse(f"\\boxed{{{gold}}}")
    if not pred or not gold_parsed:
        return False
    try:
        return bool(mv_verify(gold_parsed, pred))
    except Exception:
        # math-verify can raise on pathological/malformed LaTeX from a bad generation --
        # treat as "did not verify" rather than crashing an entire Grow batch on one row.
        return False


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    load_grow_pool: Callable[..., list[Row]]
    load_dev_pool: Callable[..., list[Row]]
    load_heldout: Callable[..., list[Row]]
    verify: Callable[[str, str], bool]


TASKS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        instruction=GSM8K_INSTRUCTION,
        load_grow_pool=load_gsm8k_grow_pool,
        load_dev_pool=load_gsm8k_dev_pool,
        load_heldout=load_gsm8k_heldout,
        verify=verify_gsm8k,
    ),
    "math": TaskSpec(
        instruction=MATH_INSTRUCTION,
        load_grow_pool=load_math_grow_pool,
        load_dev_pool=load_math_dev_pool,
        load_heldout=load_math_heldout,
        verify=verify_math,
    ),
}


def build_user_prompt(task: str, question: str) -> str:
    """The bare (no feedback prefix) user-turn content used identically at Improve-time
    training-target input, held-out eval, and Grow's Condition-A/round-0 sampling --
    Condition B's Grow prepends a feedback prefix *in front of* this, never inside it (see
    `sampling.py`)."""
    return f"{TASKS[task].instruction}\n\n{question}"
