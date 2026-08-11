"""Rule-based reward for MATH-style problems: extract the boxed answer and check equivalence.

README section 4.1 (self_correct_grpo) / section 5 (textual_momentum_grpo) both call for a
grounded, non-learned verifier -- exact-match/equivalence checking, not a reward model. This
wraps the `math-verify` library (https://github.com/huggingface/Math-Verify), which already
handles LaTeX/fraction/set normalization correctly, rather than reimplementing that logic.
"""

from __future__ import annotations

from math_verify import parse, verify


def extract_boxed_answer(solution: str) -> str | None:
    """Pull the content of the last \\boxed{...} in a MATH-style solution string.

    Returns None if no boxed answer is found. Handles nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    marker = "\\boxed{"
    start = solution.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(solution) and depth > 0:
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(chars)


def check_answer(prediction: str, label: str) -> bool:
    """True if `prediction` (a boxed-answer string or full response) is equivalent to `label`.

    `prediction` may be a raw model response (in which case the last \\boxed{...} is extracted
    first) or an already-extracted answer string. `label` is always treated as the ground-truth
    answer string (e.g. the `label` field of the vendored MATH500/AIME24/OlympiadBench jsonl).
    """
    boxed = extract_boxed_answer(prediction)
    candidate = boxed if boxed is not None else prediction
    try:
        parsed_pred = parse(f"${candidate}$")
        parsed_label = parse(f"${label}$")
        return bool(verify(parsed_label, parsed_pred))
    except Exception:
        return False


def reward_fn(response: str, label: str) -> float:
    """verl-facing reward function: 1.0 if the response's boxed answer matches `label`, else 0.0."""
    return 1.0 if check_answer(response, label) else 0.0
