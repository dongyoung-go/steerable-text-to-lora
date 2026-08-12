"""Inference-time iterative TextGrad-style refinement with T2L. See
``docs/07_iterative_t2l_application_v3.md``. This *evaluates a process*, not a fixed
(description, LoRA) pair -- unlike ``eval_accuracy.py``, which scores one fixed description's
LoRA once, this module runs several rounds of ``solve -> critique -> rewrite -> re-solve``
against the target model itself and tracks whether accuracy improves round over round.

Round 0 bootstraps with the target model solving using the task's own best training description
injected as a literal prompt (``eval_accuracy.build_prompted_prompt``), no LoRA at all -- mirrors
the ``prompted`` eval condition and gives the first critique call real generations to react to
before any LoRA exists.

Round 1..N never put the current instruction/feedback text into the target's own context
(``data/formatting.py::format_example``'s bare-question invariant is unchanged by this
experiment) -- instead that text is fed to T2L (``hypernet.generate_for_batch``) to produce a
fresh LoRA, which *replaces* the previous round's LoRA entirely rather than stacking with it: the
text itself already carries the accumulated history (each round's rewrite is conditioned on the
current text + this round's critique), so one LoRA per round already encodes the full trajectory.

``feedback_rows``/``score_rows`` are drawn once per task (disjoint, fixed across every round):
``feedback_rows`` are what the critique/rewrite calls see; ``score_rows`` are scored every round
to produce the accuracy-vs-round curve and are never shown to the critique step, so the loop is
not simply memorizing the exact rows it is graded on.
"""

from __future__ import annotations

import random

from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import Splits
from steerable_t2l.eval_accuracy import (
    ANSWER_PARSERS,
    GenerationConfig,
    build_prompted_prompt,
    classify_answer_parser,
    condition_desc,
    eval_rows_for_task,
    generate_texts,
)
from steerable_t2l.feedback_gen import critique as _critique_call
from steerable_t2l.feedback_gen import rewrite as _rewrite_call
from steerable_t2l.hypernet import SteerableHyperLoRA
from steerable_t2l.target_spec import TargetSpec


def split_feedback_and_score_rows(
    rows: list[dict], feedback_n: int, score_n: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """A fixed, disjoint ``(feedback_rows, score_rows)`` pair drawn once per task and reused
    for every round of its loop. Raises rather than silently shrinking either pool if ``rows``
    is too small -- a silently shrunk pool would quietly change what's being measured/critiqued
    round to round without any signal, the same class of hazard docs/04 §6 flags for gold joins.
    """
    if len(rows) < feedback_n + score_n:
        raise ValueError(
            f"need at least {feedback_n + score_n} rows (feedback_n={feedback_n} + "
            f"score_n={score_n}), got {len(rows)}"
        )
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:feedback_n], shuffled[feedback_n : feedback_n + score_n]


def _resolve_parser(rows: list[dict]):
    """Mirrors ``eval_accuracy.score_condition``'s embedded-gold path. This loop only ever runs
    against v2/v3-built tasks, which all carry a bare ``gold_answer`` on every row
    (``build_tasks_from_textgrad_repro_v2.py``'s change, docs/04 §11) -- the legacy
    external-GSM8K-join fallback ``score_condition`` also supports is intentionally not
    reproduced here. Classified once over the *union* of feedback/score rows so both pools are
    scored with the same parser, rather than each pool's small sample independently voting on
    its own (possibly different) parser.
    """
    if not all("gold_answer" in row for row in rows):
        raise ValueError(
            "every row must carry an embedded 'gold_answer' field (see "
            "build_tasks_from_textgrad_repro_v3.py) -- this loop does not support the legacy "
            "external-gold-index join eval_accuracy.score_condition falls back to"
        )
    gold_strs = [str(row["gold_answer"]) for row in rows]
    parser_name = classify_answer_parser(gold_strs)
    return ANSWER_PARSERS[parser_name]


def _score(rows: list[dict], responses: list[str], parse) -> dict:
    per_row = []
    n_correct = 0
    for row, response in zip(rows, responses, strict=True):
        gold = parse(str(row["gold_answer"]))
        predicted = parse(response)
        correct = predicted is not None and gold is not None and predicted == gold
        n_correct += int(correct)
        per_row.append(
            {
                "question": row.get("question"),
                "response": response,
                "predicted": None if predicted is None else str(predicted),
                "gold": None if gold is None else str(gold),
                "correct": correct,
            }
        )
    n = len(rows)
    return {"accuracy": (n_correct / n) if n else float("nan"), "n": n, "n_correct": n_correct, "rows": per_row}


def _bare_question_prompt(tokenizer, task: Task, row: dict) -> str:
    """Round 1+'s prompt: the bare question only, no instruction -- the same construction
    ``format_example`` uses for training/eval, reproduced here (not imported) since
    ``format_example`` also returns a response half this loop has no use for."""
    user_content = task.metadata.user_prompt_template.format(**row)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
    )


def run_iterative_t2l(
    hypernet: SteerableHyperLoRA,
    target,
    tokenizer,
    spec: TargetSpec,
    task: Task,
    splits: Splits,
    all_tasks: list[Task],
    feedback_llm,
    feedback_tokenizer,
    *,
    n_rounds: int = 5,
    feedback_n: int = 16,
    score_n: int = 16,
    gen_config: GenerationConfig | None = None,
    mode: str = "prompt",
    max_words: int = 150,
    seed: int = 0,
    critique_fn=None,
    rewrite_fn=None,
) -> dict:
    """Run the round-0-bootstrap + round-1..N T2L-LoRA loop for one task.

    ``critique_fn(current_text, examples) -> str`` / ``rewrite_fn(current_text, feedback) -> str``
    default to :func:`steerable_t2l.feedback_gen.critique`/``rewrite`` bound to
    ``(feedback_llm, feedback_tokenizer, mode, max_words)`` -- overridable so tests (and any
    future non-vLLM feedback backend) don't need a real engine.

    Returns a JSON-serializable report: ``{"task", "mode", "n_rounds", "rounds": [...]}``, one
    entry per round with ``held_out_accuracy`` (the number to plot against round index).
    """
    critique_fn = critique_fn or (
        lambda current_text, examples: _critique_call(feedback_llm, feedback_tokenizer, current_text, examples)
    )
    rewrite_fn = rewrite_fn or (
        lambda current_text, feedback_text: _rewrite_call(
            feedback_llm, feedback_tokenizer, current_text, feedback_text, mode=mode, max_words=max_words
        )
    )

    gen_config = gen_config or GenerationConfig()
    rng = random.Random(seed)
    device = next(target.parameters()).device

    rows = eval_rows_for_task(task, splits)
    feedback_rows, score_rows = split_feedback_and_score_rows(rows, feedback_n, score_n, rng)
    parse = _resolve_parser(feedback_rows + score_rows)

    base_prompt = condition_desc(task, splits, all_tasks, "prompted", rng)
    if base_prompt is None:
        raise ValueError(f"{task.name}: no base description available for round 0")

    current_text = base_prompt
    per_module_fixed = None  # round 0: no LoRA
    used_lora = False
    rounds: list[dict] = []

    for t in range(n_rounds + 1):
        instruction_in_context = t == 0
        if instruction_in_context:
            fb_prompts = [
                build_prompted_prompt(tokenizer, current_text, task.metadata.user_prompt_template.format(**row))
                for row in feedback_rows
            ]
            score_prompts = [
                build_prompted_prompt(tokenizer, current_text, task.metadata.user_prompt_template.format(**row))
                for row in score_rows
            ]
        else:
            fb_prompts = [_bare_question_prompt(tokenizer, task, row) for row in feedback_rows]
            score_prompts = [_bare_question_prompt(tokenizer, task, row) for row in score_rows]

        fb_responses = generate_texts(target, tokenizer, fb_prompts, spec, per_module_fixed, gen_config, device)
        score_responses = generate_texts(target, tokenizer, score_prompts, spec, per_module_fixed, gen_config, device)

        fb_score = _score(feedback_rows, fb_responses, parse)
        held_out_score = _score(score_rows, score_responses, parse)

        round_entry = {
            "round": t,
            "text": current_text,
            "used_lora": used_lora,
            "instruction_in_context": instruction_in_context,
            "feedback_pool_accuracy": fb_score["accuracy"],
            "held_out_accuracy": held_out_score["accuracy"],
            "feedback_pool_rows": fb_score["rows"],
            "held_out_rows": held_out_score["rows"],
        }

        if t < n_rounds:
            examples = [
                {"question": r["question"], "response": r["response"], "gold": r["gold"]}
                for r in fb_score["rows"]
            ]
            feedback_text = critique_fn(current_text, examples)
            next_text = rewrite_fn(current_text, feedback_text)
            round_entry["critique"] = feedback_text
            round_entry["next_text"] = next_text

            current_text = next_text
            per_module = hypernet.generate_for_batch([current_text])
            per_module_fixed = {m: (A[0], B[0]) for m, (A, B) in per_module.items()}
            used_lora = True

        rounds.append(round_entry)

    return {"task": task.name, "mode": mode, "n_rounds": n_rounds, "rounds": rounds}
