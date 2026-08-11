"""Guide-ReST Step 3: Feedback generation (Condition B only), TextGrad-inspired two-stage
structure -- see docs/guide_rest_README.md's "Feedback Generation Procedure".

Stage 1: sample N (question, incorrect completion, correct completion) triples from this
round's `grow_samples.jsonl` and, for each, prompt the model for a short critique.
Stage 2: merge the previous round's comprehensive feedback with these N local critiques
into a single, capped-length paragraph.

The critic is `M_t` -- the same checkpoint that just ran this round's Grow (self-critique,
per the user's explicit decision; no separate/fixed critic model). Loaded as its own vLLM
instance, own process, exits after writing `local_feedback.jsonl`/`feedback.txt` -- same
one-model-load-per-step-per-round discipline as `sampling.py`, see docs/01_train.md.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

STAGE1_PROMPT = """Here is a math question, an incorrect answer, and a correct answer.

Question:
{question}

Incorrect answer:
{incorrect}

Correct answer:
{correct}

Explain concisely what the incorrect answer got wrong and how to avoid this mistake. \
Do not restate the correct answer's final numeric result -- describe the reasoning error \
and the fix in general terms that would help on similar future problems."""

STAGE2_PROMPT = """Previous guidance: {previous}

New critiques from this round:
{critiques}

Merge the previous guidance (if any) and the new critiques into a single, self-contained \
paragraph of guidance for solving problems of this kind. Remove redundant points. Resolve \
any contradictions. Keep it concise: no more than {max_words} words. Output only the \
merged paragraph, nothing else."""


def load_vllm_engine(model_dir: str, gpu_memory_utilization: float, max_model_len: int, seed: int):
    from vllm import LLM

    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    return llm, llm.get_tokenizer()


def build_chat_prompt(tokenizer, user_content: str) -> str:
    # Critique/merge calls always run thinking-off -- this is text analysis over feedback,
    # not the hard task itself, same reasoning gepa_repro_common.VLLMLanguageModel gives
    # for why its reflection role never pays for a <think> block.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def sample_triples(grow_samples_path: Path, n: int, seed: int) -> list[dict]:
    by_question: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"correct": [], "incorrect": []})
    with open(grow_samples_path) as f:
        for line in f:
            row = json.loads(line)
            bucket = "correct" if row["correct"] else "incorrect"
            by_question[row["question"]][bucket].append(row["completion"])

    eligible = [q for q, buckets in by_question.items() if buckets["correct"] and buckets["incorrect"]]
    if not eligible:
        return []
    rng = random.Random(seed)
    chosen_questions = rng.choices(eligible, k=n) if len(eligible) < n else rng.sample(eligible, n)
    triples = []
    for q in chosen_questions:
        buckets = by_question[q]
        triples.append({
            "question": q,
            "incorrect": rng.choice(buckets["incorrect"]),
            "correct": rng.choice(buckets["correct"]),
        })
    return triples


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = sample_triples(Path(args.grow_samples), args.n, args.seed)
    local_feedback_path = out_dir / "local_feedback.jsonl"
    feedback_path = out_dir / "feedback.txt"

    if not triples:
        # No question in this round had both a correct and an incorrect sample (e.g. the
        # model is already at ceiling, or still at floor, on every question) -- nothing to
        # critique. Carry the previous feedback forward unchanged rather than erasing it.
        prev = Path(args.prev_feedback).read_text().strip() if args.prev_feedback and Path(args.prev_feedback).exists() else ""
        local_feedback_path.write_text("")
        feedback_path.write_text(prev)
        print("[feedback] no (correct, incorrect) pairs available this round; carried forward previous feedback")
        return

    llm, tokenizer = load_vllm_engine(
        args.checkpoint, args.gpu_memory_utilization, args.max_model_len, args.seed
    )
    from vllm import SamplingParams

    stage1_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)
    stage1_prompts = [
        build_chat_prompt(tokenizer, STAGE1_PROMPT.format(
            question=t["question"], incorrect=t["incorrect"], correct=t["correct"],
        ))
        for t in triples
    ]
    stage1_outputs = llm.generate(stage1_prompts, stage1_params, use_tqdm=False)
    local_feedbacks = [o.outputs[0].text.strip() for o in stage1_outputs]

    with open(local_feedback_path, "w") as f:
        for triple, lf in zip(triples, local_feedbacks, strict=True):
            f.write(json.dumps({**triple, "local_feedback": lf}) + "\n")

    previous = ""
    if args.prev_feedback and Path(args.prev_feedback).exists():
        previous = Path(args.prev_feedback).read_text().strip()

    critiques_block = "\n".join(f"- {lf}" for lf in local_feedbacks)
    stage2_prompt = STAGE2_PROMPT.format(previous=previous or "(none yet)", critiques=critiques_block, max_words=args.max_words)
    stage2_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)
    stage2_output = llm.generate([build_chat_prompt(tokenizer, stage2_prompt)], stage2_params, use_tqdm=False)
    merged = stage2_output[0].outputs[0].text.strip()

    feedback_path.write_text(merged)
    print(f"[feedback] n={len(triples)} merged feedback length (words)={len(merged.split())}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="M_t: the checkpoint that produced this round's grow_samples")
    p.add_argument("--grow_samples", required=True, help="this round's grow_samples.jsonl")
    p.add_argument("--prev_feedback", default=None, help="previous round's feedback.txt; omit/missing = round 0 (no prior feedback)")
    p.add_argument("--out_dir", required=True, help="round directory to write local_feedback.jsonl/feedback.txt into")
    p.add_argument("--n", type=int, default=8, help="number of Stage-1 critique triples")
    p.add_argument("--max_words", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
