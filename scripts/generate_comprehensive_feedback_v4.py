"""v4 experiment, stage A: derive a *comprehensive, generalized feedback* chain from each
TextGrad-repro run's iterations.jsonl, as an alternative T2L input to the optimized prompt text
itself (see docs/05_comprehensive_feedback_v4.md for the full design).

Read-only w.r.t. data/textgrad_repro/ (only ever opens iterations.jsonl) -- writes only under
--out-root, a brand new location, never data/textgrad_repro/ itself. Every output row carries
every key from its source iterations.jsonl row forward UNCHANGED and adds exactly one new key,
"comprehensive_feedback" -- no existing key is ever renamed, dropped, or overwritten, so nothing
about the v3 experiment (which reads iterations.jsonl / forward_outputs.jsonl directly) is
affected by this script's existence.

Algorithm (see docstring in build_tasks_from_comprehensive_feedback_v4.py for how this is
consumed downstream): each iterations.jsonl row already records, per TextGrad round, the prompt
in effect before the round ("prompt"), the 3 textual gradients computed against it
("textual_gradients"), the prompt TextGrad proposed from them, and whether that proposal was
kept or reverted for scoring worse on val ("reverted"; "updated_prompt" is already the
post-revert-decision value -- see textgrad_repro.py). We walk this in order, maintaining
cf_by_prompt: dict[prompt text -> comprehensive feedback text so far], keyed by the literal
prompt text so that when several rounds later revert back to a text this script has already
seen, they resume from the correct accumulated state instead of restarting from scratch:

  - seed cf_by_prompt[iterations[0]["prompt"]] = "" (baseline prompt, no feedback yet).
  - for each row in order:
      - cf_before = cf_by_prompt[row["prompt"]]
      - if not row["reverted"]: merge cf_before + row["textual_gradients"] into a new
        generalized-feedback paragraph via one Qwen3-14B call; store it at
        cf_by_prompt[row["updated_prompt"]]; this row's assigned feedback is the new merge.
      - if row["reverted"]: no LLM call needed -- this round's gradients are NOT folded into the
        chain (a proposal that scored worse doesn't get to extend the "generalizes well" feedback
        state); this row's assigned feedback is simply cf_before, unchanged. Note
        row["prompt"] itself is the same text next row too (TextGrad reverted the live prompt),
        so cf_by_prompt already has the right entry for the next iteration to look up.

Chains across the ~30 source dirs are independent of each other but each is internally sequential
(iteration i+1 can depend on iteration i's LLM output). We exploit the cross-chain independence by
processing all chains breadth-first by depth: at depth d, gather one merge prompt per chain that
has a non-reverted row at depth d, run them as a single batched llm.generate() call, then move to
depth d+1. This turns what could be ~1 LLM call per iterations.jsonl row (summed across every
source dir) into ~1 batched call per depth (bounded by the longest run's iteration count).

    uv run python scripts/generate_comprehensive_feedback_v4.py \\
        --src-root data/textgrad_repro \\
        --out-root data/textgrad_repro_comprehensive_feedback_v4 \\
        --model Qwen/Qwen3-14B
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SRC_DIR_RE = re.compile(r"^qwen-qwen3-14b_(?P<task>.+)_textgrad-repro$")

MERGE_PROMPT = """You are maintaining a running, generalized set of guidance notes for how to \
correctly solve problems of this kind. The notes must generalize beyond any single question -- \
do not reference specific numbers, names, or exact wording from the examples below; state the \
underlying principle or strategy instead.

Previous guidance notes: {previous}

New feedback from this round (per-example critiques of what went wrong and how to fix it):
{gradients}

Merge the previous guidance notes (if any) with the new feedback into a single, self-contained, \
generalized paragraph of guidance for solving problems of this kind. Remove redundant points. \
Resolve contradictions in favor of the more recent feedback. Do not write a system prompt or \
instruction to a model -- write guidance/feedback notes only. Keep it concise: no more than \
{max_words} words. Output only the merged paragraph, nothing else."""


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
    # Merge calls run thinking-off -- this is text analysis over feedback, not the hard task
    # itself, same reasoning guide_rest/feedback.py and gepa_repro_common.VLLMLanguageModel give
    # for why their reflection/critique roles never pay for a <think> block.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_iterations(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["iteration"])
    return rows


def build_merge_content(previous: str, textual_gradients: list[str], max_words: int) -> str:
    gradients_block = "\n".join(f"- {g}" for g in textual_gradients)
    return MERGE_PROMPT.format(
        previous=previous or "(none yet)", gradients=gradients_block, max_words=max_words
    )


def process_all_chains(
    src_dirs: list[tuple[str, Path]],
    llm,
    tokenizer,
    temperature: float,
    max_tokens: int,
    max_words: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Runs the depth-first batched chain walk across every source dir at once. Returns
    {task_name: [output rows]}, one output row per input iterations.jsonl row, in order."""
    from vllm import SamplingParams

    chains: dict[str, list[dict]] = {}
    cf_by_prompt: dict[str, dict[str, str]] = {}
    out_rows: dict[str, list[dict]] = {}
    for task_name, src_dir in src_dirs:
        rows = load_iterations(src_dir / "iterations.jsonl")
        if not rows:
            continue
        chains[task_name] = rows
        cf_by_prompt[task_name] = {rows[0]["prompt"]: ""}
        out_rows[task_name] = []

    max_depth = max((len(rows) for rows in chains.values()), default=0)
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)

    for depth in range(max_depth):
        batch_prompts: list[str] = []
        batch_keys: list[tuple[str, dict]] = []  # (task_name, row) needing an LLM merge this depth
        for task_name, rows in chains.items():
            if depth >= len(rows):
                continue
            row = rows[depth]
            if row["reverted"]:
                continue
            cf_before = cf_by_prompt[task_name][row["prompt"]]
            content = build_merge_content(cf_before, row["textual_gradients"], max_words)
            batch_prompts.append(build_chat_prompt(tokenizer, content))
            batch_keys.append((task_name, row))

        merged_by_key: dict[tuple[str, int], str] = {}
        if batch_prompts:
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for (task_name, row), output in zip(batch_keys, outputs, strict=True):
                merged_by_key[(task_name, row["iteration"])] = output.outputs[0].text.strip()

        for task_name, rows in chains.items():
            if depth >= len(rows):
                continue
            row = rows[depth]
            if row["reverted"]:
                assigned = cf_by_prompt[task_name][row["prompt"]]
            else:
                assigned = merged_by_key[(task_name, row["iteration"])]
                cf_by_prompt[task_name][row["updated_prompt"]] = assigned
            out_rows[task_name].append({**row, "comprehensive_feedback": assigned})

    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default="data/textgrad_repro")
    ap.add_argument("--out-root", default="data/textgrad_repro_comprehensive_feedback_v4")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_words", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    src_dirs: list[tuple[str, Path]] = []
    for src_dir in sorted(src_root.iterdir()):
        m = SRC_DIR_RE.match(src_dir.name)
        if not m or not (src_dir / "iterations.jsonl").exists():
            continue
        task_name = m.group("task")
        out_path = out_root / task_name / "comprehensive_feedback_v4.jsonl"
        if out_path.exists():
            print(f"  {task_name}: comprehensive_feedback_v4.jsonl already exists, skipping")
            continue
        src_dirs.append((task_name, src_dir))

    if not src_dirs:
        print("nothing to do -- every source dir already has comprehensive_feedback_v4.jsonl")
        return 0

    print(f"loading {args.model} via vLLM ...")
    llm, tokenizer = load_vllm_engine(args.model, args.gpu_memory_utilization, args.max_model_len, args.seed)

    out_rows_by_task = process_all_chains(
        src_dirs, llm, tokenizer, args.temperature, args.max_tokens, args.max_words, args.seed
    )

    for task_name, out_rows in out_rows_by_task.items():
        out_path = out_root / task_name / "comprehensive_feedback_v4.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        n_reverted = sum(1 for r in out_rows if r["reverted"])
        print(f"  {task_name}: wrote {len(out_rows)} rows ({n_reverted} reverted) -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
