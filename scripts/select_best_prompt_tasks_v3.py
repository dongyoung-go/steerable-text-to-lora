"""Pick the single winning-instruction task dir per original task/algorithm, for eval scripts
that should score only the best prompt textgrad/GEPA actually settled on -- not every `_dK`
per-instruction task dir build_tasks_from_{textgrad,gepa}_repro_v3.py writes (those exist so
training gets "LoRA per description"; this script is for restricting *eval* scope back down to
one winner per task, the same scope v2's builder baked in).

Read-only w.r.t. everything: only reads data/textgrad_repro/, data/gepa_repro/, and checks which
task dirs already exist under --tasks-out. Writes only to --out.

Winner selection, per task:
  1. PRIMARY: each source dir's own best_prompt.json records the literal instruction text
     textgrad/gepa itself settled on ("prompt" for textgrad, "candidate" for gepa) -- "the output
     of TextGrad and GEPA" in the most direct sense. Find which of the task's instruction groups
     (the same groups build_tasks_from_*_repro_v3.py forms via group_val_rows_by_prompt /
     group_val_rows_by_candidate, imported directly here so the two stay in lockstep) has that
     exact literal text, and use it if its task dir exists under --tasks-out (i.e. it survived
     the builder's --min-samples filter). Verified against the real data: 28/28 textgrad source
     dirs and 28/29 gepa source dirs have an exact-text match between best_prompt.json and some
     forward_outputs.jsonl group.
  2. FALLBACK: if best_prompt.json's text isn't found among the groups at all (rare -- one gepa
     task in the real data, apparently a quote/whitespace difference in how that one candidate
     was recorded), or its group didn't survive --min-samples, fall back to ranking every group
     by its own recorded val score and picking the best-scoring survivor:
       - textgrad: joined by iteration index (forward_outputs.jsonl row -> iterations.jsonl's
         "val_accuracy" at that same "iteration") -- verified against scripts/textgrad_repro.py
         that both files write the exact same `for iteration in range(total_iterations)` loop
         counter, a genuine shared key. See rank_groups_by_iteration.
       - gepa: NOT joinable by iteration index -- forward_outputs.jsonl's "iteration" is a global
         batch_evaluate call counter (every minibatch/val/test pass), while iterations.jsonl's
         "iteration" is an unrelated index into the accepted-candidate list (verified against
         scripts/gepa_repro.py). Joined by literal candidate text instead, against
         iterations.jsonl's own "candidate"/"val_aggregate_score" fields. See
         rank_groups_by_candidate_text.
       - a group's score is the MAX recorded score across everything it matches (a group can pool
         rows from more than one iteration/duplicate-scored candidate -- reverted textgrad
         rounds, repeated gepa candidates). A group matching nothing at all (e.g. two source dirs
         have a fully empty iterations.jsonl -- a gap in that data, confirmed harmless for the
         current dataset since both cases have only one candidate ever forward-passed against
         val, so there's no other group it could lose a ranking to) gets -inf, ranking last but
         still selectable.
  A task is skipped (with a warning) only if no group for it survived --min-samples at all.

    python scripts/select_best_prompt_tasks_v3.py \
        --textgrad-src-root data/textgrad_repro --gepa-src-root data/gepa_repro \
        --tasks-out /home/dg793/text-to-lora/tasks --out data/best_prompt_tasks_v3.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_tasks_from_gepa_repro_v3 import SRC_DIR_RE as GEPA_SRC_DIR_RE
from build_tasks_from_gepa_repro_v3 import group_val_rows_by_candidate, load_val_questions
from build_tasks_from_textgrad_repro_v3 import SRC_DIR_RE as TG_SRC_DIR_RE
from build_tasks_from_textgrad_repro_v3 import group_val_rows_by_prompt


def load_iteration_scores(iterations_path: Path, score_field: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for line in open(iterations_path):
        row = json.loads(line)
        scores[row["iteration"]] = row[score_field]
    return scores


def load_candidate_text_scores(iterations_path: Path, candidate_field: str, score_field: str) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for line in open(iterations_path):
        row = json.loads(line)
        scores.setdefault(row[candidate_field], []).append(row[score_field])
    return scores


def rank_groups_by_iteration(
    groups: dict[str, list[dict]], order: list[str], iteration_scores: dict[int, float]
) -> list[tuple[int, float]]:
    """textgrad only: forward_outputs.jsonl's "iteration" field is the exact same
    `for iteration in range(total_iterations)` loop counter iterations.jsonl's own "iteration"
    field is written from (verified directly against scripts/textgrad_repro.py) -- a genuine
    shared key, safe to join on. NOT valid for gepa, see rank_groups_by_candidate_text.

    Returns [(group_idx, score), ...] sorted by score descending, ties broken by group_idx
    ascending (first-appearance order) for determinism.

    Some forward_outputs.jsonl rows come from a pre-training baseline pass logged as
    iteration -1, which iterations.jsonl (only ever non-negative indices) has no entry for --
    such rows are excluded from the max rather than crashing on a missing key. A group with no
    scored iteration at all (only ever baseline-evaluated, never a live textgrad step) gets
    -inf so it ranks last but is still selectable as a last resort."""
    ranked = []
    for group_idx, text in enumerate(order):
        rows = groups[text]
        candidate_scores = [
            iteration_scores[r["iteration"]] for r in rows if r["iteration"] in iteration_scores
        ]
        score = max(candidate_scores) if candidate_scores else float("-inf")
        ranked.append((group_idx, score))
    return sorted(ranked, key=lambda t: (-t[1], t[0]))


def rank_groups_by_candidate_text(
    order: list[str], candidate_text_scores: dict[str, list[float]]
) -> list[tuple[int, float]]:
    """gepa only: forward_outputs.jsonl's "iteration" field is `call_counter["n"]`, a global
    batch_evaluate call counter (thousands of calls: every minibatch/val/test pass, not just
    accepted candidates) -- see scripts/gepa_repro.py's batch_evaluate(). iterations.jsonl's own
    "iteration" field is a completely different counter (`idx`, an index into
    GEPAResult.candidates, the accepted-candidate/tree-node list). These two "iteration" fields
    do NOT share a key space -- joining on iteration number (as textgrad safely can) silently
    mismatches almost every row for gepa. The only reliable join key is the literal candidate
    text itself, which both files carry verbatim.

    Multiple iterations.jsonl rows can legitimately share identical candidate text (the known
    noise-driven duplicate-val-pass case, see GEPA_VS_TEXTGRAD_COMPARISON.md) -- take the max
    recorded score across all of them, same convention as rank_groups_by_iteration."""
    ranked = []
    for group_idx, text in enumerate(order):
        candidate_scores = candidate_text_scores.get(text, [])
        score = max(candidate_scores) if candidate_scores else float("-inf")
        ranked.append((group_idx, score))
    return sorted(ranked, key=lambda t: (-t[1], t[0]))


def select_winner(
    task_name: str,
    prefix: str,
    order: list[str],
    ranked: list[tuple[int, float]],
    official_text: str,
    tasks_out_dir: Path,
) -> dict:
    """official_text is best_prompt.json's own literal winning instruction text -- tried first,
    verbatim, before falling back to the score-ranked list. Returns a summary dict; "winner" is
    None if nothing survived at all."""

    def dir_name(group_idx: int) -> str:
        return f"{prefix}_{task_name}_d{group_idx}"

    if official_text in order:
        official_idx = order.index(official_text)
        if (tasks_out_dir / dir_name(official_idx)).is_dir():
            official_score = next(s for i, s in ranked if i == official_idx)
            return {
                "task": task_name,
                "winner": dir_name(official_idx),
                "score": official_score,
                "source": "official_best_prompt",
                "n_groups": len(ranked),
            }

    for rank, (group_idx, score) in enumerate(ranked):
        if (tasks_out_dir / dir_name(group_idx)).is_dir():
            return {
                "task": task_name,
                "winner": dir_name(group_idx),
                "score": score,
                "source": "score_fallback",
                "n_groups": len(ranked),
                "n_checked_before_survivor": rank + 1,
            }

    return {
        "task": task_name,
        "winner": None,
        "score": None,
        "source": None,
        "n_groups": len(ranked),
    }


def process_textgrad(src_root: Path, tasks_out_dir: Path) -> list[dict]:
    summaries = []
    for src_dir in sorted(src_root.iterdir()):
        m = TG_SRC_DIR_RE.match(src_dir.name)
        forward_outputs_path = src_dir / "forward_outputs.jsonl"
        iterations_path = src_dir / "iterations.jsonl"
        best_prompt_path = src_dir / "best_prompt.json"
        if not m or not forward_outputs_path.exists() or not iterations_path.exists() or not best_prompt_path.exists():
            continue
        task_name = m.group("task")
        groups, order = group_val_rows_by_prompt(forward_outputs_path)
        iteration_scores = load_iteration_scores(iterations_path, "val_accuracy")
        ranked = rank_groups_by_iteration(groups, order, iteration_scores)
        official_text = json.load(open(best_prompt_path))["prompt"]
        summaries.append(select_winner(task_name, "textgrad_repro_v3", order, ranked, official_text, tasks_out_dir))
    return summaries


def process_gepa(src_root: Path, tasks_out_dir: Path) -> list[dict]:
    summaries = []
    for src_dir in sorted(src_root.iterdir()):
        m = GEPA_SRC_DIR_RE.match(src_dir.name)
        forward_outputs_path = src_dir / "forward_outputs.jsonl"
        iterations_path = src_dir / "iterations.jsonl"
        val_set_path = src_dir / "val_set.jsonl"
        best_prompt_path = src_dir / "best_prompt.json"
        if (
            not m
            or not forward_outputs_path.exists()
            or not iterations_path.exists()
            or not val_set_path.exists()
            or not best_prompt_path.exists()
        ):
            continue
        task_name = m.group("task")
        val_questions = load_val_questions(val_set_path)
        groups, order = group_val_rows_by_candidate(forward_outputs_path, val_questions)
        candidate_text_scores = load_candidate_text_scores(iterations_path, "candidate", "val_aggregate_score")
        ranked = rank_groups_by_candidate_text(order, candidate_text_scores)
        official_text = json.load(open(best_prompt_path))["candidate"]
        summaries.append(select_winner(task_name, "gepa_repro_v3", order, ranked, official_text, tasks_out_dir))
    return summaries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textgrad-src-root", default="data/textgrad_repro")
    ap.add_argument("--gepa-src-root", default="data/gepa_repro")
    ap.add_argument("--tasks-out", required=True)
    ap.add_argument("--out", default="data/best_prompt_tasks_v3.txt")
    args = ap.parse_args()

    tasks_out_dir = Path(args.tasks_out)
    summaries = process_textgrad(Path(args.textgrad_src_root), tasks_out_dir) + process_gepa(
        Path(args.gepa_src_root), tasks_out_dir
    )

    winners = [s["winner"] for s in summaries if s["winner"] is not None]
    skipped = [s for s in summaries if s["winner"] is None]

    print(f"{'task':<50} {'winner':<55} {'score':>7} {'source':>20} {'n_groups':>9}")
    for s in summaries:
        if s["winner"] is None:
            continue
        print(f"{s['task']:<50} {s['winner']:<55} {s['score']:>7.3f} {s['source']:>20} {s['n_groups']:>9}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for w in winners:
            f.write(w + "\n")
    print(f"\nwrote {len(winners)} winning task dir(s) to {args.out}")

    n_fallback = sum(1 for s in summaries if s.get("source") == "score_fallback")
    if n_fallback:
        print(f"({n_fallback} used the score-ranked fallback -- best_prompt.json's own pick didn't survive --min-samples or had no exact text match)")

    if skipped:
        print(f"\nskipped {len(skipped)} task(s) with no surviving instruction group at all:")
        for s in skipped:
            print(f"  {s['task']}: {s['n_groups']} group(s), none survived --min-samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
