"""Top up each task's `descriptions` list with LLM-generated, contrastive-sibling-aware
paraphrases of its existing (usually sole) instruction -- the "(b) optionally add
scripts/paraphrase_descs.py" step `docs/03_training_validation.md` §4 named but never built.

Why this matters: every `textgrad_repro_v3_*`/`gepa_repro_v3_*`/`comprehensive_feedback_v4_*` task
dir has exactly one description today, so the D-axis (`splits.py`'s `d_holdout`) is universally
`n/a`, and both `build_recon_batches` (recon) and `PerTaskDescDataset` (SFT) -- which already
sample uniformly from a task's *whole* `descriptions` list every step -- never see more than one
phrasing per regression/training target. Adding paraphrases here needs no other code change on
the training side; `descriptions[0]` is left untouched (so `best_description_index` stays valid),
new entries are only ever appended.

Ported and adapted from the reference repo's `/home/dg793/text-to-lora/scripts/
textgrad_repro_paraphrase_descs.py` (same idea, same contrastive-sibling filtering approach), with
two adaptations:
  1. This repo's own `scripts/gepa_repro_common.py` vLLM plumbing is used for generation instead
     of the reference's `hyper_llm_modulator.steering.textgrad_gen_backend` (not available here);
     JSON-structured output is hand-parsed (`safe_parse_json`, ported from that same reference
     backend) since nothing in this repo does guided/structured decoding.
  2. A second, coarser contrastive rule is added on top of the reference's within-task-only
     check (see "Two-tier filtering" below) to also guard against a paraphrase drifting toward a
     *different* task's instructions -- a gap the reference design (siblings scoped to one
     `task_prefix` group only) doesn't cover.

Generalized across experiment families (v3, v4, and any future one), not hardcoded to any of
them -- see "Generalization" below. `--train-tasks` accepts glob patterns exactly like every
other v3/v4 script (`discover_tasks`, the builders, `select_best_prompt_tasks_v3.py`); passing
patterns from more than one family/algorithm in one invocation pools them into the same
same-task/other-task comparison, which is the more correct scope (e.g. `aqua`'s textgrad and
gepa variants really are the same underlying task).

Generalization
--------------
"Which task dirs are siblings of the same underlying task" is recovered purely from the task-dir
naming convention every family shares, `<family>_<task>_d<K>`, using only what the CALLER passed
on the command line -- no family name is hardcoded anywhere in this file:
  1. For each `--train-tasks` glob pattern, its literal prefix (the text before the first
     wildcard) is derived automatically, e.g. `"textgrad_repro_v3_*"` -> `"textgrad_repro_v3_"`.
  2. A task's underlying task key = its dir name with (a) whichever derived prefix matches
     stripped, then (b) a trailing `_d\\d+$` stripped. E.g. `textgrad_repro_v3_bbh_causal_
     judgement_d9` -> `bbh_causal_judgement`; `comprehensive_feedback_v4_aqua_d3` -> `aqua`.
  Not using `task.metadata.domain` for this: it is a coarse category (`"bbh"` for every `bbh_*`
  task -- see `domain_for()` in the builders), not a per-task key, so it would wrongly lump
  e.g. `bbh_causal_judgement` and `bbh_snarks` into one sibling group.

Two-tier filtering (per candidate, per task T in sibling group G; outsiders = every task not in G)
--------------------------------------------------------------------------------------------------
    sim_own             = cos(candidate, T's own original)
    sim_sibling_max     = max( cos(candidate, s's original) for s in siblings )
    sim_same_task_mean  = mean( cos(candidate, x's original) for x in {T} + siblings )
    sim_other_task_max  = max( cos(candidate, o's original) for o in outsiders )

    keep if  sim_own >= sim_threshold
         AND sim_own - sim_sibling_max         >= contrast_margin      # rule 1 (reference's)
         AND sim_same_task_mean - sim_other_task_max >= cross_task_margin  # rule 2 (new)

Rule 1 (existing, ported from the reference): don't let a paraphrase of task T's instruction drift
close enough to look like a *sibling* `_dK`'s genuinely different instruction for the same task.
Rule 2 (new): don't let it drift close enough to look like it belongs to a *different* task
entirely. These don't conflict -- they're a hierarchy (own > same-task siblings > other tasks),
and different task domains naturally use different-enough vocabulary that this rarely binds; it's
a cheap safety net for the residual gap the reference design left open.

Idempotent: `needed = target_n_descs - len(existing_descriptions)` per task, so tasks already at
or above the target are skipped entirely (no generation/embedding call for them), safe to rerun
repeatedly or to top up a higher `--target-n-descs` later.

Explicitly out of scope: this script only rewrites `metadata.yaml`. It does not regenerate
`data/splits_*.json` (needed before the D-axis reflects the new descriptions -- a separate, cheap
`scripts/make_splits.py --force` step) and does not retrain any recon/SFT checkpoint (expensive,
GPU-bound, and a deliberate separate decision). Oracle LoRA training is unaffected either way --
it trains on each task's `(question, response)` rows only and never touches `descriptions` (the
hypernetwork's own input, never the target model's, per `TaskMetadata`'s `system_message=""`
invariant).

    uv run python scripts/paraphrase_descs.py \\
        --tasks-root /home/dg793/text-to-lora/tasks \\
        --train-tasks textgrad_repro_v3_* gepa_repro_v3_* \\
        --target-n-descs 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from steerable_t2l.data.registry import Task, discover_tasks  # noqa: E402

# gepa_repro_common imports vllm at module level, an optional (`gen`) extra not installed in
# every environment (e.g. CPU-only dev/test nodes) -- deferred into generate_json_batch/main
# below so importing *this* module (and unit-testing its pure logic: literal_prefix,
# underlying_task_key, safe_parse_json, filter_paraphrases) never requires vllm to be present.

# -- generalization: recovering task-sibling groups from dir names only ------------------------

_TRAILING_INDEX_RE = re.compile(r"_d\d+$")
_GLOB_WILDCARD_CHARS = "*?["


def literal_prefix(pattern: str) -> str:
    """Text before a glob pattern's first wildcard character, e.g. ``"textgrad_repro_v3_*"``
    -> ``"textgrad_repro_v3_"``. Not family-specific: derived purely from whatever pattern the
    caller passed via ``--train-tasks``."""
    idx = min((pattern.index(c) for c in _GLOB_WILDCARD_CHARS if c in pattern), default=len(pattern))
    return pattern[:idx]


def underlying_task_key(task_name: str, prefixes: list[str]) -> str:
    """Strip whichever of ``prefixes`` matches (longest first, in case one is a prefix of
    another), then a trailing ``_d<K>``. Falls back to just stripping the suffix if no prefix
    matches (shouldn't happen for tasks ``discover_tasks`` actually returned for these patterns,
    but keeps this a non-crashing degradation rather than an assertion)."""
    for p in sorted(prefixes, key=len, reverse=True):
        if task_name.startswith(p):
            return _TRAILING_INDEX_RE.sub("", task_name[len(p) :])
    return _TRAILING_INDEX_RE.sub("", task_name)


# -- generation backend (ported from the reference repo's textgrad_gen_backend.py) --------------

CONTRASTIVE_PARAPHRASE_SYSTEM_PROMPT = """\
You are creating paraphrases of an instruction prompt for training data \
augmentation. Rewrite the TARGET prompt {n} times.

Rules:
- Preserve every instruction, caveat, and behavioral detail in the TARGET \
exactly -- do not drop, soften, generalize, or merge any of them.
- Vary only surface wording: sentence structure, word choice, clause \
order. Every instruction in the TARGET must remain fully recoverable from \
your paraphrase.
- The TARGET must stay clearly distinguishable from each SIBLING prompt \
listed below. Pay special attention to preserving whatever detail in the \
TARGET is absent from every sibling -- that detail is the whole point of \
this rewrite and must never be lost or blurred.
- Do not borrow phrasing from a sibling that would make your paraphrase \
read like a match for that sibling instead of the target.

Respond with a JSON object of the form {{"paraphrases": [/* {n} strings */]}}, \
nothing else.\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def safe_parse_json(text: str) -> dict | list | None:
    """Strip ``<think>...</think>``/```` ```json ```` fences, try ``json.loads``, fall back to a
    ``{...}``/``[...]`` regex extraction. Never raises -- returns ``None`` on failure so the
    caller can retry that one item instead of the whole batch."""
    text = _THINK_RE.sub("", text)
    fence = _JSON_FENCE_RE.search(text)
    candidate = fence.group(1) if fence else text
    for attempt in (candidate.strip(), text.strip()):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            pass
        m = _JSON_OBJ_RE.search(attempt)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def build_user_content(target: str, siblings: list[str]) -> str:
    lines = [f"TARGET prompt to paraphrase:\n{target}", "", "SIBLING prompts (must remain distinguishable from these):"]
    for i, sib in enumerate(siblings, 1):
        lines.append(f"{i}. {sib}")
    return "\n".join(lines)


def generate_json_batch(
    llm,
    tokenizer,
    system_prompt: str,
    user_contents: list[str],
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
) -> tuple[list[dict | list | None], list[str | None]]:
    """One shared system prompt, many independent user prompts, batched via this repo's own
    ``gepa_repro_common.batched_generate`` -- retries only the still-unparseable subset, up to
    ``max_retries`` times. A prompt left unparseable after all retries gets ``None`` (its
    candidates list, hence stays empty -- the caller treats that task as having 0 candidates
    this round rather than crashing the whole batch). Also returns the last raw generation text
    seen for each item (``None`` only if it was never generated at all, which shouldn't happen) --
    a still-``None`` parsed result with a non-``None`` raw text is exactly the "generation ran but
    didn't produce parseable JSON" case (e.g. ``max_tokens`` too small and the JSON got cut off
    mid-string), which is otherwise silent and hard to debug from the final empty-candidates list
    alone."""
    from gepa_repro_common import batched_generate

    n = len(user_contents)
    results: list[dict | list | None] = [None] * n
    last_raw: list[str | None] = [None] * n
    pending = list(range(n))
    for _ in range(max_retries):
        if not pending:
            break
        raw = batched_generate(
            llm, tokenizer, system_prompt, [user_contents[i] for i in pending],
            temperature=temperature, max_tokens=max_tokens, enable_thinking=False,
        )
        still_pending = []
        for idx, text in zip(pending, raw, strict=True):
            last_raw[idx] = text
            parsed = safe_parse_json(text)
            if parsed is not None:
                results[idx] = parsed
            else:
                still_pending.append(idx)
        pending = still_pending
    return results, last_raw


# -- embedding backend (ported from the reference repo's model_loading.py/pooling.py) -----------


def load_emb_model(emb_model_name: str, device: str):
    """CLS pooling (both ``gte-modernbert-base`` and ``gte-large-en-v1.5`` use it) needs
    right-padding -- see the assert in the reference's ``pooling.cls_pool``."""
    model = AutoModel.from_pretrained(emb_model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model, tokenizer


def _add_full_stop(text: str) -> str:
    text = text.strip()
    if text and text[-1].isalpha():
        text += "."
    return text


@torch.no_grad()
def embed_texts(texts: list[str], model, tokenizer, device: str, batch_size: int = 32) -> torch.Tensor:
    """CLS-pooled, L2-normalized embeddings, ``[len(texts), hidden]``."""
    formatted = [_add_full_stop(t) for t in texts]
    chunks = []
    for start in range(0, len(formatted), batch_size):
        batch = formatted[start : start + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=8192, return_tensors="pt").to(device)
        out = model(**enc)
        chunks.append(out.last_hidden_state[:, 0].float().cpu())
    embs = torch.cat(chunks, dim=0)
    return torch.nn.functional.normalize(embs, dim=-1)


# -- two-tier contrastive filtering (pure function -- unit-tested directly) ---------------------


def filter_paraphrases(
    *,
    own_emb: torch.Tensor,
    sibling_embs: torch.Tensor,
    other_task_embs: torch.Tensor,
    candidates: list[str],
    candidate_embs: torch.Tensor,
    sim_threshold: float,
    contrast_margin: float,
    cross_task_margin: float,
) -> tuple[list[str], list[dict]]:
    """See the module docstring's "Two-tier filtering" section for the exact rule. All embeddings
    are assumed already L2-normalized (so a dot product is a cosine similarity). ``sibling_embs``/
    ``other_task_embs`` may be empty (``[0, d]``) -- an empty sibling/outsider set makes that
    margin check trivially pass (nothing to be too close to)."""
    same_task_embs = torch.cat([own_emb.unsqueeze(0), sibling_embs], dim=0)
    kept: list[str] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for text, c_emb in zip(candidates, candidate_embs, strict=True):
        sim_own = float(c_emb @ own_emb)
        sim_sibling_max = float((c_emb @ sibling_embs.T).max()) if sibling_embs.shape[0] else float("-inf")
        sim_same_task_mean = float((c_emb @ same_task_embs.T).mean())
        sim_other_task_max = float((c_emb @ other_task_embs.T).max()) if other_task_embs.shape[0] else float("-inf")

        reason = None
        if sim_own < sim_threshold:
            reason = f"sim_to_own={sim_own:.3f} < sim_threshold={sim_threshold}"
        elif sim_own - sim_sibling_max < contrast_margin:
            reason = f"sim_to_own={sim_own:.3f} - sim_to_sibling={sim_sibling_max:.3f} < contrast_margin={contrast_margin}"
        elif sim_same_task_mean - sim_other_task_max < cross_task_margin:
            reason = (
                f"sim_same_task={sim_same_task_mean:.3f} - sim_other_task={sim_other_task_max:.3f} "
                f"< cross_task_margin={cross_task_margin}"
            )
        elif text in seen:
            reason = "duplicate"

        if reason:
            dropped.append({"text": text, "reason": reason})
        else:
            seen.add(text)
            kept.append(text)
    return kept, dropped


# -- write-back -----------------------------------------------------------------------------


def write_descriptions(tasks_root: str | Path, task_name: str, existing: list[str], new_paraphrases: list[str]) -> None:
    """Appends ``new_paraphrases`` (if any) to a task's existing descriptions and writes it back
    -- never drops or reorders what was already there, so ``descriptions[0]``/
    ``best_description_index`` stay valid."""
    if not new_paraphrases:
        return
    path = Path(tasks_root) / task_name / "metadata.yaml"
    with open(path) as f:
        metadata = yaml.safe_load(f)
    metadata["descriptions"] = existing + new_paraphrases
    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True, width=10**6)


def own_text(task: Task) -> str:
    idx = task.metadata.best_description_index
    return task.metadata.descriptions[idx if idx is not None else 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-root", default="/home/dg793/text-to-lora/tasks")
    ap.add_argument("--train-tasks", nargs="+", required=True, help="glob patterns under --tasks-root")
    ap.add_argument("--target-n-descs", type=int, default=8)
    ap.add_argument("--n-candidates", type=int, default=None, help="default: max shortfall across tasks needing more")
    ap.add_argument("--sim-threshold", type=float, default=0.80)
    ap.add_argument("--contrast-margin", type=float, default=0.05)
    ap.add_argument("--cross-task-margin", type=float, default=0.05)
    ap.add_argument("--model-dir", default="Qwen/Qwen3-14B")
    ap.add_argument("--emb-model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens-per-paraphrase", type=int, default=40)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--log-dir", default="data/paraphrase_descs_logs")
    ap.add_argument("--seed", type=int, default=0)
    # Not resolved via a `torch.cuda.is_available()` argparse default, and not resolved right
    # after parse_args() either: that call initializes a CUDA context in this (parent) process,
    # and vLLM's engine-core subprocess (spawned by load_vllm_engine() below) is forked -- forking
    # a process that already holds an initialized CUDA context is fatal ("Cannot re-initialize
    # CUDA in forked subprocess"). Left unresolved (`None`) here; resolved lazily right before the
    # embedding model load further down, which happens after the vLLM engine is already up.
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    if not tasks:
        print("no tasks matched --train-tasks -- nothing to do")
        return 0
    prefixes = [literal_prefix(p) for p in args.train_tasks]

    groups: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        groups[underlying_task_key(t.name, prefixes)].append(t)
    print(f"{len(tasks)} task dir(s) matched, grouped into {len(groups)} underlying task(s)")

    needed = {t.name: args.target_n_descs - len(t.metadata.descriptions) for t in tasks}
    tasks_needing = [t for t in tasks if needed[t.name] > 0]
    if not tasks_needing:
        print(f"all {len(tasks)} tasks already have >= {args.target_n_descs} descriptions; nothing to do")
        return 0
    skipped = len(tasks) - len(tasks_needing)
    if skipped:
        print(f"{skipped}/{len(tasks)} tasks already have >= {args.target_n_descs} descriptions, skipping them")

    n_candidates = args.n_candidates or max(needed[t.name] for t in tasks_needing)

    user_contents = [
        build_user_content(
            own_text(t), [own_text(s) for s in groups[underlying_task_key(t.name, prefixes)] if s.name != t.name]
        )
        for t in tasks_needing
    ]

    from gepa_repro_common import load_vllm_engine

    print(f"loading generation model {args.model_dir} ...")
    llm, tokenizer = load_vllm_engine(args.model_dir, seed=args.seed)

    # --max-tokens-per-paraphrase is a FLOOR, not the actual budget: a faithful paraphrase that
    # preserves every instruction/caveat (the whole point, per the system prompt) needs at least
    # as many tokens as the original target itself -- a flat 40-token default (tuned for the
    # reference repo's short LoL-style descriptions) silently truncates our longer, formatting-
    # heavy instructions mid-JSON, producing unparseable output with 0 candidates recovered (seen
    # in an initial smoke test: 189-char/~45-token targets need ~2x that per paraphrase, but a
    # shared 80-token budget for 2 candidates left no room for either one to complete).
    longest_target_tokens = max(len(tokenizer.encode(own_text(t))) for t in tasks_needing)
    per_paraphrase_tokens = max(args.max_tokens_per_paraphrase, longest_target_tokens + 20)
    max_tokens = n_candidates * per_paraphrase_tokens + 20  # +slack for the {"paraphrases": [...]} wrapper

    print(
        f"generating {n_candidates} candidate paraphrases each for {len(tasks_needing)} task(s) (batched), "
        f"max_tokens={max_tokens} ({per_paraphrase_tokens}/paraphrase, longest target={longest_target_tokens} tokens)"
    )
    responses, raw_texts = generate_json_batch(
        llm, tokenizer,
        CONTRASTIVE_PARAPHRASE_SYSTEM_PROMPT.format(n=n_candidates),
        user_contents,
        temperature=args.temperature,
        max_tokens=max_tokens,
        max_retries=args.max_retries,
    )

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading embedding model {args.emb_model} ...")
    emb_model, emb_tokenizer = load_emb_model(args.emb_model, args.device)

    all_names = [t.name for t in tasks]
    text_by_name = {t.name: own_text(t) for t in tasks}
    all_embs = embed_texts([own_text(t) for t in tasks], emb_model, emb_tokenizer, args.device)
    emb_by_name = dict(zip(all_names, all_embs, strict=True))
    emb_dim = all_embs.shape[-1]

    os.makedirs(args.log_dir, exist_ok=True)
    total_kept = 0
    for t, resp, raw_text in zip(tasks_needing, responses, raw_texts, strict=True):
        if isinstance(resp, dict):
            candidates = resp.get("paraphrases", [])
        elif isinstance(resp, list):
            candidates = resp
        else:
            candidates = []
        candidates = [c for c in candidates if isinstance(c, str) and c.strip()]

        key = underlying_task_key(t.name, prefixes)
        own = text_by_name[t.name]
        # Exclude any sibling/other-task whose own text is byte-identical to this task's own text
        # (e.g. textgrad and gepa both starting from the same unoptimized seed instruction,
        # independently producing identical d0's in each namespace -- confirmed on real aqua
        # data) -- there is no genuinely different instruction there to protect against, so
        # including it only makes the contrast margin artificially, spuriously harder to clear.
        sibling_names = [s.name for s in groups[key] if s.name != t.name and text_by_name[s.name] != own]
        other_names = [n for n in all_names if underlying_task_key(n, prefixes) != key and text_by_name[n] != own]

        if candidates:
            cand_embs = embed_texts(candidates, emb_model, emb_tokenizer, args.device)
            sibling_embs = torch.stack([emb_by_name[n] for n in sibling_names]) if sibling_names else torch.zeros(0, emb_dim)
            other_embs = torch.stack([emb_by_name[n] for n in other_names]) if other_names else torch.zeros(0, emb_dim)
            kept, dropped = filter_paraphrases(
                own_emb=emb_by_name[t.name],
                sibling_embs=sibling_embs,
                other_task_embs=other_embs,
                candidates=candidates,
                candidate_embs=cand_embs,
                sim_threshold=args.sim_threshold,
                contrast_margin=args.contrast_margin,
                cross_task_margin=args.cross_task_margin,
            )
        else:
            kept, dropped = [], []

        kept = kept[: needed[t.name]]
        total_kept += len(kept)

        log_entry = {"original": own_text(t), "kept": kept, "dropped": dropped, "raw_candidates": candidates}
        if not candidates:
            # Generation ran but produced 0 usable candidates -- keep the last raw model output
            # around so a truncated/unparseable response is diagnosable without rerunning.
            log_entry["raw_generation_output"] = raw_text
        with open(Path(args.log_dir) / f"{t.name}.json", "w") as f:
            json.dump(log_entry, f, indent=2)

        print(f"  {t.name}: {len(kept)}/{needed[t.name]} needed paraphrases kept (of {len(candidates)} candidates)")
        write_descriptions(args.tasks_root, t.name, list(t.metadata.descriptions), kept)

    print(f"\nwrote {total_kept} new description(s) across {len(tasks_needing)} task(s); logs under {args.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
