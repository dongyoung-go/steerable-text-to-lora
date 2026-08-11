"""scripts/paraphrase_descs.py: pure-logic coverage only (task-key generalization, JSON parsing,
and the two-tier contrastive filter) -- no vllm/model loading needed, matching this repo's
existing convention of not covering the v3/v4 builder scripts' end-to-end behavior with pytest
fixtures (smoke-tested against real data instead). The filter is genuinely new logic (the
cross-task margin doesn't exist in the reference repo's script), so it gets direct coverage here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "paraphrase_descs.py"
    spec = importlib.util.spec_from_file_location("paraphrase_descs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["paraphrase_descs"] = module
    spec.loader.exec_module(module)
    return module


pd = _load_module()


# -- literal_prefix / underlying_task_key (generalization across families) ----------------------


def test_literal_prefix_stops_at_first_wildcard():
    assert pd.literal_prefix("textgrad_repro_v3_*") == "textgrad_repro_v3_"
    assert pd.literal_prefix("comprehensive_feedback_v4_*") == "comprehensive_feedback_v4_"


def test_literal_prefix_no_wildcard_is_whole_string():
    assert pd.literal_prefix("literal_only") == "literal_only"


def test_underlying_task_key_strips_family_prefix_and_index_suffix():
    prefixes = ["textgrad_repro_v3_", "gepa_repro_v3_"]
    assert pd.underlying_task_key("textgrad_repro_v3_bbh_causal_judgement_d9", prefixes) == "bbh_causal_judgement"
    assert pd.underlying_task_key("gepa_repro_v3_bbh_causal_judgement_d1", prefixes) == "bbh_causal_judgement"
    assert pd.underlying_task_key("textgrad_repro_v3_gsm8k_d4", prefixes) == "gsm8k"


def test_underlying_task_key_generalizes_to_a_different_family():
    assert pd.underlying_task_key("comprehensive_feedback_v4_aqua_d3", ["comprehensive_feedback_v4_"]) == "aqua"


def test_underlying_task_key_falls_back_gracefully_with_no_matching_prefix():
    assert pd.underlying_task_key("some_task_d2", ["unrelated_prefix_"]) == "some_task"


# -- safe_parse_json ------------------------------------------------------------------------


def test_safe_parse_json_plain():
    assert pd.safe_parse_json('{"paraphrases": ["a", "b"]}') == {"paraphrases": ["a", "b"]}


def test_safe_parse_json_strips_think_block():
    text = '<think>reasoning about the rewrite</think>{"paraphrases": ["x"]}'
    assert pd.safe_parse_json(text) == {"paraphrases": ["x"]}


def test_safe_parse_json_strips_code_fence():
    text = 'Sure, here you go:\n```json\n{"paraphrases": ["x", "y"]}\n```'
    assert pd.safe_parse_json(text) == {"paraphrases": ["x", "y"]}


def test_safe_parse_json_returns_none_on_garbage():
    assert pd.safe_parse_json("no json here whatsoever") is None


# -- filter_paraphrases (the two-tier contrastive rule) ------------------------------------------

_D = 4  # tiny embedding dim -- these are synthetic, hand-placed unit vectors, not real embeddings


def _unit(*vals: float) -> torch.Tensor:
    v = torch.tensor(vals, dtype=torch.float32)
    return v / v.norm()


def test_kept_when_close_to_own_and_far_from_siblings_and_other_tasks():
    own = _unit(1.0, 0.0, 0.0, 0.0)
    sibling = _unit(0.0, 1.0, 0.0, 0.0)  # orthogonal -- a clearly different sibling instruction
    other_task = _unit(0.0, 0.0, 1.0, 0.0)  # orthogonal -- a clearly different task
    candidate = _unit(0.99, 0.05, 0.0, 0.0)  # nearly identical to own

    kept, dropped = pd.filter_paraphrases(
        own_emb=own,
        sibling_embs=sibling.unsqueeze(0),
        other_task_embs=other_task.unsqueeze(0),
        candidates=["a good paraphrase"],
        candidate_embs=candidate.unsqueeze(0),
        sim_threshold=0.80,
        contrast_margin=0.05,
        cross_task_margin=0.05,
    )
    assert kept == ["a good paraphrase"]
    assert dropped == []


def test_dropped_when_too_close_to_a_sibling():
    own = _unit(1.0, 0.0, 0.0, 0.0)
    sibling = _unit(0.9, 0.1, 0.0, 0.0)  # close to own -- realistic "different instruction, same task"
    other_task = _unit(0.0, 0.0, 1.0, 0.0)
    # candidate sits almost exactly between own and sibling: high sim to own, but not clearly
    # closer to own than to the sibling -- must fail the within-task contrast margin.
    candidate = _unit(0.95, 0.05, 0.0, 0.0)

    kept, dropped = pd.filter_paraphrases(
        own_emb=own,
        sibling_embs=sibling.unsqueeze(0),
        other_task_embs=other_task.unsqueeze(0),
        candidates=["blurs into sibling"],
        candidate_embs=candidate.unsqueeze(0),
        sim_threshold=0.80,
        contrast_margin=0.05,
        cross_task_margin=0.05,
    )
    assert kept == []
    assert len(dropped) == 1
    assert "sim_to_sibling" in dropped[0]["reason"]


def test_dropped_when_too_close_to_another_task_despite_passing_within_task_rule():
    own = _unit(1.0, 0.0, 0.0, 0.0)
    # no siblings at all for this task (a lone _d0) -- within-task margin trivially passes
    other_task = _unit(0.9, 0.1, 0.0, 0.0)  # a different task's instruction, but similarly worded
    candidate = _unit(0.95, 0.05, 0.0, 0.0)  # high sim to own AND high sim to the other task

    kept, dropped = pd.filter_paraphrases(
        own_emb=own,
        sibling_embs=torch.zeros(0, _D),
        other_task_embs=other_task.unsqueeze(0),
        candidates=["blurs into another task"],
        candidate_embs=candidate.unsqueeze(0),
        sim_threshold=0.80,
        contrast_margin=0.05,
        cross_task_margin=0.05,
    )
    assert kept == []
    assert len(dropped) == 1
    assert "sim_other_task" in dropped[0]["reason"]


def test_dropped_when_below_base_similarity_threshold():
    own = _unit(1.0, 0.0, 0.0, 0.0)
    candidate = _unit(0.0, 1.0, 0.0, 0.0)  # unrelated to its own original entirely

    kept, dropped = pd.filter_paraphrases(
        own_emb=own,
        sibling_embs=torch.zeros(0, _D),
        other_task_embs=torch.zeros(0, _D),
        candidates=["unrelated text"],
        candidate_embs=candidate.unsqueeze(0),
        sim_threshold=0.80,
        contrast_margin=0.05,
        cross_task_margin=0.05,
    )
    assert kept == []
    assert len(dropped) == 1
    assert "sim_threshold" in dropped[0]["reason"]


def test_duplicate_candidates_after_the_first_are_dropped():
    own = _unit(1.0, 0.0, 0.0, 0.0)
    candidate = _unit(0.99, 0.05, 0.0, 0.0)

    kept, dropped = pd.filter_paraphrases(
        own_emb=own,
        sibling_embs=torch.zeros(0, _D),
        other_task_embs=torch.zeros(0, _D),
        candidates=["same text", "same text"],
        candidate_embs=torch.stack([candidate, candidate]),
        sim_threshold=0.80,
        contrast_margin=0.05,
        cross_task_margin=0.05,
    )
    assert kept == ["same text"]
    assert dropped == [{"text": "same text", "reason": "duplicate"}]
