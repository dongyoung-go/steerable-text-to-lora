"""Verify icrl_ungated/generate.py is a minimal diff against the vendored upstream file.

The Phase 1 pilot's entire premise (self_correct_grpo_README.md §1.1) depends on the ungated
variant differing from ICRL's own gated `generate.py` by exactly the gating condition and nothing
else. This test diffs the two files' bodies (from the `async def generate` function onward, past
the header/imports which mechanically differ because the ungated copy lives in a separate
top-level package) and asserts every changed line belongs to the expected gate-removal hunk.
"""

import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATED = ROOT / "vendor" / "ICRL" / "icrl" / "generate.py"
UNGATED = ROOT / "icrl_ungated" / "generate.py"

ANCHOR = "async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False):"


def _body(path: Path) -> list[str]:
    text = path.read_text()
    idx = text.index(ANCHOR)
    return text[idx:].splitlines(keepends=True)


def test_only_the_gate_condition_differs():
    gated_body = _body(GATED)
    ungated_body = _body(UNGATED)

    diff = list(difflib.unified_diff(gated_body, ungated_body, lineterm=""))
    changed_lines = [
        line
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]

    assert changed_lines, "expected at least the gate-removal diff, found no differences"

    # Every removed line from the gated file must be the original single-line gate check (plus
    # its two commented-out historical variants already present upstream); every added line from
    # the ungated file must be either the new gate check or part of the `# UNGATED:` explanatory
    # comment block replacing it. This is the actual invariant: the intended change is a comment
    # block plus a one-line condition swap, nothing structural.
    removed = [line for line in changed_lines if line.startswith("-")]
    added = [line for line in changed_lines if line.startswith("+")]

    for line in removed:
        assert "exec_success" in line or "round_id >= max_rounds" in line, (
            f"unexpected removed line outside the gate-removal change: {line!r}"
        )
    for line in added:
        stripped = line[1:].strip()
        is_gate_line = "round_id >= max_rounds" in line
        is_comment = stripped.startswith("#") or stripped == ""
        assert is_gate_line or is_comment, (
            f"unexpected added line outside the gate-removal change: {line!r}"
        )


def test_gated_file_still_has_the_oracle_gate():
    text = GATED.read_text()
    assert "if exec_success or round_id >= max_rounds:" in text


def test_ungated_file_drops_exec_success_from_the_break_condition():
    text = UNGATED.read_text()
    assert "if exec_success or round_id >= max_rounds:" not in text
    assert "if round_id >= max_rounds:" in text
