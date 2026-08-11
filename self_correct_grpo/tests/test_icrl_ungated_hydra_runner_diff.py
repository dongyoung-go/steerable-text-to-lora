"""Verify icrl_ungated/hydra_runner.py only swaps the generate-function-path, nothing else."""

import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATED = ROOT / "vendor" / "ICRL" / "icrl" / "hydra_runner.py"
UNGATED = ROOT / "icrl_ungated" / "hydra_runner.py"

ANCHOR = "def _configure_math_datasets"


def _body(path: Path) -> list[str]:
    text = path.read_text()
    idx = text.index(ANCHOR)
    return text[idx:].splitlines(keepends=True)


def test_only_the_generate_function_path_differs():
    diff = list(difflib.unified_diff(_body(GATED), _body(UNGATED), lineterm=""))
    changed_lines = [
        line
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]
    assert changed_lines

    for line in changed_lines:
        stripped = line[1:].strip()
        is_target_line = "custom-generate-function-path" in line or '"icrl.generate.generate"' in line or (
            "self_correct_grpo.icrl_ungated.generate.generate" in line
        )
        is_comment = stripped.startswith("#")
        is_config_path = "config_path=" in line  # mechanical: file lives in a different directory
        assert is_target_line or is_comment or is_config_path, f"unexpected diff line: {line!r}"


def test_gated_points_at_upstream_generate():
    assert '"icrl.generate.generate"' in GATED.read_text()


def test_ungated_points_at_local_generate():
    text = UNGATED.read_text()
    assert '"self_correct_grpo.icrl_ungated.generate.generate"' in text
    assert '"icrl.generate.generate"' not in text
