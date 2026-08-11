from scripts.render_arm_config import deep_merge, render

ALL_ARMS = [
    "arm1_floor",
    "arm2_instance_off",
    "arm3_instance_on",
    "arm4_trajectory_off",
    "arm5_trajectory_on",
]


def test_deep_merge_override_wins_on_scalar_conflict():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"a": 9, "b": {"c": 8}}
    merged = deep_merge(base, override)
    assert merged == {"a": 9, "b": {"c": 8, "d": 3}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


def test_render_every_arm_produces_expected_top_level_keys():
    for arm in ALL_ARMS:
        resolved = render(arm)
        assert "algorithm" in resolved
        assert "actor_rollout_ref" in resolved
        assert "custom" in resolved
        assert resolved["custom"]["frontier_model"] == "gpt-5-mini"


def test_render_arm1_floor_has_no_conditioning():
    resolved = render("arm1_floor")
    assert resolved["custom"]["conditioning"] == "none"
    assert resolved["custom"]["internalization"] is False
    assert resolved["custom"]["calibration"] is False


def test_render_arm3_and_arm5_enable_internalization_and_calibration():
    for arm in ("arm3_instance_on", "arm5_trajectory_on"):
        resolved = render(arm)
        assert resolved["custom"]["internalization"] is True
        assert resolved["custom"]["calibration"] is True


def test_render_arm2_and_arm4_match_off_off():
    for arm in ("arm2_instance_off", "arm4_trajectory_off"):
        resolved = render(arm)
        assert resolved["custom"]["internalization"] is False
        assert resolved["custom"]["calibration"] is False


def test_render_arm_conditioning_matches_content_column():
    assert render("arm2_instance_off")["custom"]["conditioning"] == "critique"
    assert render("arm3_instance_on")["custom"]["conditioning"] == "critique"
    assert render("arm4_trajectory_off")["custom"]["conditioning"] == "momentum"
    assert render("arm5_trajectory_on")["custom"]["conditioning"] == "momentum"


def test_render_preserves_base_hyperparameters_shared_across_arms():
    for arm in ALL_ARMS:
        resolved = render(arm)
        assert resolved["actor_rollout_ref"]["rollout"]["n"] == 8
        assert resolved["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1e-6
