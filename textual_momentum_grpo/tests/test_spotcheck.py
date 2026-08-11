from tmgrpo.spotcheck import spot_check


def test_spot_check_agrees_when_gradient_claims_improvement_and_accuracy_rose():
    result = spot_check("The policy is improving steadily.", prev_step_accuracy=0.3, this_step_accuracy=0.5)
    assert result.claimed_direction == "improving"
    assert result.measured_direction == "improving"
    assert result.agrees is True


def test_spot_check_disagrees_when_gradient_claims_improvement_but_accuracy_fell():
    result = spot_check("Clear progress this step.", prev_step_accuracy=0.6, this_step_accuracy=0.3)
    assert result.claimed_direction == "improving"
    assert result.measured_direction == "worsening"
    assert result.agrees is False


def test_spot_check_unclear_claim_always_agrees():
    result = spot_check("Mixed results across problems.", prev_step_accuracy=0.5, this_step_accuracy=0.5)
    assert result.claimed_direction == "unclear"
    assert result.agrees is True


def test_spot_check_stuck_within_flat_threshold():
    result = spot_check("Performance is stuck.", prev_step_accuracy=0.40, this_step_accuracy=0.41)
    assert result.measured_direction == "stuck"
    assert result.claimed_direction == "worsening"
    assert result.agrees is False
