import numpy as np
import pytest

from tmgrpo.calibration import apply_calibration, calibration_ratio


def test_calibration_ratio_equal_logprobs_is_one():
    logp = np.array([-1.0, -2.0, -0.5])
    ratio = calibration_ratio(logp, logp)
    np.testing.assert_allclose(ratio, np.ones_like(logp))


def test_calibration_ratio_down_weights_context_dependent_tokens():
    # Token was much more likely WITH context (logp_conditioned >> logp_unconditioned)
    # -> small ratio, i.e. down-weighted.
    logp_unconditioned = np.array([-5.0])
    logp_conditioned = np.array([-0.1])
    ratio = calibration_ratio(logp_unconditioned, logp_conditioned)
    assert ratio[0] < 0.1


def test_calibration_ratio_clips_at_w_max():
    logp_unconditioned = np.array([0.0])  # would-be ratio = exp(0 - (-10)) = e^10, huge
    logp_conditioned = np.array([-10.0])
    ratio = calibration_ratio(logp_unconditioned, logp_conditioned, w_max=2.0)
    assert ratio[0] == pytest.approx(2.0)


def test_calibration_ratio_shape_mismatch_raises():
    with pytest.raises(ValueError):
        calibration_ratio(np.array([0.0, 0.0]), np.array([0.0]))


def test_apply_calibration_multiplies_loss_and_returns_w_t():
    pg_loss = np.array([1.0, 1.0])
    logp_unconditioned = np.array([-1.0, -1.0])
    logp_conditioned = np.array([-1.0, -3.0])  # second token: ratio = exp(-1 - -3) = exp(2)
    calibrated, w_t = apply_calibration(pg_loss, logp_unconditioned, logp_conditioned, w_max=3.0)
    np.testing.assert_allclose(w_t, [1.0, 3.0])  # second clipped to w_max
    np.testing.assert_allclose(calibrated, [1.0, 3.0])
