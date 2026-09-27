import pytest
from oneseg.gain_control import set_discrete_manual_gain


class BrokenReadback:
    valid_gains_db = [-9.9, -7.3, -6.5, -5.4, 5.8]

    def __init__(self):
        self.commanded = None

    @property
    def gain(self):
        return 0.0  # FC0013 getter seen in the real 2026-09-27 survey

    @gain.setter
    def gain(self, value):
        self.commanded = value


def test_accepted_negative_manual_gain_is_not_overridden_by_zero_getter():
    device = BrokenReadback()
    result = set_discrete_manual_gain(device, -7.3)
    assert device.commanded == -7.3
    assert result["applied_gain_db"] == -7.3
    assert result["gain_readback_db"] == 0.0
    assert not result["gain_readback_matches_command"]
    assert not result["actual_analog_gain_independently_verified"]


def test_requested_gain_is_rounded_to_nearest_supported_step():
    device = BrokenReadback()
    result = set_discrete_manual_gain(device, -5.0)
    assert device.commanded == -5.4
    assert result["requested_gain_db"] == -5
    assert result["commanded_gain_db"] == -5.4


def test_matched_readback_is_confirmed_but_not_independent_rf_calibration():
    class Normal:
        valid_gains_db = [-9.9, -7.3]

        def __init__(self):
            self.gain = 0.0

    data = set_discrete_manual_gain(Normal(), -7.3)
    assert data["gain_readback_matches_command"]
    assert not data["actual_analog_gain_independently_verified"]


def test_rejects_nan_without_touching_device():
    device = BrokenReadback()
    with pytest.raises(ValueError, match="finite"):
        set_discrete_manual_gain(device, float("nan"))
    assert device.commanded is None
