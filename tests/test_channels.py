import pytest

from oneseg.channels import physical_channel_hz


@pytest.mark.parametrize("channel,hz", [
    (13, 473_142_857),
    (14, 479_142_857),
    (27, 557_142_857),
    (52, 707_142_857),
])
def test_rf_channel_center(channel, hz):
    assert physical_channel_hz(channel) == hz


@pytest.mark.parametrize("channel", [12, 53, -1])
def test_channel_range(channel):
    with pytest.raises(ValueError):
        physical_channel_hz(channel)


def test_non_integer_rejected():
    with pytest.raises(TypeError):
        physical_channel_hz(True)
