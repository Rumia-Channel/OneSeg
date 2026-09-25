import numpy as np
import pytest

from oneseg.channel_scan import (
    ChannelMeasurement,
    SCAN_READ,
    analyze_samples,
    rank_measurements,
    scan_channels,
)
from oneseg.channels import physical_channel_hz


def test_dc_offset_removed_and_full_scale_detected():
    samples = np.full(512, 0.6 + 0.4j, dtype=np.complex64)
    data = analyze_samples(27, [samples])
    assert data.physical_channel == 27
    assert data.frequency_hz == 557_142_857
    assert data.power_dbfs < -100
    loud = analyze_samples(27, [np.tile([1+0j, 0+0j], 256)])
    assert loud.clipping_percent == pytest.approx(50)


def test_relative_ranking_not_tv_declaration():
    base = [ChannelMeasurement(i, physical_channel_hz(i), -65, 0.002, 0)
            for i in range(13, 20)]
    base[2].power_dbfs = -48
    base[4].power_dbfs = -40
    base[4].clipping_percent = 49
    rank_measurements(base)
    assert base[2].status == "RF CANDIDATE (NOT TV LOCK)"
    assert base[4].status == "OVERLOAD / RETEST"
    assert all(x.status == "NO STRONG RF CONTRAST" for x in base if x not in (base[2], base[4]))


class FakeDevice:
    def __init__(self):
        self.center_freq = 82_500_000
        self.reads = 0
        self.tunes = []

    def read_samples(self, length):
        assert length == SCAN_READ
        self.reads += 1
        amp = 0.3 if self.center_freq == physical_channel_hz(14) else 0.01
        rng = np.random.default_rng(self.reads)
        return ((rng.standard_normal(length) + 1j * rng.standard_normal(length))
                * amp).astype(np.complex64)

    def __setattr__(self, name, value):
        if name == "center_freq" and "tunes" in self.__dict__:
            self.tunes.append(value)
        super().__setattr__(name, value)


def test_scan_no_hardware_restores_frequency_and_reports_candidates():
    device = FakeDevice()
    rows, cancelled = scan_channels(device, channels=range(13, 16))
    assert not cancelled
    assert device.center_freq == 82_500_000
    assert len(rows) == 3
    assert device.reads == 3 * 5
    assert rows[1].status == "RF CANDIDATE (NOT TV LOCK)"
    assert rows[1].relative_db > 6


def test_cancel_and_device_restore_on_error():
    device = FakeDevice()
    rows, cancelled = scan_channels(
        device, channels=range(13, 16),
        should_stop=lambda: device.reads >= 5,
    )
    assert cancelled
    assert len(rows) == 1
    assert device.center_freq == 82_500_000

    class BrokenDevice(FakeDevice):
        def read_samples(self, length):
            raise OSError("unplugged")

    broken = BrokenDevice()
    with pytest.raises(OSError, match="unplugged"):
        scan_channels(broken, channels=range(13, 16))
    assert broken.center_freq == 82_500_000
