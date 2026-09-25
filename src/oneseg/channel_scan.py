"""Physical UHF channel RF-energy scan for an RTL-SDR, NOT ISDB-T service scan.

A 2.048 MS/s RTL-SDR sees only the center of each 6 MHz TV channel.
Power above the scan median can highlight RF candidates, but is not
proof that ISDB-T or a particular station was decoded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np

from .channels import FIRST_CHANNEL, LAST_CHANNEL, physical_channel_hz
from .dsp import DEFAULT_SAMPLE_RATE
from .quality import block_quality

SCAN_READ = 65536
SETTLE_READS = 2
MEASURE_READS = 3


@dataclass
class ChannelMeasurement:
    physical_channel: int
    frequency_hz: int
    power_dbfs: float
    rms: float
    clipping_percent: float
    relative_db: float = 0.0
    status: str = "UNASSESSED"

    def as_dict(self) -> dict:
        return asdict(self)


def analyze_samples(channel: int, chunks: list[np.ndarray]) -> ChannelMeasurement:
    """DC-corrected wideband IQ power; exact full-scale occupancy of raw IQ."""
    if not chunks:
        raise ValueError("at least one measurement chunk required")
    total = clipped = 0
    power_sum = 0.0
    for samples in chunks:
        iq = np.asarray(samples, dtype=np.complex64)
        if len(iq) == 0:
            raise ValueError("empty measurement chunk")
        hits, n, _ = block_quality(iq)
        clipped += hits
        total += n
        # Remove the RTL2832U's DC spike rather than scoring it as a TV signal.
        centered = iq - np.mean(iq, dtype=np.complex128)
        power_sum += float(np.sum(
            centered.real.astype(np.float64) ** 2
            + centered.imag.astype(np.float64) ** 2,
            dtype=np.float64,
        ))
    power = power_sum / total
    rms = math.sqrt(power)
    # IQ full scale is max (I^2 + Q^2) == 2. Only compare at fixed tuner gain.
    dbfs = 10 * math.log10(max(power / 2, 1e-15))
    return ChannelMeasurement(
        physical_channel=channel,
        frequency_hz=physical_channel_hz(channel),
        power_dbfs=dbfs,
        rms=rms,
        clipping_percent=100 * clipped / total,
    )


def rank_measurements(
    results: list[ChannelMeasurement],
    *,
    threshold_db: float = 6.0,
    clipping_limit_percent: float = 5.0,
) -> list[ChannelMeasurement]:
    """Rank relative power across fixed-gain scan; NO television detection."""
    if not results:
        return results
    baseline = float(np.median([row.power_dbfs for row in results]))
    for row in results:
        row.relative_db = row.power_dbfs - baseline
        if row.clipping_percent > clipping_limit_percent:
            row.status = "OVERLOAD / RETEST"
        elif row.relative_db >= threshold_db:
            row.status = "RF CANDIDATE (NOT TV LOCK)"
        else:
            row.status = "NO STRONG RF CONTRAST"
    return results


def scan_channels(
    device,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    on_measurement: Callable[[ChannelMeasurement], None] | None = None,
    channels: range = range(FIRST_CHANNEL, LAST_CHANNEL + 1),
) -> tuple[list[ChannelMeasurement], bool]:
    """Scan using an *already open* RtlSdr on the owning thread.

    Keep device.gain and freq_correction fixed for the full scan. Return
    (measurements, cancelled). Always restore the pre-scan RF frequency.
    """
    before_hz = int(device.center_freq)
    results = []
    cancelled = False
    try:
        for ch in channels:
            if should_stop():
                cancelled = True
                break
            device.center_freq = physical_channel_hz(ch)
            for _ in range(SETTLE_READS):
                if should_stop():
                    cancelled = True
                    break
                device.read_samples(SCAN_READ)
            if cancelled:
                break
            chunks = []
            for _ in range(MEASURE_READS):
                if should_stop():
                    cancelled = True
                    break
                iq = np.asarray(device.read_samples(SCAN_READ), dtype=np.complex64)
                if len(iq) != SCAN_READ:
                    raise IOError(f"short scan read on {ch}ch: {len(iq)}/{SCAN_READ}")
                chunks.append(iq)
            if cancelled:
                break
            row = analyze_samples(ch, chunks)
            results.append(row)
            if on_measurement:
                on_measurement(row)
    finally:
        device.center_freq = before_hz
    return rank_measurements(results), cancelled
