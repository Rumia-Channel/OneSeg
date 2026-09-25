"""Capture a reproducible short central-one-segment IQ recording from RTL-SDR.

This intentionally does NOT claim to decode or display live ISDB-T television.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .channels import physical_channel_hz
from .dsp import DEFAULT_SAMPLE_RATE
from .ppm import PpmCorrection
from .continuous import READ_SAMPLES, record_stream

DEFAULT_SECONDS = 3.0


def capture_channel(
    output: Path,
    *,
    channel: int,
    seconds: float = DEFAULT_SECONDS,
    ppm: int = 0,
    gain: float | str = "auto",
    device_factory: Callable | None = None,
) -> dict:
    """Record raw I/Q with a dedicated USB reader and separate writer thread."""
    if not 0.25 <= seconds <= 30:
        raise ValueError("capture duration must be between 0.25 and 30 seconds")
    if not -200 <= ppm <= 200:
        raise ValueError("ppm must be -200..200")
    freq_hz = physical_channel_hz(channel)
    if gain != "auto" and not -10 <= float(gain) <= 50:
        raise ValueError("gain must be 'auto' or -10..50 dB")
    output = Path(output)
    if output.suffix.lower() != ".c64":
        raise ValueError("capture filename must have .c64 extension")
    if device_factory is None:
        from rtlsdr import RtlSdr
        device_factory = RtlSdr

    expected = round(seconds * DEFAULT_SAMPLE_RATE)
    device = None
    try:
        device = device_factory()
        device.sample_rate = DEFAULT_SAMPLE_RATE
        PpmCorrection().apply(device, ppm)
        device.center_freq = freq_hz
        device.gain = gain
        return record_stream(
            device,
            output,
            samples_required=expected,
            metadata={
                "source": "DS-DT308SV compatible RTL2832U / one-segment IQ",
                "center_frequency_hz": freq_hz,
                "physical_channel": channel,
                "ppm": ppm,
                "gain_db": gain,
                "capture_seconds": expected / DEFAULT_SAMPLE_RATE,
                "utc_completed": datetime.now(timezone.utc).isoformat(),
            },
        )
    finally:
        if device is not None:
            device.close()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, required=True, help="Japanese UHF physical ch 13..52")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--ppm", type=int, default=0)
    parser.add_argument("--gain", default="auto", help="auto or number of dB")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    target = args.output or Path(
        f"oneseg_ch{args.channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.c64"
    )
    try:
        gain = "auto" if args.gain.lower() == "auto" else float(args.gain)
        meta = capture_channel(
            target, channel=args.channel, seconds=args.seconds,
            ppm=args.ppm, gain=gain
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Capture failed: {exc}\n")
    print(
        f"Saved {meta['samples']:,} samples at {meta['center_frequency_hz']:,} Hz "
        f"({meta['capture_seconds']:.3f} sec) to {target}\n"
        f"Metadata: {target.with_suffix('.c64.json')}\n"
        "This is raw IQ, not MPEG-TS; it cannot be played as television."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
