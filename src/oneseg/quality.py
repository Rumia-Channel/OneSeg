"""RTL-SDR sample-level recording quality diagnostics.

A large number of exactly-full-scale samples strongly suggests gain overload
or clipping and makes ISDB-T synchronization unreliable. This is *not* a
television demodulator, BER estimator or proof that a station is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def block_quality(samples: np.ndarray) -> tuple[int, int, float]:
    """Return (exact-full-scale either I/Q, count, mean complex power)."""
    samples = np.asarray(samples, dtype=np.complex64)
    if not len(samples):
        return 0, 0, 0.0
    if not np.isfinite(samples).all():
        raise ValueError("capture contains non-finite I/Q values")
    clipped = (np.abs(samples.real) >= 0.99999) | (
        np.abs(samples.imag) >= 0.99999
    )
    power = np.mean(
        samples.real.astype(np.float64) ** 2
        + samples.imag.astype(np.float64) ** 2
    )
    return int(np.count_nonzero(clipped)), int(len(samples)), float(power)


def recording_quality(path: Path) -> dict:
    """Stream a .c64 capture without reading the whole file into RAM."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "complex64" or metadata.get("byte_order") != "little-endian":
        raise ValueError("expected little-endian complex64 capture and sidecar metadata")
    if path.stat().st_size % 8:
        raise ValueError("I/Q file size must be divisible by 8 bytes")
    count = clipped = 0
    mean_power_times_n = 0.0
    with path.open("rb") as stream:
        while True:
            chunk = np.fromfile(stream, dtype="<c8", count=262144)
            if len(chunk) == 0:
                break
            hits, n, power = block_quality(chunk)
            count += n
            clipped += hits
            mean_power_times_n += power * n
    if count == 0:
        raise ValueError("empty capture")
    expected = metadata.get("capture_samples") or metadata.get("samples")
    if expected is not None and int(expected) != count:
        raise ValueError(
            f"truncated/mismatched capture: metadata {expected}, file {count} samples"
        )
    clipped_fraction = clipped / count
    return {
        "samples": count,
        "sample_rate_hz": metadata.get("sample_rate_hz"),
        "center_frequency_hz": metadata.get("center_frequency_hz"),
        "full_scale_samples": clipped,
        "full_scale_fraction": clipped_fraction,
        "rms": (mean_power_times_n / count) ** 0.5,
        "high_clipping_warning": clipped_fraction > 0.05,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect I/Q full-scale occupancy; this does NOT decode ISDB-T"
    )
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        result = recording_quality(args.capture)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"I/Q quality check failed: {exc}\n")
    print(
        f"Samples: {result['samples']:,}; sample rate: {result['sample_rate_hz']}; "
        f"center Hz: {result['center_frequency_hz']}\n"
        f"I/Q full-scale hits: {result['full_scale_samples']:,} "
        f"({result['full_scale_fraction'] * 100:.2f}%); "
        f"complex RMS: {result['rms']:.3f}"
    )
    if result["high_clipping_warning"]:
        print(
            "WARNING: High full-scale occupancy. Possible tuner/ADC overload; "
            "switch off Automatic RF gain, lower gain (FC0013 supports negative dB), "
            "and recapture on a locally active physical TV channel."
        )
    print("This report does not verify ISDB-T/TMCC lock or TS recovery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
