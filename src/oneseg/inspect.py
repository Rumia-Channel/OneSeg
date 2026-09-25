"""Offline OFDM cyclic-prefix diagnostics for captured .c64 complex I/Q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dsp import DEFAULT_SAMPLE_RATE, cyclic_prefix_candidates, to_one_seg_rate


def inspect(path: Path, max_seconds: float = 0.35) -> list[dict]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rate = int(metadata["sample_rate_hz"])
    if rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("Only 2.048 MS/s recordings are supported for this diagnostic")
    count = min(int(rate * max_seconds), rate)
    iq = np.fromfile(path, dtype="<c8", count=count)
    if len(iq) < 5000:
        raise ValueError("I/Q recording too short")
    # TODO: filter and align to the central segment before treating the
    # correlation peaks as useful. This tool deliberately reports candidates.
    return cyclic_prefix_candidates(to_one_seg_rate(iq))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Path to raw little-endian complex64 .c64")
    parser.add_argument("--seconds", type=float, default=0.35)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        result = inspect(args.capture, args.seconds)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Cannot inspect recording: {exc}\n")
    print("Unverified cyclic-prefix candidates (not a valid ISDB-T lock):")
    for row in result[:6]:
        print(f"mode={row['mode']} guard={row['guard']} sample={row['sample']} correlation={row['correlation']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
