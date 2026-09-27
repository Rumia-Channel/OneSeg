"""Summarize recorded live RF/OFDM/FEC checks without the tuner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .telemetry import summarize_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="oneseg_live_*.jsonl")
    parser.add_argument("--json", type=Path, help="write aggregate report to new JSON")
    args = parser.parse_args()
    try:
        report = summarize_log(args.log)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Live diagnostic report failed: {exc}\n")
    print(f"Log: {report['path']}")
    print(f"Measured windows: {report['window_results']}; events: {report['events']}")
    for gain, values in sorted(report["by_applied_gain"].items()):
        clipping = values["fullscale_percent"]
        print(
            f"Applied gain {gain} dB: {values['windows']} windows, "
            f"{values['verified_tmcc_frames']} TMCC frames, "
            f"{values['accepted_ts']} real TS packets, "
            f"{values['failed']} failed windows; median full-scale "
            + (f"{median(clipping):.2f}%" if clipping else "unknown")
        )
    for stage, statuses in sorted(report["checks"].items()):
        print(f"{stage}: {statuses}")
    print("These metrics do not establish continuous TV playback.")
    if args.json:
        if args.json.exists():
            parser.exit(2, f"output already exists: {args.json}\n")
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
