"""Controlled FC0013 gain sweep: 20ch raw USB -> CP/pilots/TMCC -> real RS.

This is a *separate* tuner session, never an automatic change to active
live video. Each allowed hardware gain gets independent 3s windows. The
study checks raw ADC level, OFDM, pilot phase, protected TMCC and actual
RS-validated MPEG-TS before reporting evidence for a possible retest.

No I/Q or TS bytes are stored permanently. Only JSONL metadata and a
summary JSON are retained. Native asynchronous USB is never used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable

from .channels import physical_channel_hz
from .continuous import CaptureCancelled
from .decode import decode_capture
from .dsp import DEFAULT_SAMPLE_RATE
from .pilots import analyze_capture
from .ppm import PpmCorrection
from .raw_capture import (
    expand_raw_to_c64, raw_fullscale_percent, record_raw_window,
)
from .telemetry import SessionTelemetry, check_window, utc_now

FALLBACK_NEGATIVE_GAINS = (-9.9, -7.3, -6.5, -6.3, -6.0, -5.8, -5.4)


def available_survey_gains(device) -> list[float]:
    """Use ACTUAL advertised tuner gain steps, not arbitrary slider values."""
    native = getattr(device, "valid_gains_db", None)
    values = native if native is not None else FALLBACK_NEGATIVE_GAINS
    steps = sorted({
        float(item) for item in values
        if -10.0 <= float(item) <= 0.0
    })
    if not steps:
        raise ValueError(
            "tuner exposed no -10..0 dB manual gains for safe controlled scan"
        )
    return steps


def choose_retest_gain(groups: list[dict]) -> float | None:
    """A candidate only if repeated trials have real RS packets and no overload.

    This is a test priority, NOT a guaranteed RF optimum, service lock or
    automatic retune. Never reward spurious syncs or CP peaks alone.
    """
    eligible = [
        group for group in groups
        if group["successful_ts_windows"] >= 2
        and group["verified_tmcc_frames"] >= 2
        and group["max_fullscale_percent"] < 5.0
    ]
    if not eligible:
        return None
    return float(max(
        eligible,
        key=lambda row: (
            row["successful_ts_windows"],
            row["rs_packets"],
            -row["median_fullscale_percent"],
        ),
    )["applied_gain_db"])


def run_gain_survey(
    *,
    channel: int = 20,
    ppm: int = 0,
    seconds: float = 3.0,
    repeats: int = 2,
    gains: list[float] | None = None,
    output: Path | None = None,
    device_factory: Callable | None = None,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    probe_function: Callable = analyze_capture,
    decode_function: Callable = decode_capture,
    log_directory: Path | None = None,
) -> dict:
    """Sweep only when no other receiver owns USB. Leave its gain untouched
    after completing the experiment; no live tuner is launched."""
    if not 13 <= channel <= 52:
        raise ValueError("physical channel must be 13..52")
    if not -200 <= ppm <= 200:
        raise ValueError("ppm out of range")
    if not 1.0 <= seconds <= 3.0:
        raise ValueError("gain survey needs 1..3 seconds per window")
    if not 1 <= repeats <= 5:
        raise ValueError("repeats must be 1..5")
    if gains is not None and (
        not gains or len(gains) > 12
        or any(not -10 <= value <= 0 for value in gains)
    ):
        raise ValueError("gains must be 1..12 requested values in -10..0 dB")
    if device_factory is None:
        from rtlsdr import RtlSdr
        device_factory = RtlSdr
    device = None
    log = None
    results = []
    report = None
    try:
        log = SessionTelemetry(log_directory)
        output = (
            Path(output) if output is not None
            else log.path.with_suffix(".survey.json")
        )
        partial = output.with_name(output.name + ".partial")
        if output.exists() or partial.exists():
            raise FileExistsError(f"survey report already exists: {output}")
        log.emit(
            "survey_start", channel=channel,
            center_frequency_hz=physical_channel_hz(channel),
            ppm=ppm, window_seconds=seconds, repeats=repeats,
            log_path=str(log.path),
        )
        if progress:
            progress(f"Gain survey log: {log.path}")
        with TemporaryDirectory(prefix="oneseg-gain-survey-") as folder:
            root = Path(folder)
            device = device_factory()
            device.sample_rate = DEFAULT_SAMPLE_RATE
            PpmCorrection().apply(device, ppm)
            device.center_freq = physical_channel_hz(channel)
            supported = available_survey_gains(device)
            requested = (
                supported if gains is None
                else sorted({
                    min(supported, key=lambda step: abs(step - value))
                    for value in gains
                })
            )
            log.emit(
                "survey_gain_plan", supported_gains_db=supported,
                requested_gains_db=requested,
                total_windows=len(requested) * repeats,
            )
            for gain in requested:
                if cancelled():
                    break
                device.gain = gain  # only tuner-owning thread sets gain
                try:
                    applied = float(device.gain)
                except (TypeError, ValueError, AttributeError):
                    applied = gain
                for trial in range(repeats):
                    if cancelled():
                        break
                    index = len(results)
                    raw = root / f"survey_{index:04d}.u8iq"
                    c64 = root / f"survey_{index:04d}.c64"
                    ts = root / f"survey_{index:04d}.ts"
                    measurement = {
                        "window_seconds": seconds,
                        "gain_requested_db": gain,
                        "applied_gain_db": applied,
                        "trial": trial + 1,
                        "accepted_chunk": None,
                        "has_pat": None,
                        "has_pmt": None,
                    }
                    try:
                        if progress:
                            progress(
                                f"RF gain {applied:g} dB trial {trial+1}/"
                                f"{repeats}: recording {seconds:g}s..."
                            )
                        record_raw_window(
                            device, raw,
                            samples_required=round(
                                DEFAULT_SAMPLE_RATE * seconds
                            ),
                            cancelled=cancelled,
                            warmup_buffers=1,
                        )
                        measurement["fullscale_percent"] = raw_fullscale_percent(raw)
                        expand_raw_to_c64(
                            raw, c64, metadata={
                                "center_frequency_hz":
                                    physical_channel_hz(channel),
                                "ppm": ppm,
                                "gain_db": applied,
                                "gain_requested_db": gain,
                                "source": "automated_gain_survey",
                            },
                        )
                        evidence = probe_function(c64, seconds=seconds)
                        measurement.update({
                            "cp_quality": evidence.get("cp_quality"),
                            "pilot_coherence": evidence.get(
                                "pilot_coherence"
                            ),
                            "rms": evidence.get("iq_rms"),
                            "tmcc_parity_verified_frames": len(
                                evidence.get("bch_parity_verified_frames", [])
                            ),
                            "tmcc_soft_source": evidence.get(
                                "tmcc_soft_source"
                            ),
                            "per_carrier_verified_counts": evidence.get(
                                "tmcc_per_carrier_verified_counts"
                            ),
                            "integer_offset_bins": evidence.get(
                                "integer_offset_bins"
                            ),
                            "fractional_cfo_hz": evidence.get(
                                "fractional_cfo_hz"
                            ),
                        })
                        if measurement["tmcc_parity_verified_frames"]:
                            try:
                                recovered = decode_function(
                                    c64, ts, seconds=seconds,
                                    max_ofdm_symbols=1020,
                                )
                            except (OSError, ValueError, KeyError) as exc:
                                measurement["accepted_chunk"] = 0
                                measurement["failure_reason"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                            else:
                                measurement.update({
                                    "accepted_chunk": recovered[
                                        "rs_and_ts_accepted_packets"
                                    ],
                                    "rejected_chunk": recovered[
                                        "rejected_rs_or_invalid_ts_packets"
                                    ],
                                    "has_pat": bool(
                                        recovered["pat_programs"]
                                    ),
                                    "has_pmt": bool(
                                        recovered["pmt_elementary_streams"]
                                    ),
                                    "stage_seconds": recovered.get(
                                        "stage_seconds", {}
                                    ),
                                })
                        else:
                            measurement["failure_reason"] = (
                                "no parity-verified TMCC frames"
                            )
                    except CaptureCancelled:
                        if progress:
                            progress("Gain survey stopped by user.")
                        break
                    except Exception as exc:
                        measurement["failure_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        log.emit(
                            "survey_window_error", window=index,
                            error=measurement["failure_reason"],
                        )
                    finally:
                        for path in (
                            raw, raw.with_name(raw.name + ".partial"),
                            c64, c64.with_suffix(".c64.json"),
                            ts, ts.with_suffix(".ts.json"),
                        ):
                            path.unlink(missing_ok=True)
                    measurement["checks"] = check_window(measurement)
                    results.append(measurement)
                    log.emit(
                        "window_result", window=index,
                        metrics=measurement,
                        checks=measurement["checks"],
                    )
                    if progress:
                        progress(
                            f"{applied:g} dB: "
                            f"clip {measurement.get('fullscale_percent')}%, "
                            f"CP {measurement.get('cp_quality')}, "
                            f"pilots {measurement.get('pilot_coherence')}, "
                            f"TMCC {measurement.get('tmcc_parity_verified_frames', 0)}, "
                            f"RS TS {measurement.get('accepted_chunk') or 0}"
                        )
            groups = []
            for gain in sorted(set(r["applied_gain_db"] for r in results)):
                rows = [
                    r for r in results if r["applied_gain_db"] == gain
                ]
                if not rows:
                    continue
                fullscale = [
                    r["fullscale_percent"] for r in rows
                    if r.get("fullscale_percent") is not None
                ]
                groups.append({
                    "applied_gain_db": gain,
                    "windows": len(rows),
                    "verified_tmcc_frames": sum(
                        r.get("tmcc_parity_verified_frames") or 0
                        for r in rows
                    ),
                    "successful_ts_windows": sum(
                        (r.get("accepted_chunk") or 0) > 0 for r in rows
                    ),
                    "rs_packets": sum(
                        r.get("accepted_chunk") or 0 for r in rows
                    ),
                    "max_fullscale_percent": (
                        max(fullscale) if fullscale else None
                    ),
                    "median_fullscale_percent": (
                        median(fullscale) if fullscale else None
                    ),
                })
            recommendation = choose_retest_gain(groups)
            report = {
                "format": "oneseg_gain_survey_v1",
                "utc_finished": utc_now(),
                "center_frequency_hz": physical_channel_hz(channel),
                "channel": channel,
                "ppm": ppm,
                "sample_rate_hz": DEFAULT_SAMPLE_RATE,
                "seconds_per_window": seconds,
                "repeats": repeats,
                "gains_advertised_db": supported,
                "log_jsonl": str(log.path),
                "report_json": str(output),
                "windows": results,
                "by_gain": groups,
                "promising_gain_for_manual_retest_db": recommendation,
                "retest_selection_requires_two_rs_positive_windows":
                    True,
                "continuous_tv_verified": False,
                "samples_continuity_verified": False,
                "stop_requested": bool(cancelled()),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("x", encoding="utf-8") as stream:
                json.dump(
                    report, stream, ensure_ascii=False, indent=2,
                    allow_nan=False,
                )
            os.replace(partial, output)
            log.emit(
                "survey_complete", report_json=str(output),
                surveyed_windows=len(results),
                promising_gain_db=recommendation,
            )
            if progress:
                progress(
                    f"Gain survey saved: {output}; "
                    f"possible retest gain {recommendation} dB "
                    "(none means insufficient RS/ADC evidence)."
                )
    finally:
        if device is not None:
            device.close()
        if log is not None:
            log.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, default=20)
    parser.add_argument("--ppm", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--gains", type=str,
        help="optional comma-separated -10..0 dB requested levels",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        gains = (
            [float(x) for x in args.gains.split(",")]
            if args.gains else None
        )
        result = run_gain_survey(
            channel=args.channel, ppm=args.ppm, seconds=args.seconds,
            repeats=args.repeats, gains=gains, output=args.output,
            progress=print,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Gain survey failed: {exc}\n")
    print(f"Result: {result['report_json']}")
    print(f"Full event log: {result['log_jsonl']}")
    if result["promising_gain_for_manual_retest_db"] is None:
        print("No validated gain for retest; do not infer TV reception.")
    else:
        print(
            "Evidence-supported *manual retest* gain: "
            f"{result['promising_gain_for_manual_retest_db']} dB. "
            "Not automatically applied to live reception."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
