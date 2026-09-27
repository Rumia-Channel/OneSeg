"""Capture and extract reproducible one-seg RF + full offline stage evidence.

Unlike oneseg-survey, retain ALL input and intermediate files, including
failed OFDM/TMCC/FEC cases. Requires exclusive Windows RTL-SDR USB ownership;
safe synchronous read_bytes only; no native asynchronous transfer/cancel.
This does not make a gapless or confirmed playable live TV stream.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

from .channels import physical_channel_hz
from .continuous import CaptureCancelled
from .deinterleave import process_fixture
from .dsp import DEFAULT_SAMPLE_RATE
from .gain_control import set_discrete_manual_gain
from .pilots import analyze_capture
from .ppm import PpmCorrection
from .quality import recording_quality
from .raw_capture import expand_raw_to_c64, record_raw_window
from .recover import recover_file


def _write_json_atomic(path: Path, value: dict) -> None:
    """Replace only this tool's manifest, never pre-existing user files."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(partial, path)


def _error(stage: str, exc: Exception) -> dict:
    result = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    diagnostics = getattr(exc, "diagnostics", None)
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def extract_channel(
    output_dir: Path, *,
    channel: int = 20,
    gain: float = -6.5,
    ppm: int = 0,
    seconds: float = 3.0,
    captures: int = 3,
    max_ofdm_symbols: int = 2200,
    device_factory: Callable | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Save raw IQ, c64, quality, full TMCC/Layer-A/FEC/TS, per-window results.

    First capture all windows with ONE tuner owner; close USB before starting
    high-CPU conversion and FEC. Completed recordings are kept on failure.
    Existing destination directories are never overwritten or reused.
    """
    if not 13 <= channel <= 52:
        raise ValueError("physical channel must be 13..52")
    if not -10 <= gain <= 50:
        raise ValueError("gain must be -10..50 dB")
    if not -200 <= ppm <= 200:
        raise ValueError("ppm must be -200..200")
    if not 0.5 <= seconds <= 3.0:
        raise ValueError("seconds must be 0.5..3")
    if not 1 <= captures <= 8:
        raise ValueError("captures must be 1..8")
    if max_ofdm_symbols < 204:
        raise ValueError("max_ofdm_symbols must be at least 204")
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(
            f"output directory exists, choose a new one: {directory}"
        )
    if device_factory is None:
        from rtlsdr import RtlSdr
        device_factory = RtlSdr
    directory.mkdir(parents=True)
    frequency = physical_channel_hz(channel)
    started = datetime.now(timezone.utc).isoformat()
    manifest_path = directory / "extract_manifest.json"
    manifest = {
        "format": "oneseg_extract_v1",
        "utc_started": started,
        "channel": channel,
        "center_frequency_hz": frequency,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE,
        "requested_gain_db": float(gain),
        "ppm": ppm,
        "seconds_per_capture": seconds,
        "requested_captures": captures,
        "max_ofdm_symbols": max_ofdm_symbols,
        "continuous_tv_verified": False,
        "mpeg_ts_packets_are_rs_validated_only": True,
        "missing_packets_or_pat_pmt_synthesized": False,
        "files_retained_on_failed_decode": True,
        "windows": [],
        "status": "starting",
    }

    def announce(message: str):
        if progress:
            progress(message)

    def flush():
        _write_json_atomic(manifest_path, manifest)

    flush()
    device = None
    capture_error = None
    try:
        device = device_factory()
        device.sample_rate = DEFAULT_SAMPLE_RATE
        PpmCorrection().apply(device, ppm)
        device.center_freq = frequency
        gain_state = set_discrete_manual_gain(device, gain)
        manifest["gain"] = gain_state
        announce(
            f"FC0013 gain requested {gain:g}; commanded "
            f"{gain_state['commanded_gain_db']:g} dB; driver getter "
            f"{gain_state['gain_readback_db']} dB"
        )
        manifest["status"] = "capturing"
        flush()
        for index in range(captures):
            root = directory / f"capture_{index+1:02d}"
            root.mkdir()
            raw = root / "iq.u8iq"
            row = {
                "index": index + 1,
                "status": "capturing",
                "capture_dir": str(root),
                "raw_iq": str(raw),
            }
            manifest["windows"].append(row)
            flush()
            announce(
                f"[{index+1}/{captures}] acquiring raw ch{channel} "
                f"{seconds:g}s -> {raw}"
            )
            began = perf_counter()
            try:
                row["sample_count"] = record_raw_window(
                    device, raw,
                    samples_required=round(DEFAULT_SAMPLE_RATE * seconds),
                    cancelled=lambda: False,
                    warmup_buffers=1 if index == 0 else 0,
                )
                row["raw_bytes"] = raw.stat().st_size
                row["usb_seconds"] = round(perf_counter() - began, 3)
                row["status"] = "captured"
                flush()
            except Exception as exc:
                row["status"] = "capture_failed"
                row["error"] = _error("raw_usb", exc)
                manifest["status"] = "capture_failed"
                capture_error = exc
                flush()
                announce(
                    f"[{index+1}/{captures}] USB capture failed; "
                    "previous completed raw I/Q remains saved"
                )
                break
    except Exception as exc:
        manifest["status"] = "capture_failed"
        manifest["capture_setup_error"] = _error("tuner_setup", exc)
        capture_error = exc
        flush()
    finally:
        if device is not None:
            try:
                device.close()
            except Exception as exc:
                manifest["tuner_close_error"] = _error(
                    "tuner_close", exc
                )
                flush()
        manifest["utc_usb_closed"] = datetime.now(
            timezone.utc
        ).isoformat()
        flush()

    # No active USB here: long NumPy/Viterbi/RS runs cannot block reception.
    manifest["status"] = "decoding" if any(
        w["status"] == "captured" for w in manifest["windows"]
    ) else "capture_failed"
    flush()
    for row in manifest["windows"]:
        if row["status"] != "captured":
            continue
        folder = Path(row["capture_dir"])
        raw = Path(row["raw_iq"])
        c64 = folder / "iq.c64"
        layer = folder / "layer_a.npz"
        tmcc = folder / "tmcc.json"
        deinterleaved = folder / "deinterleaved.npz"
        ts = folder / "partial.ts"
        row["status"] = "decoding"
        row["artifacts"] = {
            "raw_iq": str(raw),
            "c64": str(c64),
            "c64_metadata": str(c64.with_suffix(".c64.json")),
            "quality": str(folder / "quality.json"),
            "tmcc": str(tmcc),
            "layer_a": str(layer),
            "deinterleaved": str(deinterleaved),
            "partial_ts": str(ts),
            "partial_ts_metadata": str(ts.with_suffix(".ts.json")),
        }
        flush()
        stage = "u8_to_c64"
        began = perf_counter()
        try:
            expand_raw_to_c64(
                raw, c64,
                metadata={
                    "source": "oneseg_extract",
                    "center_frequency_hz": frequency,
                    "physical_channel": channel,
                    "gain_db": manifest["gain"]["commanded_gain_db"],
                    "gain_requested_db": gain,
                    "gain_readback_db": manifest["gain"][
                        "gain_readback_db"
                    ],
                    "ppm": ppm,
                },
            )
            stage = "iq_quality"
            quality = recording_quality(c64)
            _write_json_atomic(folder / "quality.json", quality)
            row["quality"] = quality
            stage = "ofdm_pilots_tmcc"
            announce(f"[{row['index']}/{captures}] extracting OFDM/TMCC...")
            report = analyze_capture(
                c64, seconds=seconds, layer_a_output=layer,
                compress_fixture=False,
            )
            _write_json_atomic(tmcc, report)
            row["cp_quality"] = report["cp_quality"]
            row["pilot_coherence"] = report["pilot_coherence"]
            row["tmcc_verified_frames"] = len(
                report["bch_parity_verified_frames"]
            )
            row["tmcc_soft_source"] = report.get("tmcc_soft_source")
            row["tmcc_per_carrier_verified_counts"] = report.get(
                "tmcc_per_carrier_verified_counts"
            )
            row["iq_rms"] = report["iq_rms"]
            row["fullscale_percent"] = report["iq_fullscale_percent"]
            flush()
            if not row["tmcc_verified_frames"]:
                row["status"] = "tmcc_unverified"
                row["failure"] = (
                    "No protected TMCC frame; unverified layer settings "
                    "MUST NOT be used to synthesize a TS"
                )
                announce(
                    f"[{row['index']}/{captures}] TMCC not verified. "
                    "Kept full I/Q and OFDM/Layer-A intermediates."
                )
                continue
            stage = "deinterleave"
            announce(f"[{row['index']}/{captures}] deinterleaving QPSK...")
            stage_data = process_fixture(
                layer, tmcc, deinterleaved, compress=False
            )
            row["deinterleaved_bits"] = stage_data["output_bit_count"]
            flush()
            stage = "viterbi_rs_ts"
            announce(
                f"[{row['index']}/{captures}] Viterbi/RS partial TS "
                f"(max {max_ofdm_symbols} OFDM rows)..."
            )
            recovered = recover_file(
                deinterleaved, ts,
                max_ofdm_symbols=max_ofdm_symbols,
            )
            row["rs_verified_packets"] = recovered[
                "rs_and_ts_accepted_packets"
            ]
            row["rejected_blocks"] = recovered[
                "rejected_rs_or_invalid_ts_packets"
            ]
            row["pat_found"] = bool(recovered["pat_programs"])
            row["pmt_found"] = bool(recovered["pmt_elementary_streams"])
            row["pid_packet_counts"] = recovered["pid_packet_counts"]
            row["transport_continuity_errors"] = recovered[
                "continuity_errors_due_to_missing_or_bad_packets"
            ]
            row["status"] = "partial_ts_extracted"
            announce(
                f"[{row['index']}/{captures}] "
                f"{row['rs_verified_packets']} genuine TS packets; "
                f"PAT {row['pat_found']}, PMT {row['pmt_found']}"
            )
        except Exception as exc:
            row["status"] = "decode_failed"
            row["error"] = _error(stage, exc)
            announce(
                f"[{row['index']}/{captures}] {stage} failed: {exc}; "
                "all existing raw and intermediate artifacts retained"
            )
        finally:
            row["decode_seconds"] = round(
                perf_counter() - began, 3
            )
            flush()

    manifest["utc_finished"] = datetime.now(timezone.utc).isoformat()
    manifest["total_rs_verified_packets"] = sum(
        w.get("rs_verified_packets", 0) for w in manifest["windows"]
    )
    manifest["partial_ts_files"] = [
        w["artifacts"]["partial_ts"] for w in manifest["windows"]
        if w["status"] == "partial_ts_extracted"
    ]
    manifest["status"] = (
        "finished_with_capture_error" if capture_error is not None
        else "finished"
    )
    flush()
    announce(
        f"Extraction evidence saved: {manifest_path}; "
        f"{manifest['total_rs_verified_packets']} total RS-verified "
        "partial TS packets. No fake/missing packets added."
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, default=20)
    parser.add_argument("--gain", type=float, default=-6.5)
    parser.add_argument("--ppm", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--captures", type=int, default=3)
    parser.add_argument("--max-ofdm-symbols", type=int, default=2200)
    parser.add_argument(
        "--output-dir", type=Path,
        help="NEW destination folder. Existing folders are not overwritten."
    )
    args = parser.parse_args()
    dest = args.output_dir or Path(
        "oneseg_extract_ch"
        + str(args.channel)
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    try:
        result = extract_channel(
            dest, channel=args.channel, gain=args.gain, ppm=args.ppm,
            seconds=args.seconds, captures=args.captures,
            max_ofdm_symbols=args.max_ofdm_symbols, progress=print,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"RF extraction failed: {exc}\n")
    print("Manifest:", dest / "extract_manifest.json")
    for window in result["windows"]:
        print(
            "  capture", window["index"],
            window["status"],
            "RS", window.get("rs_verified_packets", 0),
            "output", window.get("artifacts", {}).get("partial_ts"),
        )
    print(
        "Offline partial MPEG-TS only; no continuous broadcast "
        "or video playback is guaranteed."
    )
    return int(result["status"] == "finished_with_capture_error")


if __name__ == "__main__":
    raise SystemExit(main())
