"""Offline recorded-I/Q -> parity-verified TMCC -> PARTIAL MPEG-TS workflow.

Not live television: processes an existing, matching .c64/.c64.json,
rejects absent TMCC or insufficient RS-validated packets, and never
invents missing MPEG transport packets or PSI tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable

from .deinterleave import process_fixture
from .pilots import analyze_capture
from .recover import recover_file


class UnverifiedTmccError(ValueError):
    """Cannot infer Layer-A bits if no protected TMCC frame was recovered."""

    def __init__(self, report: dict):
        self.diagnostics = {
            "cp_quality": report.get("cp_quality"),
            "pilot_coherence": report.get("pilot_coherence"),
            "iq_rms": report.get("iq_rms"),
            "iq_fullscale_percent": report.get("iq_fullscale_percent"),
            "single_sync_candidates": report.get("single_sync_candidates"),
            "repeated_sync_candidates": len(
                report.get("repeated_sync_candidates", [])
            ),
            "gain_db": report.get("gain_db"),
            "tmcc_parity_verified_frames": len(
                report.get("bch_parity_verified_frames", [])
            ),
            "tmcc_soft_source": report.get("tmcc_soft_source"),
            "tmcc_per_carrier_verified_counts": report.get(
                "tmcc_per_carrier_verified_counts"
            ),
            "integer_offset_bins": report.get("integer_offset_bins"),
            "fractional_cfo_hz": report.get("fractional_cfo_hz"),
            "fft_symbols": report.get("fft_symbols"),
        }
        diag = self.diagnostics
        super().__init__(
            "no parity-verified TMCC frames; "
            f"CP {diag['cp_quality']:.3f}, "
            f"pilots {diag['pilot_coherence']:.3f}, "
            f"I/Q RMS {diag['iq_rms']:.4f}, "
            f"full-scale {diag['iq_fullscale_percent']:.2f}%, "
            f"sync candidates {diag['single_sync_candidates']} "
            f"(repeated {diag['repeated_sync_candidates']}); "
            "no MPEG-TS produced"
        )


class DecodeStageError(ValueError):
    """Carry verified earlier stages when a subsequent FEC/TS stage fails."""

    def __init__(self, stage: str, report: dict, error: Exception):
        self.stage = stage
        self.diagnostics = UnverifiedTmccError(report).diagnostics
        self.diagnostics["failed_stage"] = stage
        self.diagnostics["original_error"] = str(error)
        super().__init__(f"{stage} failed: {error}")


def decode_capture(
    capture: Path,
    output: Path,
    *,
    seconds: float = 3.0,
    max_ofdm_symbols: int = 1020,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Perform all offline stages without leaving transient huge NPZ files.

    The output contains only authentic RS-validated MPEG-TS packets and may
    contain gaps. PAT/PMT and decodable video are NOT guaranteed.
    """
    capture, output = Path(capture), Path(output)
    if capture.suffix.lower() != ".c64":
        raise ValueError("input must be a .c64 raw IQ recording")
    if output.suffix.lower() != ".ts":
        raise ValueError("output must end in .ts")
    input_meta = capture.with_suffix(capture.suffix + ".json")
    output_meta = output.with_suffix(output.suffix + ".json")
    if not capture.is_file() or not input_meta.is_file():
        raise FileNotFoundError(
            f"requires capture and adjacent metadata: {capture} / {input_meta}"
        )
    if output.exists() or output_meta.exists():
        raise FileExistsError(f"output or sidecar already exists: {output}")
    if not 0.5 <= seconds <= 3.0:
        raise ValueError("seconds must be between 0.5 and 3.0")
    if max_ofdm_symbols < 204:
        raise ValueError("max_ofdm_symbols must be >= 204")
    start = perf_counter()
    if progress:
        progress("Analyzing Mode-3 OFDM and pilot positions (no tuner access)...")
    with TemporaryDirectory(prefix="oneseg-offline-") as temporary:
        root = Path(temporary)
        layer_a = root / "layer_a.npz"
        tmcc_path = root / "tmcc.json"
        deinterleaved = root / "deinterleaved.npz"
        report = analyze_capture(
            capture, seconds=seconds, layer_a_output=layer_a,
            compress_fixture=False,
        )
        after_pilots = perf_counter()
        frames = report["bch_parity_verified_frames"]
        if not frames:
            raise UnverifiedTmccError(report)
        tmcc_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if progress:
            progress(
                f"Verified {len(frames)} TMCC frames; "
                "extracting and deinterleaving Layer-A QPSK..."
            )
        try:
            stage = process_fixture(
                layer_a, tmcc_path, deinterleaved, compress=False
            )
        except (OSError, ValueError, KeyError) as exc:
            raise DecodeStageError(
                "deinterleave", report, exc
            ) from exc
        after_deinterleave = perf_counter()
        if progress:
            progress(
                f"Deinterleaved {stage['output_bit_count']:,} coded bits; "
                "running rate-2/3 Viterbi and outer Reed–Solomon..."
            )
        try:
            summary = recover_file(
                deinterleaved,
                output,
                max_ofdm_symbols=max_ofdm_symbols,
            )
        except (OSError, ValueError, KeyError) as exc:
            raise DecodeStageError(
                "viterbi_rs_ts", report, exc
            ) from exc
        after_fec = perf_counter()
        summary.update({
            "stage_seconds": {
                "ofdm_pilots_tmcc": round(after_pilots - start, 2),
                "deinterleave": round(
                    after_deinterleave - after_pilots, 2
                ),
                "fec_rs_ts": round(after_fec - after_deinterleave, 2),
                "total_to_ts": round(after_fec - start, 2),
            },
            "input_capture": str(capture),
            "input_capture_metadata": str(input_meta),
            "ofdm_mode": report["mode_assumed"],
            "ofdm_guard": report["guard_assumed"],
            "tmcc_parity_verified_frames": len(frames),
            "tmcc_soft_source": report.get("tmcc_soft_source"),
            "tmcc_per_carrier_verified_counts": report.get(
                "tmcc_per_carrier_verified_counts"
            ),
            "tmcc_repeated_verified_pairs": report.get(
                "tmcc_repeated_verified_pairs"
            ),
            "tmcc_sync_candidates": report.get("single_sync_candidates"),
            "integer_offset_bins": report.get("integer_offset_bins"),
            "fractional_cfo_hz": report.get("fractional_cfo_hz"),
            "layer_a": frames[0]["layer_A"],
            "input_fullscale_percent": report["iq_fullscale_percent"],
            "input_rms": report["iq_rms"],
            "input_cp_quality": report["cp_quality"],
            "input_pilot_coherence": report["pilot_coherence"],
            "input_overload_warning": report["iq_overload_warning"],
            "intermediate_npz_files_retained": False,
            "not_live_reception": True,
            "may_not_be_playable_without_pat_pmt": not bool(
                summary["pat_programs"]
            ),
        })
        # recover_file has already published output.ts and output.ts.json;
        # rewrite the sidecar with durable source provenance, not temp paths.
        summary["source_npz"] = "temporary stage, removed after decode"
        summary["source_metadata"] = "temporary stage, removed after decode"
        output_meta.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if progress:
            progress(
                f"Saved {summary['rs_and_ts_accepted_packets']} "
                f"RS-verified packets; PAT present: "
                f"{bool(summary['pat_programs'])}. Offline partial TS only."
            )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="raw .c64 with adjacent .c64.json")
    parser.add_argument("--output", type=Path, default=Path("oneseg_partial.ts"))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--max-ofdm-symbols", type=int, default=1020)
    args = parser.parse_args()
    try:
        result = decode_capture(
            args.capture, args.output,
            seconds=args.seconds,
            max_ofdm_symbols=args.max_ofdm_symbols,
            progress=print,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Offline decode failed: {exc}\n")
    print(
        f"Offline RS-validated TS: {args.output}; "
        f"{result['rs_and_ts_accepted_packets']} packets, "
        f"{result['rejected_rs_or_invalid_ts_packets']} rejected; "
        f"PAT present: {bool(result['pat_programs'])}. "
        "Video playback is NOT guaranteed; NOT live reception."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
