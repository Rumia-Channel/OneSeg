"""Pilot-based integer-carrier alignment and experimental TMCC sync diagnostics.

MODE 3 / 1/8 only until additional modes are validated with real recordings.
No BCH-verified TMCC, transport stream, or live television is produced.

A 2.048MS/s capture is centrally filtered and resampled by 125/252.
The nominal 433 active carriers of segment 0 are searched over integer FFT
bin offsets, using the expected DBPSK/scattered pilot pattern. Equalization
then supports an *unverified* search for repeated TMCC sync words.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .dsp import DEFAULT_SAMPLE_RATE, ONESEG_RATE
from .ofdm import extract_ofdm_symbols, find_symbol_lock

MODE = 3
ACTIVE = 433
BASE = 512 - 216
TMCC_CARRIERS = np.array([101, 131, 286, 349], dtype=np.intp)
TMCC_SYNC_EVEN = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
TMCC_SYNC_ODD = 1 - TMCC_SYNC_EVEN


def central_pilot_polarities() -> np.ndarray:
    """Reference BPSK pilot symbols at global active carriers 2592..3024.

    ISDB-T mode 3 uses a width-433 central segment. The standard pilot
    sequence uses the 11-bit PRBS with initial state all ones and LSB
    output; it is *not* the 15-bit MPEG energy-dispersal PRBS.
    """
    state = 0x7FF
    bits = np.empty(5617, dtype=np.float32)
    for index in range(5617):
        bits[index] = 1 - 2 * (state & 1)
        feedback = ((state >> 2) ^ state) & 1
        state = (state >> 1) | (feedback << 10)
    return bits[2592:3025] * (4 / 3)


@dataclass(frozen=True)
class PilotAlignment:
    integer_offset_bins: int
    symbol_phase: int
    pilot_coherence: float

    @property
    def integer_offset_hz(self) -> float:
        return self.integer_offset_bins * ONESEG_RATE / 1024


def scattered_pilot_indices(symbol_number: int, phase: int) -> np.ndarray:
    return np.arange(3 * ((symbol_number + phase) % 4), 432, 12, dtype=np.intp)


def _candidate_coherence(
    fft: np.ndarray,
    integer_offset: int,
    phase: int,
    polarities: np.ndarray,
    max_symbols: int,
) -> float:
    values = []
    for symbol in range(min(max_symbols, len(fft))):
        reference = scattered_pilot_indices(symbol, phase)
        actual = BASE + integer_offset + reference
        if np.any(actual < 0) or np.any(actual >= fft.shape[1]):
            return -1.0
        channel = fft[symbol, actual] / polarities[reference]
        adjacent = channel[1:] * np.conj(channel[:-1])
        normalized = adjacent / np.maximum(np.abs(adjacent), 1e-12)
        # Magnitude tolerates an unknown linear channel phase across
        # adjacent pilots; wrong PRBS / pilot phase should not match.
        values.append(np.abs(np.mean(normalized)))
    return float(np.mean(values)) if values else 0.0


def find_pilot_alignment(
    fft: np.ndarray,
    *,
    max_shift: int = 32,
    max_symbols: int = 32,
) -> PilotAlignment:
    """Search central segment's integer frequency-bin offset and 4-symbol phase.

    FFT input is the *full*, shifted 1024-bin OFDM spectrum per symbol.
    Pilot coherence alone does not prove a broadcast lock.
    """
    fft = np.asarray(fft)
    if fft.ndim != 2 or fft.shape[1] != 1024 or len(fft) < 8:
        raise ValueError("expected at least 8 full mode-3 OFDM FFT symbols")
    if not 0 <= max_shift <= 200 or max_symbols < 8:
        raise ValueError("invalid pilot search bounds")
    pol = central_pilot_polarities()
    best = PilotAlignment(0, 0, -1.0)
    for shift in range(-max_shift, max_shift + 1):
        for phase in range(4):
            score = _candidate_coherence(fft, shift, phase, pol, max_symbols)
            if score > best.pilot_coherence:
                best = PilotAlignment(shift, phase, score)
    return best


def equalize_segment(fft: np.ndarray, alignment: PilotAlignment) -> np.ndarray:
    """Linear pilot interpolation for each OFDM symbol (experimental)."""
    fft = np.asarray(fft)
    if fft.ndim != 2 or fft.shape[1] != 1024:
        raise ValueError("expected a (symbols, 1024) FFT array")
    offsets = BASE + alignment.integer_offset_bins + np.arange(ACTIVE)
    if offsets[0] < 0 or offsets[-1] >= 1024:
        raise ValueError("central segment outside FFT")
    received = fft[:, offsets]
    reference = central_pilot_polarities()
    out = np.empty(received.shape, dtype=np.complex64)
    for symbol in range(len(fft)):
        positions = scattered_pilot_indices(symbol, alignment.symbol_phase)
        estimate_at_pilots = received[symbol, positions] / reference[positions]
        # np.interp extrapolates the closest estimated edge channel.
        estimate = (
            np.interp(np.arange(ACTIVE), positions, estimate_at_pilots.real)
            + 1j * np.interp(np.arange(ACTIVE), positions, estimate_at_pilots.imag)
        )
        if np.any(np.abs(estimate) < 1e-9):
            raise ValueError("near-zero pilot channel estimate")
        out[symbol] = received[symbol] / estimate
    return out


def tmcc_differential_soft_bits(equalized: np.ndarray) -> np.ndarray:
    """One tentative DBPSK decision per OFDM symbol, 4 central TMCC carriers.

    Positive output represents 0 and negative 1. No BCH parity validation
    or sustained frame lock is implied.
    """
    equalized = np.asarray(equalized)
    if equalized.ndim != 2 or equalized.shape[1] != ACTIVE:
        raise ValueError("expected equalized central 433-carrier symbols")
    data = equalized[:, TMCC_CARRIERS]
    differential = data[1:] * np.conj(data[:-1])
    unit = differential / np.maximum(np.abs(differential), 1e-12)
    return np.mean(unit.real, axis=1).astype(np.float32)


def candidate_tmcc_sync(soft: np.ndarray, *, max_errors: int = 2) -> dict:
    """Look for repeated 16-bit sync patterns 204 symbols apart.

    This is deliberately a *candidate*, never 'TMCC OK' without BCH.
    """
    soft = np.asarray(soft, dtype=np.float32).reshape(-1)
    bits = (soft < 0).astype(np.uint8)
    matches = []
    for position in range(max(0, len(bits) - 16 + 1)):
        word = bits[position : position + 16]
        even = int(np.count_nonzero(word != TMCC_SYNC_EVEN))
        odd = int(np.count_nonzero(word != TMCC_SYNC_ODD))
        best = min(even, odd)
        if best <= max_errors:
            matches.append({
                "bit_index": position,
                "polarity": "even" if even < odd else "odd",
                "bit_errors": best,
                "confidence": float(np.mean(np.abs(soft[position : position + 16]))),
            })
    hits = {item["bit_index"]: item for item in matches}
    pairs = []
    for item in matches:
        opposite = "odd" if item["polarity"] == "even" else "even"
        other = hits.get(item["bit_index"] + 204)
        if other and other["polarity"] == opposite:
            pairs.append({
                "bit_index": item["bit_index"],
                "next_bit_index": other["bit_index"],
                "combined_bit_errors": item["bit_errors"] + other["bit_errors"],
            })
    return {
        "single_sync_candidates": len(matches),
        "repeated_sync_candidates": sorted(
            pairs, key=lambda row: (row["combined_bit_errors"], row["bit_index"])
        )[:10],
        "tmcc_bch_verified": False,
        "mpeg_ts_recovered": False,
    }


def analyze_capture(path: Path, *, seconds: float = 1.2) -> dict:
    """Process capture off line, finding pilot positions before attempting TMCC."""
    if not 0.5 <= seconds <= 3.0:
        raise ValueError("seconds must be 0.5..3.0")
    path = Path(path)
    metadata = json.loads(
        path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8")
    )
    if (
        int(metadata["sample_rate_hz"]) != DEFAULT_SAMPLE_RATE
        or metadata.get("format") != "complex64"
    ):
        raise ValueError("requires 2.048 MS/s complex64 I/Q capture")
    samples = np.fromfile(path, dtype="<c8", count=int(seconds * DEFAULT_SAMPLE_RATE))
    if len(samples) < int(0.5 * DEFAULT_SAMPLE_RATE):
        raise ValueError("not enough captured I/Q")
    raw = samples.astype(np.complex128)
    raw -= raw.mean()
    lowpass = butter(8, 205_000, fs=DEFAULT_SAMPLE_RATE, output="sos")
    baseband = resample_poly(sosfilt(lowpass, raw), 125, 252).astype(np.complex64)
    # Mode and GI are currently *assumed* from the verified repeating CP
    # experiment on ch20. Do not silently handle other modes.
    lock = find_symbol_lock(
        baseband[:min(len(baseband), 120000)],
        modes=(3,), guards=((1, 8),), min_symbols=30,
    )
    fft = extract_ofdm_symbols(baseband, lock)
    if len(fft) < 204 * 2 + 17:
        raise ValueError("need enough OFDM symbols for repeated TMCC sync")
    alignment = find_pilot_alignment(fft)
    equalized = equalize_segment(fft, alignment)
    soft = tmcc_differential_soft_bits(equalized)
    result = {
        "capture": path.name,
        "center_frequency_hz": metadata["center_frequency_hz"],
        "mode_assumed": MODE,
        "guard_assumed": "1/8",
        "cp_quality": lock.cp_quality,
        "fractional_cfo_hz": lock.coarse_cfo_hz,
        "fft_symbols": len(fft),
        **asdict(alignment),
        "integer_offset_hz": alignment.integer_offset_hz,
        "combined_offset_hz": alignment.integer_offset_hz + lock.coarse_cfo_hz,
        "signal_stage": "pilot carrier alignment + experimental DBPSK TMCC sync only",
        **candidate_tmcc_sync(soft),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--seconds", type=float, default=1.2)
    parser.add_argument("--json", type=Path, help="optional output report JSON")
    args = parser.parse_args()
    try:
        report = analyze_capture(args.capture, seconds=args.seconds)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Pilot/TMCC diagnostic failed: {exc}\n")
    print(
        f"CP: {report['cp_quality']:.3f}; fractional CFO "
        f"{report['fractional_cfo_hz']:.1f} Hz"
    )
    print(
        f"Pilot alignment: {report['integer_offset_bins']:+d} bins / "
        f"{report['integer_offset_hz']:+.1f} Hz, "
        f"phase {report['symbol_phase']}, "
        f"coherence {report['pilot_coherence']:.3f}"
    )
    print(
        f"TMCC sync candidates: {report['single_sync_candidates']} singles, "
        f"{len(report['repeated_sync_candidates'])} repeated. "
        "BCH NOT VERIFIED. MPEG-TS NOT RECOVERED."
    )
    if args.json:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
