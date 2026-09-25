"""First actual ISDB-T one-segment OFDM demodulation stage (offline).

This extracts synchronized FFT *constellations*. It is deliberately not a
complete demodulator: it cannot produce MPEG-TS until pilot equalization,
TMCC, segment mapping, deinterleaving and inner FEC are implemented.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dsp import DEFAULT_SAMPLE_RATE, MODE_FFT, ONESEG_RATE, to_one_seg_rate


@dataclass(frozen=True)
class SymbolLock:
    mode: int
    guard: str
    fft_size: int
    guard_samples: int
    symbol_samples: int
    first_sample: int
    cp_quality: float
    coarse_cfo_hz: float


def _moving_sum(values: np.ndarray, length: int) -> np.ndarray:
    prefix = np.empty(len(values) + 1, dtype=np.result_type(values, np.float64))
    prefix[0] = 0
    np.cumsum(values, out=prefix[1:])
    return prefix[length:] - prefix[:-length]


def find_symbol_lock(
    iq: np.ndarray,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    guards: tuple[tuple[int, int], ...] = ((1, 4), (1, 8), (1, 16), (1, 32)),
    min_symbols: int = 5,
) -> SymbolLock:
    """Search periodic CP correlation over >=5 successive OFDM symbols.

    Assumes input is centered, filtered and resampled to ~1.015873 MS/s.
    The guard estimate is only a *candidate*, not TMCC-verified lock.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if min_symbols < 3:
        raise ValueError("at least three symbols required")
    best = None
    best_score = -1.0
    for mode in modes:
        fft_size = MODE_FFT[mode]
        if len(iq) < min_symbols * (fft_size + fft_size // 4):
            continue
        a = iq[:-fft_size]
        b = iq[fft_size:]
        cross = np.conj(a) * b
        aa = np.abs(a) ** 2
        bb = np.abs(b) ** 2
        for numer, denom in guards:
            guard = fft_size * numer // denom
            spacing = fft_size + guard
            max_start = len(iq) - fft_size - guard
            if max_start < (min_symbols - 1) * spacing:
                continue
            cov = _moving_sum(cross, guard)
            power = np.sqrt(
                np.maximum(_moving_sum(aa, guard), 0)
                * np.maximum(_moving_sum(bb, guard), 0)
            )
            norm_cov = cov / np.maximum(power, 1e-12)
            # Noncoherent accumulation avoids cancelling a legitimate CFO.
            # A single high noise peak across the buffer is insufficient.
            offsets = np.arange(spacing)[:, None]
            periods = np.arange(min_symbols)[None, :] * spacing
            positions = offsets + periods
            permitted = np.all(positions <= max_start, axis=1)
            if not np.any(permitted):
                continue
            valid_offsets = offsets[:, 0][permitted]
            aligned = norm_cov[positions[permitted]]
            scores = np.abs(np.mean(aligned, axis=1))
            at = int(np.argmax(scores))
            score = float(scores[at])
            if score > best_score:
                first = int(valid_offsets[at])
                phase = float(np.angle(np.mean(cov[positions[permitted][at]])))
                best = SymbolLock(
                    mode=mode,
                    guard=f"{numer}/{denom}",
                    fft_size=fft_size,
                    guard_samples=guard,
                    symbol_samples=spacing,
                    first_sample=first,
                    cp_quality=score,
                    coarse_cfo_hz=phase * ONESEG_RATE / (2 * np.pi * fft_size),
                )
                best_score = score
    if best is None:
        raise ValueError("insufficient data for periodic OFDM correlation")
    return best


def extract_ofdm_symbols(
    iq: np.ndarray,
    lock: SymbolLock,
    *,
    max_symbols: int | None = None,
) -> np.ndarray:
    """FFT and coarse fractional-CFO removal using cyclic-prefix lock."""
    iq = np.asarray(iq, dtype=np.complex64)
    n = lock.fft_size
    first = lock.first_sample + lock.guard_samples
    spacing = lock.symbol_samples
    number = max(0, (len(iq) - first - n) // spacing + 1)
    if max_symbols is not None:
        number = min(number, max_symbols)
    if not number:
        return np.empty((0, n), dtype=np.complex64)
    start = first + np.arange(number, dtype=np.int64) * spacing
    indexes = start[:, None] + np.arange(n)[None, :]
    data = iq[indexes]
    phase = -2j * np.pi * lock.coarse_cfo_hz * indexes / ONESEG_RATE
    corrected = data * np.exp(phase)
    return np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1).astype(
        np.complex64
    )


def central_carriers(constellations: np.ndarray, mode: int) -> np.ndarray:
    """Return the ~429-kHz central segment, including its edge carrier."""
    if mode not in MODE_FFT:
        raise ValueError("invalid OFDM mode")
    if constellations.ndim != 2 or constellations.shape[1] != MODE_FFT[mode]:
        raise ValueError("incorrect FFT shape")
    half = 54 * mode
    middle = MODE_FFT[mode] // 2
    return constellations[:, middle - half : middle + half + 1]


def analyze_file(path: Path, output: Path, seconds: float = 0.5) -> SymbolLock:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    meta_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("sample_rate_hz") != DEFAULT_SAMPLE_RATE:
        raise ValueError("expected .c64 capture at 2.048 MS/s")
    iq = np.fromfile(path, dtype="<c8", count=int(DEFAULT_SAMPLE_RATE * seconds))
    if len(iq) < 8192:
        raise ValueError("capture too short")
    baseband = to_one_seg_rate(iq)
    lock = find_symbol_lock(baseband)
    constellations = central_carriers(
        extract_ofdm_symbols(baseband, lock), lock.mode
    )
    if not len(constellations):
        raise ValueError("no complete OFDM symbols")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, central_carriers=constellations)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "input_capture": str(path),
                "source": metadata,
                "stage": "OFDM FFT constellation only; no TS decoder",
                "lock_candidate": asdict(lock),
                "symbols": len(constellations),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline 1seg OFDM FFT investigation (does not produce MPEG-TS)"
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ofdm_symbols.npz"))
    parser.add_argument("--seconds", type=float, default=0.5)
    args = parser.parse_args()
    try:
        result = analyze_file(args.capture, args.output, args.seconds)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Cannot analyze I/Q: {exc}\n")
    print(
        f"CP candidate: mode {result.mode}, guard {result.guard}, "
        f"correlation {result.cp_quality:.3f}, CFO {result.coarse_cfo_hz:.1f} Hz"
    )
    print(f"Saved FFT carriers to: {args.output} (not a transport stream)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
