"""Recover the 384 payload carriers of ISDB-T Mode-3 central segment.

This is *carrier extraction*, not post-deinterleaving QPSK bits, channel
decoding, transport stream or playback. Carrier frequencies are kept in
ascending segment order before ISDB-T frequency deinterleaving.
"""
from __future__ import annotations

import numpy as np

from .pilots import ACTIVE, TMCC_CARRIERS, scattered_pilot_indices

# ARIB STD-B31 Mode-3 center segment (segment 0): AC1 positions.
# The four TMCC positions and 36 symbol-varying SP carriers are excluded.
AC1_CARRIERS = np.array(
    [7, 89, 206, 209, 226, 244, 377, 407], dtype=np.intp
)
PAYLOAD_CARRIERS = 384
SEGMENT_CARRIERS_WITHOUT_EDGE = ACTIVE - 1


def data_carrier_indices(symbol: int, phase: int = 0) -> np.ndarray:
    """Ascending RF-frequency indices for 384 payload carriers per symbol."""
    if symbol < 0 or phase not in range(4):
        raise ValueError("symbol must be >=0 and phase must be 0..3")
    available = np.ones(SEGMENT_CARRIERS_WITHOUT_EDGE, dtype=np.bool_)
    available[scattered_pilot_indices(symbol, phase)] = False
    available[TMCC_CARRIERS] = False
    available[AC1_CARRIERS] = False
    indices = np.flatnonzero(available)
    if len(indices) != PAYLOAD_CARRIERS:
        raise ValueError(
            f"expected 384 payload carriers, got {len(indices)}"
        )
    return indices


def extract_layer_a_carriers(
    equalized: np.ndarray, *, pilot_phase: int
) -> np.ndarray:
    """Extract unmapped QPSK constellation symbols in RF carrier order.

    Rows correspond to input FFT symbols; columns are the 384 payload
    positions for that row's scattered-pilot phase. This step neither
    frequency/time/bit-deinterleaves nor FEC-decodes the broadcast.
    """
    equalized = np.asarray(equalized)
    if equalized.ndim != 2 or equalized.shape[1] != ACTIVE:
        raise ValueError("expected (N,433) equalized center segment")
    if pilot_phase not in range(4):
        raise ValueError("pilot phase must be 0..3")
    selected = np.empty(
        (len(equalized), PAYLOAD_CARRIERS), dtype=np.complex64
    )
    for symbol in range(len(equalized)):
        selected[symbol] = equalized[
            symbol, data_carrier_indices(symbol, pilot_phase)
        ]
    return selected


def save_layer_a_fixture(
    path,
    *,
    payload: np.ndarray,
    pilot_phase: int,
    tmcc_frames: list[dict],
    integer_offset_bins: int,
):
    """Save carrier symbols and BCH-verified frame-start annotations.

    Filename must be a new .npz (no overwrite). The first DIFFERENTIAL
    TMCC soft bit is for FFT symbol row 1 (row 1 vs row 0);
    therefore a TMCC frame start at soft index k maps to FFT row k+1.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Layer A fixture filename must end in .npz")
    if path.exists():
        raise FileExistsError(f"will not overwrite {path}")
    payload = np.asarray(payload, dtype=np.complex64)
    if payload.ndim != 2 or payload.shape[1] != PAYLOAD_CARRIERS:
        raise ValueError("expected (N,384) Layer A carrier array")
    frame_fft_rows = np.array(
        sorted({
            int(row["frame_start_bit_index"]) + 1
            for row in tmcc_frames
            if row.get("bch_parity_verified")
            and 0 <= int(row["frame_start_bit_index"]) + 1
            and int(row["frame_start_bit_index"]) + 1 + 204 <= len(payload)
        }),
        dtype=np.int32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        np.savez_compressed(
            destination,
            equalized_payload_carriers=payload,
            verified_tmcc_frame_start_fft_rows=frame_fft_rows,
            pilot_phase=np.int32(pilot_phase),
            integer_offset_bins=np.int32(integer_offset_bins),
            carrier_order="ascending RF frequency, no deinterleaving",
            stage="unmapped 384 complex data carriers per OFDM symbol; NOT TS",
        )
    return {
        "symbols": len(payload),
        "payload_carriers_per_symbol": PAYLOAD_CARRIERS,
        "verified_tmcc_frame_start_fft_rows": frame_fft_rows.tolist(),
        "file": str(path),
        "mpeg_ts_recovered": False,
    }
