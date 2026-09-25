"""Verify ISDB-T TMCC difference-set cyclic parity and parse validated frames.

ARIB STD-B31 B20..B121 is 102 message bits, B122..B203 is 82
parity bits of the shortened (184,102) (273,191) cyclic code.
This module CHECKS a zero syndrome; it does not correct bit errors.
B0 differential reference, B1..B16 sync, B17..B19 identification
are outside the protected data. No TS is recovered here.

Generator polynomial is specified by Japan's terrestrial digital
broadcasting transmission regulations, Annex 12 paragraph 2:
https://laws.e-gov.go.jp/law/423M60000008087
"""
from __future__ import annotations

import numpy as np

# Polynomial coefficients listed by descending nonzero exponent in law:
_GENERATOR_EXPONENTS = (
    82, 77, 76, 71, 67, 66, 56, 52, 48,
    40, 36, 34, 24, 22, 18, 10, 4, 0,
)
_GENERATOR = sum(1 << power for power in _GENERATOR_EXPONENTS)
TMCC_BITS_PER_FRAME = 204
INFO_START = 20
INFO_STOP = 122
PARITY_STOP = 204

MODULATION = {
    0: "DQPSK", 1: "QPSK", 2: "16QAM", 3: "64QAM",
    7: "UNUSED",
}
CODE_RATE = {
    0: "1/2", 1: "2/3", 2: "3/4", 3: "5/6",
    4: "7/8", 7: "UNUSED",
}
TIME_INTERLEAVE_MODE3 = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: "NOT_USED", 7: "UNUSED",
}


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for digit in bits:
        value = (value << 1) | int(digit)
    return value


def _modulo_generator(codeword: int) -> int:
    """Binary polynomial long division over GF(2), MSB is highest power."""
    while codeword.bit_length() > 82:
        codeword ^= _GENERATOR << (codeword.bit_length() - 83)
    return codeword


def tmcc_syndrome(frame: np.ndarray) -> int:
    """Zero means parity-consistent shortened 184-bit TMCC codeword.

    This does NOT correct errors or independently verify B1..B16 sync.
    """
    frame = np.asarray(frame, dtype=np.uint8).reshape(-1)
    if len(frame) != TMCC_BITS_PER_FRAME or np.any(frame > 1):
        raise ValueError("TMCC frame must be 204 binary bits")
    return _modulo_generator(_bits_to_int(frame[INFO_START:PARITY_STOP]))


def _field(frame: np.ndarray, first: int, last: int) -> int:
    """Inclusive B-numbered field, MSB transmitted first."""
    return _bits_to_int(frame[first:last + 1])


def _layer(frame: np.ndarray, first: int) -> dict:
    modulation = _field(frame, first, first + 2)
    code = _field(frame, first + 3, first + 5)
    interleaving = _field(frame, first + 6, first + 8)
    segments = _field(frame, first + 9, first + 12)
    return {
        "modulation": MODULATION.get(modulation, "RESERVED"),
        "code_rate": CODE_RATE.get(code, "RESERVED"),
        "time_interleaving_mode3": TIME_INTERLEAVE_MODE3.get(
            interleaving, "RESERVED"
        ),
        "segments": (
            "UNUSED" if segments == 15
            else segments if 1 <= segments <= 13
            else "RESERVED"
        ),
        "raw_bits": "".join(str(int(v)) for v in frame[first:first + 13]),
    }


def parse_tmcc_frame(frame: np.ndarray) -> dict:
    """Parse only a frame whose cyclic parity syndrome is zero."""
    frame = np.asarray(frame, dtype=np.uint8).reshape(-1)
    syndrome = tmcc_syndrome(frame)
    if syndrome:
        raise ValueError(f"TMCC parity check failed (syndrome {syndrome:x})")
    return {
        "bch_parity_verified": True,
        "bit_error_correction_performed": False,
        "partial_reception_flag": bool(frame[27]),
        "layer_A": _layer(frame, 28),
        "layer_B": _layer(frame, 41),
        "layer_C": _layer(frame, 54),
        "info_bits_hex": hex(_bits_to_int(frame[INFO_START:INFO_STOP])),
        "parity_bits_hex": hex(_bits_to_int(frame[INFO_STOP:PARITY_STOP])),
    }


def verify_frames_from_soft(soft: np.ndarray, *, max_sync_errors: int = 2) -> dict:
    """Use B1..B16 sync to align frame boundaries, then check B20..B203.

    Inputs are differential soft bits, one per consecutive OFDM symbol.
    Index of B0 is sync_bit_index - 1. Every returned frame passes
    polynomial parity, but there is no multi-frame lock or error correction.
    """
    from .pilots import TMCC_SYNC_EVEN, TMCC_SYNC_ODD

    soft = np.asarray(soft, dtype=np.float32).reshape(-1)
    if max_sync_errors < 0 or max_sync_errors > 16:
        raise ValueError("max_sync_errors must be 0..16")
    bits = (soft < 0).astype(np.uint8)
    accepted = []
    tested = 0
    for sync_pos in range(1, len(bits) - 15):
        if sync_pos - 1 + TMCC_BITS_PER_FRAME > len(bits):
            break
        word = bits[sync_pos:sync_pos + 16]
        even_errors = int(np.count_nonzero(word != TMCC_SYNC_EVEN))
        odd_errors = int(np.count_nonzero(word != TMCC_SYNC_ODD))
        errors = min(even_errors, odd_errors)
        if errors > max_sync_errors:
            continue
        tested += 1
        frame = bits[sync_pos - 1:sync_pos - 1 + TMCC_BITS_PER_FRAME]
        syndrome = tmcc_syndrome(frame)
        if syndrome:
            continue
        parsed = parse_tmcc_frame(frame)
        accepted.append({
            "sync_bit_index": sync_pos,
            "polarity": "even" if even_errors < odd_errors else "odd",
            "sync_bit_errors": errors,
            "sync_confidence": float(
                np.mean(np.abs(soft[sync_pos:sync_pos + 16]))
            ),
            "frame_start_bit_index": sync_pos - 1,
            "syndrome": 0,
            **parsed,
        })
    return {
        "frames_tested_with_sync": tested,
        "bch_parity_verified_frames": accepted,
        "tmcc_bch_verified": bool(accepted),
        "bit_error_correction_performed": False,
        "mpeg_ts_recovered": False,
    }
