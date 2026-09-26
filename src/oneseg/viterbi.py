"""Rate-1/2 K=7 convolutional Viterbi with ISDB-T puncturing patterns.

This is a reference *offline* implementation. A single Viterbi stage does not
produce a transport stream until the ISDB-T byte/frame alignment chain works.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

# K=7 equivalent NASA generator polynomials (bit-reversed canonical forms).
POLYNOMIALS = (0x4F, 0x6D)

PUNCTURE = {
    "1/2": (1, 1),
    "2/3": (1, 1, 0, 1),
    "3/4": (1, 1, 0, 1, 1, 0),
    "5/6": (1, 1, 0, 1, 1, 0, 0, 1, 1, 0),
    "7/8": (1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0),
}


@lru_cache(maxsize=1)
def _transitions():
    state = np.arange(64, dtype=np.uint8)
    pred0 = state >> 1
    pred1 = pred0 | 32
    incoming = state & 1

    def parity(reg):
        bits = np.zeros(len(reg), dtype=np.uint8)
        for pos in range(7):
            bits ^= (reg >> pos) & 1
        return bits

    branches = np.empty((2, 64, 2), dtype=np.uint8)
    for i, predecessor in enumerate((pred0, pred1)):
        registers = (predecessor.astype(np.uint16) << 1) | incoming
        for j, polynomial in enumerate(POLYNOMIALS):
            branches[i, :, j] = parity(registers & polynomial)
    return pred0, pred1, branches


def depuncture(soft_bits: np.ndarray, rate: str = "1/2") -> np.ndarray:
    """Return [N,2] float soft decisions in [0,1], erasures represented by 0.5."""
    if rate not in PUNCTURE:
        raise ValueError(f"unsupported code rate: {rate}")
    mask = np.asarray(PUNCTURE[rate], dtype=bool)
    received = np.asarray(soft_bits, dtype=np.float32).reshape(-1)
    if not len(received) or len(received) % int(mask.sum()):
        raise ValueError("soft bits do not align to a complete puncturing period")
    if np.any(~np.isfinite(received)) or np.any((received < 0) | (received > 1)):
        raise ValueError("soft decisions must be in [0,1]")
    periods = len(received) // int(mask.sum())
    output = np.full((periods, len(mask)), 0.5, dtype=np.float32)
    output[:, mask] = received.reshape(periods, -1)
    return output.reshape(-1, 2)


@njit(cache=True, nogil=True)
def _fast_trellis(symbols, pred0, pred1, expected, end_state_zero):
    """Native-code equivalent of the 64-state reference Viterbi recurrence.

    The packed history is one uint8 per state/step, not an unbounded
    Python object per trellis step. Compiled code releases the GIL while
    a separate thread reads the Windows USB tuner.
    """
    total = len(symbols)
    history = np.empty((total, 64), dtype=np.uint8)
    scores = np.full(64, 1.0e30, dtype=np.float64)
    scores[0] = 0.0
    for step in range(total):
        a = float(symbols[step, 0])
        b = float(symbols[step, 1])
        next_scores = np.empty(64, dtype=np.float64)
        least = 1.0e30
        for state in range(64):
            v00 = float(expected[0, state, 0])
            v01 = float(expected[0, state, 1])
            v10 = float(expected[1, state, 0])
            v11 = float(expected[1, state, 1])
            left = scores[pred0[state]] + abs(v00 - a) + abs(v01 - b)
            right = scores[pred1[state]] + abs(v10 - a) + abs(v11 - b)
            choose_right = right < left
            history[step, state] = 1 if choose_right else 0
            cost = right if choose_right else left
            next_scores[state] = cost
            if cost < least:
                least = cost
        scores = next_scores - least
    state = 0
    if not end_state_zero:
        state = int(np.argmin(scores))
    decoded = np.empty(total, dtype=np.uint8)
    for step in range(total - 1, -1, -1):
        decoded[step] = state & 1
        state = (state >> 1) | (int(history[step, state]) << 5)
    return decoded


def decode_soft_bits(
    received: np.ndarray, rate: str = "1/2", *, end_state_zero: bool = False
) -> np.ndarray:
    """Viterbi decode punctured soft decisions, producing one decoded bit per trellis step.

    The trellis starts at state zero (for a known block boundary). The
    numba-compiled loop releases the GIL for concurrent USB acquisition.
    For random midstream samples, state acquisition remains approximate.
    """
    symbols = depuncture(received, rate)
    pred0, pred1, expected = _transitions()
    return _fast_trellis(symbols, pred0, pred1, expected, end_state_zero)

def convolutional_encode_test(bits: np.ndarray, rate: str = "1/2") -> np.ndarray:
    """Synthetic test encoder with puncturing; not an ISDB-T transmitter."""
    if rate not in PUNCTURE:
        raise ValueError("invalid code rate")
    state = 0
    outputs = []
    for bit in np.asarray(bits, dtype=np.uint8):
        reg = ((state << 1) | int(bit)) & 0x7F
        outputs.extend(((reg & p).bit_count() & 1) for p in POLYNOMIALS)
        state = reg & 63
    mask = np.asarray(PUNCTURE[rate], dtype=bool)
    if len(outputs) % len(mask):
        raise ValueError("bit count does not align to a puncturing period")
    return np.asarray(outputs, dtype=np.uint8)[np.tile(mask, len(outputs) // len(mask))]
