"""Rate-1/2 K=7 convolutional Viterbi with ISDB-T puncturing patterns.

This is a reference *offline* implementation. A single Viterbi stage does not
produce a transport stream until the ISDB-T byte/frame alignment chain works.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

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


def decode_soft_bits(
    received: np.ndarray, rate: str = "1/2", *, end_state_zero: bool = False
) -> np.ndarray:
    """Viterbi decode punctured soft decisions, producing one decoded bit per trellis step.

    The trellis starts at state zero (for a known block boundary). For random
    midstream samples, acquisition/traceback boundary logic remains to be added.
    """
    symbols = depuncture(received, rate)
    pred0, pred1, expected = _transitions()
    scores = np.full(64, np.inf, dtype=np.float64)
    scores[0] = 0.0
    history = np.empty((len(symbols), 64), dtype=np.uint8)
    rows = np.arange(64)
    for step, pair in enumerate(symbols):
        # Missing punctured bits have the same penalty for 0 and 1.
        score0 = scores[pred0] + np.sum(
            np.abs(expected[0] - pair[None, :]), axis=1
        )
        score1 = scores[pred1] + np.sum(
            np.abs(expected[1] - pair[None, :]), axis=1
        )
        choose1 = score1 < score0
        history[step] = choose1
        scores = np.where(choose1, score1, score0)
        scores -= np.min(scores)
    state = 0 if end_state_zero else int(np.argmin(scores))
    decoded = np.empty(len(symbols), dtype=np.uint8)
    for step in range(len(symbols) - 1, -1, -1):
        decoded[step] = state & 1
        state = int((state >> 1) | (int(history[step, state]) << 5))
    return decoded


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
