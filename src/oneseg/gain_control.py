"""Set discrete RTL tuner gain without treating unreliable readback as truth.

On the tested FC0013, librtlsdr's gain getter returned 0.0 dB after
accepted manual negative gain commands. Preserve BOTH the intended
nearest tuner step and the observed readback; the former is a successful
command, not a hardware-independent measurement of analog RF gain.
"""
from __future__ import annotations

from math import isclose, isfinite


def set_discrete_manual_gain(
    device, requested: float, *, supported: list[float] | None = None,
) -> dict:
    """Set nearest advertised gain; return requested, commanded, readback.

    NEVER group trials by a contradictory getter response. An accepted
    setter and a changing ADC level support, but do not prove, the actual
    analog gain. The readback_matches_command flag preserves uncertainty.
    """
    requested = float(requested)
    if not isfinite(requested):
        raise ValueError("manual tuner gain must be finite")
    steps = (
        supported if supported is not None
        else getattr(device, "valid_gains_db", None)
    )
    steps = sorted({float(value) for value in (steps or [])})
    if any(not isfinite(value) for value in steps):
        raise ValueError("tuner advertised nonfinite gain values")
    commanded = (
        min(steps, key=lambda value: abs(value - requested))
        if steps else requested
    )
    device.gain = commanded
    try:
        readback = float(device.gain)
        if not isfinite(readback):
            readback = None
    except (AttributeError, TypeError, ValueError, OSError):
        readback = None
    matches = (
        readback is not None
        and isclose(readback, commanded, abs_tol=0.051, rel_tol=0)
    )
    return {
        "requested_gain_db": requested,
        "commanded_gain_db": commanded,
        "applied_gain_db": commanded,  # legacy alias for per-gain aggregators
        "gain_readback_db": readback,
        "gain_readback_matches_command": matches,
        "gain_value_source": (
            "nearest_supported_setter_accepted"
            if steps else "requested_setter_accepted_no_gain_list"
        ),
        "actual_analog_gain_independently_verified": False,
    }
