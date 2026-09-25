"""Japanese terrestrial physical RF channel calculations (not TV service IDs)."""

FIRST_CHANNEL = 13
LAST_CHANNEL = 52
FIRST_CENTER_HZ = 473_142_857
CHANNEL_SPACING_HZ = 6_000_000


def physical_channel_hz(channel: int) -> int:
    """Return nominal RF center frequency (Hz) for UHF physical channel 13..52."""
    if isinstance(channel, bool) or not isinstance(channel, int):
        raise TypeError("physical channel must be an integer")
    if not FIRST_CHANNEL <= channel <= LAST_CHANNEL:
        raise ValueError(f"physical channel must be {FIRST_CHANNEL}..{LAST_CHANNEL}")
    return FIRST_CENTER_HZ + CHANNEL_SPACING_HZ * (channel - FIRST_CHANNEL)
