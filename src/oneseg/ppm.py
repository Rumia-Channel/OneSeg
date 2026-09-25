"""Manage librtlsdr PPM changes without writing an unchanged value.

librtlsdr's rtlsdr_set_freq_correction() returns -2 (INVALID_PARAM)
when dev->corr == ppm. Its freshly opened devices start at 0 ppm.
"""

class PpmCorrection:
    def __init__(self, initial_ppm: int = 0):
        self.current = int(initial_ppm)

    def apply(self, device, requested_ppm: int) -> bool:
        """Apply a real PPM change only; return True when setter was invoked."""
        requested = int(requested_ppm)
        if requested == self.current:
            return False
        device.freq_correction = requested
        self.current = requested
        return True
