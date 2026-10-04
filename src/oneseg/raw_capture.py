"""Low-overhead live raw-u8 I/Q capture; conversion runs AFTER USB reads.

The synchronous RTL-SDR reader copies 8-bit interleaved I/Q into a
buffered file directly. No Python queue, float conversion or filesystem
metadata serialization occurs inside the high-rate USB read loop.
After an entire window is captured, another thread can expand it to
legacy complex64 .c64 for the existing, validated offline decoder.

We deliberately do NOT call librtlsdr's native async/cancel methods,
which previously access-violated with this RTL2832U+FC0013 on Windows.
Buffering on a normal file does not establish hardware sample continuity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .file_output import publish_new_file

from .continuous import CaptureCancelled, READ_SAMPLES, convert_u8_to_c64
from .dsp import DEFAULT_SAMPLE_RATE


def record_raw_window(
    device,
    target: Path,
    *,
    samples_required: int,
    cancelled: Callable[[], bool] = lambda: False,
    warmup_buffers: int = 0,
    carry: bytearray | None = None,
) -> int:
    """Publish an exact-sized interleaved 8-bit IQ window atomically.

    Keep the native USB API on the calling thread; only copy bytes and
    write to a 1MiB-buffered stream. 3sec at 2.048MS/s = 12,288,000
    bytes on disk (versus 49,152,000 bytes for complex64).
    Reuse ``carry`` for successive windows to retain the unused tail of
    a full USB read. This prevents software-created gaps, but cannot
    establish whether the hardware itself dropped samples.
    """
    target = Path(target)
    if target.suffix.lower() != ".u8iq":
        raise ValueError("raw live window must end in .u8iq")
    if samples_required <= 0 or warmup_buffers < 0:
        raise ValueError("invalid capture length or warmup")
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"raw capture already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if carry is not None and len(carry) % 2:
        raise ValueError("incomplete I/Q pair in carry buffer")
    if warmup_buffers and carry:
        raise ValueError("cannot warm up a continuing raw stream")
    recorded = 0
    owns_partial = False
    try:
        with partial.open("xb", buffering=1024 * 1024) as out:
            owns_partial = True
            for _ in range(warmup_buffers):
                if cancelled():
                    raise CaptureCancelled("raw I/Q capture cancelled")
                warmup = bytes(device.read_bytes(2 * READ_SAMPLES))
                if len(warmup) != 2 * READ_SAMPLES:
                    raise IOError(
                        f"short warmup USB buffer: {len(warmup)}"
                    )
            if cancelled():
                raise CaptureCancelled("raw I/Q capture cancelled")
            if carry:
                take = min(samples_required * 2, len(carry))
                out.write(carry[:take])
                del carry[:take]
                recorded += take // 2
            while recorded < samples_required:
                if cancelled():
                    raise CaptureCancelled("raw I/Q capture cancelled")
                raw = bytes(device.read_bytes(2 * READ_SAMPLES))
                if len(raw) != 2 * READ_SAMPLES:
                    raise IOError(
                        f"short raw USB buffer: {len(raw)}/{2 * READ_SAMPLES}"
                    )
                take = min(READ_SAMPLES, samples_required - recorded)
                out.write(raw[:take * 2])
                recorded += take
                if carry is not None and take < READ_SAMPLES:
                    carry.extend(raw[take * 2:])
        publish_new_file(partial, target)
        partial.unlink(missing_ok=True)
        return recorded
    except Exception:
        if owns_partial:
            partial.unlink(missing_ok=True)
        raise


def expand_raw_to_c64(
    raw_window: Path,
    target: Path,
    *,
    metadata: dict,
) -> dict:
    """Convert captured u8 IQ into c64 without accessing USB hardware."""
    raw_window, target = Path(raw_window), Path(target)
    if raw_window.suffix.lower() != ".u8iq":
        raise ValueError("expected .u8iq input")
    if target.suffix.lower() != ".c64":
        raise ValueError("expected .c64 output")
    if not raw_window.is_file():
        raise FileNotFoundError(raw_window)
    byte_count = raw_window.stat().st_size
    if byte_count <= 0 or byte_count % 2:
        raise ValueError("raw window has incomplete interleaved I/Q bytes")
    sidecar = target.with_suffix(target.suffix + ".json")
    partial = target.with_name(target.name + ".partial")
    if target.exists() or sidecar.exists() or partial.exists():
        raise FileExistsError(f"complex64 capture already exists: {target}")
    details = {
        **metadata,
        "format": "complex64",
        "byte_order": "little-endian",
        "sample_rate_hz": DEFAULT_SAMPLE_RATE,
        "samples": byte_count // 2,
        "capture_samples": byte_count // 2,
        "capture_seconds": (byte_count // 2) / DEFAULT_SAMPLE_RATE,
        "acquisition": "synchronous_raw_u8iq_direct_write",
        "sample_continuity_verified": False,
        "decoding_status": "RAW_IQ_NOT_TS",
    }
    owns_partial = owns_sidecar = published = False
    try:
        with raw_window.open("rb", buffering=1024 * 1024) as source:
            with partial.open("xb", buffering=1024 * 1024) as output:
                owns_partial = True
                while raw := source.read(2 * READ_SAMPLES):
                    if len(raw) % 2:
                        raise ValueError("truncated u8 I/Q pair")
                    converted = convert_u8_to_c64(raw)
                    output.write(converted.tobytes())
        if partial.stat().st_size != byte_count * 4:
            raise IOError("raw IQ expansion has incorrect output byte count")
        with sidecar.open("x", encoding="utf-8") as report:
            owns_sidecar = True
            report.write(json.dumps(details, ensure_ascii=False, indent=2))
        publish_new_file(partial, target)
        published = True
        partial.unlink(missing_ok=True)
        return details
    except Exception:
        if owns_partial:
            partial.unlink(missing_ok=True)
        if owns_sidecar and not published:
            sidecar.unlink(missing_ok=True)
        raise



def raw_fullscale_percent(window: Path) -> float:
    """Exact-full-scale I/Q occupancy on an acquired raw-u8 window.

    Memory-map only after capture; never allocate or calculate this in
    the time-critical USB read loop. A component u8 0/255 maps to +/-1.
    """
    path = Path(window)
    size = path.stat().st_size
    if size <= 0 or size % 2:
        raise ValueError("invalid raw I/Q component count")
    samples = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, 2)
    fullscale = (
        (samples[:, 0] == 0) | (samples[:, 0] == 255)
        | (samples[:, 1] == 0) | (samples[:, 1] == 255)
    )
    return float(100 * np.count_nonzero(fullscale) / len(samples))
