"""Crash-avoidance capture path for pyrtlsdr on Windows.

Do not use pyrtlsdr's read_samples_async()/cancel_read_async() here:
those paths can fail on some WinUSB + librtlsdr combinations and close
the native device from inside an error path. The dedicated USB reader
copies each synchronous unsigned-byte buffer promptly. A separate writer
thread converts raw u8 I/Q and writes complex64 .c64 data.

Unlike hardware-timestamped acquisition, this does NOT guarantee zero
dropped USB samples; compare pilots/symbol continuity in real captures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Callable

import numpy as np

READ_SAMPLES = 131_072
_MAX_QUEUED_BUFFERS = 8


class CaptureCancelled(RuntimeError):
    """User cancelled capture."""


def convert_u8_to_c64(raw: bytes) -> np.ndarray:
    """Match pyrtlsdr read_samples() scaling: u8 / 127.5 - (1+1j)."""
    iq = np.frombuffer(raw, dtype=np.uint8)
    if len(iq) % 2:
        raise ValueError("raw interleaved I/Q must have an even byte count")
    values = iq.reshape(-1, 2).astype(np.float32)
    out = np.empty(len(values), dtype="<c8")
    out.real = values[:, 0] / 127.5 - 1.0
    out.imag = values[:, 1] / 127.5 - 1.0
    return out


def stream_exact_samples(
    device,
    stream,
    samples_required: int,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    warmup_buffers: int = 1,
) -> int:
    """Write exact sample count; hardware access stays on calling thread.

    Reader copies librtlsdr's reused ctypes buffer immediately, and never
    performs numpy conversion, FFT, file I/O, or USB async cancellation.
    Writer failure, short read and queue overflow are errors, never success.
    """
    if samples_required <= 0 or warmup_buffers < 0:
        raise ValueError("invalid capture length/warmup")
    queued: Queue[bytes | None] = Queue(maxsize=_MAX_QUEUED_BUFFERS)
    writer_errors: list[Exception] = []
    writer_count = [0]

    def write_buffers():
        try:
            while True:
                packet = queued.get()
                if packet is None:
                    break
                # Drain the queue even after a writer error so the reader can
                # always submit the sentinel and join without deadlocking.
                if writer_errors:
                    continue
                try:
                    converted = convert_u8_to_c64(packet)
                    take = min(
                        samples_required - writer_count[0], len(converted)
                    )
                    if take:
                        converted[:take].tofile(stream)
                        writer_count[0] += take
                except Exception as exc:
                    writer_errors.append(exc)
        except Exception as exc:
            writer_errors.append(exc)

    writer = Thread(
        target=write_buffers, name="OneSeg-IQ-Writer", daemon=True
    )
    writer.start()
    acquired = 0
    read_error = None
    try:
        for _ in range(warmup_buffers):
            if cancelled():
                raise CaptureCancelled("I/Q capture cancelled")
            data = bytes(device.read_bytes(2 * READ_SAMPLES))
            if len(data) != 2 * READ_SAMPLES:
                raise IOError(
                    f"short warmup USB read: {len(data)}/{2*READ_SAMPLES}"
                )
        while acquired < samples_required:
            if cancelled():
                raise CaptureCancelled("I/Q capture cancelled")
            if writer_errors:
                raise IOError(
                    f"I/Q writer thread failed: {writer_errors[0]}"
                )
            data = bytes(device.read_bytes(2 * READ_SAMPLES))
            if len(data) != 2 * READ_SAMPLES:
                raise IOError(
                    f"short USB read: {len(data)}/{2*READ_SAMPLES}"
                )
            acquired += min(READ_SAMPLES, samples_required - acquired)
            try:
                queued.put_nowait(data)
            except Full as exc:
                raise IOError(
                    "I/Q writer queue overran; capture cannot be trusted"
                ) from exc
    except Exception as exc:
        read_error = exc
    finally:
        # Reader is no longer touching the device; let the writer drain.
        # The writer always drains after an error, so this cannot get stuck
        # behind queued data when the output file fails.
        queued.put(None)
        writer.join()

    if read_error is not None:
        raise read_error
    if writer_errors:
        raise IOError(f"I/Q writer failed: {writer_errors[0]}") from writer_errors[0]
    if writer_count[0] != samples_required:
        raise IOError(
            f"incomplete capture: {writer_count[0]}/{samples_required} samples"
        )
    return writer_count[0]


def record_stream(
    device,
    output: Path,
    *,
    samples_required: int,
    metadata: dict,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    """Atomically publish completed .c64 and matching sidecar, no overwrite."""
    output = Path(output)
    if output.suffix.lower() != ".c64":
        raise ValueError("capture must have .c64 extension")
    partial = output.with_name(output.name + ".partial")
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or partial.exists() or sidecar.exists():
        raise FileExistsError(
            f"capture, partial file or metadata already exists for {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        **metadata,
        "format": "complex64",
        "byte_order": "little-endian",
        "sample_rate_hz": 2_048_000,
        "samples": samples_required,
        "capture_samples": samples_required,
        "acquisition": "sync_usb_reader_threaded_iq_writer",
        "sample_continuity_verified": False,
        "decoding_status": "RAW_IQ_NOT_TS",
    }
    try:
        with partial.open("xb") as destination:
            stream_exact_samples(
                device, destination, samples_required, cancelled=cancelled
            )
        sidecar.write_text(
            json.dumps(details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(partial, output)
        return details
    except Exception:
        partial.unlink(missing_ok=True)
        if not output.exists():
            sidecar.unlink(missing_ok=True)
        raise
