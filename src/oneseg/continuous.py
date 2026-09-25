"""Continuous RTL-SDR callback recorder for I/Q continuity experiments.

The callback avoids repeated synchronous read_samples() calls which may
allow USB FIFO data to be lost while GUI FFT processing takes place.
This cannot guarantee an overrun-free recording without tuner timestamps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

READ_SAMPLES = 131_072


class CaptureCancelled(RuntimeError):
    """User requested capture cancellation."""


def stream_exact_samples(
    device,
    stream,
    samples_required: int,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    warmup_buffers: int = 1,
) -> int:
    """Read from a single librtlsdr asynchronous stream, with no FFT/UI work.

    All file I/O occurs on the receiver worker callback thread. An exception
    raised by the Python callback cannot be left to ctypes to silently log:
    we store it, cancel the async USB transfer and re-raise afterward.
    """
    if samples_required <= 0 or warmup_buffers < 0:
        raise ValueError("invalid capture length/warmup")
    written = 0
    skip = warmup_buffers
    failure = None

    def cancel():
        nonlocal failure
        try:
            device.cancel_read_async()
        except Exception as exc:
            if failure is None:
                failure = exc

    def callback(samples, context):
        nonlocal written, skip, failure
        if failure is not None:
            cancel()
            return
        try:
            if cancelled():
                raise CaptureCancelled("I/Q capture cancelled")
            iq = np.asarray(samples, dtype=np.complex64)
            if len(iq) != READ_SAMPLES:
                raise IOError(
                    f"short asynchronous RTL-SDR buffer: {len(iq)}/{READ_SAMPLES}"
                )
            if skip:
                skip -= 1
                return
            take = min(len(iq), samples_required - written)
            if take:
                np.asarray(iq[:take], dtype="<c8").tofile(stream)
                written += take
            if written >= samples_required:
                cancel()
        except Exception as exc:
            failure = exc
            cancel()

    device.read_samples_async(callback, num_samples=READ_SAMPLES)
    if failure is not None:
        raise failure
    if written != samples_required:
        raise IOError(f"incomplete asynchronous capture: {written}/{samples_required}")
    return written


def record_stream(
    device,
    output: Path,
    *,
    samples_required: int,
    metadata: dict,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    """Write file and sidecar atomically; never publish incomplete .c64 files."""
    output = Path(output)
    if output.suffix.lower() != ".c64":
        raise ValueError("capture must have .c64 extension")
    partial = output.with_name(output.name + ".partial")
    sidecar = output.with_suffix(output.suffix + ".json")
    # Do not overwrite user's prior recordings or metadata.
    if output.exists() or partial.exists() or sidecar.exists():
        raise FileExistsError(f"recording/partial/sidecar already exists for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        **metadata,
        "format": "complex64",
        "byte_order": "little-endian",
        "sample_rate_hz": 2_048_000,
        "samples": samples_required,
        "capture_samples": samples_required,
        "acquisition": "continuous_async",
        "decoding_status": "RAW_IQ_NOT_TS",
    }
    try:
        with partial.open("xb") as destination:
            stream_exact_samples(
                device, destination, samples_required, cancelled=cancelled
            )
        sidecar.write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(partial, output)
        return details
    except Exception:
        partial.unlink(missing_ok=True)
        # Never delete a preexisting sidecar (checked above).
        if not output.exists():
            sidecar.unlink(missing_ok=True)
        raise
