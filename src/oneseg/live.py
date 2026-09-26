"""Experimental LIVE RTL-SDR -> offline-window decode -> genuine TS packets.

This is a *first live integration*, NOT gapless/verified TV playback:
each 3-second raw u8 IQ window is expanded off the USB thread,
its time-deinterleaver warms up again, and USB reads have no timestamps. The decoder may lag or fail
and the GUI must report this instead of inventing an uninterrupted TS.
No native pyrtlsdr async transfer/cancel is used: the device is opened,
read synchronously, and closed by this one Qt worker thread.
"""
from __future__ import annotations

from collections import deque
from io import RawIOBase
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import TemporaryDirectory
from time import perf_counter
from threading import Condition, Event, Thread
from typing import Callable

from PySide6.QtCore import QThread, Signal

from .continuous import CaptureCancelled
from .raw_capture import record_raw_window, expand_raw_to_c64
from .decode import decode_capture
from .dsp import DEFAULT_SAMPLE_RATE
from .ppm import PpmCorrection


class LiveTsBuffer(RawIOBase):
    """Blocking non-seekable TS input for PyAV's custom-file reader.

    Backpressure is explicit: exceeding the memory bound rejects data,
    rather than silently dropping TS bytes and hiding a discontinuity.
    close() wakes a blocked FFmpeg read on Stop/exit.
    """
    def __init__(self, limit_bytes: int = 188 * 32768):
        super().__init__()
        self._condition = Condition()
        self._parts: deque[bytes] = deque()
        self._front_offset = 0
        self._pending_bytes = 0
        self._limit_bytes = limit_bytes
        self.name = "Experimental live one-seg TS"

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def push(self, packet_stream: bytes) -> None:
        packet_stream = bytes(packet_stream)
        if not packet_stream or len(packet_stream) % 188:
            raise ValueError("live input requires complete 188-byte TS packets")
        with self._condition:
            if self.closed:
                raise ValueError("live TS input is closed")
            if self._pending_bytes + len(packet_stream) > self._limit_bytes:
                raise BufferError(
                    "live TS player is behind; refusing to silently drop packets"
                )
            self._parts.append(packet_stream)
            self._pending_bytes += len(packet_stream)
            self._condition.notify_all()

    def read(self, size: int = -1):
        if size is None or size < 0:
            size = 64 * 1024
        if size == 0:
            return b""
        with self._condition:
            while self._pending_bytes == 0 and not self.closed:
                self._condition.wait()
            if self._pending_bytes == 0:
                return b""
            remaining = min(size, self._pending_bytes)
            parts = []
            while remaining and self._parts:
                front = self._parts[0]
                count = min(remaining, len(front) - self._front_offset)
                parts.append(front[self._front_offset:self._front_offset+count])
                self._front_offset += count
                remaining -= count
                self._pending_bytes -= count
                if self._front_offset == len(front):
                    self._parts.popleft()
                    self._front_offset = 0
            return b"".join(parts)

    def close(self):
        with self._condition:
            super().close()
            self._condition.notify_all()


class ExperimentalLiveReceiver(QThread):
    """Keep RTL USB read separate from NumPy/Numba decoding and PyAV."""

    status = Signal(str)
    failed = Signal(str)
    transport = Signal(bytes)
    progress = Signal(object)

    def __init__(
        self,
        *,
        frequency_hz: int,
        ppm: int,
        gain: float,
        chunk_seconds: float = 3.0,
        max_ofdm_symbols: int = 3000,
        device_factory: Callable | None = None,
        decode_function: Callable = decode_capture,
    ):
        super().__init__()
        if not 0.5 <= chunk_seconds <= 10.0:
            raise ValueError("chunk_seconds must be 0.5..10")
        if max_ofdm_symbols < 204:
            raise ValueError("max_ofdm_symbols must be >=204")
        if not -200 <= ppm <= 200 or not -10 <= gain <= 50:
            raise ValueError("invalid RTL tuner settings")
        self.frequency_hz = int(frequency_hz)
        self.ppm = int(ppm)
        self.gain = float(gain)
        self.chunk_seconds = float(chunk_seconds)
        self.max_ofdm_symbols = max_ofdm_symbols
        self.device_factory = device_factory
        self.decode_function = decode_function
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        device = None
        decoded = 0
        missing = 0
        capture_queue: Queue[Path | None] = Queue(maxsize=2)
        decoder: Thread | None = None
        temporary: TemporaryDirectory | None = None
        capture_timings: dict[Path, float] = {}

        try:
            temporary = TemporaryDirectory(prefix="oneseg-live-")
            root = Path(temporary.name)

            def consume():
                nonlocal decoded, missing
                while True:
                    item = capture_queue.get()
                    if item is None:
                        return
                    target = item.with_suffix(".ts")
                    iq = item.with_suffix(".c64")
                    capture_elapsed = capture_timings.pop(item, None)
                    decode_started = perf_counter()
                    try:
                        if not self.stop_event.is_set():
                            # The potentially CPU-intensive u8->float
                            # conversion is outside the USB reader thread.
                            expand_raw_to_c64(
                                item, iq,
                                metadata={
                                    "source": "experimental_live_window",
                                    "center_frequency_hz": self.frequency_hz,
                                    "gain_db": self.gain, "ppm": self.ppm,
                                },
                            )
                            result = self.decode_function(
                                iq, target,
                                seconds=self.chunk_seconds,
                                max_ofdm_symbols=self.max_ofdm_symbols,
                            )
                            data = target.read_bytes()
                            if len(data) != 188 * result["rs_and_ts_accepted_packets"]:
                                raise IOError("recovered TS byte count mismatch")
                            decoded += result["rs_and_ts_accepted_packets"]
                            self.progress.emit({
                                "accepted_total": decoded,
                                "accepted_chunk": result["rs_and_ts_accepted_packets"],
                                "rejected_chunk": result["rejected_rs_or_invalid_ts_packets"],
                                "has_pat": bool(result["pat_programs"]),
                                "has_pmt": bool(result["pmt_elementary_streams"]),
                                "overloaded": bool(result["input_overload_warning"]),
                                "windows_failed": missing,
                                "usb_window_seconds": capture_elapsed,
                                "decoder_window_seconds": round(
                                    perf_counter() - decode_started, 2
                                ),
                                "queued_windows": capture_queue.qsize(),
                            })
                            self.transport.emit(data)
                            self.status.emit(
                                f"LIVE EXPERIMENT: {decoded} genuine TS packets; "
                                f"latest +{result['rs_and_ts_accepted_packets']}, "
                                f"PAT {'found' if result['pat_programs'] else 'missing'}. "
                                "Window gaps possible; video NOT yet verified."
                            )
                    except Exception as exc:
                        missing += 1
                        self.status.emit(
                            f"Live window rejected ({type(exc).__name__}: {exc}); "
                            "continuing RF reads; no fake TS packets."
                        )
                    finally:
                        for name in (
                            item, item.with_name(item.name + ".partial"),
                            iq, iq.with_suffix(".c64.json"),
                            target, target.with_suffix(".ts.json"),
                        ):
                            name.unlink(missing_ok=True)

            # JIT initialization can briefly hold the Python GIL. Do it
            # BEFORE opening USB, not during the first subsequent RF read
            # where blocking Python callbacks could overflow RTL buffers.
            import numpy as np
            from .viterbi import decode_soft_bits
            self.status.emit(
                "Preparing compiled 64-state Viterbi decoder "
                "before RTL-SDR capture..."
            )
            decode_soft_bits(
                np.zeros(3 * 64, dtype=np.float32), "2/3"
            )
            if self.stop_event.is_set():
                return
            decoder = Thread(
                target=consume, name="OneSeg-Live-Decoder", daemon=False
            )
            decoder.start()
            if self.device_factory is None:
                from rtlsdr import RtlSdr
                factory = RtlSdr
            else:
                factory = self.device_factory
            device = factory()
            device.sample_rate = DEFAULT_SAMPLE_RATE
            PpmCorrection().apply(device, self.ppm)
            device.center_freq = self.frequency_hz
            device.gain = self.gain
            self.status.emit(
                f"EXPERIMENTAL live 1seg: {self.frequency_hz/1e6:.6f} MHz "
                f"gain {self.gain:g} dB. Buffering {self.chunk_seconds:g}s "
                "I/Q windows; playback and continuity NOT guaranteed."
            )

            ordinal = 0
            while not self.stop_event.is_set():
                capture = root / f"window_{ordinal:06d}.u8iq"
                ordinal += 1
                try:
                    record_raw_window(
                        device, capture,
                        samples_required=round(
                            DEFAULT_SAMPLE_RATE * self.chunk_seconds
                        ),
                        warmup_buffers=1 if ordinal == 1 else 0,
                        cancelled=self.stop_event.is_set,
                    )
                except CaptureCancelled:
                    break
                except Exception as exc:
                    self.failed.emit(
                        f"RTL-SDR live USB capture failed: {exc}"
                    )
                    break
                try:
                    capture_queue.put_nowait(capture)
                except Full:
                    missing += 1
                    capture.unlink(missing_ok=True)
                    capture.with_name(
                        capture.name + ".partial"
                    ).unlink(missing_ok=True)
                    self.status.emit(
                        "LIVE decoder behind RF by more than two "
                        "3-second windows; dropped an ENTIRE window. "
                        "CPU too slow for this experimental mode."
                    )
                    # Never block the USB read loop to wait for DSP.
        except Exception as exc:
            self.failed.emit(f"Could not start live RTL-SDR: {exc}")
        finally:
            # The only thread that opens the tuner also closes it; do not
            # invoke librtlsdr_cancel_async or close it from consumer/GUI.
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    self.failed.emit(f"RTL-SDR close failed: {exc}")
            # Drain any queued work without decoding after stop. Stop
            # typically means an immediate USB release, not processing
            # another several seconds of old video.
            self.stop_event.set()
            while True:
                try:
                    capture_queue.put(None, timeout=0.2)
                    break
                except Full:
                    try:
                        dropped = capture_queue.get_nowait()
                    except Empty:
                        continue
                    if dropped is not None:
                        dropped.unlink(missing_ok=True)
                        dropped.with_name(
                            dropped.name + ".partial"
                        ).unlink(missing_ok=True)
            if decoder is not None:
                decoder.join()
            if temporary is not None:
                temporary.cleanup()
            self.status.emit(
                f"Experimental live session ended: {decoded} "
                "real TS packets; NOT a confirmed gapless TV receiver."
            )
