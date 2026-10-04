"""Decode and display recovered MPEG-TS using PyAV's FFmpeg wheels.

This is an output stage, not proof of uninterrupted RF television reception.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import BinaryIO

import av
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class TransportPlayer(QThread):
    image_ready = Signal(QImage)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, path: Path | BinaryIO):
        super().__init__()
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()
        if not isinstance(self.path, Path):
            # Wake blocking FFmpeg reads of a non-seekable live TS buffer.
            # A source may already have been closed by the receiver.
            try:
                self.path.close()
            except Exception as exc:
                self.status.emit(f"Could not close TS input: {exc}")

    def run(self):
        if self.stop_event.is_set():
            return
        # Delay audio device initialization until the TS actually contains audio.
        from .wfm import AudioQueue

        audio_sink = None
        audio_unavailable = False
        resampler = None
        container = None
        start_clock = None
        first_timestamp = None
        try:
            source = str(self.path) if isinstance(self.path, Path) else self.path
            container = av.open(source, format="mpegts")
            if self.stop_event.is_set():
                return
            video_count = 0
            audio_count = 0
            name = getattr(self.path, "name", "MPEG-TS stream")
            self.status.emit(f"Reading MPEG-TS: {name}")
            for frame in container.decode():
                if self.stop_event.is_set():
                    break
                if frame.time is not None:
                    if first_timestamp is None:
                        first_timestamp = float(frame.time)
                        # FFmpeg may spend seconds probing a live input. Playback
                        # starts with its first frame, not before that probe.
                        start_clock = time.monotonic()
                    target = start_clock + max(0.0, float(frame.time) - first_timestamp)
                    delay = target - time.monotonic()
                    if delay > 0 and self.stop_event.wait(delay):
                        break
                if self.stop_event.is_set():
                    break
                if isinstance(frame, av.video.frame.VideoFrame):
                    image_array = frame.to_ndarray(format="rgb24")
                    height, width, _ = image_array.shape
                    image = QImage(
                        image_array.data,
                        width,
                        height,
                        int(image_array.strides[0]),
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.image_ready.emit(image)
                    video_count += 1
                elif isinstance(frame, av.audio.frame.AudioFrame):
                    audio_count += 1
                    if audio_sink is None and not audio_unavailable:
                        try:
                            resampler = av.AudioResampler(
                                format="flt", layout="mono", rate=48000
                            )
                            audio_sink = AudioQueue()
                        except Exception as exc:
                            audio_unavailable = True
                            self.status.emit(f"No audio output device: {exc}")
                    if audio_sink is not None and resampler is not None:
                        for out_frame in resampler.resample(frame):
                            audio = np.asarray(
                                out_frame.to_ndarray(), dtype=np.float32
                            ).reshape(-1)
                            audio_sink.feed(audio)
            self.status.emit(
                f"TS player stopped: {video_count} video, {audio_count} audio frames"
            )
        except Exception as exc:
            # Closing a live input interrupts av.open/decode. This is expected
            # on Stop; don't report it as an RF or media failure.
            if not self.stop_event.is_set():
                self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # A failing sound device must not prevent FFmpeg/file cleanup or
            # escape QThread.run as an unhandled exception.
            for resource, label in ((audio_sink, "audio output"), (container, "TS input")):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        if not self.stop_event.is_set():
                            self.failed.emit(f"Could not close {label}: {exc}")
