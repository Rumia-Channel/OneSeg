"""Decode and display already recovered MPEG-TS using PyAV's FFmpeg wheels.

This is an *output* stage and not the missing I/Q -> ISDB-T TS demodulator.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import BinaryIO
from threading import Event

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
            self.path.close()

    def run(self):
        # Delay audio device initialization until the TS actually contains audio.
        from .wfm import AudioQueue

        audio_sink = None
        resampler = None
        container = None
        start_clock = time.monotonic()
        first_timestamp = None
        try:
            source = str(self.path) if isinstance(self.path, Path) else self.path
            container = av.open(source, format="mpegts")
            video_count = 0
            audio_count = 0
            self.status.emit(f"Reading MPEG-TS: {self.path.name}")
            for frame in container.decode():
                if self.stop_event.is_set():
                    break
                if frame.time is not None:
                    if first_timestamp is None:
                        first_timestamp = float(frame.time)
                    target = start_clock + max(0.0, float(frame.time) - first_timestamp)
                    while not self.stop_event.is_set() and target > time.monotonic():
                        time.sleep(min(0.025, target - time.monotonic()))
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
                    if audio_sink is None:
                        try:
                            audio_sink = AudioQueue()
                            resampler = av.AudioResampler(
                                format="flt", layout="mono", rate=48000
                            )
                        except Exception as exc:
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
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if audio_sink is not None:
                audio_sink.close()
            if container is not None:
                container.close()
