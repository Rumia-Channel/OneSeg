"""Simple mono broadcast-FM demodulation; SDR mode only, NOT 1seg audio."""

from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np
from scipy import signal

from .dsp import DEFAULT_SAMPLE_RATE


class MonoWfm:
    """Continuous mono FM discriminator; 2.048 MS/s -> 48 kHz audio."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError("This prototype requires a 2.048 MS/s input")
        self.sample_rate = sample_rate
        self.rf_sos = signal.butter(4, 100_000, fs=sample_rate, output="sos")
        self.audio_sos = signal.butter(4, 15_000, fs=sample_rate, output="sos")
        self.reset()

    def reset(self):
        self.rf_zi = np.zeros((len(self.rf_sos), 2), dtype=np.complex128)
        self.audio_zi = np.zeros((len(self.audio_sos), 2), dtype=np.float64)
        self.previous = np.complex128(0)

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size == 0:
            return np.empty(0, dtype=np.float32)
        filtered, self.rf_zi = signal.sosfilt(self.rf_sos, iq, zi=self.rf_zi)
        prev = np.concatenate(([self.previous], filtered[:-1]))
        self.previous = filtered[-1]
        phase = np.angle(filtered * np.conj(prev))
        audio = phase * (self.sample_rate / (2 * np.pi * 75_000))
        audio, self.audio_zi = signal.sosfilt(self.audio_sos, audio, zi=self.audio_zi)
        # 2,048,000 * 3 / 128 = 48,000 exactly.
        downsampled = signal.resample_poly(audio, 3, 128)
        return np.clip(downsampled * 0.6, -1.0, 1.0).astype(np.float32)


class AudioQueue:
    """Small bounded FIFO consumed by sounddevice's PortAudio callback."""

    def __init__(self):
        import sounddevice as sd

        self.lock = Lock()
        self.chunks = deque()
        self.offset = 0
        self.stream = sd.OutputStream(
            samplerate=48_000,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def feed(self, audio: np.ndarray):
        with self.lock:
            if len(self.chunks) >= 8:
                self.chunks.popleft()
                self.offset = 0
            self.chunks.append(np.asarray(audio, dtype=np.float32).copy())

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        index = 0
        with self.lock:
            while index < frames and self.chunks:
                chunk = self.chunks[0]
                size = min(frames - index, len(chunk) - self.offset)
                outdata[index:index + size, 0] = chunk[self.offset:self.offset + size]
                index += size
                self.offset += size
                if self.offset == len(chunk):
                    self.chunks.popleft()
                    self.offset = 0

    def close(self):
        self.stream.stop()
        self.stream.close()
