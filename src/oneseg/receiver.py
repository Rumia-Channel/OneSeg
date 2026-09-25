"""RTL-SDR acquisition worker. All USB access happens on a single QThread."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event

import numpy as np
from PySide6.QtCore import QThread, Signal

from .dsp import DEFAULT_SAMPLE_RATE, power_spectrum
from .wfm import AudioQueue, MonoWfm

READ_SIZE = 131_072


class Receiver(QThread):
    spectrum = Signal(object)
    message = Signal(str)
    failed = Signal(str)
    recording = Signal(bool)
    device_ready = Signal()

    def __init__(
        self,
        frequency_hz: int,
        mode: str = "sdr",
        ppm: int = 0,
        gain: float | str = "auto",
    ):
        super().__init__()
        self.frequency_hz = frequency_hz
        self.mode = mode
        self.ppm = ppm
        self.gain = gain
        self.commands = Queue()
        self.stop_event = Event()

    def request(self, command: str, *args):
        self.commands.put((command, args))

    def stop(self):
        self.stop_event.set()

    def run(self):
        device = None
        file = None
        audio = None
        demod = MonoWfm()
        audio_enabled = False

        def close_file():
            nonlocal file
            if file is not None:
                file.close()
                file = None
                self.recording.emit(False)

        def close_audio():
            nonlocal audio
            if audio is not None:
                audio.close()
                audio = None

        try:
            # Import lazily so the GUI can show a helpful error if the USB DLL
            # or hardware is missing rather than failing at process startup.
            from rtlsdr import RtlSdr

            device = RtlSdr()
            device.sample_rate = DEFAULT_SAMPLE_RATE
            device.freq_correction = self.ppm
            device.center_freq = self.frequency_hz
            device.gain = self.gain
            self.device_ready.emit()
            self.message.emit("RTL-SDR connected")

            while not self.stop_event.is_set():
                try:
                    while True:
                        command, args = self.commands.get_nowait()
                        if command == "tune":
                            next_hz = int(args[0])
                            if next_hz != self.frequency_hz:
                                close_file()
                                self.frequency_hz = next_hz
                                device.center_freq = next_hz
                                demod.reset()
                                self.message.emit(f"Tuned to {next_hz / 1e6:.6f} MHz")
                        elif command == "settings":
                            self.ppm, self.gain = args
                            device.freq_correction = int(self.ppm)
                            device.gain = self.gain
                        elif command == "mode":
                            self.mode = str(args[0])
                            close_file()
                            demod.reset()
                            if self.mode != "sdr":
                                close_audio()
                        elif command == "audio":
                            audio_enabled = bool(args[0])
                            if not audio_enabled:
                                close_audio()
                            elif self.mode == "sdr" and audio is None:
                                try:
                                    audio = AudioQueue()
                                    demod.reset()
                                except Exception as exc:
                                    audio_enabled = False
                                    self.message.emit(f"Audio unavailable: {exc}")
                        elif command == "record":
                            close_file()
                            path = Path(args[0])
                            path.parent.mkdir(parents=True, exist_ok=True)
                            file = path.open("wb")
                            metadata = {
                                "format": "complex64",
                                "byte_order": "little-endian",
                                "sample_rate_hz": DEFAULT_SAMPLE_RATE,
                                "center_frequency_hz": self.frequency_hz,
                                "ppm": int(self.ppm),
                                "gain_db": self.gain,
                                "mode": self.mode,
                                "started_utc": datetime.now(timezone.utc).isoformat(),
                            }
                            try:
                                path.with_suffix(path.suffix + ".json").write_text(
                                    json.dumps(metadata, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                            except Exception:
                                close_file()
                                raise
                            self.recording.emit(True)
                            self.message.emit(f"Recording: {path.name}")
                        elif command == "record_stop":
                            close_file()
                except Empty:
                    pass

                samples = np.asarray(device.read_samples(READ_SIZE), dtype=np.complex64)
                if file is not None:
                    np.asarray(samples, dtype="<c8").tofile(file)
                if audio_enabled and self.mode == "sdr" and audio is not None:
                    audio.feed(demod.process(samples))
                x_mhz, y_db = power_spectrum(
                    samples, DEFAULT_SAMPLE_RATE, self.frequency_hz
                )
                self.spectrum.emit((x_mhz, y_db))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                close_file()
                close_audio()
            finally:
                if device is not None:
                    device.close()
            self.message.emit("Receiver stopped")
