"""Offscreen Qt lifecycle regressions using real controllably delayed workers."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from threading import Event
import time

import pytest
from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from oneseg import gui
from oneseg.live import LiveTsBuffer
from oneseg.ts import mpeg_crc32


@pytest.fixture
def app(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui, "QSettings", lambda *args: QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat,
    ))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    return app


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


class DelayedReceiver(QThread):
    spectrum = Signal(object)
    device_ready = Signal()
    message = Signal(str)
    failed = Signal(str)
    recording = Signal(bool)
    scan_started = Signal()
    scan_measurement = Signal(object)
    scan_complete = Signal(object)

    def __init__(self, **kwargs):
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.stop_requested = Event()
        self.commands = []

    def run(self):
        self.entered.set()
        self.device_ready.emit()
        self.release.wait(5)

    def stop(self):
        self.stop_requested.set()

    def request(self, *args):
        self.commands.append(args)


def test_repeated_receiver_start_stop_and_late_ready(app, monkeypatch):
    monkeypatch.setattr(gui, "Receiver", DelayedReceiver)
    window = gui.MainWindow()
    window.show()
    window._start()
    first = window.worker
    try:
        wait_until(app, lambda: window.record_btn.isEnabled())
        window._start()
        assert window.worker is first
        window._stop()
        assert first.stop_requested.is_set()
        assert not window.record_btn.isEnabled()
        first.device_ready.emit()
        app.processEvents()
        assert not window.record_btn.isEnabled()
        assert not window.short_capture_btn.isEnabled()
        window._start()
        assert window.worker is first
        first.release.set()
        wait_until(app, lambda: window.worker is None)
        assert window.start_btn.isEnabled()
        window._start()
        assert window.worker is not None and window.worker is not first
        wait_until(app, lambda: window.record_btn.isEnabled())
    finally:
        if window.worker is not None:
            window.worker.release.set()
            window.worker.wait(1000)
            app.processEvents()
        window.close()
        app.processEvents()


def test_close_requests_stop_without_blocking_and_closes_when_done(app, monkeypatch):
    monkeypatch.setattr(gui, "Receiver", DelayedReceiver)
    window = gui.MainWindow()
    window.show()
    window._start()
    worker = window.worker
    try:
        wait_until(app, worker.entered.is_set)
        before = time.monotonic()
        assert not window.close()
        assert time.monotonic() - before < 0.2
        assert window.isVisible()
        assert worker.stop_requested.is_set()
        assert not window.centralWidget().isEnabled()
        window._start()
        assert window.worker is worker
        worker.release.set()
        wait_until(app, lambda: not window.isVisible())
        assert window.worker is None
    finally:
        worker.release.set()
        app.processEvents()
        window.close()


def test_close_waits_for_offline_decode_then_finishes_itself(app, monkeypatch, tmp_path):
    entered, release = Event(), Event()
    source = tmp_path / "capture.c64"
    source.write_bytes(bytes(8))
    source.with_suffix(".c64.json").write_text("{}")
    target = tmp_path / "decoded.ts"
    def delayed_decode(*args, **kwargs):
        entered.set()
        release.wait(5)
        return {"rs_and_ts_accepted_packets": 1, "pat_programs": {}}
    monkeypatch.setattr(gui, "decode_capture", delayed_decode)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(source), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window = gui.MainWindow()
    window.show()
    window._decode_saved_iq()
    try:
        wait_until(app, entered.is_set)
        assert not window.close()
        assert window.isVisible()
        release.set()
        wait_until(app, lambda: not window.isVisible())
        assert window.decoder is None
    finally:
        release.set()
        if window.decoder is not None:
            window.decoder.wait(1000)
            app.processEvents()
        window.close()


def start_mock_live(window, monkeypatch):
    monkeypatch.setattr(gui.ExperimentalLiveReceiver, "start", lambda self: None)
    starts = []
    monkeypatch.setattr(gui.TransportPlayer, "start", lambda self: starts.append(self))
    window.mode.setCurrentIndex(window.mode.findData("oneseg"))
    window.auto_gain.setChecked(False)
    window._toggle_live()
    return starts


def report(**kwargs):
    return {
        "accepted_total": 1, "accepted_chunk": 1, "rejected_chunk": 0,
        "windows_failed": 0, "has_pat": True, "has_pmt": False,
        "overloaded": False, **kwargs,
    }


def section_packet(table_id, pid, contents):
    section = bytes((table_id, 0xb0, len(contents) + 4)) + contents
    payload = b"\0" + section + mpeg_crc32(section).to_bytes(4, "big")
    return bytes((0x47, 0x40 | (pid >> 8), pid & 255, 0x10)) + payload.ljust(184, b"\xff")


def test_live_keeps_original_pat_until_later_pmt(app, monkeypatch):
    window = gui.MainWindow()
    starts = start_mock_live(window, monkeypatch)
    pat = section_packet(0, 0, bytes.fromhex("0001c100000007e100"))
    pmt = section_packet(2, 0x100, bytes.fromhex("0007c10000e200f0001be200f000"))
    window._live_progress(report())
    window._live_transport(pat)
    assert starts == []
    assert window.live_ts_buffered_waiting_psi == 1
    window._live_progress(report(has_pmt=True, accepted_total=2))
    window._live_transport(pmt)
    assert len(starts) == 1
    assert window.live_stream.read(376) == pat + pmt
    window._stop_live()
    window._player_finished()
    window._live_finished()
    window.close()


def test_live_buffer_overflow_before_psi_stops_instead_of_silent_drop(app, monkeypatch):
    window = gui.MainWindow()
    starts = start_mock_live(window, monkeypatch)
    window.live_stream.close()
    window.live_stream = LiveTsBuffer(limit_bytes=188)
    data = b"\x47\x1f\xff\x10" + b"\xff" * 184
    window._live_transport(data)
    window._live_transport(data)
    assert starts == []
    assert window.live.stop_event.is_set()
    assert window.live_stream.closed
    assert "waiting for PAT/PMT" in window.status.text()
    window._live_finished()
    window.close()


def test_late_live_progress_and_packets_cannot_restart_stopped_player(app, monkeypatch):
    window = gui.MainWindow()
    starts = start_mock_live(window, monkeypatch)
    window._stop_live()
    status = window.status.text()
    window._live_progress(report(has_pmt=True))
    window._live_transport(b"\x47\x1f\xff\x10" + b"\xff" * 184)
    window._update_live_button()
    assert starts == []
    assert window.player is None
    assert window.status.text() == status
    assert not window.live_btn.isEnabled()
    window._live_finished()
    window.close()


def test_file_playback_error_is_not_presented_as_usb_failure(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    window = gui.MainWindow()
    window._player_error("Invalid MPEG-TS")
    assert warnings[0][1] == "TS playback error"
    assert warnings[0][2] == "Invalid MPEG-TS"
    assert window._receiver_error == ""
    window.close()


def test_natural_live_completion_releases_player_that_never_started(app, monkeypatch):
    window = gui.MainWindow()
    starts = start_mock_live(window, monkeypatch)
    stream = window.live_stream
    window._live_finished()
    assert starts == []
    assert window.player is None
    assert stream.closed
    assert window.live_btn.isEnabled()
    window.close()


def test_stopping_receiver_preserves_reported_failure(app, monkeypatch):
    monkeypatch.setattr(gui, "Receiver", DelayedReceiver)
    window = gui.MainWindow()
    window._start()
    worker = window.worker
    try:
        wait_until(app, worker.entered.is_set)
        worker.failed.emit("USB device disconnected")
        worker.release.set()
        wait_until(app, lambda: window.worker is None)
        assert window.status.text() == "USB device disconnected"
    finally:
        worker.release.set()
        window.close()
        app.processEvents()
