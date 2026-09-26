"""Offline GUI decode button must never open a tuner or block the Qt event loop."""
import os
import sys

import pytest

if sys.platform != "win32":
    pytest.skip("GUI Qt Windows CI only", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from oneseg.gui import MainWindow, OfflineDecodeWorker


def test_offline_button_requires_adjacent_iq_metadata(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "capture.c64"
    source.write_bytes(b"\0" * 8)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        lambda *a, **k: (str(source), "Complex64 I/Q (*.c64)")
    )
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a: warnings.append(a)
    )
    window._decode_saved_iq()
    assert window.decoder is None
    assert warnings and "metadata" in warnings[0][1].lower()
    window.close()


def test_offline_button_uses_background_thread_and_new_output(
    tmp_path, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = tmp_path / "capture.c64"
    source.write_bytes(b"\0" * 8)
    source.with_suffix(".c64.json").write_text("{}")
    output = tmp_path / "partial.ts"
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        lambda *a, **k: (str(source), "Complex64 I/Q (*.c64)")
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(output), "MPEG-TS (*.ts)")
    )
    starts = []
    monkeypatch.setattr(
        OfflineDecodeWorker, "start", lambda self: starts.append(
            (self.capture, self.target)
        )
    )
    window._decode_saved_iq()
    assert starts == [(source, output)]
    assert not window.decode_iq_btn.isEnabled()
    assert window.decoder is not None
    assert window.worker is None
    window.decoder = None
    window.close()
