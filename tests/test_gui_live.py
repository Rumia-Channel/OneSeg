"""Windows offscreen GUI experimental live demodulation controls."""

import os
import sys
import pytest

if sys.platform != "win32":
    pytest.skip("GUI Qt Windows CI only", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from oneseg.gui import MainWindow, ExperimentalLiveReceiver, TransportPlayer


def test_live_toggle_does_not_open_native_tuner_in_gui_thread(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.mode.setCurrentIndex(
        window.mode.findData("oneseg")
    )
    window.channel.setValue(20)
    window.auto_gain.setChecked(False)
    window.gain.setValue(-3.0)
    assert window.live_btn.isEnabled()

    started=[]
    monkeypatch.setattr(
        ExperimentalLiveReceiver, "start",
        lambda self: started.append(
            (self.frequency_hz, self.ppm, self.gain)
        )
    )
    monkeypatch.setattr(
        TransportPlayer, "start", lambda self: started.append("player")
    )
    window._toggle_live()
    assert window.live is not None
    assert started == ["player", (515142857, 0, -3.0)]
    assert not window.start_btn.isEnabled()
    assert not window.play_ts_btn.isEnabled()
    assert not window.decode_iq_btn.isEnabled()
    assert window.live_btn.text() == "Stop experimental LIVE 1seg"
    window._stop_live()
    assert window.live.stop_event.is_set()
    assert window.live_stream.closed
    window._player_finished()
    window._live_finished()
    assert window.live is None
    assert window.start_btn.isEnabled()
    assert window.live_btn.isEnabled()
    window.close()


def test_live_not_available_with_automatic_gain_or_sdr_mode():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert not window.live_btn.isEnabled()
    window.mode.setCurrentIndex(window.mode.findData("oneseg"))
    assert not window.live_btn.isEnabled()
    window.auto_gain.setChecked(False)
    assert window.live_btn.isEnabled()
    window.mode.setCurrentIndex(window.mode.findData("sdr"))
    assert not window.live_btn.isEnabled()
    window.close()
