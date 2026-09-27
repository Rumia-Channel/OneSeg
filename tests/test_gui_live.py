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
    assert started == [(515142857, 0, -3.0)]
    assert not window.live_player_started
    no_psi = {
        "accepted_total": 12, "accepted_chunk": 12,
        "rejected_chunk": 2, "windows_failed": 0,
        "has_pat": False, "has_pmt": False, "overloaded": False,
    }
    sample = b"\x47" + b"\0" * 187
    window._live_progress(no_psi)
    window._live_transport(sample)
    assert window.live_ts_skipped_waiting_psi == 1
    assert started == [(515142857, 0, -3.0)]
    window._live_progress({**no_psi, "has_pat": True, "has_pmt": True})
    assert window.live_player_started
    assert started[-1] == "player"
    window._live_transport(sample)
    assert window.live_stream.read(188) == sample
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



def test_failed_live_window_surfaces_actual_adc_and_stage_bottleneck():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._live_progress({
        "accepted_total": 29,
        "accepted_chunk": 0,
        "rejected_chunk": 0,
        "windows_failed": 2,
        "has_pat": False,
        "has_pmt": False,
        "overloaded": True,
        "fullscale_percent": 11.32,
        "usb_window_seconds": 3.02,
        "decoder_window_seconds": 5.83,
        "queued_windows": 1,
        "failure_reason": "ValueError: no parity-verified TMCC frames",
    })
    assert "11.32%" in window.live_metrics.text()
    assert "DSP 5.83s" in window.live_metrics.text()
    assert "Latest window FAILED" in window.live_metrics.text()
    assert "no parity-verified TMCC frames" in window.video.text()
    window.close()
