"""Windows offscreen GUI test: scan workflow without a connected RTL2832U."""
import os
import sys
from threading import Event

import pytest

if sys.platform != "win32":
    pytest.skip("QtGui test needs the Windows Qt runtime in CI", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from oneseg.gui import MainWindow


class StubWorker:
    def __init__(self):
        self.commands = []
        self.scan_cancel = Event()

    def request(self, command, *args):
        self.commands.append((command, args))

    def cancel_scan(self):
        self.scan_cancel.set()


def test_manual_gain_scan_and_select_physical_channel():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    stub = StubWorker()
    window.worker = stub
    window.record_btn.setEnabled(True)
    window.mode.setCurrentIndex(1)
    window.auto_gain.setChecked(False)
    window._update_scan_button()
    assert window.scan_button.isEnabled()

    window._scan_clicked()
    assert window.scanning
    assert ("scan", ()) in stub.commands
    assert window.scan_button.text() == "Cancel RF scan"

    entry = {
        "physical_channel": 14, "frequency_hz": 479_142_857,
        "power_dbfs": -45.0, "rms": 0.008,
        "clipping_percent": 0.0, "relative_db": 12.0,
        "status": "RF CANDIDATE (NOT TV LOCK)",
    }
    window._scan_measurement({**entry, "status": "UNASSESSED"})
    assert window.scan_table.rowCount() == 1
    window._scan_complete({"rows": [entry], "cancelled": False, "error": ""})
    assert not window.scanning
    assert window.scan_export_button.isEnabled()
    window.scan_table.selectRow(0)
    window._tune_scan_selection()
    assert window.channel.value() == 14
    assert abs(window.frequency.value() - 479.142857) < 0.00001

    window.worker = None
    window.close()
