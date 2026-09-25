"""Windows desktop UI; TV mode is RF research, not decoded video."""

from __future__ import annotations

import sys
import csv
from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .channels import physical_channel_hz
from .receiver import Receiver
from .player import TransportPlayer
from .diagnostics import explain_receiver_error


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OneSeg — DS-DT308SV SDR / 1seg research")
        self.resize(1050, 720)
        self.worker: Receiver | None = None
        self.player: TransportPlayer | None = None
        self.scanning = False
        self.scan_rows = []
        self.settings = QSettings("Rumia-Channel", "OneSeg")

        shell = QWidget()
        self.setCentralWidget(shell)
        layout = QVBoxLayout(shell)

        self.notice = QLabel(
            "1seg live video decoding is not integrated. Offline TS, audio/video "
            "playback and RF/I-Q analysis are available."
        )
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(
            "background: #3b2a10; color: #fff0c0; padding: 11px; border-radius: 4px;"
        )
        layout.addWidget(self.notice)

        controls = QGroupBox("Receiver")
        form = QFormLayout(controls)
        self.mode = QComboBox()
        self.mode.addItem("SDR (spectrum / mono WFM)", "sdr")
        self.mode.addItem("1seg RF research (no video decoder)", "oneseg")
        form.addRow("Mode", self.mode)

        self.channel = QSpinBox()
        self.channel.setRange(13, 52)
        self.channel.setValue(27)
        form.addRow("Physical RF channel", self.channel)

        self.frequency = QDoubleSpinBox()
        self.frequency.setRange(22.0, 1100.0)
        self.frequency.setDecimals(6)
        self.frequency.setSuffix(" MHz")
        self.frequency.setSingleStep(0.1)
        self.frequency.setValue(82.5)
        form.addRow("RF center", self.frequency)

        self.ppm = QSpinBox()
        self.ppm.setRange(-200, 200)
        self.ppm.setSuffix(" ppm")
        self.ppm.setValue(int(self.settings.value("ppm", 0)))
        form.addRow("Frequency correction", self.ppm)

        self.auto_gain = QCheckBox("Automatic")
        self.auto_gain.setChecked(True)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-10, 20)
        self.gain.setDecimals(1)
        self.gain.setSuffix(" dB")
        self.gain.setToolTip("FC0013 gain can be negative; try -9.9 dB if samples clip.")
        gain_row = QWidget()
        gain_layout = QHBoxLayout(gain_row)
        gain_layout.setContentsMargins(0, 0, 0, 0)
        gain_layout.addWidget(self.auto_gain)
        gain_layout.addWidget(self.gain)
        form.addRow("RF gain", gain_row)

        self.wfm = QCheckBox("Listen to mono WFM (SDR mode)")
        form.addRow("FM audio", self.wfm)
        layout.addWidget(controls)

        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget(background="#10151b")
        self.plot.setLabel("bottom", "Frequency", units="MHz")
        self.plot.setLabel("left", "Power", units="dBFS")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setYRange(-110, 5)
        self.trace = self.plot.plot(pen=pg.mkPen("#57c8ed", width=1.5))
        layout.addWidget(self.plot, 1)

        scan_header = QHBoxLayout()
        self.scan_button = QPushButton("Scan UHF 13–52 (RF)")
        self.scan_button.setEnabled(False)
        self.scan_button.setToolTip(
            "Use 1seg mode and fixed manual RF gain; candidates are NOT decoded TV stations."
        )
        self.scan_tune_button = QPushButton("Tune selected")
        self.scan_tune_button.setEnabled(False)
        self.scan_export_button = QPushButton("Export scan CSV…")
        self.scan_export_button.setEnabled(False)
        scan_header.addWidget(self.scan_button)
        scan_header.addWidget(self.scan_tune_button)
        scan_header.addWidget(self.scan_export_button)
        layout.addLayout(scan_header)
        self.scan_note = QLabel(
            "RF power scan only: this does not identify TV stations or validate ISDB-T lock. "
            "Set manual RF gain before starting."
        )
        self.scan_note.setWordWrap(True)
        layout.addWidget(self.scan_note)
        self.scan_table = QTableWidget(0, 6)
        self.scan_table.setHorizontalHeaderLabels([
            "Physical ch", "MHz", "Power dBFS", "Above median dB",
            "Full-scale %", "RF assessment",
        ])
        self.scan_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scan_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.scan_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.scan_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.scan_table.setMinimumHeight(135)
        layout.addWidget(self.scan_table, 1)

        self.video = QLabel("Open an already decoded .ts file to play video/audio")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumHeight(140)
        self.video.setStyleSheet("background:#10151b;color:#b0b7be;")
        layout.addWidget(self.video, 1)

        actions = QHBoxLayout()
        self.start_btn = QPushButton("Start receiver")
        self.stop_btn = QPushButton("Stop")
        self.record_btn = QPushButton("Record I/Q…")
        self.play_ts_btn = QPushButton("Play decoded TS…")
        self.short_capture_btn = QPushButton("Capture 3s for decoder…")
        self.short_capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        actions.addWidget(self.start_btn)
        actions.addWidget(self.stop_btn)
        actions.addWidget(self.record_btn)
        actions.addWidget(self.short_capture_btn)
        actions.addWidget(self.play_ts_btn)
        layout.addLayout(actions)

        self.status = QLabel("Stopped")
        layout.addWidget(self.status)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.channel.valueChanged.connect(self._channel_changed)
        self.frequency.valueChanged.connect(self._frequency_changed)
        self.ppm.valueChanged.connect(self._settings_changed)
        self.auto_gain.toggled.connect(self._settings_changed)
        self.auto_gain.toggled.connect(self.gain.setDisabled)
        self.gain.setDisabled(self.auto_gain.isChecked())
        self.gain.valueChanged.connect(self._settings_changed)
        self.wfm.toggled.connect(self._audio_changed)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.record_btn.clicked.connect(self._record)
        self.short_capture_btn.clicked.connect(self._capture_short)
        self.play_ts_btn.clicked.connect(self._play_ts)
        self.scan_button.clicked.connect(self._scan_clicked)
        self.scan_tune_button.clicked.connect(self._tune_scan_selection)
        self.scan_export_button.clicked.connect(self._export_scan)
        self.scan_table.itemSelectionChanged.connect(self._scan_selection_changed)
        self._mode_changed()
        self._update_scan_button()

    def _mode_changed(self, *args):
        research = self.mode.currentData() == "oneseg"
        self.channel.setEnabled(research)
        self.wfm.setEnabled(not research)
        if research:
            self.wfm.setChecked(False)
            self.notice.show()
            self._channel_changed()
        else:
            self.notice.hide()
        if self.worker:
            self.worker.request("mode", self.mode.currentData())
        self._update_scan_button()

    def _channel_changed(self, *args):
        mhz = physical_channel_hz(self.channel.value()) / 1e6
        self.frequency.setValue(mhz)

    def _frequency_changed(self, *args):
        if self.worker:
            self.worker.request("tune", round(self.frequency.value() * 1e6))

    def _settings_changed(self, *args):
        self.settings.setValue("ppm", self.ppm.value())
        if self.worker:
            self.worker.request("settings", self.ppm.value(), self._gain())
        self._update_scan_button()

    def _gain(self):
        return "auto" if self.auto_gain.isChecked() else self.gain.value()

    def _audio_changed(self, enabled):
        if self.worker:
            self.worker.request("audio", enabled)

    def _start(self):
        if self.worker is not None:
            return
        self.worker = Receiver(
            frequency_hz=round(self.frequency.value() * 1e6),
            mode=self.mode.currentData(),
            ppm=self.ppm.value(),
            gain=self._gain(),
        )
        self.worker.spectrum.connect(self._update_spectrum)
        self.worker.device_ready.connect(self._ready)
        self.worker.message.connect(self.status.setText)
        self.worker.failed.connect(self._error)
        self.worker.recording.connect(self._recording)
        self.worker.scan_started.connect(self._scan_started)
        self.worker.scan_measurement.connect(self._scan_measurement)
        self.worker.scan_complete.connect(self._scan_complete)
        self.worker.finished.connect(self._finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Opening RTL-SDR…")
        self.worker.start()

    def _ready(self):
        self.record_btn.setEnabled(True)
        self.short_capture_btn.setEnabled(True)
        self._update_scan_button()
        if self.wfm.isChecked() and self.worker:
            self.worker.request("audio", True)

    def _update_spectrum(self, values):
        x_mhz, y_db = values
        self.trace.setData(x_mhz, y_db)

    def _update_scan_button(self):
        if self.scanning:
            return
        ready = self.worker is not None and self.record_btn.isEnabled()
        permitted = (
            ready
            and self.mode.currentData() == "oneseg"
            and not self.auto_gain.isChecked()
            and not bool(self.record_btn.property("active"))
        )
        self.scan_button.setEnabled(permitted)
        self.scan_button.setToolTip(
            "Scan is available after Start receiver, in 1seg RF research mode, "
            "with Automatic gain OFF and no active I/Q recording."
        )

    def _scan_controls(self, running: bool):
        self.mode.setEnabled(not running)
        self.channel.setEnabled(not running and self.mode.currentData() == "oneseg")
        self.frequency.setEnabled(not running)
        self.ppm.setEnabled(not running)
        self.auto_gain.setEnabled(not running)
        self.gain.setEnabled(not running and not self.auto_gain.isChecked())
        self.record_btn.setEnabled(not running and self.worker is not None)
        self.short_capture_btn.setEnabled(not running and self.worker is not None)
        self.wfm.setEnabled(not running and self.mode.currentData() == "sdr")
        self.scan_tune_button.setEnabled(not running and bool(
            self.scan_table.selectedItems()
        ))
        self.scan_export_button.setEnabled(not running and bool(self.scan_rows))

    def _scan_clicked(self):
        if self.worker is None:
            return
        if self.scanning:
            self.worker.cancel_scan()
            self.scan_button.setEnabled(False)
            self.scan_note.setText("Stopping RF scan after current USB read…")
            return
        if self.mode.currentData() != "oneseg" or self.auto_gain.isChecked():
            QMessageBox.information(
                self, "Fixed gain required",
                "Select 1seg RF research and switch Automatic RF gain OFF first. "
                "Use a fixed gain (start around -9.9 dB for overloaded signals)."
            )
            return
        if self.record_btn.property("active"):
            QMessageBox.information(
                self, "Stop recording", "Stop I/Q recording before running the RF scan."
            )
            return
        self.scan_rows = []
        self.scan_table.setRowCount(0)
        self.scanning = True
        self._scan_controls(True)
        self.scan_button.setText("Cancel RF scan")
        self.scan_button.setEnabled(True)
        self.scan_note.setText(
            "Scanning 40 physical RF channels with fixed gain; "
            "no broadcast/ISDB-T identity is being decoded."
        )
        self.worker.scan_cancel.clear()
        self.worker.request("scan")

    def _scan_started(self):
        self.status.setText("Scanning physical UHF channels 13–52…")

    def _scan_measurement(self, data):
        self.scan_note.setText(
            f"Measured {self.scan_table.rowCount() + 1}/40: "
            f"{data['physical_channel']}ch; assessment after baseline scan."
        )
        self._append_scan_row(data)

    def _append_scan_row(self, data):
        index = self.scan_table.rowCount()
        self.scan_table.insertRow(index)
        columns = [
            str(data["physical_channel"]),
            f"{data['frequency_hz'] / 1e6:.6f}",
            f"{data['power_dbfs']:.2f}",
            f"{data['relative_db']:+.2f}",
            f"{data['clipping_percent']:.2f}",
            data["status"],
        ]
        for col, value in enumerate(columns):
            cell = QTableWidgetItem(value)
            cell.setData(Qt.ItemDataRole.UserRole, int(data["physical_channel"]))
            self.scan_table.setItem(index, col, cell)

    def _scan_complete(self, result):
        self.scanning = False
        priority = {
            "RF CANDIDATE (NOT TV LOCK)": 0,
            "OVERLOAD / RETEST": 1,
            "NO STRONG RF CONTRAST": 2,
        }
        self.scan_rows = sorted(
            result.get("rows", []),
            key=lambda row: (
                priority.get(row["status"], 3),
                -row["power_dbfs"],
            ),
        )
        self.scan_table.setRowCount(0)
        for row in self.scan_rows:
            self._append_scan_row(row)
        self._scan_controls(False)
        self.scan_button.setText("Scan UHF 13–52 (RF)")
        self._update_scan_button()
        candidates = sum(
            row["status"] == "RF CANDIDATE (NOT TV LOCK)"
            for row in self.scan_rows
        )
        overloaded = sum(
            row["status"] == "OVERLOAD / RETEST" for row in self.scan_rows
        )
        if result.get("error"):
            self.scan_note.setText(f"Scan error: {result['error']}")
        else:
            state = "cancelled" if result.get("cancelled") else "complete"
            strongest = [
                f"{row['physical_channel']}ch ({row['relative_db']:+.1f} dB)"
                for row in self.scan_rows
                if row["status"] == "RF CANDIDATE (NOT TV LOCK)"
            ][:6]
            examples = ", ".join(strongest) if strongest else "none"
            self.scan_note.setText(
                f"RF scan {state}: {len(self.scan_rows)} measured, "
                f"{candidates} RF candidates, {overloaded} overloaded. "
                f"Strongest candidates: {examples}. "
                "Not yet identified TV services."
            )

    def _scan_selection_changed(self):
        self.scan_tune_button.setEnabled(
            not self.scanning and bool(self.scan_table.selectedItems())
        )

    def _tune_scan_selection(self):
        row = self.scan_table.currentRow()
        if row < 0 or self.scanning or self.worker is None:
            return
        cell = self.scan_table.item(row, 0)
        if cell is None:
            return
        ch = int(cell.data(Qt.ItemDataRole.UserRole))
        self.channel.setValue(ch)
        self.status.setText(
            f"Tuning RF {ch}ch (not yet an identified TV service)"
        )

    def _export_scan(self):
        if not self.scan_rows or self.scanning:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save RF scan measurements", str(Path.home() / "oneseg_rf_scan.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        if path.exists():
            if QMessageBox.question(
                self, "Overwrite scan?", f"Overwrite {path}?"
            ) != QMessageBox.StandardButton.Yes:
                return
        columns = [
            "physical_channel", "frequency_hz", "power_dbfs", "rms",
            "clipping_percent", "relative_db", "status",
        ]
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.scan_rows)
        except OSError as exc:
            QMessageBox.warning(self, "Save scan failed", str(exc))
            return
        self.status.setText(f"RF scan saved: {path.name}")

    def _play_ts(self):
        if self.player is not None:
            self.player.stop()
            self.play_ts_btn.setEnabled(False)
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open already decoded MPEG-TS", str(Path.home()),
            "MPEG Transport Stream (*.ts);;All files (*.*)",
        )
        if not filename:
            return
        self.player = TransportPlayer(Path(filename))
        self.player.image_ready.connect(self._show_frame)
        self.player.status.connect(self.status.setText)
        self.player.failed.connect(self._error)
        self.player.finished.connect(self._player_finished)
        self.play_ts_btn.setText("Stop TS player")
        self.player.start()

    def _show_frame(self, image):
        image_size = self.video.size()
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(pixmap.scaled(
            image_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _player_finished(self):
        self.player = None
        self.play_ts_btn.setEnabled(True)
        self.play_ts_btn.setText("Play decoded TS…")

    def _capture_short(self):
        if self.worker is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ch{self.channel.value()}" if self.mode.currentData() == "oneseg" else "sdr"
        suggested = str(Path.home() / f"oneseg_{name}_{stamp}.c64")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save 3 seconds of raw I/Q (not video)",
            suggested, "Complex64 I/Q (*.c64)"
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".c64":
            path = path.with_suffix(".c64")
        if path.exists():
            QMessageBox.warning(self, "Already exists", "Choose a new capture filename.")
            return
        self.worker.request("capture_short", str(path), 3.0)

    def _record(self):
        if self.worker is None:
            return
        if self.record_btn.property("active"):
            self.worker.request("record_stop")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"oneseg_{stamp}.c64")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save raw I/Q", default, "Complex64 I/Q (*.c64)"
        )
        if filename:
            path = Path(filename)
            if path.suffix.lower() != ".c64":
                path = path.with_suffix(".c64")
            if path.exists():
                result = QMessageBox.question(
                    self, "Overwrite?", f"Overwrite existing file?\n{path}"
                )
                if result != QMessageBox.StandardButton.Yes:
                    return
            self.worker.request("record", str(path))

    def _recording(self, recording: bool):
        self.record_btn.setProperty("active", recording)
        self.record_btn.setText("Stop recording" if recording else "Record I/Q…")
        self.short_capture_btn.setEnabled(not recording and self.worker is not None)
        self._update_scan_button()

    def _stop(self):
        if self.worker:
            self.stop_btn.setEnabled(False)
            self.worker.stop()
            self.status.setText("Stopping…")

    def _finished(self):
        self.worker = None
        self.scanning = False
        self._scan_controls(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.short_capture_btn.setEnabled(False)
        self._recording(False)
        self.status.setText("Stopped")
        self._update_scan_button()

    def _error(self, details):
        self.status.setText(details)
        QMessageBox.warning(
            self,
            "Receiver error",
            explain_receiver_error(details),
        )

    def closeEvent(self, event):
        if self.player and self.player.isRunning():
            self.player.stop()
            if not self.player.wait(3000):
                event.ignore()
                return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(3000):
                self.status.setText("Waiting for RTL-SDR to release…")
                event.ignore()
                return
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
