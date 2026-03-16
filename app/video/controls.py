"""Playback control widgets - buttons, timeline slider, speed selector."""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider,
    QLabel, QComboBox, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.config import PLAYBACK_SPEEDS


class PlaybackControls(QWidget):
    """Video playback control bar with transport buttons and timeline."""

    play_pause_clicked = pyqtSignal()
    step_forward_clicked = pyqtSignal()
    step_backward_clicked = pyqtSignal()
    fast_forward_clicked = pyqtSignal()
    reverse_clicked = pyqtSignal()
    slider_moved = pyqtSignal(int)
    speed_changed = pyqtSignal(float)
    record_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_playing = False
        self._is_recording = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Timeline slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self.slider_moved.emit)
        layout.addWidget(self._slider)

        # Controls row
        controls_row = QHBoxLayout()

        # Reverse button
        self._btn_reverse = QPushButton("Rev")
        self._btn_reverse.setFixedWidth(50)
        self._btn_reverse.setToolTip("Play in reverse")
        self._btn_reverse.clicked.connect(self.reverse_clicked.emit)
        controls_row.addWidget(self._btn_reverse)

        # Step backward
        self._btn_step_back = QPushButton("< |")
        self._btn_step_back.setFixedWidth(40)
        self._btn_step_back.setToolTip("Previous frame")
        self._btn_step_back.clicked.connect(self.step_backward_clicked.emit)
        controls_row.addWidget(self._btn_step_back)

        # Play/Pause
        self._btn_play = QPushButton("Play")
        self._btn_play.setFixedWidth(60)
        self._btn_play.setToolTip("Play / Pause (Space)")
        self._btn_play.clicked.connect(self._on_play_pause)
        controls_row.addWidget(self._btn_play)

        # Step forward
        self._btn_step_fwd = QPushButton("| >")
        self._btn_step_fwd.setFixedWidth(40)
        self._btn_step_fwd.setToolTip("Next frame")
        self._btn_step_fwd.clicked.connect(self.step_forward_clicked.emit)
        controls_row.addWidget(self._btn_step_fwd)

        # Fast forward
        self._btn_ff = QPushButton("FF")
        self._btn_ff.setFixedWidth(50)
        self._btn_ff.setToolTip("Fast forward")
        self._btn_ff.clicked.connect(self.fast_forward_clicked.emit)
        controls_row.addWidget(self._btn_ff)

        controls_row.addStretch()

        # Frame counter
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(100)
        controls_row.addWidget(self._frame_label)

        # Time display
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setMinimumWidth(120)
        controls_row.addWidget(self._time_label)

        controls_row.addStretch()

        # Speed selector
        speed_label = QLabel("Speed:")
        controls_row.addWidget(speed_label)
        self._speed_combo = QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self._speed_combo.addItem(f"{speed}x", speed)
        self._speed_combo.setCurrentIndex(2)  # 1.0x default
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        controls_row.addWidget(self._speed_combo)

        # Record button
        self._btn_record = QPushButton("Rec")
        self._btn_record.setFixedWidth(50)
        self._btn_record.setCheckable(True)
        self._btn_record.setToolTip("Record voice commentary (R)")
        self._btn_record.setStyleSheet("")
        self._btn_record.toggled.connect(self._on_record_toggled)
        controls_row.addWidget(self._btn_record)

        layout.addLayout(controls_row)

    def _on_play_pause(self):
        self._is_playing = not self._is_playing
        self._btn_play.setText("Pause" if self._is_playing else "Play")
        self.play_pause_clicked.emit()

    def _on_speed_changed(self, index):
        speed = self._speed_combo.currentData()
        if speed is not None:
            self.speed_changed.emit(speed)

    def _on_record_toggled(self, checked):
        self._is_recording = checked
        if checked:
            self._btn_record.setStyleSheet("background-color: #cc3333; color: white;")
            self._btn_record.setText("Stop")
        else:
            self._btn_record.setStyleSheet("")
            self._btn_record.setText("Rec")
        self.record_toggled.emit(checked)

    def set_playing(self, playing: bool):
        """Update play/pause button state externally."""
        self._is_playing = playing
        self._btn_play.setText("Pause" if playing else "Play")

    def set_range(self, total_frames: int):
        """Set the slider range."""
        self._slider.setMaximum(max(0, total_frames - 1))

    def set_position(self, frame_index: int, fps: float = 30.0, total_frames: int = 0):
        """Update slider position and frame/time labels."""
        self._slider.blockSignals(True)
        self._slider.setValue(frame_index)
        self._slider.blockSignals(False)

        total = total_frames or self._slider.maximum() + 1
        self._frame_label.setText(f"{frame_index} / {total}")

        if fps > 0:
            current_sec = frame_index / fps
            total_sec = total / fps
            self._time_label.setText(
                f"{self._format_time(current_sec)} / {self._format_time(total_sec)}"
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def get_speed(self) -> float:
        return self._speed_combo.currentData() or 1.0
