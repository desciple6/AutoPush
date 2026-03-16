"""Velocity graph widget - displays velocity over time using pyqtgraph."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class VelocityGraph(QWidget):
    """Embedded velocity chart showing barbell velocity over time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Barbell Velocity")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ddd; font-weight: bold;")
        layout.addWidget(title)

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground('#1a1a1a')
        self._plot_widget.setLabel('left', 'Velocity', units='px/frame')
        self._plot_widget.setLabel('bottom', 'Frame')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Velocity curve
        self._velocity_curve = self._plot_widget.plot(
            pen=pg.mkPen(color='#55aaff', width=2)
        )

        # Current frame marker
        self._frame_marker = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color='#ffffff', width=1, style=Qt.PenStyle.DashLine)
        )
        self._plot_widget.addItem(self._frame_marker)

        # Zero line
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color='#666666', width=1)
        )
        self._plot_widget.addItem(self._zero_line)

        # Rep boundary markers
        self._rep_lines = []

        layout.addWidget(self._plot_widget)

        # Stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

    def update_data(self, velocities: dict, rep_boundaries: list, current_frame: int):
        """Update the graph with new velocity data."""
        if not velocities:
            return

        # Sort by frame
        sorted_frames = sorted(velocities.keys())
        frames = np.array(sorted_frames, dtype=np.float64)
        vels = np.array([velocities[f] for f in sorted_frames], dtype=np.float64)

        # Smooth the velocity curve (moving average)
        if len(vels) > 5:
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            vels_smooth = np.convolve(vels, kernel, mode='same')
        else:
            vels_smooth = vels

        self._velocity_curve.setData(frames, vels_smooth)
        self._frame_marker.setValue(current_frame)

        # Update rep boundary lines
        for line in self._rep_lines:
            self._plot_widget.removeItem(line)
        self._rep_lines.clear()

        for boundary in rep_boundaries:
            line = pg.InfiniteLine(
                pos=boundary, angle=90,
                pen=pg.mkPen(color='#ff5555', width=1, style=Qt.PenStyle.DotLine)
            )
            self._plot_widget.addItem(line)
            self._rep_lines.append(line)

        # Update stats
        self._update_stats(vels, rep_boundaries)

    def _update_stats(self, velocities: np.ndarray, rep_boundaries: list):
        """Calculate and display velocity statistics."""
        if len(velocities) == 0:
            return

        peak_vel = np.max(np.abs(velocities))
        avg_vel = np.mean(np.abs(velocities))
        num_reps = max(0, len(rep_boundaries) // 2)

        stats_text = f"Peak: {peak_vel:.1f} | Avg: {avg_vel:.1f} | Reps detected: {num_reps}"

        # Rep-by-rep velocity comparison
        if len(rep_boundaries) >= 2:
            rep_peaks = []
            for i in range(0, len(rep_boundaries) - 1, 2):
                start = rep_boundaries[i]
                end = rep_boundaries[i + 1] if i + 1 < len(rep_boundaries) else len(velocities)
                # Find peak velocity in this rep range
                mask = (np.arange(len(velocities)) >= start) & (np.arange(len(velocities)) < end)
                if np.any(mask):
                    rep_peak = np.max(np.abs(velocities[mask]))
                    rep_peaks.append(rep_peak)

            if len(rep_peaks) >= 2:
                drop = ((rep_peaks[0] - rep_peaks[-1]) / rep_peaks[0]) * 100
                stats_text += f"\nVelocity drop: {drop:.1f}%"

        self._stats_label.setText(stats_text)

    def clear(self):
        """Clear the graph."""
        self._velocity_curve.setData([], [])
        for line in self._rep_lines:
            self._plot_widget.removeItem(line)
        self._rep_lines.clear()
        self._stats_label.setText("")
