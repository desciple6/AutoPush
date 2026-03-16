"""Velocity graph widget - displays velocity over time with RPE/RIR estimation."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class VelocityGraph(QWidget):
    """Embedded velocity chart showing barbell velocity, per-rep stats, and RPE/RIR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracker = None  # Reference to VelocityTracker for rep stats
        self._setup_ui()

    def set_tracker(self, tracker):
        """Set reference to VelocityTracker for per-rep analysis."""
        self._tracker = tracker

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
        self._plot_widget.setLabel('left', 'Velocity', units='px/s')
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

        # Stats label - per-rep breakdown
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        # RPE/RIR estimation label
        self._rpe_label = QLabel("")
        self._rpe_label.setStyleSheet("color: #ffcc55; font-size: 12px; font-weight: bold;")
        self._rpe_label.setWordWrap(True)
        layout.addWidget(self._rpe_label)

    def update_data(self, velocities: dict, rep_boundaries: list,
                    current_frame: int):
        """Update the graph with new velocity data."""
        if not velocities:
            return

        fps = self._tracker.fps if self._tracker else 30.0

        # Sort by frame
        sorted_frames = sorted(velocities.keys())
        frames = np.array(sorted_frames, dtype=np.float64)
        vels = np.array([velocities[f] for f in sorted_frames], dtype=np.float64)

        # Convert to px/s for display
        vels_ps = vels * fps

        # Smooth the velocity curve (moving average)
        if len(vels_ps) > 5:
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            vels_smooth = np.convolve(vels_ps, kernel, mode='same')
        else:
            vels_smooth = vels_ps

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

        # Update detailed stats
        self._update_stats()

    def _update_stats(self):
        """Calculate and display per-rep velocity stats and RPE/RIR."""
        if not self._tracker:
            return

        rep_stats = self._tracker.get_rep_stats()

        if not rep_stats:
            # No reps detected yet — show overall stats
            frames, vels = self._tracker.get_velocity_arrays()
            if len(vels) == 0:
                return
            fps = self._tracker.fps
            abs_vels = np.abs(vels)
            peak_ps = float(np.max(abs_vels) * fps)
            avg_ps = float(np.mean(abs_vels) * fps)
            self._stats_label.setText(
                f"Peak: {peak_ps:.0f} px/s | Avg: {avg_ps:.0f} px/s | "
                f"Tracking... (no reps detected yet)"
            )
            self._rpe_label.setText("")
            return

        # Per-rep breakdown with per-rep RPE
        rpe_data = self._tracker.estimate_rpe_from_velocity_loss(rep_stats)
        per_rep = rpe_data.get('per_rep', [])

        lines = []
        for rs in rep_stats:
            per = next((p for p in per_rep if p['rep_num'] == rs['rep_num']), None)
            rpe_str = f" | RPE ~{per['estimated_rpe']:.1f}" if per else ""
            lines.append(
                f"Rep {rs['rep_num']}: "
                f"Peak {rs['peak_velocity_per_sec']:.0f} px/s | "
                f"Avg {rs['avg_velocity_per_sec']:.0f} px/s | "
                f"{rs['duration_sec']:.2f}s{rpe_str}"
            )

        self._stats_label.setText("\n".join(lines))

        # Overall RPE/RIR estimation and average RPE
        loss = rpe_data['velocity_loss_pct']
        overall_rpe = rpe_data.get('overall_estimated_rpe')
        overall_rir = rpe_data.get('overall_estimated_rir')
        avg_rpe = rpe_data.get('average_rpe')

        if overall_rpe is not None:
            rpe_text = (
                f"Velocity Loss (set): {loss:.1f}% "
                f"(Rep 1 avg: {rpe_data['first_rep_avg']:.0f} → "
                f"Rep {len(rep_stats)} avg: {rpe_data['last_rep_avg']:.0f} px/s)\n"
                f"Overall RPE: ~{overall_rpe:.1f} | Overall RIR: ~{overall_rir}"
            )
            if avg_rpe is not None:
                rpe_text += f"\nAverage per-rep RPE: ~{avg_rpe:.1f}"

            # Color code by overall RPE
            if overall_rpe >= 9.5:
                color = "#ff4444"  # Red - maximal
            elif overall_rpe >= 8.5:
                color = "#ff8844"  # Orange - very hard
            elif overall_rpe >= 7.5:
                color = "#ffcc44"  # Yellow - hard
            else:
                color = "#44cc44"  # Green - moderate

            self._rpe_label.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: bold;"
            )
            self._rpe_label.setText(rpe_text)
        else:
            # Single-rep case: still show its RPE
            if per_rep:
                only = per_rep[0]
                self._rpe_label.setStyleSheet(
                    "color: #44cc44; font-size: 12px; font-weight: bold;"
                )
                self._rpe_label.setText(
                    f"Single rep avg: {only['avg_velocity_per_sec']:.0f} px/s | "
                    f"Estimated RPE: ~{only['estimated_rpe']:.1f} "
                    f"(based on velocity vs first rep)"
                )
            else:
                self._rpe_label.setText("")

    def clear(self):
        """Clear the graph."""
        self._velocity_curve.setData([], [])
        for line in self._rep_lines:
            self._plot_widget.removeItem(line)
        self._rep_lines.clear()
        self._stats_label.setText("")
        self._rpe_label.setText("")
