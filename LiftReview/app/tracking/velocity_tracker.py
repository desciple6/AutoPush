"""Velocity tracker - tracks a point across frames using optical flow."""

import cv2
import numpy as np
from app.config import OPTICAL_FLOW_WIN_SIZE, OPTICAL_FLOW_MAX_LEVEL


class VelocityTracker:
    """Tracks a user-selected point across video frames and computes velocity."""

    def __init__(self):
        self._tracking = False
        self._positions = {}      # frame_index -> (x, y)
        self._velocities = {}     # frame_index -> velocity (pixels/frame)
        self._prev_gray = None
        self._prev_point = None
        self._prev_frame_idx = -1
        self._start_frame = 0
        self._rep_boundaries = []  # Frame indices where reps start/end
        self._fps = 30.0           # Set when tracking starts

        # Optical flow parameters
        self._lk_params = dict(
            winSize=OPTICAL_FLOW_WIN_SIZE,
            maxLevel=OPTICAL_FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    @property
    def is_tracking(self):
        return self._tracking

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def rep_boundaries(self):
        return self._rep_boundaries

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value if value > 0 else 30.0

    def start_tracking(self, frame: np.ndarray, point: tuple, frame_index: int):
        """Begin tracking from the given point on the given frame."""
        self.reset()
        self._tracking = True
        self._start_frame = frame_index
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_point = np.array([[point]], dtype=np.float32)
        self._prev_frame_idx = frame_index
        self._positions[frame_index] = point

    def update(self, frame: np.ndarray, frame_index: int):
        """Track the point to the new frame and compute velocity."""
        if not self._tracking or self._prev_gray is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        next_point, status, error = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_point, None, **self._lk_params
        )

        if next_point is not None and status is not None and status[0][0] == 1:
            new_pos = (float(next_point[0][0][0]), float(next_point[0][0][1]))
            self._positions[frame_index] = new_pos

            # Compute velocity (pixel displacement per frame)
            if self._prev_frame_idx in self._positions:
                prev_pos = self._positions[self._prev_frame_idx]
                dy = new_pos[1] - prev_pos[1]
                # Negate so upward movement is positive
                vertical_velocity = -dy
                self._velocities[frame_index] = vertical_velocity

                # Detect rep boundaries (velocity sign changes)
                self._detect_rep_boundary(frame_index)

            self._prev_point = next_point
        else:
            # Tracking lost - keep last known position
            if self._prev_frame_idx in self._positions:
                self._positions[frame_index] = self._positions[self._prev_frame_idx]
                self._velocities[frame_index] = 0.0

        self._prev_gray = gray
        self._prev_frame_idx = frame_index

    def _detect_rep_boundary(self, frame_index: int):
        """Detect rep boundaries based on velocity sign changes."""
        sorted_frames = sorted(self._velocities.keys())
        if len(sorted_frames) < 3:
            return

        recent = sorted_frames[-10:]
        if len(recent) >= 3:
            v_curr = self._velocities.get(recent[-1], 0)
            v_prev = self._velocities.get(recent[-2], 0)

            if v_prev > 2 and v_curr < -2:
                if not self._rep_boundaries or frame_index - self._rep_boundaries[-1] > 10:
                    self._rep_boundaries.append(frame_index)
            elif v_prev < -2 and v_curr > 2:
                if not self._rep_boundaries or frame_index - self._rep_boundaries[-1] > 10:
                    self._rep_boundaries.append(frame_index)

    def get_rep_stats(self) -> list[dict]:
        """Compute per-rep velocity statistics.

        Returns a list of dicts, one per detected rep, with:
          - rep_num: 1-indexed rep number
          - start_frame, end_frame
          - peak_velocity: max absolute velocity (px/frame)
          - avg_velocity: mean absolute velocity (px/frame)
          - peak_velocity_per_sec: peak * fps (px/sec)
          - avg_velocity_per_sec: avg * fps (px/sec)
          - duration_frames: number of frames in this rep
          - duration_sec: duration in seconds
        """
        if len(self._rep_boundaries) < 2:
            return []

        sorted_frames = sorted(self._velocities.keys())
        if not sorted_frames:
            return []

        all_vels = np.array([self._velocities[f] for f in sorted_frames])

        stats = []
        # Reps are between pairs of boundaries (up-down-up = one rep)
        for i in range(0, len(self._rep_boundaries) - 1, 2):
            start = self._rep_boundaries[i]
            end = self._rep_boundaries[i + 1] if i + 1 < len(self._rep_boundaries) else sorted_frames[-1]

            # Get velocities in this rep range
            rep_frames = [f for f in sorted_frames if start <= f <= end]
            if not rep_frames:
                continue
            rep_vels = np.array([self._velocities[f] for f in rep_frames])
            abs_vels = np.abs(rep_vels)

            peak = float(np.max(abs_vels))
            avg = float(np.mean(abs_vels))
            duration_frames = len(rep_frames)
            duration_sec = duration_frames / self._fps

            stats.append({
                'rep_num': len(stats) + 1,
                'start_frame': start,
                'end_frame': end,
                'peak_velocity': peak,
                'avg_velocity': avg,
                'peak_velocity_per_sec': peak * self._fps,
                'avg_velocity_per_sec': avg * self._fps,
                'duration_frames': duration_frames,
                'duration_sec': duration_sec,
            })

        return stats

    @staticmethod
    def _map_velocity_loss_to_rpe(loss_pct: float) -> tuple[float, str]:
        """Map velocity loss percentage to (RPE, RIR band)."""
        if loss_pct < 5:
            return 6.5, "4+"
        if loss_pct < 10:
            return 7.5, "2-3"
        if loss_pct < 20:
            return 8.5, "1-2"
        if loss_pct < 30:
            return 9.0, "0.5-1"
        if loss_pct < 40:
            return 9.5, "0-0.5"
        return 10.0, "0"

    @classmethod
    def estimate_rpe_from_velocity_loss(cls, rep_stats: list[dict]) -> dict:
        """Estimate RPE/RIR per rep and overall from velocity loss.

        For each rep i, compares its avg velocity to rep 1 and
        assigns an RPE/RIR. Also returns an overall RPE based on
        total loss from first to last rep and the average RPE
        across all reps.
        """
        if not rep_stats:
            return {
                'velocity_loss_pct': 0.0,
                'overall_estimated_rpe': None,
                'overall_estimated_rir': None,
                'first_rep_avg': 0.0,
                'last_rep_avg': 0.0,
                'per_rep': [],
                'average_rpe': None,
            }

        first_avg = rep_stats[0]['avg_velocity_per_sec']
        per_rep = []

        for rs in rep_stats:
            rep_avg = rs['avg_velocity_per_sec']
            if first_avg == 0:
                loss_pct = 0.0
            else:
                loss_pct = ((first_avg - rep_avg) / first_avg) * 100
            rpe, rir = cls._map_velocity_loss_to_rpe(loss_pct)
            per_rep.append({
                'rep_num': rs['rep_num'],
                'avg_velocity_per_sec': rep_avg,
                'velocity_loss_pct': loss_pct,
                'estimated_rpe': rpe,
                'estimated_rir': rir,
            })

        # Overall loss from first to last rep
        last_avg = rep_stats[-1]['avg_velocity_per_sec']
        if first_avg == 0:
            total_loss = 0.0
        else:
            total_loss = ((first_avg - last_avg) / first_avg) * 100

        overall_rpe, overall_rir = cls._map_velocity_loss_to_rpe(total_loss)

        # Average RPE across all reps
        rpes = [p['estimated_rpe'] for p in per_rep if p['estimated_rpe'] is not None]
        avg_rpe = float(np.mean(rpes)) if rpes else None

        return {
            'velocity_loss_pct': total_loss,
            'overall_estimated_rpe': overall_rpe,
            'overall_estimated_rir': overall_rir,
            'first_rep_avg': first_avg,
            'last_rep_avg': last_avg,
            'per_rep': per_rep,
            'average_rpe': avg_rpe,
        }

    def render_marker(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw the tracking marker on the frame."""
        pos = self._positions.get(frame_index)
        if pos is None:
            return frame

        center = (int(pos[0]), int(pos[1]))
        # Crosshair marker
        cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, center, 10, (0, 255, 255), 1)

        # Show velocity value with px/s conversion
        vel = self._velocities.get(frame_index, 0)
        vel_ps = vel * self._fps
        vel_text = f"v: {vel_ps:.0f} px/s"
        cv2.putText(frame, vel_text, (center[0] + 15, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def render_path(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw the bar path (trajectory of tracked point) up to the current frame."""
        if not self._positions:
            return frame

        # Collect positions in frame order up to current frame
        sorted_frames = sorted(f for f in self._positions.keys() if f <= frame_index)
        if len(sorted_frames) < 2:
            return frame

        pts = np.array(
            [[self._positions[f][0], self._positions[f][1]] for f in sorted_frames],
            dtype=np.int32
        ).reshape(-1, 1, 2)

        # Cyan path line
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        return frame

    def stop_tracking(self):
        """Stop tracking without clearing data."""
        self._tracking = False

    def reset(self):
        """Clear all tracking data."""
        self._tracking = False
        self._positions.clear()
        self._velocities.clear()
        self._prev_gray = None
        self._prev_point = None
        self._prev_frame_idx = -1
        self._rep_boundaries.clear()

    def get_velocity_arrays(self):
        """Get sorted frame indices and velocity values for graphing."""
        if not self._velocities:
            return np.array([]), np.array([])
        sorted_frames = sorted(self._velocities.keys())
        frames = np.array(sorted_frames)
        vels = np.array([self._velocities[f] for f in sorted_frames])
        return frames, vels

    def to_dict(self) -> dict:
        return {
            "positions": {str(k): list(v) for k, v in self._positions.items()},
            "velocities": {str(k): v for k, v in self._velocities.items()},
            "rep_boundaries": self._rep_boundaries,
            "start_frame": self._start_frame,
            "fps": self._fps
        }
