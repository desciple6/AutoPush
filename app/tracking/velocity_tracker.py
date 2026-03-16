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

            # Compute velocity (pixel displacement)
            if self._prev_frame_idx in self._positions:
                prev_pos = self._positions[self._prev_frame_idx]
                dx = new_pos[0] - prev_pos[0]
                dy = new_pos[1] - prev_pos[1]
                displacement = np.sqrt(dx ** 2 + dy ** 2)
                # Use signed vertical velocity (positive = up in image coords means down)
                vertical_velocity = -dy  # Negate so upward movement is positive
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

        # Look at recent velocities for sign change
        recent = sorted_frames[-10:]  # Last 10 frames
        if len(recent) >= 3:
            v_curr = self._velocities.get(recent[-1], 0)
            v_prev = self._velocities.get(recent[-2], 0)

            # Detect zero-crossing (sign change) with some smoothing
            if v_prev > 2 and v_curr < -2:  # Was going up, now going down
                if not self._rep_boundaries or frame_index - self._rep_boundaries[-1] > 10:
                    self._rep_boundaries.append(frame_index)
            elif v_prev < -2 and v_curr > 2:  # Was going down, now going up
                if not self._rep_boundaries or frame_index - self._rep_boundaries[-1] > 10:
                    self._rep_boundaries.append(frame_index)

    def render_marker(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw the tracking marker on the frame."""
        pos = self._positions.get(frame_index)
        if pos is None:
            return frame

        center = (int(pos[0]), int(pos[1]))
        # Crosshair marker
        cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, center, 10, (0, 255, 255), 1)

        # Show velocity value
        vel = self._velocities.get(frame_index, 0)
        vel_text = f"v: {vel:.1f} px/f"
        cv2.putText(frame, vel_text, (center[0] + 15, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
            "start_frame": self._start_frame
        }
