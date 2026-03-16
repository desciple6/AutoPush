"""Pose estimator - wraps MediaPipe for skeleton detection."""

import cv2
import numpy as np
import mediapipe as mp

from app.config import POSE_CONFIDENCE_THRESHOLD


class PoseEstimator:
    """Detects human pose landmarks using MediaPipe Pose."""

    # MediaPipe landmark indices for key body parts
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32,
    }

    # Skeleton connections for drawing
    CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('left_ankle', 'left_heel'),
        ('right_ankle', 'right_heel'),
        ('left_heel', 'left_foot_index'),
        ('right_heel', 'right_foot_index'),
    ]

    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=POSE_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=POSE_CONFIDENCE_THRESHOLD
        )

    def process(self, frame: np.ndarray) -> dict | None:
        """Process a frame and return landmark positions.

        Returns dict mapping landmark name -> (x_pixel, y_pixel, visibility)
        or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = {}
        for name, idx in self.LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = (
                int(lm.x * w),
                int(lm.y * h),
                lm.visibility
            )

        return landmarks

    def close(self):
        self._pose.close()
