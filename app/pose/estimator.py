"""Pose estimator - wraps MediaPipe PoseLandmarker (Tasks API) for skeleton detection."""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

from app.config import (
    POSE_CONFIDENCE_THRESHOLD,
    POSE_MODEL_URL,
    POSE_MODEL_DIR,
    POSE_MODEL_PATH,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class PoseEstimator:
    """Detects human pose landmarks using MediaPipe PoseLandmarker (Tasks API)."""

    # MediaPipe landmark indices for key body parts (same 33-point model)
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
        self._ensure_model_downloaded()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_CONFIDENCE_THRESHOLD,
            min_pose_presence_confidence=POSE_CONFIDENCE_THRESHOLD,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model_downloaded():
        """Download the pose landmarker model if not already present."""
        if os.path.exists(POSE_MODEL_PATH):
            return
        os.makedirs(POSE_MODEL_DIR, exist_ok=True)
        print(f"Downloading pose model to {POSE_MODEL_PATH}...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Pose model downloaded.")

    def process(self, frame: np.ndarray) -> dict | None:
        """Process a frame and return landmark positions.

        Returns dict mapping landmark name -> (x_pixel, y_pixel, visibility)
        or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        h, w = frame.shape[:2]
        pose = result.pose_landmarks[0]  # First detected pose
        landmarks = {}
        for name, idx in self.LANDMARKS.items():
            lm = pose[idx]
            landmarks[name] = (
                int(lm.x * w),
                int(lm.y * h),
                lm.visibility
            )

        return landmarks

    def close(self):
        self._landmarker.close()
