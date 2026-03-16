"""Application configuration and constants."""

import os

APP_NAME = "LiftReview"
APP_VERSION = "1.0.0"

# Supported video formats
VIDEO_EXTENSIONS = "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.flv *.webm);;All Files (*)"

# Playback speeds available
PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Default annotation colors
DEFAULT_ANNOTATION_COLOR = (0, 255, 0)  # Green BGR
ANNOTATION_THICKNESS = 2
ANNOTATION_FONT_SCALE = 0.7

# Pose estimation - MediaPipe Tasks API model
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
POSE_MODEL_PATH = os.path.join(POSE_MODEL_DIR, 'pose_landmarker_lite.task')
POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_SKELETON_COLOR_GOOD = (0, 255, 0)     # Green
POSE_SKELETON_COLOR_WARNING = (0, 255, 255)  # Yellow
POSE_SKELETON_COLOR_BAD = (0, 0, 255)       # Red

# Velocity tracker
OPTICAL_FLOW_WIN_SIZE = (21, 21)
OPTICAL_FLOW_MAX_LEVEL = 3

# Audio
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

# UI
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800
