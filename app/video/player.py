"""Video player engine wrapping OpenCV VideoCapture."""

import cv2
import numpy as np


class VideoPlayer:
    """Handles video file loading, seeking, and frame retrieval."""

    def __init__(self):
        self._cap = None
        self._frame_index = 0
        self._total_frames = 0
        self._fps = 30.0
        self._width = 0
        self._height = 0
        self._file_path = ""

    @property
    def is_loaded(self):
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def fps(self):
        return self._fps

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def file_path(self):
        return self._file_path

    @property
    def duration_seconds(self):
        if self._fps > 0:
            return self._total_frames / self._fps
        return 0.0

    def load(self, path: str) -> bool:
        """Load a video file. Returns True on success."""
        self.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            return False

        self._file_path = path
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_index = 0
        return True

    def get_frame(self, index: int = None) -> np.ndarray | None:
        """Get frame at given index (or current). Returns BGR numpy array."""
        if not self.is_loaded:
            return None

        if index is not None:
            index = max(0, min(index, self._total_frames - 1))
            if index != self._frame_index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self._frame_index = index

        ret, frame = self._cap.read()
        if ret:
            self._frame_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            return frame
        return None

    def next_frame(self) -> np.ndarray | None:
        """Advance to next frame."""
        if self._frame_index < self._total_frames - 1:
            return self.get_frame(self._frame_index + 1)
        return self.get_frame(self._frame_index)

    def prev_frame(self) -> np.ndarray | None:
        """Go to previous frame."""
        if self._frame_index > 1:
            return self.get_frame(self._frame_index - 2)
        return self.get_frame(0)

    def seek(self, index: int) -> np.ndarray | None:
        """Seek to specific frame index."""
        return self.get_frame(index)

    def release(self):
        """Release video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_index = 0
        self._total_frames = 0
        self._file_path = ""
