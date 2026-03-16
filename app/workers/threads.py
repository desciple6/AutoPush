"""Worker threads for background processing."""

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class PoseWorker(QThread):
    """Background worker for pose estimation on video frames."""

    result_ready = pyqtSignal(int, object)  # frame_index, landmarks

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None
        self._frame_index = 0
        self._running = False
        self._estimator = None

    def set_estimator(self, estimator):
        self._estimator = estimator

    def process_frame(self, frame: np.ndarray, frame_index: int):
        """Queue a frame for pose estimation."""
        self._frame = frame.copy()
        self._frame_index = frame_index
        if not self.isRunning():
            self._running = True
            self.start()

    def run(self):
        while self._running and self._frame is not None:
            frame = self._frame
            index = self._frame_index
            self._frame = None

            if self._estimator is not None:
                landmarks = self._estimator.process(frame)
                self.result_ready.emit(index, landmarks)

            if self._frame is None:
                self._running = False

    def stop(self):
        self._running = False
        self.wait()


class ExportWorker(QThread):
    """Background worker for video export."""

    progress = pyqtSignal(int)       # percentage 0-100
    finished = pyqtSignal(str)       # output file path
    error = pyqtSignal(str)          # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._export_func = None
        self._kwargs = {}

    def setup(self, export_func, **kwargs):
        self._export_func = export_func
        self._kwargs = kwargs

    def run(self):
        try:
            if self._export_func:
                self._export_func(progress_callback=self.progress.emit, **self._kwargs)
        except Exception as e:
            self.error.emit(str(e))
