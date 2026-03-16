"""Worker threads for background processing."""

import traceback
import cv2
import numpy as np
from PyQt6.QtCore import QThread, QMutex, pyqtSignal


class PoseWorker(QThread):
    """Background worker for pose estimation on video frames.

    Processes one frame at a time. If a new frame arrives while processing,
    the old request is dropped and the new one is processed instead.
    """

    result_ready = pyqtSignal(int, object)  # frame_index, landmarks

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._frame = None
        self._frame_index = 0
        self._running = False
        self._estimator = None

    def set_estimator(self, estimator):
        self._estimator = estimator

    def process_frame(self, frame: np.ndarray, frame_index: int):
        """Queue a frame for pose estimation. Drops any pending request."""
        self._mutex.lock()
        self._frame = frame.copy()
        self._frame_index = frame_index
        self._mutex.unlock()

        if not self.isRunning():
            self._running = True
            self.start()

    def run(self):
        while self._running:
            self._mutex.lock()
            frame = self._frame
            index = self._frame_index
            self._frame = None  # Consume the request
            self._mutex.unlock()

            if frame is None:
                # No pending work, exit the loop
                break

            if self._estimator is not None:
                try:
                    landmarks = self._estimator.process(frame)
                    self.result_ready.emit(index, landmarks)
                except Exception:
                    pass  # Don't crash the worker on pose errors

            # Check if a new frame arrived while we were processing
            self._mutex.lock()
            has_more = self._frame is not None
            self._mutex.unlock()
            if not has_more:
                break

        self._running = False

    def stop(self):
        self._running = False
        self._mutex.lock()
        self._frame = None
        self._mutex.unlock()
        self.wait(2000)  # Wait up to 2 seconds


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
