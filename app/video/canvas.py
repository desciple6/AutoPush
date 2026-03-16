"""Video canvas widget for displaying frames and handling mouse events."""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QImage, QPainter, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize


class VideoCanvas(QWidget):
    """Displays video frames and captures mouse events for annotations."""

    mouse_pressed = pyqtSignal(QPoint)
    mouse_moved = pyqtSignal(QPoint)
    mouse_released = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_frame = None
        self._display_image = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._video_width = 0
        self._video_height = 0

        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1a1a1a;")

    def set_frame(self, frame: np.ndarray):
        """Set and display a new frame (BGR numpy array)."""
        self._current_frame = frame
        if frame is not None:
            self._video_width = frame.shape[1]
            self._video_height = frame.shape[0]
        self._update_display()

    def display_composited(self, frame: np.ndarray):
        """Display a pre-composited frame (with annotations/pose already drawn)."""
        if frame is None:
            return
        self._video_width = frame.shape[1]
        self._video_height = frame.shape[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._display_image = qimg.copy()
        self._compute_scaling()
        self.update()

    def _update_display(self):
        """Convert current frame to QImage for painting."""
        if self._current_frame is None:
            self._display_image = None
            self.update()
            return

        frame = self._current_frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._display_image = qimg.copy()
        self._compute_scaling()
        self.update()

    def _compute_scaling(self):
        """Compute scale and offset to fit video in widget with aspect ratio."""
        if self._video_width == 0 or self._video_height == 0:
            return

        widget_w = self.width()
        widget_h = self.height()
        scale_x = widget_w / self._video_width
        scale_y = widget_h / self._video_height
        self._scale = min(scale_x, scale_y)

        scaled_w = int(self._video_width * self._scale)
        scaled_h = int(self._video_height * self._scale)
        self._offset_x = (widget_w - scaled_w) // 2
        self._offset_y = (widget_h - scaled_h) // 2

    def widget_to_video_coords(self, pos: QPoint) -> QPoint | None:
        """Convert widget coordinates to video frame coordinates."""
        if self._scale == 0:
            return None

        vx = (pos.x() - self._offset_x) / self._scale
        vy = (pos.y() - self._offset_y) / self._scale

        if 0 <= vx < self._video_width and 0 <= vy < self._video_height:
            return QPoint(int(vx), int(vy))
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._display_image is not None:
            scaled_w = int(self._video_width * self._scale)
            scaled_h = int(self._video_height * self._scale)
            scaled = self._display_image.scaled(
                scaled_w, scaled_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.drawImage(self._offset_x, self._offset_y, scaled)
        else:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Open a video file to begin")

        painter.end()

    def resizeEvent(self, event):
        self._compute_scaling()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        video_pos = self.widget_to_video_coords(event.pos())
        if video_pos is not None:
            self.mouse_pressed.emit(video_pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        video_pos = self.widget_to_video_coords(event.pos())
        if video_pos is not None:
            self.mouse_moved.emit(video_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        video_pos = self.widget_to_video_coords(event.pos())
        if video_pos is not None:
            self.mouse_released.emit(video_pos)

    def sizeHint(self):
        return QSize(960, 540)
