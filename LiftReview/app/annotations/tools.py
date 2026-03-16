"""Annotation drawing tools - each handles mouse events and produces annotations."""

import math
from PyQt6.QtCore import QPoint
from app.annotations.manager import Annotation


class BaseTool:
    """Base class for annotation tools."""

    tool_type = "base"

    def __init__(self, color=(0, 255, 0), thickness=2):
        self.color = color
        self.thickness = thickness
        self._points = []
        self._active = False

    def on_press(self, pos: QPoint, frame_index: int):
        pass

    def on_move(self, pos: QPoint, frame_index: int):
        pass

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        return None

    def get_preview(self) -> Annotation | None:
        """Return a temporary annotation for live preview while drawing."""
        return None

    def reset(self):
        self._points = []
        self._active = False


class LineTool(BaseTool):
    """Draw a straight line between two points."""

    tool_type = "line"

    def on_press(self, pos: QPoint, frame_index: int):
        self._points = [(pos.x(), pos.y())]
        self._active = True

    def on_move(self, pos: QPoint, frame_index: int):
        if self._active and len(self._points) >= 1:
            if len(self._points) > 1:
                self._points[1] = (pos.x(), pos.y())
            else:
                self._points.append((pos.x(), pos.y()))

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if not self._active:
            return None
        self._points = [self._points[0], (pos.x(), pos.y())]
        self._active = False
        ann = Annotation(
            tool_type="line",
            points=list(self._points),
            color=self.color,
            thickness=self.thickness,
            start_frame=frame_index
        )
        self._points = []
        return ann

    def get_preview(self):
        if self._active and len(self._points) == 2:
            return Annotation(tool_type="line", points=list(self._points),
                              color=self.color, thickness=self.thickness)
        return None


class ArrowTool(BaseTool):
    """Draw an arrow from start to end point."""

    tool_type = "arrow"

    def on_press(self, pos: QPoint, frame_index: int):
        self._points = [(pos.x(), pos.y())]
        self._active = True

    def on_move(self, pos: QPoint, frame_index: int):
        if self._active:
            if len(self._points) > 1:
                self._points[1] = (pos.x(), pos.y())
            else:
                self._points.append((pos.x(), pos.y()))

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if not self._active:
            return None
        self._points = [self._points[0], (pos.x(), pos.y())]
        self._active = False
        ann = Annotation(
            tool_type="arrow",
            points=list(self._points),
            color=self.color,
            thickness=self.thickness,
            start_frame=frame_index
        )
        self._points = []
        return ann

    def get_preview(self):
        if self._active and len(self._points) == 2:
            return Annotation(tool_type="arrow", points=list(self._points),
                              color=self.color, thickness=self.thickness)
        return None


class CircleTool(BaseTool):
    """Draw a circle from center point with drag radius."""

    tool_type = "circle"

    def on_press(self, pos: QPoint, frame_index: int):
        self._points = [(pos.x(), pos.y())]
        self._active = True
        self._radius_point = None

    def on_move(self, pos: QPoint, frame_index: int):
        if self._active:
            self._radius_point = (pos.x(), pos.y())

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if not self._active or not self._points:
            return None
        center = self._points[0]
        end = (pos.x(), pos.y())
        self._active = False
        ann = Annotation(
            tool_type="circle",
            points=[center, end],
            color=self.color,
            thickness=self.thickness,
            start_frame=frame_index
        )
        self._points = []
        return ann

    def get_preview(self):
        if self._active and self._points and self._radius_point:
            return Annotation(tool_type="circle",
                              points=[self._points[0], self._radius_point],
                              color=self.color, thickness=self.thickness)
        return None


class AngleTool(BaseTool):
    """Three-click angle measurement tool."""

    tool_type = "angle"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_count = 0

    def on_press(self, pos: QPoint, frame_index: int):
        self._points.append((pos.x(), pos.y()))
        self._click_count += 1
        self._active = True

    def on_move(self, pos: QPoint, frame_index: int):
        pass

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if self._click_count >= 3:
            # Calculate angle
            p1, p2, p3 = self._points[:3]
            angle = self._calc_angle(p1, p2, p3)
            ann = Annotation(
                tool_type="angle",
                points=list(self._points[:3]),
                color=self.color,
                thickness=self.thickness,
                angle_degrees=angle,
                start_frame=frame_index
            )
            self.reset()
            self._click_count = 0
            return ann
        return None

    def get_preview(self):
        if self._active and len(self._points) >= 2:
            return Annotation(tool_type="angle", points=list(self._points),
                              color=self.color, thickness=self.thickness)
        return None

    @staticmethod
    def _calc_angle(p1, p2, p3):
        """Calculate angle at p2 formed by lines p1-p2 and p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def reset(self):
        super().reset()
        self._click_count = 0


class FreehandTool(BaseTool):
    """Freehand drawing tool."""

    tool_type = "freehand"

    def on_press(self, pos: QPoint, frame_index: int):
        self._points = [(pos.x(), pos.y())]
        self._active = True

    def on_move(self, pos: QPoint, frame_index: int):
        if self._active:
            self._points.append((pos.x(), pos.y()))

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if not self._active or len(self._points) < 2:
            self._active = False
            return None
        self._active = False
        ann = Annotation(
            tool_type="freehand",
            points=list(self._points),
            color=self.color,
            thickness=self.thickness,
            start_frame=frame_index
        )
        self._points = []
        return ann

    def get_preview(self):
        if self._active and len(self._points) >= 2:
            return Annotation(tool_type="freehand", points=list(self._points),
                              color=self.color, thickness=self.thickness)
        return None


class CurveTool(BaseTool):
    """Bezier curve tool - click to add control points, double-click to finish."""

    tool_type = "curve"

    def on_press(self, pos: QPoint, frame_index: int):
        self._points.append((pos.x(), pos.y()))
        self._active = True

    def on_move(self, pos: QPoint, frame_index: int):
        pass

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        # Finish on 4th point (cubic bezier)
        if len(self._points) >= 4:
            ann = Annotation(
                tool_type="curve",
                points=list(self._points[:4]),
                color=self.color,
                thickness=self.thickness,
                start_frame=frame_index
            )
            self.reset()
            return ann
        return None

    def get_preview(self):
        if self._active and len(self._points) >= 2:
            return Annotation(tool_type="curve", points=list(self._points),
                              color=self.color, thickness=self.thickness)
        return None


class TextTool(BaseTool):
    """Text annotation - click to place, enters text via dialog."""

    tool_type = "text"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_text = ""

    def set_text(self, text: str):
        self._pending_text = text

    def on_press(self, pos: QPoint, frame_index: int):
        self._points = [(pos.x(), pos.y())]
        self._active = True

    def on_release(self, pos: QPoint, frame_index: int) -> Annotation | None:
        if not self._active or not self._pending_text:
            self._active = False
            return None
        self._active = False
        ann = Annotation(
            tool_type="text",
            points=list(self._points),
            color=self.color,
            thickness=self.thickness,
            text=self._pending_text,
            start_frame=frame_index
        )
        self._points = []
        return ann
