"""Annotation renderer - draws annotations onto OpenCV frames."""

import math
import cv2
import numpy as np

from app.annotations.manager import Annotation, AnnotationManager


class AnnotationRenderer:
    """Renders annotations onto video frames using OpenCV drawing functions."""

    def render(self, frame: np.ndarray, manager: AnnotationManager,
               frame_index: int) -> np.ndarray:
        """Draw all visible annotations onto the frame."""
        # Draw completed annotations
        for ann in manager.get_visible(frame_index):
            frame = self._draw_annotation(frame, ann)

        # Draw in-progress annotation (preview)
        if manager.temp_annotation is not None:
            frame = self._draw_annotation(frame, manager.temp_annotation)

        return frame

    def _draw_annotation(self, frame: np.ndarray, ann: Annotation) -> np.ndarray:
        """Draw a single annotation based on its type."""
        color = tuple(ann.color) if isinstance(ann.color, (list, tuple)) else (0, 255, 0)
        thickness = ann.thickness

        if ann.tool_type == "line":
            if len(ann.points) >= 2:
                pt1 = tuple(int(c) for c in ann.points[0])
                pt2 = tuple(int(c) for c in ann.points[1])
                cv2.line(frame, pt1, pt2, color, thickness)

        elif ann.tool_type == "arrow":
            if len(ann.points) >= 2:
                pt1 = tuple(int(c) for c in ann.points[0])
                pt2 = tuple(int(c) for c in ann.points[1])
                cv2.arrowedLine(frame, pt1, pt2, color, thickness, tipLength=0.05)

        elif ann.tool_type == "circle":
            if len(ann.points) >= 2:
                center = tuple(int(c) for c in ann.points[0])
                edge = ann.points[1]
                radius = int(math.sqrt(
                    (center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2
                ))
                cv2.circle(frame, center, radius, color, thickness)

        elif ann.tool_type == "angle":
            if len(ann.points) >= 3:
                pts = [tuple(int(c) for c in p) for p in ann.points[:3]]
                cv2.line(frame, pts[0], pts[1], color, thickness)
                cv2.line(frame, pts[1], pts[2], color, thickness)
                # Draw arc
                self._draw_angle_arc(frame, pts[0], pts[1], pts[2], color)
                # Draw degree text
                angle_text = f"{ann.angle_degrees:.1f}°"
                text_pos = (pts[1][0] + 15, pts[1][1] - 10)
                cv2.putText(frame, angle_text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif len(ann.points) == 2:
                pts = [tuple(int(c) for c in p) for p in ann.points]
                cv2.line(frame, pts[0], pts[1], color, thickness)

        elif ann.tool_type == "freehand":
            if len(ann.points) >= 2:
                pts = np.array(ann.points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, color, thickness)

        elif ann.tool_type == "curve":
            if len(ann.points) >= 2:
                frame = self._draw_bezier(frame, ann.points, color, thickness)

        elif ann.tool_type == "text":
            if ann.points and ann.text:
                pos = tuple(int(c) for c in ann.points[0])
                cv2.putText(frame, ann.text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    @staticmethod
    def _draw_angle_arc(frame, p1, vertex, p3, color, radius=30):
        """Draw a small arc at the vertex of an angle."""
        angle1 = math.degrees(math.atan2(p1[1] - vertex[1], p1[0] - vertex[0]))
        angle2 = math.degrees(math.atan2(p3[1] - vertex[1], p3[0] - vertex[0]))
        # Ensure we draw the shorter arc
        if angle1 > angle2:
            angle1, angle2 = angle2, angle1
        if angle2 - angle1 > 180:
            angle1, angle2 = angle2, angle1 + 360
        cv2.ellipse(frame, vertex, (radius, radius), 0, angle1, angle2, color, 1)

    @staticmethod
    def _draw_bezier(frame, points, color, thickness, num_segments=50):
        """Draw a bezier curve through control points."""
        if len(points) < 2:
            return frame

        pts = np.array(points, dtype=np.float64)
        curve_points = []
        for t in np.linspace(0, 1, num_segments):
            # De Casteljau algorithm
            temp = pts.copy()
            n = len(temp)
            for k in range(1, n):
                for i in range(n - k):
                    temp[i] = (1 - t) * temp[i] + t * temp[i + 1]
            curve_points.append(temp[0].astype(int))

        if len(curve_points) >= 2:
            curve_arr = np.array(curve_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [curve_arr], False, color, thickness)

        # Draw control points
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)

        return frame
