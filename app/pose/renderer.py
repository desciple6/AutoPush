"""Pose renderer - draws skeleton and joint overlays on video frames."""

import cv2
import numpy as np

from app.pose.estimator import PoseEstimator
from app.config import (
    POSE_SKELETON_COLOR_GOOD,
    POSE_SKELETON_COLOR_WARNING,
    POSE_SKELETON_COLOR_BAD,
    POSE_CONFIDENCE_THRESHOLD
)


class PoseRenderer:
    """Draws the pose skeleton and form analysis results on frames."""

    def render(self, frame: np.ndarray, landmarks: dict,
               analysis: dict = None) -> np.ndarray:
        """Draw skeleton overlay with optional form analysis coloring."""
        if landmarks is None:
            return frame

        # Draw skeleton connections
        for name1, name2 in PoseEstimator.CONNECTIONS:
            if name1 in landmarks and name2 in landmarks:
                pt1 = landmarks[name1]
                pt2 = landmarks[name2]
                if pt1[2] > POSE_CONFIDENCE_THRESHOLD and pt2[2] > POSE_CONFIDENCE_THRESHOLD:
                    color = self._get_connection_color(name1, name2, analysis)
                    cv2.line(frame,
                             (pt1[0], pt1[1]),
                             (pt2[0], pt2[1]),
                             color, 2)

        # Draw joint dots
        for name, (x, y, vis) in landmarks.items():
            if vis > POSE_CONFIDENCE_THRESHOLD:
                color = self._get_joint_color(name, analysis)
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

        # Draw analysis text overlays
        if analysis:
            self._draw_analysis_labels(frame, landmarks, analysis)

        return frame

    def _get_joint_color(self, joint_name: str, analysis: dict = None) -> tuple:
        """Get color for a joint based on form analysis."""
        if analysis is None:
            return POSE_SKELETON_COLOR_GOOD

        # Check if this joint is flagged
        issues = analysis.get('issues', {})
        for issue_name, issue_data in issues.items():
            if joint_name in issue_data.get('affected_joints', []):
                severity = issue_data.get('severity', 'ok')
                if severity == 'bad':
                    return POSE_SKELETON_COLOR_BAD
                elif severity == 'warning':
                    return POSE_SKELETON_COLOR_WARNING

        return POSE_SKELETON_COLOR_GOOD

    def _get_connection_color(self, name1: str, name2: str,
                              analysis: dict = None) -> tuple:
        """Get color for a skeleton connection."""
        if analysis is None:
            return POSE_SKELETON_COLOR_GOOD

        issues = analysis.get('issues', {})
        for issue_name, issue_data in issues.items():
            affected = issue_data.get('affected_joints', [])
            if name1 in affected or name2 in affected:
                severity = issue_data.get('severity', 'ok')
                if severity == 'bad':
                    return POSE_SKELETON_COLOR_BAD
                elif severity == 'warning':
                    return POSE_SKELETON_COLOR_WARNING

        return POSE_SKELETON_COLOR_GOOD

    def _draw_analysis_labels(self, frame: np.ndarray, landmarks: dict,
                              analysis: dict):
        """Draw text labels for detected form issues."""
        issues = analysis.get('issues', {})
        y_offset = 30

        for issue_name, issue_data in issues.items():
            severity = issue_data.get('severity', 'ok')
            if severity == 'ok':
                continue

            message = issue_data.get('message', issue_name)
            color = POSE_SKELETON_COLOR_BAD if severity == 'bad' else POSE_SKELETON_COLOR_WARNING

            # Draw at the top of the frame
            cv2.putText(frame, message, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

            # Also draw near the affected joint if available
            affected = issue_data.get('affected_joints', [])
            if affected and affected[0] in landmarks:
                jx, jy, _ = landmarks[affected[0]]
                cv2.putText(frame, issue_data.get('short_label', '!'),
                            (jx + 10, jy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
