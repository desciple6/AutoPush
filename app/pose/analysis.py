"""Form analysis - detects common lifting form issues from pose landmarks."""

import math
import numpy as np


class FormAnalysis:
    """Analyzes pose landmarks for common lifting form issues."""

    # Thresholds (in degrees or pixels)
    KNEE_CAVE_THRESHOLD_WARNING = 10   # pixels of medial drift
    KNEE_CAVE_THRESHOLD_BAD = 20
    BACK_ROUNDING_THRESHOLD_WARNING = 160  # degrees (straight = 180)
    BACK_ROUNDING_THRESHOLD_BAD = 145
    HIP_SHIFT_THRESHOLD_WARNING = 15   # pixel asymmetry
    HIP_SHIFT_THRESHOLD_BAD = 30

    def analyze(self, landmarks: dict, frame_shape: tuple) -> dict:
        """Run all form checks on the given landmarks.

        Returns dict with 'issues' key containing detected problems.
        """
        if landmarks is None:
            return {'issues': {}}

        issues = {}

        # Knee cave check
        knee_cave = self._check_knee_cave(landmarks)
        if knee_cave:
            issues['knee_cave'] = knee_cave

        # Back rounding check
        back_rounding = self._check_back_rounding(landmarks)
        if back_rounding:
            issues['back_rounding'] = back_rounding

        # Hip shift check
        hip_shift = self._check_hip_shift(landmarks)
        if hip_shift:
            issues['hip_shift'] = hip_shift

        return {'issues': issues}

    def _check_knee_cave(self, landmarks: dict) -> dict | None:
        """Check if knees are caving inward relative to ankle-hip midline.

        Knee cave = knee X-position drifts medially (toward center) compared to
        the midpoint of ankle and hip on the same side.
        """
        result = {'affected_joints': [], 'severity': 'ok', 'message': '', 'short_label': ''}

        for side in ['left', 'right']:
            hip = landmarks.get(f'{side}_hip')
            knee = landmarks.get(f'{side}_knee')
            ankle = landmarks.get(f'{side}_ankle')

            if not all([hip, knee, ankle]):
                continue
            if min(hip[2], knee[2], ankle[2]) < 0.5:
                continue

            # Expected knee X is roughly between hip and ankle X
            expected_x = (hip[0] + ankle[0]) / 2
            actual_x = knee[0]

            # For left side, cave = knee drifts right (toward center)
            # For right side, cave = knee drifts left (toward center)
            if side == 'left':
                drift = actual_x - expected_x  # Positive = drifting right (caving)
            else:
                drift = expected_x - actual_x  # Positive = drifting left (caving)

            if drift > self.KNEE_CAVE_THRESHOLD_BAD:
                result['severity'] = 'bad'
                result['affected_joints'].extend([f'{side}_knee', f'{side}_ankle'])
                result['message'] = f'Knee Cave ({side.title()}): {drift:.0f}px drift'
                result['short_label'] = 'CAVE'
            elif drift > self.KNEE_CAVE_THRESHOLD_WARNING:
                if result['severity'] != 'bad':
                    result['severity'] = 'warning'
                result['affected_joints'].extend([f'{side}_knee'])
                result['message'] = f'Knee Cave ({side.title()}): {drift:.0f}px drift'
                result['short_label'] = 'cave?'

        return result if result['severity'] != 'ok' else None

    def _check_back_rounding(self, landmarks: dict) -> dict | None:
        """Check for back rounding by measuring shoulder-hip-knee angle.

        A straight back during squats/deadlifts should have the shoulder-hip
        line roughly straight (close to 180 degrees with the hip-knee line).
        """
        for side in ['left', 'right']:
            shoulder = landmarks.get(f'{side}_shoulder')
            hip = landmarks.get(f'{side}_hip')
            knee = landmarks.get(f'{side}_knee')

            if not all([shoulder, hip, knee]):
                continue
            if min(shoulder[2], hip[2], knee[2]) < 0.5:
                continue

            angle = self._calc_angle(
                (shoulder[0], shoulder[1]),
                (hip[0], hip[1]),
                (knee[0], knee[1])
            )

            if angle < self.BACK_ROUNDING_THRESHOLD_BAD:
                return {
                    'severity': 'bad',
                    'affected_joints': [f'{side}_shoulder', f'{side}_hip'],
                    'message': f'Back Rounding: {angle:.0f}° (ideal >160°)',
                    'short_label': 'ROUND'
                }
            elif angle < self.BACK_ROUNDING_THRESHOLD_WARNING:
                return {
                    'severity': 'warning',
                    'affected_joints': [f'{side}_shoulder', f'{side}_hip'],
                    'message': f'Back Rounding: {angle:.0f}° (watch form)',
                    'short_label': 'round?'
                }

        return None

    def _check_hip_shift(self, landmarks: dict) -> dict | None:
        """Check for lateral hip shift (asymmetry between left and right hips)."""
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')
        left_ankle = landmarks.get('left_ankle')
        right_ankle = landmarks.get('right_ankle')

        if not all([left_hip, right_hip, left_ankle, right_ankle]):
            return None
        if min(left_hip[2], right_hip[2], left_ankle[2], right_ankle[2]) < 0.5:
            return None

        # Hip midpoint vs ankle midpoint
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
        shift = abs(hip_mid_x - ankle_mid_x)

        direction = "left" if hip_mid_x < ankle_mid_x else "right"

        if shift > self.HIP_SHIFT_THRESHOLD_BAD:
            return {
                'severity': 'bad',
                'affected_joints': ['left_hip', 'right_hip'],
                'message': f'Hip Shift {direction}: {shift:.0f}px',
                'short_label': 'SHIFT'
            }
        elif shift > self.HIP_SHIFT_THRESHOLD_WARNING:
            return {
                'severity': 'warning',
                'affected_joints': ['left_hip', 'right_hip'],
                'message': f'Hip Shift {direction}: {shift:.0f}px',
                'short_label': 'shift?'
            }

        return None

    @staticmethod
    def _calc_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return 180.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))
