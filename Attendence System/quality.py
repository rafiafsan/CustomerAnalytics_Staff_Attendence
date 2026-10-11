from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from config import (
    MIN_FACE_WIDTH,
    MIN_FACE_HEIGHT,
    MIN_BLUR_SCORE,
    MAX_ROLL_DEGREES,
    MIN_YAW_RATIO,
    BORDER_MARGIN_PX
)

@dataclass
class QualityResult:
    passed: bool
    reasons: List[str]
    width: float
    height: float
    blur_score: float
    roll_angle: float
    yaw_ratio: float
    raw_crop: Optional[np.ndarray] = None


class FaceQualityAssessor:
    """
    Evaluates detected face quality and orientation before embedding extraction.
    Rejects blurry, tiny, distorted, or severely angled faces.
    """

    def __init__(
        self,
        min_width: float = MIN_FACE_WIDTH,
        min_height: float = MIN_FACE_HEIGHT,
        min_blur_score: float = MIN_BLUR_SCORE,
        max_roll: float = MAX_ROLL_DEGREES,
        min_yaw_ratio: float = MIN_YAW_RATIO,
        border_margin: int = BORDER_MARGIN_PX
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_blur_score = min_blur_score
        self.max_roll = max_roll
        self.min_yaw_ratio = min_yaw_ratio
        self.border_margin = border_margin

    def assess(self, frame: np.ndarray, face_data: np.ndarray) -> QualityResult:
        """
        Assess face candidate.
        
        face_data layout from YuNet:
          [0:4]: [x, y, w, h]
          [4:14]: landmarks:
             [4, 5]: right eye
             [6, 7]: left eye
             [8, 9]: nose tip
             [10, 11]: right mouth corner
             [12, 13]: left mouth corner
          [14]: confidence score
        """
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = float(face_data[0]), float(face_data[1]), float(face_data[2]), float(face_data[3])
        reasons = []

        # 1. Size Check
        if w < self.min_width or h < self.min_height:
            reasons.append(f"Face too small ({w:.0f}x{h:.0f} < {self.min_width}x{self.min_height})")

        aspect_ratio = h / max(w, 1e-4)
        if aspect_ratio < 0.65 or aspect_ratio > 2.2:
            reasons.append(f"Abnormal aspect ratio ({aspect_ratio:.2f})")

        # 2. Border Margin Check (ensure face isn't clipped out of frame)
        if (x < self.border_margin or 
            y < self.border_margin or 
            (x + w) > (frame_w - self.border_margin) or 
            (y + h) > (frame_h - self.border_margin)):
            reasons.append("Face clipped at frame edge")

        # Extract face crop for blur computation
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(frame_w, int(round(x + w)))
        y2 = min(frame_h, int(round(y + h)))

        crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

        # 3. Blur Check (Laplacian Variance)
        blur_score = 0.0
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < self.min_blur_score:
                reasons.append(f"Face blurry (Laplacian variance {blur_score:.1f} < {self.min_blur_score:.1f})")
        else:
            reasons.append("Empty face crop")

        # 4. Orientation & Pose Checks using Landmarks
        # Right Eye (4, 5), Left Eye (6, 7), Nose (8, 9), Mouth R (10, 11), Mouth L (12, 13)
        rx, ry = float(face_data[4]), float(face_data[5])
        lx, ly = float(face_data[6]), float(face_data[7])
        nx, ny = float(face_data[8]), float(face_data[9])
        rmx, rmy = float(face_data[10]), float(face_data[11])
        lmx, lmy = float(face_data[12]), float(face_data[13])

        # Roll Angle: Tilt between right eye and left eye
        dx = lx - rx
        dy = ly - ry
        roll_angle = float(abs(np.degrees(np.arctan2(dy, dx))))
        if roll_angle > self.max_roll:
            reasons.append(f"Excessive roll angle ({roll_angle:.1f}° > {self.max_roll:.1f}°)")

        # Yaw Symmetry: Nose horizontal offset between eyes
        dist_right = abs(nx - rx)
        dist_left = abs(lx - nx)
        yaw_ratio = float(min(dist_right, dist_left) / max(dist_right, dist_left, 1e-4))
        if yaw_ratio < self.min_yaw_ratio:
            reasons.append(f"Extreme yaw angle (symmetry ratio {yaw_ratio:.2f} < {self.min_yaw_ratio:.2f})")

        # Pitch Sanity Check: Nose should be vertically between eye line and mouth line
        eye_y = (ry + ly) / 2.0
        mouth_y = (rmy + lmy) / 2.0
        if not (eye_y < ny < mouth_y):
            reasons.append("Abnormal vertical pitch (landmarks out of sequence)")

        passed = len(reasons) == 0
        return QualityResult(
            passed=passed,
            reasons=reasons,
            width=w,
            height=h,
            blur_score=blur_score,
            roll_angle=roll_angle,
            yaw_ratio=yaw_ratio,
            raw_crop=crop
        )
