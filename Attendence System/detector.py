from typing import List, Tuple, Optional
import numpy as np
import cv2

from config import (
    YUNET_MODEL_PATH,
    YUNET_CONFIDENCE_THRESHOLD,
    YUNET_NMS_THRESHOLD,
    DETECTION_INPUT_SIZE
)

class YuNetFaceDetector:
    """
    OpenCV FaceDetectorYN wrapper for fast, robust face & landmark detection.
    Detects bounding box (x, y, w, h) + 5 facial landmarks + detection confidence.
    """

    def __init__(
        self,
        model_path: str = YUNET_MODEL_PATH,
        conf_threshold: float = YUNET_CONFIDENCE_THRESHOLD,
        nms_threshold: float = YUNET_NMS_THRESHOLD,
        input_size: Tuple[int, int] = DETECTION_INPUT_SIZE
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.current_input_size = input_size

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.current_input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def detect(self, frame: np.ndarray, score_threshold: Optional[float] = None) -> np.ndarray:
        """
        Detect faces in frame.
        Returns numpy array of shape (N, 15) where each row is:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        if frame is None or frame.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        h, w = frame.shape[:2]
        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        if score_threshold is not None and score_threshold != self.conf_threshold:
            self.detector.setScoreThreshold(score_threshold)
            status, faces = self.detector.detect(frame)
            self.detector.setScoreThreshold(self.conf_threshold)
        else:
            status, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)

        return faces

    @staticmethod
    def draw_face_annotations(
        frame: np.ndarray,
        face_data: np.ndarray,
        label: str = "Face",
        passed_quality: bool = True,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Draw bounding box, landmarks and status label on frame."""
        vis = frame.copy()
        x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
        score = float(face_data[14])

        if color is None:
            color = (0, 220, 100) if passed_quality else (0, 100, 255)

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Draw 5 landmarks
        # Right eye, left eye (blue/cyan)
        cv2.circle(vis, (int(face_data[4]), int(face_data[5])), 3, (255, 200, 0), -1)
        cv2.circle(vis, (int(face_data[6]), int(face_data[7])), 3, (255, 200, 0), -1)
        # Nose tip (green)
        cv2.circle(vis, (int(face_data[8]), int(face_data[9])), 3, (0, 255, 0), -1)
        # Mouth corners (red/magenta)
        cv2.circle(vis, (int(face_data[10]), int(face_data[11])), 3, (0, 0, 255), -1)
        cv2.circle(vis, (int(face_data[12]), int(face_data[13])), 3, (0, 0, 255), -1)

        # Label background pill
        display_text = f"{label} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        bg_y1 = max(0, y - th - 8)
        bg_y2 = y
        cv2.rectangle(vis, (x, bg_y1), (x + tw + 10, bg_y2), color, -1)
        cv2.putText(vis, display_text, (x + 5, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return vis
