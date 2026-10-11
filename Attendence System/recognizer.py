from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2

from config import (
    SFACE_MODEL_PATH,
    COSINE_SIMILARITY_THRESHOLD,
    L2_DISTANCE_THRESHOLD
)

class SFaceRecognizer:
    """
    OpenCV FaceRecognizerSF wrapper for SFace 128-d face embedding and verification.
    Handles alignment to 112x112 chip, feature extraction, and cosine/L2 identity matching.
    """

    def __init__(
        self,
        model_path: str = SFACE_MODEL_PATH,
        cosine_threshold: float = COSINE_SIMILARITY_THRESHOLD,
        l2_threshold: float = L2_DISTANCE_THRESHOLD
    ):
        self.model_path = model_path
        self.cosine_threshold = cosine_threshold
        self.l2_threshold = l2_threshold

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def align_crop(self, frame: np.ndarray, face_data: np.ndarray) -> np.ndarray:
        """
        Align and crop detected face into standard 112x112 frontal face representation.
        Uses 5 facial landmarks for affine transformation.
        """
        # OpenCV FaceRecognizerSF expects float32 face data
        face_input = np.array(face_data, dtype=np.float32)
        aligned = self.recognizer.alignCrop(frame, face_input)
        return aligned

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract 128-dimensional feature embedding from 112x112 aligned face.
        Returns 1x128 float32 vector, L2-normalized.
        """
        raw_feat = self.recognizer.feature(aligned_face)
        # Ensure L2 normalization
        norm = np.linalg.norm(raw_feat)
        if norm > 0:
            normalized_feat = raw_feat / norm
        else:
            normalized_feat = raw_feat
        return normalized_feat.astype(np.float32)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two 128-d embeddings.
        Returns float in range [-1.0, 1.0], typically > 0.38 for same identity.
        """
        # OpenCV built-in SFace cosine match
        f1 = feat1.reshape(1, -1).astype(np.float32)
        f2 = feat2.reshape(1, -1).astype(np.float32)
        score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def compute_l2_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute L2 Norm distance between two embeddings."""
        f1 = feat1.reshape(1, -1).astype(np.float32)
        f2 = feat2.reshape(1, -1).astype(np.float32)
        dist = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_NORM_L2)
        return float(dist)

    def match_against_gallery(
        self,
        query_embedding: np.ndarray,
        gallery: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], str, float, bool]:
        """
        Compare query embedding against a list of enrolled profiles:
        Each entry in gallery should have 'id', 'name', 'embedding'.
        
        Returns:
            (matched_id, matched_name, highest_similarity, is_match_above_threshold)
        """
        if not gallery:
            return None, "UNKNOWN", 0.0, False

        best_sim = -1.0
        best_id = None
        best_name = "UNKNOWN"

        for profile in gallery:
            embeddings_list = profile.get("embeddings")
            if embeddings_list is None:
                single_emb = profile.get("embedding")
                embeddings_list = [single_emb] if single_emb is not None else []

            if not embeddings_list:
                continue

            # Compare query against all registered reference embeddings for this person
            person_best_sim = -1.0
            for enrolled_emb in embeddings_list:
                if enrolled_emb is None:
                    continue
                sim = self.compute_similarity(query_embedding, enrolled_emb)
                if sim > person_best_sim:
                    person_best_sim = sim

            if person_best_sim > best_sim:
                best_sim = person_best_sim
                best_id = profile.get("id")
                best_name = profile.get("name", "Unknown")

        is_match = best_sim >= self.cosine_threshold
        if not is_match:
            best_name = "UNKNOWN"

        return best_id, best_name, float(best_sim), is_match
