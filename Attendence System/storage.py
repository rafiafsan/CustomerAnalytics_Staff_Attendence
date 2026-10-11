import os
import json
import time
import uuid
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

from config import FACES_DIR, EMBEDDINGS_DIR

class StorageManager:
    """Manages file storage for aligned face images, crops, and 128-d SFace embeddings."""

    def __init__(self, faces_dir: Path = FACES_DIR, embeddings_dir: Path = EMBEDDINGS_DIR):
        self.faces_dir = Path(faces_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save_face_and_embedding(
        self,
        aligned_face: np.ndarray,
        embedding: np.ndarray,
        raw_crop: Optional[np.ndarray] = None,
        camera_id: str = "CAM_01",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Saves aligned face image, raw crop, numpy embedding (.npy), and metadata JSON.
        Returns dictionary of saved relative/absolute file paths.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = uuid.uuid4().hex[:8]
        base_name = f"face_{timestamp_str}_{unique_id}"

        # 1. Save 112x112 aligned face
        aligned_filename = f"{base_name}_aligned.jpg"
        aligned_path = self.faces_dir / aligned_filename
        cv2.imwrite(str(aligned_path), aligned_face)

        # 2. Save raw crop if provided
        crop_path = None
        if raw_crop is not None and raw_crop.size > 0:
            crop_filename = f"{base_name}_crop.jpg"
            crop_path = self.faces_dir / crop_filename
            cv2.imwrite(str(crop_path), raw_crop)

        # 3. Save raw numpy embedding (.npy)
        emb_filename = f"{base_name}_emb.npy"
        emb_path = self.embeddings_dir / emb_filename
        np.save(str(emb_path), embedding.astype(np.float32))

        # 4. Save metadata JSON with embedded vector for inspection & easy verification
        meta_filename = f"{base_name}_meta.json"
        meta_path = self.embeddings_dir / meta_filename
        meta_payload = {
            "id": unique_id,
            "created_at": datetime.datetime.now().isoformat(),
            "camera_id": camera_id,
            "aligned_image": str(aligned_path),
            "raw_crop_image": str(crop_path) if crop_path else None,
            "embedding_npy": str(emb_path),
            "embedding_dim": int(embedding.size),
            "embedding_vector": embedding.flatten().tolist(),
            "quality_metrics": metadata or {}
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        return {
            "aligned_image_path": str(aligned_path),
            "raw_crop_path": str(crop_path) if crop_path else "",
            "embedding_path": str(emb_path),
            "metadata_path": str(meta_path),
            "base_name": base_name,
            "id": unique_id
        }

    def load_embedding(self, file_path: str) -> np.ndarray:
        """Load an embedding from either a .npy or a .json file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        if path.suffix == ".npy":
            emb = np.load(str(path))
            return emb.reshape(1, -1).astype(np.float32)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vec = np.array(data["embedding_vector"], dtype=np.float32)
            return vec.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported embedding format: {path.suffix}")

    def list_saved_embeddings(self) -> list:
        """List all saved embedding metadata files."""
        json_files = sorted(self.embeddings_dir.glob("*_meta.json"), key=os.path.getmtime, reverse=True)
        results = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception:
                continue
        return results
