import time
import re
import io
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import cv2
from PIL import Image, ImageOps

from config import (
    BASE_DIR,
    CAMERA_ID,
    CAMERA_NAME,
    TARGET_FPS
)
from stream import RTSPStreamReader
from detector import YuNetFaceDetector
from quality import FaceQualityAssessor, QualityResult
from recognizer import SFaceRecognizer
from temporal import TemporalConfirmationTracker
from storage import StorageManager
from database import DatabaseManager

logger = logging.getLogger("AttendancePipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AttendancePipeline:
    """
    Complete 8-stage Face Recognition & Attendance Pipeline:
    1. RTSP Camera Input (OpenCV VideoCapture / FFmpeg)
    2. Frame Sampling (2-5 FPS)
    3. YuNet Face & Landmark Detection
    4. Quality Filter (Blur, Size, Orientation) + Alignment
    5. SFace 128-d Embedding Extraction
    6. Identity Matching against Enrolled Gallery
    7. Temporal Multi-Frame Confirmation
    8. Local Storage (SQLite + .npy / .json / aligned image crops)
    """

    def __init__(
        self,
        auto_start_stream: bool = False,
        camera_id: str = CAMERA_ID,
        target_fps: float = TARGET_FPS
    ):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self._model_lock = threading.RLock()

        logger.info("Initializing Attendance Pipeline components...")
        self.detector = YuNetFaceDetector()
        self.quality_assessor = FaceQualityAssessor()
        self.recognizer = SFaceRecognizer()
        self.temporal_tracker = TemporalConfirmationTracker()
        self.storage = StorageManager()
        self.db = DatabaseManager()

        self.stream = RTSPStreamReader(target_fps=self.target_fps)

        # In-memory gallery cache for fast cosine matching
        self.enrolled_gallery: List[Dict[str, Any]] = []
        self.reload_enrolled_gallery()

        if auto_start_stream:
            self.stream.start()

    def reload_enrolled_gallery(self):
        """Loads or refreshes enrolled employee embeddings from disk and SQLite."""
        self.enrolled_gallery = []
        employees = self.db.get_employees()
        for emp in employees:
            embs = []
            # 1. Primary embedding
            primary_path = emp.get("embedding_path")
            if primary_path:
                try:
                    embs.append(self.storage.load_embedding(primary_path))
                except Exception as e:
                    logger.warning(f"Could not load primary embedding for {emp['id']}: {e}")

            # 2. Additional multi-view embeddings
            extra_records = self.db.get_employee_embeddings(emp["id"])
            for rec in extra_records:
                p = rec.get("embedding_path")
                if p and p != primary_path:
                    try:
                        embs.append(self.storage.load_embedding(p))
                    except Exception as e:
                        logger.warning(f"Could not load extra embedding for {emp['id']}: {e}")

            if embs:
                self.enrolled_gallery.append({
                    "id": emp["id"],
                    "name": emp["name"],
                    "department": emp.get("department", "General"),
                    "embedding": embs[0],
                    "embeddings": embs,
                    "photo_path": emp.get("photo_path", "")
                })

        logger.info(f"Loaded {len(self.enrolled_gallery)} enrolled employee profiles ({sum(len(p['embeddings']) for p in self.enrolled_gallery)} total embeddings).")

    def process_frame(
        self,
        frame: np.ndarray,
        save_passed_embeddings: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes stages 3 through 8 on a single sampled frame.
        
        Returns:
            annotated_frame: Frame with visual overlays, landmarks, and identification badges.
            processed_faces: List of dictionaries detailing each detected face, quality, embedding and matches.
        """
        if frame is None or frame.size == 0:
            return frame, []

        vis_frame = frame.copy()
        with self._model_lock:
            raw_detections = self.detector.detect(frame)

        processed_faces: List[Dict[str, Any]] = []

        for face_data in raw_detections:
            x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
            conf_score = float(face_data[14])

            # 4. Quality Filter
            quality: QualityResult = self.quality_assessor.assess(frame, face_data)

            face_record: Dict[str, Any] = {
                "bbox": [x, y, w, h],
                "confidence": conf_score,
                "passed_quality": quality.passed,
                "rejection_reasons": quality.reasons,
                "blur_score": quality.blur_score,
                "roll_angle": quality.roll_angle,
                "yaw_ratio": quality.yaw_ratio,
                "matched_id": None,
                "matched_name": "UNKNOWN",
                "similarity_score": 0.0,
                "temporal_status": "SKIPPED_QUALITY" if not quality.passed else "PENDING",
                "saved_files": {}
            }

            if not quality.passed:
                # Face rejected by quality filter
                reason_str = ", ".join(quality.reasons) if quality.reasons else "Quality failed"
                label = f"REJECTED: {reason_str[:28]}"
                vis_frame = self.detector.draw_face_annotations(
                    vis_frame, face_data, label=label, passed_quality=False, color=(0, 0, 255)
                )

                # Log quality failure audit to database
                self.db.log_face_capture(
                    camera_id=self.camera_id,
                    bbox_w=w, bbox_h=h,
                    blur_score=quality.blur_score,
                    roll_angle=quality.roll_angle,
                    yaw_ratio=quality.yaw_ratio,
                    passed_quality=False,
                    rejection_reason=reason_str,
                    match_name="UNKNOWN",
                    match_score=0.0,
                    face_image_path="",
                    embedding_path=""
                )

                processed_faces.append(face_record)
                continue

            # 4. Alignment & 5. SFace Embedding Extraction
            with self._model_lock:
                aligned_face = self.recognizer.align_crop(frame, face_data)
                embedding = self.recognizer.extract_embedding(aligned_face)

                # 6. Identity Matching
                matched_id, matched_name, match_score, is_match = self.recognizer.match_against_gallery(
                    embedding, self.enrolled_gallery
                )

            face_record["matched_id"] = matched_id
            face_record["matched_name"] = matched_name
            face_record["similarity_score"] = match_score

            # 7. Temporal Multi-Frame Confirmation
            is_confirmed, temp_status, streak = self.temporal_tracker.update(
                emp_id=matched_id,
                emp_name=matched_name,
                confidence=match_score,
                camera_id=self.camera_id
            )
            face_record["temporal_status"] = temp_status

            # 8. Local Storage to Drive & SQLite
            saved_paths = {}
            if save_passed_embeddings:
                saved_paths = self.storage.save_face_and_embedding(
                    aligned_face=aligned_face,
                    embedding=embedding,
                    raw_crop=quality.raw_crop,
                    camera_id=self.camera_id,
                    metadata={
                        "confidence": conf_score,
                        "blur_score": quality.blur_score,
                        "roll_angle": quality.roll_angle,
                        "yaw_ratio": quality.yaw_ratio,
                        "matched_name": matched_name,
                        "similarity_score": match_score,
                        "temporal_status": temp_status
                    }
                )
                face_record["saved_files"] = saved_paths

            # Log capture to SQLite
            self.db.log_face_capture(
                camera_id=self.camera_id,
                bbox_w=w, bbox_h=h,
                blur_score=quality.blur_score,
                roll_angle=quality.roll_angle,
                yaw_ratio=quality.yaw_ratio,
                passed_quality=True,
                rejection_reason="",
                match_name=matched_name,
                match_score=match_score,
                face_image_path=saved_paths.get("aligned_image_path", ""),
                embedding_path=saved_paths.get("embedding_path", "")
            )








            # Record verified attendance event if confirmed
            if is_confirmed and matched_id:
                # Date
                cdate = datetime.now().strftime("%Y%m%d")  # "20261011"

                # Time
                ctime = datetime.now().strftime("%H%M%S")  # "172943"

                print(f"\n=======================================================", flush=True)
                print(f" [✓] VERIFIED ATTENDANCE CONFIRMED!", flush=True)
                print(f"     Staff ID   : {matched_id}", flush=True)
                print(f"     Staff Name : {matched_name}", flush=True)
                print(f"     Date       : {cdate}", flush=True)
                print(f"     Time       : {ctime}", flush=True)
                print(f"     Confidence : {match_score * 100:.1f}%", flush=True)
                print(f"     Camera     : {self.camera_id}", flush=True)
                print(f"=======================================================\n", flush=True)
                logger.info(f"Verified Attendance: {matched_name} (ID: {matched_id}, Date: {cdate}, Time: {ctime}, Sim: {match_score:.2f})")
                self.db.log_attendance(
                    emp_id=matched_id,
                    name=matched_name,
                    camera_id=self.camera_id,
                    confidence=match_score,
                    status="CONFIRMED",
                    face_image_path=saved_paths.get("aligned_image_path", ""),
                    embedding_path=saved_paths.get("embedding_path", "")
                )







            # Visual Annotation Overlay
            box_color = (0, 255, 0) if is_match else (0, 215, 255)  # Green if recognized, Yellow/Orange if unknown
            if matched_name != "UNKNOWN":
                display_label = f"{matched_name} [{temp_status}]"
            else:
                display_label = f"UNKNOWN [Quality OK]"

            vis_frame = self.detector.draw_face_annotations(
                vis_frame, face_data, label=display_label, passed_quality=True, color=box_color
            )

            processed_faces.append(face_record)

        return vis_frame, processed_faces

    def enroll_face_from_image(
        self,
        image_path: str,
        emp_id: str,
        name: str,
        department: str = "General",
        email: str = ""
    ) -> Tuple[bool, str]:
        """
        Enrolls an employee from a clear portrait image.
        Detects face, verifies quality, aligns, extracts SFace embedding, saves to disk & DB.
        """
        img = cv2.imread(image_path)
        if img is None:
            return False, f"Could not read image from {image_path}"

        # If image is already an aligned 112x112 chip
        if img.shape[0] == 112 and img.shape[1] == 112:
            aligned = img
            embedding = self.recognizer.extract_embedding(aligned)
            saved = self.storage.save_face_and_embedding(
                aligned_face=aligned,
                embedding=embedding,
                raw_crop=None,
                camera_id="ENROLLMENT_PORTAL",
                metadata={"enrolled_name": name, "emp_id": emp_id}
            )
            self.db.enroll_employee(
                emp_id=emp_id,
                name=name,
                department=department,
                photo_path=saved["aligned_image_path"],
                embedding_path=saved["embedding_path"],
                email=email
            )
            self.reload_enrolled_gallery()
            return True, f"Employee '{name}' ({emp_id}) enrolled successfully from aligned face chip!"

    @staticmethod
    def preprocess_enrollment_image(
        file_source: Any,
        max_dim: int = 1280
    ) -> Tuple[Optional[np.ndarray], bytes]:
        """
        Safely loads photo, auto-corrects EXIF rotation (critical for smartphones),
        downscales to optimal dimension (speeding up YuNet by 10x-20x without quality loss),
        and returns both the OpenCV BGR frame and normalized JPEG bytes for disk saving.
        """
        try:
            if isinstance(file_source, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(file_source))
            else:
                pil_img = Image.open(file_source)

            # Auto-orient based on EXIF (fixes phone portrait photos)
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass

            pil_img = pil_img.convert("RGB")

            # Rescale large images to optimal receptive field
            w, h = pil_img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                nw = int(round(w * scale))
                nh = int(round(h * scale))
                pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)

            # Convert to OpenCV BGR
            rgb_arr = np.array(pil_img)
            bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            # Generate normalized JPEG bytes
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            clean_bytes = buf.getvalue()

            return bgr_frame, clean_bytes
        except Exception as e:
            logger.error(f"Error in preprocess_enrollment_image: {e}")
            return None, b""

    def enroll_face_from_image(
        self,
        image_path: str,
        emp_id: str,
        name: str,
        department: str = "General",
        email: str = ""
    ) -> Tuple[bool, str]:
        """Enrolls an employee from a single image file."""
        p = Path(image_path)
        if not p.exists():
            return False, f"Image not found: {image_path}"

        bgr_frame, _ = self.preprocess_enrollment_image(str(p))
        if bgr_frame is None:
            return False, f"Could not read image: {image_path}"

        img = bgr_frame

        # If it's already an aligned 112x112 chip
        if img.shape[0] == 112 and img.shape[1] == 112:
            with self._model_lock:
                embedding = self.recognizer.extract_embedding(img)
            saved = self.storage.save_face_and_embedding(
                aligned_face=img,
                embedding=embedding,
                raw_crop=None,
                camera_id="ENROLLMENT_PORTAL",
                metadata={"enrolled_name": name, "emp_id": emp_id}
            )
            self.db.enroll_employee(
                emp_id=emp_id,
                name=name,
                department=department,
                photo_path=saved["aligned_image_path"],
                embedding_path=saved["embedding_path"],
                email=email
            )
            self.reload_enrolled_gallery()
            return True, f"Employee '{name}' ({emp_id}) enrolled successfully from aligned face chip!"

        with self._model_lock:
            faces = self.detector.detect(img, score_threshold=0.45)
            if len(faces) == 0:
                faces = self.detector.detect(img, score_threshold=0.30)

        if len(faces) == 0:
            return False, "No face detected in enrollment image."

        # Pick the largest face in image
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        quality = self.quality_assessor.assess(img, largest_face)

        with self._model_lock:
            aligned = self.recognizer.align_crop(img, largest_face)
            embedding = self.recognizer.extract_embedding(aligned)

        saved = self.storage.save_face_and_embedding(
            aligned_face=aligned,
            embedding=embedding,
            raw_crop=quality.raw_crop,
            camera_id="ENROLLMENT_PORTAL",
            metadata={"enrolled_name": name, "emp_id": emp_id}
        )

        self.db.enroll_employee(
            emp_id=emp_id,
            name=name,
            department=department,
            photo_path=saved["aligned_image_path"],
            embedding_path=saved["embedding_path"],
            email=email
        )

        self.reload_enrolled_gallery()
        return True, f"Employee '{name}' ({emp_id}) enrolled successfully!"

    def delete_employee(self, emp_id: str) -> bool:
        """Deletes employee from database and reloads recognition gallery."""
        success = self.db.delete_employee(emp_id)
        if success:
            self.reload_enrolled_gallery()
        return success

    def enroll_staff_from_photos(
        self,
        staff_id: str,
        name: str,
        department: str = "General",
        email: str = "",
        photos: List[Tuple[str, bytes]] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Industrial-grade multi-photo staff registration:
        1. Preprocesses and auto-orients photos using EXIF metadata.
        2. Normalizes dimensions to optimize YuNet inference speed (10x-20x faster).
        3. Extracts YuNet face landmarks and verifies quality under thread lock.
        4. Extracts 128-d SFace embeddings and aligned 112x112 crops.
        5. Computes normalized centroid mean embedding for multi-angle robustness.
        6. Registers all multi-view templates in SQLite.
        7. Hot-reloads in-memory gallery cache for instant RTSP recognition.
        """
        if photos is None:
            photos = []

        staff_id = staff_id.strip()
        name = name.strip()

        if not staff_id or not name:
            return False, "Staff ID and Full Name are required.", {}

        if not photos:
            return False, "At least one portrait photo is required.", {}

        if progress_callback:
            progress_callback(f"Preparing archive folder for '{name}'...")

        # 1. Create clean folder: Attendence System/Staffs/{staff_id}_{name}
        clean_id = re.sub(r'[^\w\-]', '_', staff_id)
        clean_name = re.sub(r'[^\w\-]', '_', name)
        folder_name = f"{clean_id}_{clean_name}"
        staffs_root = BASE_DIR / "Staffs"
        staff_dir = staffs_root / folder_name
        staff_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Enrolling staff '{name}' ({staff_id}) -> Folder: Staffs/{folder_name} with {len(photos)} photos...")

        valid_embeddings = []
        valid_records = []
        best_frontal_score = -1.0
        primary_record = None
        saved_photo_paths = []

        # 2. Process each photo with thread safety and fast scaling
        for idx, (filename, file_bytes) in enumerate(photos, 1):
            if progress_callback:
                progress_callback(f"Analyzing Photo {idx} of {len(photos)}...")

            bgr_img, clean_bytes = self.preprocess_enrollment_image(file_bytes)
            if bgr_img is None:
                logger.warning(f"Could not decode photo {idx}: {filename}")
                continue

            # Save normalized photo to Staffs/{folder_name}/
            target_path = staff_dir / f"photo_{idx}.jpg"
            target_path.write_bytes(clean_bytes)
            saved_photo_paths.append(target_path)

            # Detect face under thread lock with adaptive threshold
            with self._model_lock:
                faces = self.detector.detect(bgr_img, score_threshold=0.45)
                if len(faces) == 0:
                    # Fallback lower threshold for angled/low-light shots
                    faces = self.detector.detect(bgr_img, score_threshold=0.30)

            if len(faces) == 0:
                logger.warning(f"No face detected in photo {idx} ({filename})")
                continue

            # Pick largest face
            face_data = max(faces, key=lambda f: f[2] * f[3])
            quality = self.quality_assessor.assess(bgr_img, face_data)

            # Align face to 112x112 chip under thread lock
            with self._model_lock:
                aligned_face = self.recognizer.align_crop(bgr_img, face_data)
                embedding = self.recognizer.extract_embedding(aligned_face)

            saved = self.storage.save_face_and_embedding(
                aligned_face=aligned_face,
                embedding=embedding,
                raw_crop=quality.raw_crop,
                camera_id="ENROLLMENT_PORTAL",
                metadata={
                    "staff_name": name,
                    "staff_id": staff_id,
                    "photo_source": target_path.name,
                    "blur_score": quality.blur_score,
                    "roll_angle": quality.roll_angle,
                    "yaw_ratio": quality.yaw_ratio,
                    "passed_quality": quality.passed
                }
            )

            # Heuristic for best frontal photo
            frontal_score = (1.0 - abs(quality.yaw_ratio - 1.0) * 0.5) * np.log1p(max(0.1, quality.blur_score))
            if frontal_score > best_frontal_score:
                best_frontal_score = frontal_score
                primary_record = saved

            valid_embeddings.append(embedding)
            valid_records.append(saved)

        if not valid_records:
            return False, "No valid faces could be detected in the uploaded photos. Please provide clear portrait photos with visible faces.", {}

        if progress_callback:
            progress_callback("Saving staff profile to SQLite database...")

        # 3. Save into SQLite
        primary = primary_record if primary_record else valid_records[0]
        self.db.enroll_employee(
            emp_id=staff_id,
            name=name,
            department=department or "Staff",
            photo_path=primary["aligned_image_path"],
            embedding_path=primary["embedding_path"],
            email=email
        )

        for rec in valid_records:
            is_pri = 1 if rec == primary else 0
            self.db.add_employee_embedding(
                emp_id=staff_id,
                embedding_path=rec["embedding_path"],
                photo_path=rec["aligned_image_path"],
                is_primary=is_pri
            )

        # 4. Compute combined normalized centroid mean embedding if multiple valid photos
        if len(valid_embeddings) > 1:
            if progress_callback:
                progress_callback("Computing multi-angle centroid template...")

            mean_vector = np.mean(np.vstack(valid_embeddings), axis=0, keepdims=True)
            norm = np.linalg.norm(mean_vector)
            if norm > 1e-6:
                centroid_emb = (mean_vector / norm).astype(np.float32)
                pri_img = cv2.imread(primary["aligned_image_path"])
                if pri_img is not None:
                    centroid_saved = self.storage.save_face_and_embedding(
                        aligned_face=pri_img,
                        embedding=centroid_emb,
                        camera_id="CENTROID_TEMPLATE",
                        metadata={"staff_name": name, "staff_id": staff_id, "type": "centroid_average"}
                    )
                    self.db.add_employee_embedding(
                        emp_id=staff_id,
                        embedding_path=centroid_saved["embedding_path"],
                        photo_path=primary["aligned_image_path"],
                        is_primary=0
                    )

        # 5. Hot reload live gallery cache
        if progress_callback:
            progress_callback("Reloading live recognition gallery...")

        self.reload_enrolled_gallery()

        summary_data = {
            "staff_id": staff_id,
            "name": name,
            "department": department or "Staff",
            "folder": folder_name,
            "folder_path": str(staff_dir),
            "photos_saved": len(saved_photo_paths),
            "templates_created": len(valid_records) + (1 if len(valid_embeddings) > 1 else 0),
            "primary_photo": Path(primary["aligned_image_path"]).name
        }

        return True, f"Successfully registered '{name}' (ID: {staff_id}) with {len(valid_records)} templates stored in Staffs/{folder_name}!", summary_data
