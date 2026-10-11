import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2

from config import (
    BASE_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    DB_PATH,
    COSINE_SIMILARITY_THRESHOLD
)
from detector import YuNetFaceDetector
from quality import FaceQualityAssessor
from recognizer import SFaceRecognizer
from storage import StorageManager
from database import DatabaseManager

def enroll_all_staffs(staffs_root: Path = BASE_DIR / "Staffs", clear_existing: bool = True):
    """
    Processes all staff portrait folders under Staffs/, extracts high-quality
    128-d SFace embeddings for each photo, and registers them in SQLite and local disk.
    """
    staffs_root = Path(staffs_root)
    if not staffs_root.exists():
        print(f"[!] Error: Staffs directory not found at {staffs_root}")
        return

    print("\n" + "=" * 78)
    print("      Enrolling Staff Profiles from 'Staffs' Folder (Multi-Template)")
    print("=" * 78)
    print(f" Source Directory: {staffs_root.resolve()}")
    print(f" Target Database : {DB_PATH.resolve()}")
    print(f" Embeddings Dir  : {EMBEDDINGS_DIR.resolve()}")
    print("=" * 78 + "\n")

    detector = YuNetFaceDetector()
    quality_assessor = FaceQualityAssessor()
    recognizer = SFaceRecognizer()
    storage = StorageManager()
    db = DatabaseManager()

    if clear_existing:
        print("[*] Clearing existing employees and embeddings from database...")
        db.clear_employees()
        print("[+] Existing employees cleared.\n")

    staff_folders = sorted([f for f in staffs_root.iterdir() if f.is_dir()])
    if not staff_folders:
        print("[!] No subdirectories found in Staffs folder.")
        return

    enrolled_summary = []

    for staff_dir in staff_folders:
        if "_" in staff_dir.name:
            parts = staff_dir.name.split("_", 1)
            emp_id = parts[0].strip()
            raw_name = parts[1].replace("_", " ").strip()
        else:
            raw_name = staff_dir.name.replace("_", " ")
            emp_id = staff_dir.name.upper()

        image_files = sorted(
            [f for f in staff_dir.glob("*") if f.suffix.lower() in [".jpeg", ".jpg", ".png", ".bmp", ".webp"]]
        )

        if not image_files:
            print(f"[!] Skipping {staff_dir.name}: No image files found.")
            continue

        print(f"\n[*] Processing [{staff_dir.name}] ({len(image_files)} photos)...")
        valid_embeddings = []
        valid_records = []

        best_frontal_score = -1.0
        primary_record = None

        for idx, img_path in enumerate(image_files, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [-] Image {img_path.name}: Unable to read file.")
                continue

            faces = detector.detect(img)
            if len(faces) == 0:
                print(f"    [-] Image {img_path.name}: No face detected.")
                continue

            # Pick largest face
            face_data = max(faces, key=lambda f: f[2] * f[3])
            w, h = face_data[2], face_data[3]
            quality = quality_assessor.assess(img, face_data)

            # Align face to 112x112 chip
            aligned_face = recognizer.align_crop(img, face_data)
            embedding = recognizer.extract_embedding(aligned_face)

            # Save individual embedding and aligned crop
            tag = f"{staff_dir.name.lower()}_v{idx}"
            saved = storage.save_face_and_embedding(
                aligned_face=aligned_face,
                embedding=embedding,
                raw_crop=quality.raw_crop,
                camera_id="ENROLLMENT_PORTAL",
                metadata={
                    "staff_name": raw_name,
                    "staff_id": emp_id,
                    "photo_source": img_path.name,
                    "blur_score": quality.blur_score,
                    "roll_angle": quality.roll_angle,
                    "yaw_ratio": quality.yaw_ratio,
                    "passed_quality": quality.passed
                }
            )

            # Heuristic for best frontal photo: high blur score + frontal yaw ratio (closer to 1.0)
            frontal_score = (1.0 - abs(quality.yaw_ratio - 1.0) * 0.5) * np.log1p(quality.blur_score)
            if frontal_score > best_frontal_score:
                best_frontal_score = frontal_score
                primary_record = saved

            valid_embeddings.append(embedding)
            valid_records.append(saved)

            status_str = f"\033[92mFrontal (Q-OK)\033[0m" if quality.passed else f"\033[93mAngled/Side\033[0m (Yaw: {quality.yaw_ratio:.2f})"
            print(f"    [+] Photo {idx}: {img_path.name[:25]:<25} | {int(w)}x{int(h)}px | {status_str} -> {Path(saved['embedding_path']).name}")

        if not valid_records:
            print(f"    [!] No valid faces could be extracted for {staff_dir.name}.")
            continue

        # Register in SQLite database
        primary = primary_record if primary_record else valid_records[0]
        db.enroll_employee(
            emp_id=emp_id,
            name=raw_name,
            department="Staff",
            photo_path=primary["aligned_image_path"],
            embedding_path=primary["embedding_path"]
        )

        # Register all multi-view embeddings for this staff member
        for rec in valid_records:
            is_pri = 1 if rec == primary else 0
            db.add_employee_embedding(
                emp_id=emp_id,
                embedding_path=rec["embedding_path"],
                photo_path=rec["aligned_image_path"],
                is_primary=is_pri
            )

        # Also compute & save a combined normalized mean centroid embedding for this staff member
        if len(valid_embeddings) > 1:
            mean_vector = np.mean(np.vstack(valid_embeddings), axis=0, keepdims=True)
            norm = np.linalg.norm(mean_vector)
            centroid_emb = (mean_vector / norm).astype(np.float32)

            centroid_saved = storage.save_face_and_embedding(
                aligned_face=cv2.imread(primary["aligned_image_path"]),
                embedding=centroid_emb,
                camera_id="CENTROID_TEMPLATE",
                metadata={"staff_name": raw_name, "staff_id": emp_id, "type": "centroid_average"}
            )
            db.add_employee_embedding(
                emp_id=emp_id,
                embedding_path=centroid_saved["embedding_path"],
                photo_path=primary["aligned_image_path"],
                is_primary=0
            )
            print(f"    [+] Generated Centroid Mean Embedding: {Path(centroid_saved['embedding_path']).name}")

        enrolled_summary.append({
            "id": emp_id,
            "name": raw_name,
            "count": len(valid_records),
            "primary_photo": Path(primary["aligned_image_path"]).name
        })

    print("\n" + "=" * 78)
    print("                     Enrollment Summary")
    print("=" * 78)
    for s in enrolled_summary:
        print(f"  {s['id']:<10} | {s['name']:<16} | {s['count']} Templates Enrolled | Primary: {s['primary_photo']}")
    print("=" * 78)
    print(f"\n[+] Successfully enrolled {len(enrolled_summary)} staff members into VisionGuard AI.")
    print("    All embeddings and SQLite database records are ready for live RTSP recognition!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll staff profiles from Staffs directory")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing database employees before enrollment")
    args = parser.parse_args()
    enroll_all_staffs(clear_existing=not args.no_clear)
