import argparse
import sys
import time
from pathlib import Path
import cv2

from pipeline import AttendancePipeline
from config import CAMERA_ID

def enroll_from_camera(emp_id: str, name: str, department: str = "General", email: str = ""):
    """Captures a clear frame from the RTSP stream and enrolls the detected face."""
    print("\nConnecting to RTSP stream for live face enrollment...")
    pipeline = AttendancePipeline(auto_start_stream=True, camera_id=CAMERA_ID)

    # Wait for stream
    retries = 0
    while not pipeline.stream.is_connected and retries < 15:
        time.sleep(0.5)
        retries += 1

    if not pipeline.stream.is_connected:
        print("[!] Camera connection failed.")
        return False

    print("Please look directly at the camera. Capturing candidate frames for 5 seconds...")
    best_candidate = None
    best_score = -1.0
    start_time = time.time()

    while time.time() - start_time < 5.0:
        ret, frame, _ = pipeline.stream.get_latest_frame()
        if ret and frame is not None:
            faces = pipeline.detector.detect(frame)
            if len(faces) > 0:
                for f in faces:
                    q = pipeline.quality_assessor.assess(frame, f)
                    if q.passed:
                        score = q.blur_score * (f[2] * f[3])
                        if score > best_score:
                            best_score = score
                            best_candidate = (frame.copy(), f)
        time.sleep(0.1)

    pipeline.stream.stop()

    if best_candidate is None:
        print("[!] No face passing quality filters was detected. Ensure good lighting and frontal orientation.")
        return False

    frame, face = best_candidate
    aligned = pipeline.recognizer.align_crop(frame, face)
    embedding = pipeline.recognizer.extract_embedding(aligned)

    saved = pipeline.storage.save_face_and_embedding(
        aligned_face=aligned,
        embedding=embedding,
        raw_crop=pipeline.quality_assessor.assess(frame, face).raw_crop,
        camera_id="LIVE_ENROLLMENT",
        metadata={"enrolled_name": name, "emp_id": emp_id}
    )

    pipeline.db.enroll_employee(
        emp_id=emp_id,
        name=name,
        department=department,
        photo_path=saved["aligned_image_path"],
        embedding_path=saved["embedding_path"],
        email=email
    )

    print(f"\n[+] Successfully enrolled '{name}' (ID: {emp_id})!")
    print(f"    Aligned Face : {saved['aligned_image_path']}")
    print(f"    Embedding    : {saved['embedding_path']}\n")
    return True

def enroll_from_file(image_path: str, emp_id: str, name: str, department: str = "General", email: str = ""):
    pipeline = AttendancePipeline(auto_start_stream=False)
    success, msg = pipeline.enroll_face_from_image(
        image_path=image_path,
        emp_id=emp_id,
        name=name,
        department=department,
        email=email
    )
    if success:
        print(f"\n[+] {msg}\n")
    else:
        print(f"\n[!] Enrollment failed: {msg}\n")
    return success

def main():
    parser = argparse.ArgumentParser(description="Enroll Employee Face Profile into Database")
    parser.add_argument("--id", type=str, required=True, help="Employee ID (e.g. EMP001)")
    parser.add_argument("--name", type=str, required=True, help="Full Name (e.g. Alice Johnson)")
    parser.add_argument("--dept", type=str, default="Engineering", help="Department")
    parser.add_argument("--email", type=str, default="", help="Email address")
    parser.add_argument("--image", type=str, help="Path to portrait photo file (if omitted, captures from live RTSP)")

    args = parser.parse_args()

    if args.image:
        enroll_from_file(args.image, args.id, args.name, args.dept, args.email)
    else:
        enroll_from_camera(args.id, args.name, args.dept, args.email)

if __name__ == "__main__":
    main()
