import time
import argparse
import sys
from pathlib import Path

from config import (
    RTSP_URL,
    CAMERA_ID,
    CAMERA_NAME,
    TARGET_FPS,
    EMBEDDINGS_DIR,
    FACES_DIR,
    DB_PATH
)
from pipeline import AttendancePipeline

def main():
    parser = argparse.ArgumentParser(description="Live RTSP Face Detection, Quality Filter & SFace Embedding Demo")
    parser.add_argument("--url", type=str, default=RTSP_URL, help="RTSP Camera URL")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="Frame sampling rate (2.0 - 5.0 FPS)")
    parser.add_argument("--seconds", type=int, default=30, help="Run duration in seconds (0 = run indefinitely)")
    parser.add_argument("--save-all", action="store_true", default=True, help="Store embeddings of all quality-passed faces")
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("   AI Face Attendance System - Live RTSP Capture & Embedding Demo")
    print("=" * 76)
    print(f" Camera Stream : {args.url}")
    print(f" Sampling Rate : {args.fps:.1f} FPS (Sample interval: {1.0 / args.fps:.2f}s)")
    print(f" Models Loaded : YuNet (Face & 5 Landmarks) + SFace (128-d Embeddings)")
    print(f" Storage Path  : Faces -> '{FACES_DIR}', Embeddings -> '{EMBEDDINGS_DIR}'")
    print(f" SQLite DB     : '{DB_PATH}'")
    print("=" * 76)
    print(" Connecting to RTSP stream... Please wait.\n")

    pipeline = AttendancePipeline(auto_start_stream=True, camera_id=CAMERA_ID, target_fps=args.fps)

    # Wait for connection
    retries = 0
    while not pipeline.stream.is_connected and retries < 15:
        time.sleep(0.5)
        retries += 1

    if not pipeline.stream.is_connected:
        print("[!] Camera connection could not be established immediately. Checking live stream...")

    start_time = time.time()
    sampled_frame_count = 0
    total_faces_detected = 0
    total_quality_passed = 0
    total_embeddings_stored = 0

    print(f"[+] RTSP Stream Connected! Running capture loop (Duration: {args.seconds}s, Ctrl+C to stop)...\n")
    print(f"{'Time':<10} | {'Faces':<6} | {'Status & Details':<45} | {'Saved Files'}")
    print("-" * 88)

    try:
        while True:
            # Check elapsed time
            elapsed = time.time() - start_time
            if args.seconds > 0 and elapsed >= args.seconds:
                print(f"\n[*] Demo duration of {args.seconds}s reached.")
                break

            # 2. Frame Sampling at 2–5 FPS
            has_sample, frame, frame_time = pipeline.stream.get_sampled_frame()
            if not has_sample or frame is None:
                time.sleep(0.01)
                continue

            sampled_frame_count += 1
            timestamp_str = time.strftime("%H:%M:%S")

            # 3-8. Full pipeline processing
            annotated_frame, faces = pipeline.process_frame(frame, save_passed_embeddings=args.save_all)
            num_faces = len(faces)
            total_faces_detected += num_faces

            if num_faces == 0:
                # No face detected in this sampled frame
                if sampled_frame_count % 5 == 0:
                    print(f"{timestamp_str:<10} | {0:<6} | No face in frame (scanning...)                | -")
            else:
                for idx, face in enumerate(faces):
                    passed = face["passed_quality"]
                    w, h = face["bbox"][2], face["bbox"][3]
                    conf = face["confidence"]

                    if passed:
                        total_quality_passed += 1
                        name = face["matched_name"]
                        sim = face["similarity_score"]
                        temp = face["temporal_status"]
                        saved = face.get("saved_files", {})
                        emb_file = Path(saved.get("embedding_path", "")).name if saved else "N/A"
                        total_embeddings_stored += 1

                        status_msg = f"\033[92mPASS\033[0m: {w}x{h}px | Match: {name} ({sim:.2f}) | {temp}"
                        print(f"{timestamp_str:<10} | {idx+1}/{num_faces:<4} | {status_msg:<54} | {emb_file}")
                    else:
                        reasons = ", ".join(face["rejection_reasons"])
                        status_msg = f"\033[93mREJECT\033[0m: {w}x{h}px | {reasons[:38]}"
                        print(f"{timestamp_str:<10} | {idx+1}/{num_faces:<4} | {status_msg:<54} | (Filtered out)")

    except KeyboardInterrupt:
        print("\n\n[!] User interrupted demo.")
    finally:
        pipeline.stream.stop()

    print("\n" + "=" * 76)
    print("                     Capture & Embedding Summary")
    print("=" * 76)
    print(f" Sampled Frames Processed   : {sampled_frame_count}")
    print(f" Total Raw Faces Detected   : {total_faces_detected}")
    print(f" Quality Passed Faces       : {total_quality_passed}")
    print(f" SFace Embeddings Stored    : {total_embeddings_stored}")
    print(f" Embeddings Saved In        : {EMBEDDINGS_DIR.resolve()}")
    print(f" Aligned Face Images In     : {FACES_DIR.resolve()}")
    print(f" SQLite Attendance Database : {DB_PATH.resolve()}")
    print("=" * 76)
    print(" Next Step: Verify stored embeddings using:")
    print("   python verify_embeddings.py --list")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    main()
