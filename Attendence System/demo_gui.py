import time
import cv2
import numpy as np

from config import (
    RTSP_URL,
    CAMERA_ID,
    TARGET_FPS,
    SAVE_LIVE_CAPTURES_TO_DISK
)
from pipeline import AttendancePipeline

def main():
    print("Initializing GUI Stream Demo...")
    pipeline = AttendancePipeline(auto_start_stream=True, camera_id=CAMERA_ID, target_fps=TARGET_FPS)

    window_name = "AI Face Attendance & Embedding Verification (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps_history = []
    last_proc_time = time.time()
    last_annotated_frame = None

    print("\nControls:")
    print(" [Q] / [ESC] - Quit")
    print(" [S] - Manually save snapshot of current frame")
    print(" [R] - Reload enrolled employees database\n")

    try:
        while True:
            # 1. Grab fresh stream frame
            is_valid, frame, frame_time = pipeline.stream.get_latest_frame()
            if not is_valid or frame is None:
                # Waiting for stream
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Connecting to RTSP Camera Stream...", (380, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow(window_name, blank)
                if cv2.waitKey(30) & 0xFF in [27, ord('q'), ord('Q')]:
                    break
                continue

            # 2. Sample frame at target FPS for detection & embedding
            has_sample, sample_frame, _ = pipeline.stream.get_sampled_frame()
            if has_sample and sample_frame is not None:
                t0 = time.time()
                last_annotated_frame, faces = pipeline.process_frame(sample_frame, save_passed_embeddings=SAVE_LIVE_CAPTURES_TO_DISK)
                proc_duration = time.time() - t0
                fps_history.append(1.0 / max(proc_duration, 1e-4))
                if len(fps_history) > 20:
                    fps_history.pop(0)

            display_frame = last_annotated_frame if last_annotated_frame is not None else frame

            # Overlay stats bar on display frame
            vis = display_frame.copy()
            avg_fps = np.mean(fps_history) if fps_history else TARGET_FPS
            stats_text = f"CAM: {CAMERA_ID} | Sample Rate: {TARGET_FPS:.1f} FPS | Engine: {avg_fps:.1f} FPS | Press 'Q' to Exit"
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (20, 20, 20), -1)
            cv2.putText(vis, stats_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('s'), ord('S')]:
                snap_path = f"captures/snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[+] Saved snapshot to {snap_path}")
            elif key in [ord('r'), ord('R')]:
                pipeline.reload_enrolled_gallery()
                print("[+] Reloaded enrolled database.")

    finally:
        pipeline.stream.stop()
        cv2.destroyAllWindows()
        print("GUI Demo closed.")

if __name__ == "__main__":
    main()
