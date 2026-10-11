import asyncio
import io
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    BASE_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    STAFFS_DIR,
    CAMERA_ID,
    CAMERA_NAME,
    TARGET_FPS,
    WEB_HOST,
    WEB_PORT,
    SAVE_LIVE_CAPTURES_TO_DISK
)
from pipeline import AttendancePipeline

pipeline: Optional[AttendancePipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = AttendancePipeline(auto_start_stream=True, camera_id=CAMERA_ID, target_fps=TARGET_FPS)
    yield
    if pipeline and pipeline.stream:
        pipeline.stream.stop()

app = FastAPI(title="AI Face Attendance System", lifespan=lifespan)

# Mount static and templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/faces", StaticFiles(directory=str(FACES_DIR)), name="faces")
app.mount("/staffs_media", StaticFiles(directory=str(STAFFS_DIR)), name="staffs_media")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "camera_id": CAMERA_ID,
        "camera_name": CAMERA_NAME,
        "target_fps": TARGET_FPS
    })

def generate_video_frames():
    """Generator yielding annotated MJPEG video frames for the web monitor."""
    global pipeline
    last_annotated = None
    while True:
        if pipeline is None:
            time.sleep(0.1)
            continue

        # Get latest sampled frame for detection
        has_sample, sample_frame, _ = pipeline.stream.get_sampled_frame()
        if has_sample and sample_frame is not None:
            last_annotated, _ = pipeline.process_frame(sample_frame, save_passed_embeddings=SAVE_LIVE_CAPTURES_TO_DISK)

        # Get latest video frame for smooth display
        is_fresh, raw_frame, _ = pipeline.stream.get_latest_frame()
        if not is_fresh or raw_frame is None:
            # Placeholder frame
            placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Connecting to RTSP Camera...", (250, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            ret, jpeg = cv2.imencode(".jpg", placeholder)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        frame_to_show = last_annotated if last_annotated is not None else raw_frame
        
        # Resize for smooth web streaming
        resized = cv2.resize(frame_to_show, (960, 540))
        ret, jpeg = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

        time.sleep(0.04)  # ~25 FPS stream delivery

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/status")
def get_status():
    global pipeline
    if pipeline is None:
        return {"status": "offline"}

    db_stats = pipeline.db.get_stats()
    return {
        "status": "online" if pipeline.stream.is_connected else "connecting",
        "camera_id": CAMERA_ID,
        "camera_name": CAMERA_NAME,
        "target_fps": TARGET_FPS,
        "enrolled_employees": len(pipeline.enrolled_gallery),
        "total_attendance": db_stats.get("total_attendance", 0),
        "total_captures": db_stats.get("total_captures", 0),
        "passed_quality": db_stats.get("passed_quality", 0),
        "quality_pass_rate": db_stats.get("quality_pass_rate", 0.0)
    }

@app.get("/api/attendance")
def get_attendance():
    global pipeline
    if pipeline is None:
        return []
    return pipeline.db.get_recent_attendance(limit=25)

@app.get("/api/captures")
def get_captures():
    global pipeline
    if pipeline is None:
        return []
    return pipeline.db.get_recent_captures(limit=25)

@app.get("/api/employees")
def get_employees():
    global pipeline
    if pipeline is None:
        return []
    employees = pipeline.db.get_employees_with_attendance_status()
    for emp in employees:
        photo_path = emp.get("photo_path", "")
        if photo_path:
            p = Path(photo_path)
            if "faces" in p.parts:
                emp["avatar_url"] = f"/faces/{p.name}"
            elif "Staffs" in p.parts:
                try:
                    rel = p.relative_to(STAFFS_DIR)
                    emp["avatar_url"] = f"/staffs_media/{rel.as_posix()}"
                except Exception:
                    emp["avatar_url"] = f"/faces/{p.name}"
            else:
                emp["avatar_url"] = f"/faces/{p.name}"
        else:
            emp["avatar_url"] = ""
    return employees

@app.get("/api/embeddings")
def get_embeddings():
    global pipeline
    if pipeline is None:
        return []
    return pipeline.storage.list_saved_embeddings()[:30]

@app.post("/api/verify")
async def verify_embeddings_api(payload: dict):
    """Compare two embeddings provided by file path or IDs."""
    global pipeline
    file1 = payload.get("file1")
    file2 = payload.get("file2")

    if not file1 or not file2:
        return JSONResponse({"error": "Both file1 and file2 are required"}, status_code=400)

    try:
        emb1 = pipeline.storage.load_embedding(file1)
        emb2 = pipeline.storage.load_embedding(file2)
        sim = pipeline.recognizer.compute_similarity(emb1, emb2)
        l2_dist = pipeline.recognizer.compute_l2_distance(emb1, emb2)
        is_match = sim >= pipeline.recognizer.cosine_threshold

        return {
            "cosine_similarity": round(float(sim), 4),
            "l2_distance": round(float(l2_dist), 4),
            "threshold": pipeline.recognizer.cosine_threshold,
            "is_match": is_match,
            "confidence_percent": round(max(0.0, min(100.0, float(sim) * 100)), 1)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/enroll")
async def enroll_api(
    emp_id: str = Form(...),
    name: str = Form(...),
    department: str = Form("General"),
    email: str = Form(""),
    photo1: Optional[UploadFile] = File(None),
    photo2: Optional[UploadFile] = File(None),
    photo3: Optional[UploadFile] = File(None),
    photos: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None)
):
    global pipeline
    if pipeline is None:
        return JSONResponse({"error": "Pipeline offline"}, status_code=500)

    # Gather all uploaded photos
    collected_photos = []
    for p_upload in [photo1, photo2, photo3]:
        if p_upload and p_upload.filename:
            content = await p_upload.read()
            if len(content) > 0:
                collected_photos.append((p_upload.filename, content))

    if photos:
        for p_upload in photos:
            if p_upload and p_upload.filename:
                content = await p_upload.read()
                if len(content) > 0:
                    collected_photos.append((p_upload.filename, content))

    if not collected_photos and file and file.filename:
        content = await file.read()
        if len(content) > 0:
            collected_photos.append((file.filename, content))

    if collected_photos:
        success, msg, info = await asyncio.to_thread(
            pipeline.enroll_staff_from_photos,
            staff_id=emp_id,
            name=name,
            department=department,
            email=email,
            photos=collected_photos
        )
        if success:
            return {"success": True, "message": msg, "data": info}
        else:
            return JSONResponse({"error": msg}, status_code=400)
    else:
        # Fallback to current RTSP live frame
        is_fresh, frame, _ = pipeline.stream.get_latest_frame()
        if not is_fresh or frame is None:
            return JSONResponse({"error": "No camera frame or photos provided"}, status_code=400)

        temp_path = BASE_DIR / "data" / f"temp_cam_{emp_id}.jpg"
        cv2.imwrite(str(temp_path), frame)
        success, msg = await asyncio.to_thread(
            pipeline.enroll_face_from_image,
            image_path=str(temp_path),
            emp_id=emp_id,
            name=name,
            department=department,
            email=email
        )
        temp_path.unlink(missing_ok=True)

        if success:
            return {"success": True, "message": msg}
        else:
            return JSONResponse({"error": msg}, status_code=400)

@app.delete("/api/employees/{emp_id}")
def delete_employee_api(emp_id: str):
    global pipeline
    if pipeline is None:
        return JSONResponse({"error": "Pipeline offline"}, status_code=500)
    success = pipeline.delete_employee(emp_id)
    if success:
        return {"success": True, "message": f"Employee {emp_id} deleted."}
    return JSONResponse({"error": f"Failed to delete employee {emp_id}"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="VisionGuard Web Dashboard Server")
    parser.add_argument("--port", type=int, default=WEB_PORT, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default=WEB_HOST, help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()
    print(f"\n=================================================================")
    print(f" VisionGuard Web Dashboard starting at: http://{args.host}:{args.port}")
    print(f"=================================================================\n")
    uvicorn.run("web_app:app", host=args.host, port=args.port, reload=False)
