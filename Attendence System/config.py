import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"
STAFFS_DIR = BASE_DIR / "Staffs"
DB_PATH = DATA_DIR / "attendance.db"

# Create required directories
for directory in [DATA_DIR, FACES_DIR, EMBEDDINGS_DIR, MODELS_DIR, STAFFS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 1. RTSP Camera Settings
RTSP_URL = os.getenv("RTSP_URL", "rtsp://ctrl:ctl%40123456@10.1.56.11:554/Streaming/Channels/1201")
CAMERA_ID = "CAM_01_MAIN"
CAMERA_NAME = "Main Entrance Camera"
RTSP_TRANSPORT = "tcp"  # 'tcp' or 'udp'
RECONNECT_TIMEOUT_SEC = 5.0

# 2. Frame Sampling Settings
# Sample at 2–5 FPS to keep processing efficient while capturing all motion
TARGET_FPS = 3.0  # Frames per second to sample (2.0 - 5.0)
FRAME_INTERVAL_SEC = 1.0 / TARGET_FPS

# 3. Face Detection (YuNet) Settings
YUNET_MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
YUNET_CONFIDENCE_THRESHOLD = 0.60
YUNET_NMS_THRESHOLD = 0.30
DETECTION_INPUT_SIZE = (1920, 1080)

# 4. Quality Filtering & Alignment Settings
# Reject blurry, tiny or poorly oriented faces
MIN_FACE_WIDTH = int(os.getenv("MIN_FACE_WIDTH", "25"))          # Minimum width in pixels
MIN_FACE_HEIGHT = int(os.getenv("MIN_FACE_HEIGHT", "25"))        # Minimum height in pixels
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "50.0"))      # Laplacian variance threshold
MAX_ROLL_DEGREES = float(os.getenv("MAX_ROLL_DEGREES", "30.0"))  # Maximum head tilt in degrees
MIN_YAW_RATIO = float(os.getenv("MIN_YAW_RATIO", "0.35"))        # Ratio min(eye_nose_dist)/max(eye_nose_dist)
BORDER_MARGIN_PX = int(os.getenv("BORDER_MARGIN_PX", "8"))       # Distance from frame edge

# 5. Face Recognition (SFace) Settings
SFACE_MODEL_PATH = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
# OpenCV benchmark recommendation for SFace cosine similarity is ~0.363
# Threshold >= 0.38 provides robust recognition while minimizing false accepts
COSINE_SIMILARITY_THRESHOLD = 0.380
L2_DISTANCE_THRESHOLD = 1.128

# 6. Temporal Confirmation Settings
REQUIRED_CONSECUTIVE_MATCHES = 3   # Must match 3 times across sampled frames
CONFIRMATION_WINDOW_SEC = 4.0      # Within a 4-second time window
ATTENDANCE_COOLDOWN_SEC = 300.0    # 5 minutes cooldown before logging same person again

# 7. Web Dashboard Settings
WEB_HOST = "127.0.0.1"
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

# 8. Live RTSP Capture Logging
# False = Keep live streaming in RAM memory (stops filling disk space).
# True  = Save every camera frame face crop and embedding to disk.
SAVE_LIVE_CAPTURES_TO_DISK = False
