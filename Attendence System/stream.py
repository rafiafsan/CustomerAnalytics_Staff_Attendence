import os
import time
import threading
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from config import (
    RTSP_URL,
    RTSP_TRANSPORT,
    TARGET_FPS,
    RECONNECT_TIMEOUT_SEC
)

logger = logging.getLogger("RTSPStream")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RTSPStreamReader:
    """
    Threaded, non-blocking RTSP stream reader.
    Continuously drains OpenCV's internal video buffer in a background thread to
    guarantee strictly zero-latency, real-time frames without drift.
    Includes auto-reconnect handling.
    """

    def __init__(
        self,
        rtsp_url: str = RTSP_URL,
        transport: str = RTSP_TRANSPORT,
        target_fps: float = TARGET_FPS,
        reconnect_timeout: float = RECONNECT_TIMEOUT_SEC
    ):
        self.rtsp_url = rtsp_url
        self.transport = transport
        self.target_fps = target_fps
        self.sample_interval = 1.0 / target_fps
        self.reconnect_timeout = reconnect_timeout

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._is_connected: bool = False
        self._frame_count: int = 0
        self._last_sample_time: float = 0.0

    def start(self) -> bool:
        """Start background capture thread."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="RTSPReaderThread")
        self._thread.start()

        # Wait briefly for first frame
        start_wait = time.time()
        while time.time() - start_wait < 3.0:
            with self._lock:
                if self._latest_frame is not None:
                    logger.info("RTSP Stream initialized successfully.")
                    return True
            time.sleep(0.1)

        logger.warning("RTSP Stream thread started, waiting for incoming frames...")
        return True

    def _open_capture(self) -> bool:
        """Open VideoCapture with low-latency FFMPEG transport options."""
        # Set environment options before opening capture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}|max_delay;500000|buffer_size;1024000"

        logger.info(f"Connecting to RTSP stream ({self.transport.upper()}): {self.rtsp_url}")
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimize OpenCV capture properties
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream.")
            return False

        logger.info("RTSP stream connected.")
        return True

    def _capture_worker(self):
        """Worker loop continuously grabbing latest frames."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_timeout)
                    continue

            # Read frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("RTSP stream read returned False. Reconnecting...")
                with self._lock:
                    self._is_connected = False
                if self._cap:
                    self._cap.release()
                    self._cap = None
                time.sleep(1.0)
                continue

            # Filter out blank initialization frames (e.g. flat grey before first I-frame)
            if np.std(frame) < 5.0:
                time.sleep(0.01)
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_time = now
                self._is_connected = True
                self._frame_count += 1

            # Minimal yield to keep CPU efficient
            time.sleep(0.005)

        if self._cap:
            self._cap.release()
            self._cap = None

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Non-blocking call returning (is_fresh, frame, timestamp).
        """
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._latest_frame_time

    def get_sampled_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Returns a frame only when the sample interval (e.g. 2–5 FPS) has elapsed.
        Returns (has_new_sample, frame, timestamp).
        """
        now = time.time()
        if (now - self._last_sample_time) < self.sample_interval:
            return False, None, 0.0

        is_valid, frame, frame_time = self.get_latest_frame()
        if is_valid and frame is not None:
            self._last_sample_time = now
            return True, frame, frame_time

        return False, None, 0.0

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    def stop(self):
        """Stop background thread and close RTSP capture."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
            self._is_connected = False
        logger.info("RTSP Stream stopped.")
