"""Ultra-light periodic health logger (one line every `interval_sec`)."""
import logging
import time

import psutil

logger = logging.getLogger("Health")


class LightweightMonitor:
    def __init__(self, interval_sec: float = 30.0):
        self.interval = interval_sec
        self.start_time = time.time()
        self._last = 0.0
        self.process = psutil.Process()
        self.process.cpu_percent(None)  # prime
        self.error_count = 0
        self.detection_total = 0

    def add_detections(self, n: int) -> None:
        self.detection_total += n

    def add_error(self) -> None:
        self.error_count += 1

    def update(self, ai_fps: float, inf_ms: float, reconnects: int, camera_online: bool) -> None:
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now
        try:
            ram = self.process.memory_info().rss / 1048576
            cpu = self.process.cpu_percent(None)
        except Exception:
            ram, cpu = 0.0, 0.0
        msg = (
            f"uptime={int(now - self.start_time)}s ai_fps={ai_fps:.1f} inf_ms={inf_ms:.0f} "
            f"ram={ram:.0f}MB cpu={cpu:.0f}% cam={'ON' if camera_online else 'OFF'} "
            f"reconnects={reconnects} detections={self.detection_total} errors={self.error_count}"
        )
        logger.info(msg)
        print(f"[Health] {msg}")
