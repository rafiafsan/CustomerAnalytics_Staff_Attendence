"""
Retail AI Analytics - API Gateway Manager
=========================================
Dispatches edge tracking and demographic events to the central FastAPI Gateway.
Uses a non-blocking background queue with automated retry so video inference never drops FPS.
"""

import json
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple
import requests
# Default FastAPI Gateway Endpoint & Credentials.
# API_KEY must come from the environment (or production.yaml); no secret is hardcoded here.
API_BASE_URL = os.getenv("API_GATEWAY_URL", "http://ai.prangroup.com:8021").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
SHOP_ID = int(os.getenv("SHOP_ID", "2"))
SPILL_DB_PATH = os.getenv("SPILL_DB_PATH", "edge_events.db")
SPILL_MAX_ROWS = 200000  # hard cap to protect disk


class ApiManager:
    """Asynchronous event writer that streams tracking events to the FastAPI Gateway."""

    def __init__(
        self,
        credentials_file: Optional[Any] = None,
        enable: bool = True,
        shop_id: Optional[int] = None,
        camera_id: str = "cam_01",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        # Backward-compatibility guard if boolean passed positionally as first arg
        if isinstance(credentials_file, bool):
            enable = credentials_file

        self.enabled = enable
        self.shop_id = shop_id if shop_id is not None else SHOP_ID
        self.camera_id = camera_id
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        # Background non-blocking queue (keeps video processing running at 100% FPS)
        self.job_queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._spill_lock = threading.Lock()
        self._spill_conn: Optional[sqlite3.Connection] = None
        if self.enabled:
            self._init_spill_db()

        if self.enabled:
            is_online = self.check_connection()
            status_str = "ONLINE" if is_online else "OFFLINE (buffering events in queue)"
            print(f"[ApiManager] Gateway: {self.base_url} | Outlet ID: {self.shop_id} | Status: {status_str}")
            if self.base_url.lower().startswith("http://") and not any(
                h in self.base_url for h in ("localhost", "127.0.0.1")
            ):
                print("[ApiManager] SECURITY WARNING: gateway uses cleartext HTTP; API key and data can be intercepted. Use https://.")
            if not self.api_key:
                print("[ApiManager] WARNING: no API key configured (set the API_KEY environment variable).")
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="API-HTTP-WriterThread",
            )
            self._worker_thread.start()

    def check_connection(self) -> bool:
        """Perform immediate health check against the API Gateway."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Local SQLite spill buffer (zero data loss during network outages)
    # -------------------------------------------------------------------------

    def _init_spill_db(self) -> None:
        try:
            conn = sqlite3.connect(SPILL_DB_PATH, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pending_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, method TEXT NOT NULL, "
                "endpoint TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            conn.commit()
            self._spill_conn = conn
        except Exception as exc:
            print(f"[ApiManager] WARNING: spill DB unavailable ({exc}); events will only be retried in memory.")
            self._spill_conn = None

    def _spill(self, method: str, endpoint: str, payload: Dict[str, Any]) -> None:
        if self._spill_conn is None:
            return
        try:
            with self._spill_lock:
                n = self._spill_conn.execute("SELECT COUNT(*) FROM pending_events").fetchone()[0]
                if n >= SPILL_MAX_ROWS:
                    # Drop oldest to bound disk usage
                    self._spill_conn.execute(
                        "DELETE FROM pending_events WHERE id IN (SELECT id FROM pending_events ORDER BY id LIMIT 1000)"
                    )
                self._spill_conn.execute(
                    "INSERT INTO pending_events (method, endpoint, payload) VALUES (?, ?, ?)",
                    (method, endpoint, json.dumps(payload)),
                )
                self._spill_conn.commit()
        except Exception as exc:
            print(f"[ApiManager] Spill write failed: {exc}")

    def _send(self, method: str, endpoint: str, payload: Dict[str, Any]) -> bool:
        """Try to deliver one event. Returns False only for transient failures (retry later)."""
        url = f"{self.base_url}{endpoint}"
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if method == "PATCH":
                    resp = self.session.patch(url, json=payload, headers=self.headers, timeout=5)
                else:
                    resp = self.session.post(url, json=payload, headers=self.headers, timeout=5)

                if resp.status_code in (200, 201):
                    return True
                if resp.status_code >= 500 or resp.status_code in (408, 429):
                    print(f"[ApiManager] Transient HTTP {resp.status_code} on {endpoint}")
                    return False
                # Permanent client error (400/401/404/422...): retrying will not help
                print(f"[ApiManager] Warning: HTTP {resp.status_code} on {endpoint}: {resp.text[:200]}")
                return True
            except Exception as exc:
                if attempt < max_retries and not self._stop_event.is_set():
                    time.sleep(0.5)
                else:
                    print(f"[ApiManager] Network error sending to {url}: {exc}")
        return False

    def _replay_spilled(self, batch: int = 50) -> None:
        """Resend persisted events in order; stop at first transient failure."""
        if self._spill_conn is None:
            return
        try:
            with self._spill_lock:
                rows = self._spill_conn.execute(
                    "SELECT id, method, endpoint, payload FROM pending_events ORDER BY id LIMIT ?", (batch,)
                ).fetchall()
        except Exception:
            return
        if not rows:
            return
        if not self.check_connection():
            return
        for row_id, method, endpoint, payload_str in rows:
            if self._stop_event.is_set():
                break
            try:
                payload = json.loads(payload_str)
            except Exception:
                payload = None
            if payload is not None and not self._send(method, endpoint, payload):
                break
            with self._spill_lock:
                self._spill_conn.execute("DELETE FROM pending_events WHERE id = ?", (row_id,))
                self._spill_conn.commit()

    def _worker_loop(self):
        """Background thread: sends events; failures persist to SQLite and replay when idle."""
        last_replay = 0.0
        while not self._stop_event.is_set() or not self.job_queue.empty():
            try:
                method, endpoint, payload = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if not self._stop_event.is_set() and now - last_replay >= 10.0:
                    last_replay = now
                    self._replay_spilled()
                continue

            try:
                if not self._send(method, endpoint, payload):
                    self._spill(method, endpoint, payload)
            finally:
                self.job_queue.task_done()

    # -------------------------------------------------------------------------
    # Public Event Ingestion Methods:
    # -------------------------------------------------------------------------

    def insert_entry(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "ENTRY",
        age_group: Optional[str] = None,
        gender: Optional[str] = None,
        camera_id: str = "cam_01",
        shop_id: Optional[int] = None,
        **kwargs,
    ):
        """Queue entry event to POST /api/v1/entries."""
        if not self.enabled:
            return
        payload = {
            "track_id": track_id,
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "age_group": age_group or "Unknown",
            "gender": gender or "Unknown",
            "camera_id": camera_id,
            "shop_id": shop_id if shop_id is not None else self.shop_id,
        }
        self.job_queue.put(("POST", "/api/v1/entries", payload))

    def insert_exit(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        direction: str = "EXIT",
        shop_id: Optional[int] = None,
        **kwargs,
    ):
        """Queue exit event to POST /api/v1/exits."""
        if not self.enabled:
            return
        payload = {
            "track_id": track_id,
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "shop_id": shop_id if shop_id is not None else self.shop_id,
        }
        self.job_queue.put(("POST", "/api/v1/exits", payload))

    def insert_demographic(
        self,
        track_id: int,
        timestamp: Optional[str] = None,
        frame_no: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        direction: str = "ENTRY",
        result_dict: Optional[Dict[str, Any]] = None,
        shop_id: Optional[int] = None,
        **kwargs,
    ):
        """Queue demographic inference to POST /api/v1/demographics."""
        if not self.enabled or not result_dict:
            return
        exact_age = result_dict.get("exact_age") or result_dict.get("age")
        try:
            exact_age_int = int(exact_age) if exact_age is not None else None
        except (ValueError, TypeError):
            exact_age_int = None

        payload = {
            "track_id": track_id,
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_no": frame_no,
            "bbox": list(bbox) if bbox else None,
            "direction": direction,
            "age_group": result_dict.get("age_group", result_dict.get("age", "Unknown")),
            "exact_age": exact_age_int,
            "gender": result_dict.get("gender", result_dict.get("apparent_gender", "Unknown")),
            "visibility": result_dict.get("visibility", "Unknown"),
            "confidence": result_dict.get("confidence", "Unknown"),
            "shop_id": shop_id if shop_id is not None else self.shop_id,
        }
        self.job_queue.put(("POST", "/api/v1/demographics", payload))

    def update_entry_demographics(self, track_id: int, age_group: str, gender: str, **kwargs):
        """Queue patch to PATCH /api/v1/entries/demographics."""
        if not self.enabled:
            return
        payload = {
            "track_id": track_id,
            "age_group": age_group,
            "gender": gender,
        }
        self.job_queue.put(("PATCH", "/api/v1/entries/demographics", payload))

    def insert_occupancy(
        self,
        timestamp: Optional[str] = None,
        camera_id: str = "cam_01",
        valid_entries: int = 0,
        valid_exits: int = 0,
        inside_visitors: Optional[int] = None,
        interval_minutes: float = 20.0,
        shop_id: Optional[int] = None,
        **kwargs,
    ):
        """Queue store occupancy snapshot to POST /api/v1/occupancy."""
        if not self.enabled:
            return
        payload = {
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": camera_id,
            "valid_entries": valid_entries,
            "valid_exits": valid_exits,
            "inside_visitors": inside_visitors,
            "interval_minutes": interval_minutes,
            "shop_id": shop_id if shop_id is not None else self.shop_id,
        }
        self.job_queue.put(("POST", "/api/v1/occupancy", payload))

    def upsert_daily_metrics(
        self,
        date_val: Optional[str] = None,
        in_count: int = 0,
        out_count: int = 0,
        person_with_bag_exit: int = 0,
        demographics_count: int = 0,
        shop_id: Optional[int] = None,
        **kwargs,
    ):
        """Queue daily metrics rollup to POST /api/v1/metrics/daily."""
        if not self.enabled:
            return
        payload = {
            "date_val": date_val or date.today().strftime("%Y-%m-%d"),
            "in_count": in_count,
            "out_count": out_count,
            "person_with_bag_exit": person_with_bag_exit,
            "demographics_count": demographics_count,
            "shop_id": shop_id if shop_id is not None else self.shop_id,
        }
        self.job_queue.put(("POST", "/api/v1/metrics/daily", payload))

    def close(self, timeout: float = 5.0, **kwargs):
        """Drain queue on application shutdown."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)


# Backward-compatible alias for codebase modules importing DatabaseManager
DatabaseManager = ApiManager

__all__ = ["ApiManager", "DatabaseManager", "API_BASE_URL", "API_KEY", "SHOP_ID"]
