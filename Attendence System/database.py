import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from config import DB_PATH

class DatabaseManager:
    """Manages SQLite storage for employees, attendance events, and capture logs."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize all required SQLite tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enrolled Employees Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT DEFAULT 'General',
                email TEXT DEFAULT '',
                enrolled_at TEXT NOT NULL,
                photo_path TEXT DEFAULT '',
                embedding_path TEXT DEFAULT ''
            )
            """)

            # Multi-view Employee Embeddings Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS employee_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                embedding_path TEXT NOT NULL,
                photo_path TEXT DEFAULT '',
                is_primary INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employees(id)
            )
            """)

            # Attendance Log Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                employee_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                face_image_path TEXT,
                embedding_path TEXT,
                FOREIGN KEY(employee_id) REFERENCES employees(id)
            )
            """)

            # Raw Face Captures & Quality Audits Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_captures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                bbox_w REAL,
                bbox_h REAL,
                blur_score REAL,
                roll_angle REAL,
                yaw_ratio REAL,
                passed_quality INTEGER NOT NULL,
                rejection_reason TEXT DEFAULT '',
                match_name TEXT DEFAULT 'UNKNOWN',
                match_score REAL DEFAULT 0.0,
                face_image_path TEXT,
                embedding_path TEXT
            )
            """)
            conn.commit()

    def enroll_employee(self, emp_id: str, name: str, department: str,
                        photo_path: str, embedding_path: str, email: str = "") -> bool:
        """Add or update an enrolled employee."""
        now = datetime.datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO employees (id, name, department, email, enrolled_at, photo_path, embedding_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                department=excluded.department,
                email=excluded.email,
                enrolled_at=excluded.enrolled_at,
                photo_path=excluded.photo_path,
                embedding_path=excluded.embedding_path
            """, (emp_id, name, department, email, now, photo_path, embedding_path))
            conn.commit()
            return True

    def add_employee_embedding(self, emp_id: str, embedding_path: str, photo_path: str = "", is_primary: int = 0) -> int:
        """Add an additional reference embedding for an enrolled employee."""
        now = datetime.datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO employee_embeddings (employee_id, embedding_path, photo_path, is_primary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (emp_id, embedding_path, photo_path, is_primary, now))
            conn.commit()
            return cursor.lastrowid

    def get_employee_embeddings(self, emp_id: str) -> List[Dict[str, Any]]:
        """Retrieve all reference embeddings associated with an employee."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employee_embeddings WHERE employee_id = ? ORDER BY is_primary DESC, id ASC", (emp_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_employees(self) -> List[Dict[str, Any]]:
        """Retrieve all enrolled employees."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employees ORDER BY enrolled_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_employees_with_attendance_status(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all enrolled employees along with their confirmation status for a given day (defaults to today).
        """
        if not target_date:
            target_date = datetime.date.today().isoformat()  # 'YYYY-MM-DD'

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employees ORDER BY name ASC")
            employees = [dict(row) for row in cursor.fetchall()]

            # Fetch today's confirmed events
            cursor.execute("""
                SELECT employee_id, MAX(timestamp) as latest_time, confidence, status, COUNT(*) as punches_today
                FROM attendance_events
                WHERE timestamp LIKE ? AND status = 'CONFIRMED'
                GROUP BY employee_id
            """, (f"{target_date}%",))
            attendance_map = {row["employee_id"]: dict(row) for row in cursor.fetchall()}

            results = []
            for emp in employees:
                emp_data = dict(emp)
                att = attendance_map.get(emp["id"])
                if att:
                    emp_data["attended_today"] = True
                    emp_data["last_attendance_time"] = att["latest_time"]
                    emp_data["last_confidence"] = float(att["confidence"])
                    emp_data["punches_today"] = att["punches_today"]
                else:
                    emp_data["attended_today"] = False
                    emp_data["last_attendance_time"] = None
                    emp_data["last_confidence"] = None
                    emp_data["punches_today"] = 0
                results.append(emp_data)
            return results

    def get_employee(self, emp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single employee by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employees WHERE id = ?", (emp_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_employee(self, emp_id: str) -> bool:
        """Delete employee and associated reference embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM employee_embeddings WHERE employee_id = ?", (emp_id,))
            cursor.execute("DELETE FROM employees WHERE id = ?", (emp_id,))
            conn.commit()
            return True

    def clear_employees(self) -> bool:
        """Clear all employees and reference embeddings from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM employee_embeddings")
            cursor.execute("DELETE FROM employees")
            conn.commit()
            return True

    def log_attendance(self, emp_id: str, name: str, camera_id: str,
                       confidence: float, status: str,
                       face_image_path: str = "", embedding_path: str = "") -> int:
        """Record an attendance confirmation event."""
        now = datetime.datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO attendance_events 
            (employee_id, employee_name, timestamp, camera_id, confidence, status, face_image_path, embedding_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (emp_id, name, now, camera_id, float(confidence), status, face_image_path, embedding_path))
            conn.commit()
            return cursor.lastrowid

    def log_face_capture(self, camera_id: str, bbox_w: float, bbox_h: float,
                         blur_score: float, roll_angle: float, yaw_ratio: float,
                         passed_quality: bool, rejection_reason: str,
                         match_name: str, match_score: float,
                         face_image_path: str, embedding_path: str) -> int:
        """Audit log for each detected face and quality metrics."""
        now = datetime.datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO face_captures
            (timestamp, camera_id, bbox_w, bbox_h, blur_score, roll_angle, yaw_ratio, 
             passed_quality, rejection_reason, match_name, match_score, face_image_path, embedding_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, camera_id, bbox_w, bbox_h, blur_score, roll_angle, yaw_ratio,
                  1 if passed_quality else 0, rejection_reason, match_name,
                  float(match_score), face_image_path, embedding_path))
            conn.commit()
            return cursor.lastrowid

    def get_recent_attendance(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch latest attendance records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM attendance_events ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_captures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch latest face capture audits."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM face_captures ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for monitoring."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM employees")
            total_employees = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM attendance_events")
            total_attendance = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM face_captures")
            total_captures = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM face_captures WHERE passed_quality = 1")
            passed_quality = cursor.fetchone()[0]

            return {
                "total_employees": total_employees,
                "total_attendance": total_attendance,
                "total_captures": total_captures,
                "passed_quality": passed_quality,
                "quality_pass_rate": round(passed_quality / total_captures * 100, 1) if total_captures > 0 else 0.0
            }
