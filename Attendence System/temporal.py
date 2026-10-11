import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config import (
    REQUIRED_CONSECUTIVE_MATCHES,
    CONFIRMATION_WINDOW_SEC,
    ATTENDANCE_COOLDOWN_SEC
)

@dataclass
class MatchObservation:
    timestamp: float
    confidence: float
    camera_id: str

class TemporalConfirmationTracker:
    """
    Tracks face recognition matches across consecutive sampled frames.
    Confirms identity only after repeated consistent detections within a time window.
    Applies cooldown to prevent duplicate attendance marks.
    """

    def __init__(
        self,
        required_matches: int = REQUIRED_CONSECUTIVE_MATCHES,
        window_sec: float = CONFIRMATION_WINDOW_SEC,
        cooldown_sec: float = ATTENDANCE_COOLDOWN_SEC
    ):
        self.required_matches = required_matches
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec

        # emp_id -> List[MatchObservation]
        self._history: Dict[str, List[MatchObservation]] = {}
        # emp_id -> last_confirmed_timestamp
        self._last_confirmed: Dict[str, float] = {}

    def update(
        self,
        emp_id: Optional[str],
        emp_name: str,
        confidence: float,
        camera_id: str
    ) -> Tuple[bool, str, int]:
        """
        Update tracker with a frame observation.
        
        Returns:
            (is_confirmed_now, status_message, streak_count)
            status_message can be:
            - 'CONFIRMED': Attendance marked right now!
            - 'IN_PROGRESS': Matching in progress (streak < required)
            - 'COOLDOWN': Already recorded recently
            - 'UNCONFIRMED_UNKNOWN': Unknown or unverified face
        """
        now = time.time()

        if not emp_id or emp_name == "UNKNOWN":
            return False, "UNKNOWN", 0

        # Check cooldown
        last_time = self._last_confirmed.get(emp_id, 0.0)
        if (now - last_time) < self.cooldown_sec:
            remaining = int(self.cooldown_sec - (now - last_time))
            return False, f"COOLDOWN ({remaining}s)", 0

        # Maintain observation list within sliding time window
        if emp_id not in self._history:
            self._history[emp_id] = []

        # Prune entries outside time window
        self._history[emp_id] = [
            obs for obs in self._history[emp_id]
            if (now - obs.timestamp) <= self.window_sec
        ]

        # Add current match
        self._history[emp_id].append(MatchObservation(now, confidence, camera_id))
        streak_count = len(self._history[emp_id])

        if streak_count >= self.required_matches:
            # Trigger confirmation!
            self._last_confirmed[emp_id] = now
            self._history[emp_id] = []  # reset window after confirmation
            return True, "CONFIRMED", streak_count

        return False, f"CONFIRMING ({streak_count}/{self.required_matches})", streak_count

    def get_streak(self, emp_id: str) -> int:
        now = time.time()
        obs = self._history.get(emp_id, [])
        valid = [o for o in obs if (now - o.timestamp) <= self.window_sec]
        return len(valid)

    def is_in_cooldown(self, emp_id: str) -> bool:
        now = time.time()
        last_time = self._last_confirmed.get(emp_id, 0.0)
        return (now - last_time) < self.cooldown_sec
