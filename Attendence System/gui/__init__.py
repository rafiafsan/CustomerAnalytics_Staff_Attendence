"""VisionGuard AI — Windows GUI Module
=====================================
Contains the native graphical user interface for real-time face attendance,
embedding verification, employee roster, quality auditing, and 3-photo staff enrollment.
"""

from .attendance_gui import AttendanceApp, launch_gui

__all__ = ["AttendanceApp", "launch_gui"]
