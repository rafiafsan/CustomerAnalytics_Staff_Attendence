"""VisionGuard AI — Standalone Windows Native GUI Application
============================================================
Full-featured desktop application for real-time edge face attendance,
quality assessment, SFace embedding verification, and staff enrollment.
Zero web browser or external web server required.
"""

import os
import sys
import time
import queue
import logging
import threading
import datetime
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

# Adjust root paths
PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    STAFFS_DIR,
    CAMERA_ID,
    CAMERA_NAME,
    RTSP_URL,
    TARGET_FPS,
    SAVE_LIVE_CAPTURES_TO_DISK,
    COSINE_SIMILARITY_THRESHOLD,
    L2_DISTANCE_THRESHOLD
)
from pipeline import AttendancePipeline

logger = logging.getLogger("AttendanceGUI")

# ==============================================================================
# Modern Dark Theme Palette (Consistent with FootFall_AgeGender)
# ==============================================================================
BG_MAIN = "#0f172a"       # Slate 900
BG_CARD = "#1e293b"       # Slate 800
BG_CARD_LIGHT = "#334155" # Slate 700
BORDER_COLOR = "#475569"  # Slate 600
TEXT_WHITE = "#f8fafc"    # Slate 50
TEXT_MUTED = "#94a3b8"    # Slate 400
ACCENT_BLUE = "#3b82f6"   # Blue 500
ACCENT_HOVER = "#2563eb"  # Blue 600
ACCENT_GREEN = "#10b981"  # Emerald 500
ACCENT_RED = "#ef4444"    # Red 500
ACCENT_AMBER = "#f59e0b"  # Amber 500
ACCENT_PURPLE = "#a855f7" # Purple 500


class GuiLogHandler(logging.Handler):
    """Thread-safe logging handler redirecting logs into Tkinter queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


class AttendanceApp:
    """Main Standalone Native GUI Application for VisionGuard AI."""

    def __init__(self, root: tk.Tk, parent_frame: Optional[tk.Frame] = None):
        self.root = root
        self.is_embedded = parent_frame is not None
        if not self.is_embedded:
            self.root.title("VisionGuard AI — Real-Time Face Attendance & Verification")
            self.root.geometry("1480 Central" if False else "1460x900")
            self.root.minsize(1200, 760)
            self.root.configure(bg=BG_MAIN)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.container = parent_frame if parent_frame is not None else self.root

        # Threading queues & state
        self.log_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = True
        self.stream_paused = False
        self.current_rtsp_url = RTSP_URL

        # Caches
        self.latest_display_frame: Optional[np.ndarray] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.avatar_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.photo_preview_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.roster_filter_mode = "all"  # 'all', 'confirmed', 'pending'
        self.roster_search_query = ""

        # Setup Logging redirection
        self._setup_logging()

        # Initialize Attendance Pipeline
        logger.info("Initializing Attendance Pipeline...")
        self.pipeline = AttendancePipeline(auto_start_stream=True, camera_id=CAMERA_ID, target_fps=TARGET_FPS)

        # Configure TTK Styles
        self._setup_styles()

        # Build Main UI Components
        self._build_header()
        self._build_kpi_row()
        self._build_main_workspace()
        self._build_lower_data_tabs()

        # Start background threads & scheduled timers
        self._start_processing_thread()
        self._schedule_periodic_tasks()
        logger.info("VisionGuard Native GUI initialized successfully.")

    # --------------------------------------------------------------------------
    # Style System
    # --------------------------------------------------------------------------
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Base Frame & Label styles
        self.style.configure(".", background=BG_MAIN, foreground=TEXT_WHITE, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=BG_MAIN)
        self.style.configure("Card.TFrame", background=BG_CARD, relief="flat")
        self.style.configure("CardLight.TFrame", background=BG_CARD_LIGHT, relief="flat")

        # Labels
        self.style.configure("TLabel", background=BG_MAIN, foreground=TEXT_WHITE, font=("Segoe UI", 9))
        self.style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Segoe UI", 9))
        self.style.configure("CardMuted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        self.style.configure("CardHeader.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Segoe UI", 11, "bold"))
        self.style.configure("KpiVal.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Segoe UI", 16, "bold"))
        self.style.configure("KpiSub.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 8))

        # Notebook (Tabs)
        self.style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_MUTED, padding=[14, 6], font=("Segoe UI", 9, "bold"))
        self.style.map("TNotebook.Tab",
                       background=[("selected", ACCENT_BLUE), ("active", BG_CARD_LIGHT)],
                       foreground=[("selected", TEXT_WHITE), ("active", TEXT_WHITE)])

        # Treeview (Data Tables)
        self.style.configure("Treeview",
                             background=BG_CARD,
                             foreground=TEXT_WHITE,
                             fieldbackground=BG_CARD,
                             rowheight=26,
                             font=("Segoe UI", 9),
                             borderwidth=0)
        self.style.map("Treeview", background=[("selected", ACCENT_BLUE)], foreground=[("selected", TEXT_WHITE)])
        self.style.configure("Treeview.Heading",
                             background=BG_CARD_LIGHT,
                             foreground=TEXT_WHITE,
                             font=("Segoe UI", 9, "bold"),
                             relief="flat")
        self.style.map("Treeview.Heading", background=[("active", ACCENT_BLUE)])

        # Combobox
        self.style.configure("TCombobox", fieldbackground=BG_CARD_LIGHT, background=BG_CARD_LIGHT, foreground=TEXT_WHITE)

    # --------------------------------------------------------------------------
    # Logging Configuration
    # --------------------------------------------------------------------------
    def _setup_logging(self):
        handler = GuiLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    # --------------------------------------------------------------------------
    # Header Section
    # --------------------------------------------------------------------------
    def _build_header(self):
        header_frame = tk.Frame(self.container, bg=BG_CARD, height=64, padx=16, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        # Left Brand
        brand_left = tk.Frame(header_frame, bg=BG_CARD)
        brand_left.pack(side=tk.LEFT)

        logo_lbl = tk.Label(brand_left, text="🛡️", font=("Segoe UI Emoji", 18), bg=BG_CARD, fg=ACCENT_BLUE)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        title_box = tk.Frame(brand_left, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(title_box, text="VisionGuard AI", font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(title_box, text="RTSP Face Attendance & SFace Verification System", font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED)
        sub_lbl.pack(anchor="w")

        # Right Actions & Status
        actions_frame = tk.Frame(header_frame, bg=BG_CARD)
        actions_frame.pack(side=tk.RIGHT)

        # Connection Status Pill
        self.status_pill = tk.Label(actions_frame, text="● Connecting...", font=("Segoe UI", 9, "bold"),
                                    bg=BG_CARD_LIGHT, fg=ACCENT_AMBER, padx=10, pady=5)
        self.status_pill.pack(side=tk.LEFT, padx=(0, 10))

        # Register Staff Button
        reg_btn = tk.Button(actions_frame, text="+ Register Staff (3 Photos)", font=("Segoe UI", 9, "bold"),
                            bg=ACCENT_BLUE, fg=TEXT_WHITE, activebackground=ACCENT_HOVER, activeforeground=TEXT_WHITE,
                            relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=self.open_enroll_dialog)
        reg_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Settings / Camera source
        settings_btn = tk.Button(actions_frame, text="⚙️ Camera Settings", font=("Segoe UI", 9),
                                 bg=BG_CARD_LIGHT, fg=TEXT_WHITE, activebackground=BORDER_COLOR, activeforeground=TEXT_WHITE,
                                 relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.open_settings_dialog)
        settings_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Reload Gallery Button
        reload_btn = tk.Button(actions_frame, text="🔄 Reload DB", font=("Segoe UI", 9),
                               bg=BG_CARD_LIGHT, fg=TEXT_WHITE, activebackground=BORDER_COLOR, activeforeground=TEXT_WHITE,
                               relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self.reload_gallery)
        reload_btn.pack(side=tk.LEFT)

    # --------------------------------------------------------------------------
    # KPI Metric Cards
    # --------------------------------------------------------------------------
    def _build_kpi_row(self):
        kpi_frame = tk.Frame(self.container, bg=BG_MAIN, padx=16, pady=10)
        kpi_frame.pack(fill=tk.X, side=tk.TOP)

        self.kpi_cards = {}
        cards_meta = [
            ("fps", "Frame Sampling", f"{TARGET_FPS:.1f} FPS", "YuNet + SFace Engine", ACCENT_BLUE),
            ("captures", "Total Captures", "0", "RTSP Live Stream", ACCENT_PURPLE),
            ("quality", "Quality Pass Rate", "0.0%", "Blur & Pose Filter", ACCENT_GREEN),
            ("enrolled", "Enrolled Staff", "0 Staff", "Gallery DB Profiles", ACCENT_BLUE),
            ("attendance", "Verified Today", "0 Present", "Temporal Confirmed", ACCENT_AMBER),
        ]

        for idx, (cid, title, init_val, sub, color) in enumerate(cards_meta):
            card = tk.Frame(kpi_frame, bg=BG_CARD, padx=12, pady=8, highlightbackground=BORDER_COLOR, highlightthickness=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if idx == 0 else 8, 0))

            top_row = tk.Frame(card, bg=BG_CARD)
            top_row.pack(fill=tk.X)

            lbl_t = tk.Label(top_row, text=title.upper(), font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_MUTED)
            lbl_t.pack(side=tk.LEFT)

            indicator = tk.Label(top_row, text="●", font=("Segoe UI", 9), bg=BG_CARD, fg=color)
            indicator.pack(side=tk.RIGHT)

            val_lbl = tk.Label(card, text=init_val, font=("Segoe UI", 16, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
            val_lbl.pack(anchor="w", pady=(2, 0))

            sub_lbl = tk.Label(card, text=sub, font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED)
            sub_lbl.pack(anchor="w")

            self.kpi_cards[cid] = val_lbl

    # --------------------------------------------------------------------------
    # Main Workspace (Split: Left Video Canvas, Right Roster & Verifier)
    # --------------------------------------------------------------------------
    def _build_main_workspace(self):
        workspace = tk.Frame(self.container, bg=BG_MAIN, padx=16, pady=4)
        workspace.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # 1. Left Video Panel (62% width)
        left_panel = tk.Frame(workspace, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_header = tk.Frame(left_panel, bg=BG_CARD, padx=12, pady=8)
        video_header.pack(fill=tk.X)

        live_tag = tk.Label(video_header, text="● LIVE", font=("Segoe UI", 8, "bold"), bg="#15803d", fg=TEXT_WHITE, padx=6, pady=2)
        live_tag.pack(side=tk.LEFT, padx=(0, 8))

        cam_title = tk.Label(video_header, text=f"{CAMERA_NAME} ({CAMERA_ID})", font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
        cam_title.pack(side=tk.LEFT)

        # Snap button & pause
        snap_btn = tk.Button(video_header, text="📷 Snapshot", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                             relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=self.save_snapshot)
        snap_btn.pack(side=tk.RIGHT, padx=(4, 0))

        self.pause_btn = tk.Button(video_header, text="⏸ Pause", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                                  relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=self.toggle_stream_pause)
        self.pause_btn.pack(side=tk.RIGHT)

        # Canvas for video
        self.video_canvas = tk.Canvas(left_panel, bg="#050811", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Video Footer / Pipeline stages
        stages_bar = tk.Frame(left_panel, bg=BG_CARD, padx=8, pady=6)
        stages_bar.pack(fill=tk.X)

        stages = ["1. RTSP Input", "2. 2-5 FPS Sample", "3. YuNet Detect", "4. Quality Filter", "5. SFace Embed", "6. Gallery Match", "7. Temporal Conf.", "8. SQLite"]
        for stg in stages:
            chip = tk.Label(stages_bar, text=stg, font=("Segoe UI", 7, "bold"), bg=BG_CARD_LIGHT, fg=TEXT_MUTED, padx=5, pady=2)
            chip.pack(side=tk.LEFT, padx=2)

        # 2. Right Panel: Roster & Verification Tester (38% width)
        right_panel = tk.Frame(workspace, bg=BG_MAIN, width=480)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)

        # Right Notebook with 2 tabs: "Staff Roster" and "Embedding Tester"
        self.right_notebook = ttk.Notebook(right_panel)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_roster = tk.Frame(self.right_notebook, bg=BG_CARD, padx=8, pady=8)
        self.tab_verifier = tk.Frame(self.right_notebook, bg=BG_CARD, padx=8, pady=8)

        self.right_notebook.add(self.tab_roster, text="👥 Staff Roster & Attendance")
        self.right_notebook.add(self.tab_verifier, text="🔬 Embedding Verifier")

        self._build_roster_tab()
        self._build_verifier_tab()

    # --------------------------------------------------------------------------
    # Right Tab 1: Staff Roster & Live Attendance
    # --------------------------------------------------------------------------
    def _build_roster_tab(self):
        # Roster Header
        header = tk.Frame(self.tab_roster, bg=BG_CARD)
        header.pack(fill=tk.X, pady=(0, 6))

        self.roster_meta_lbl = tk.Label(header, text="0 Staff Registered", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
        self.roster_meta_lbl.pack(side=tk.LEFT)

        self.roster_present_lbl = tk.Label(header, text="0 Present Today", font=("Segoe UI", 8, "bold"), bg="#15803d", fg=TEXT_WHITE, padx=6, pady=2)
        self.roster_present_lbl.pack(side=tk.RIGHT)

        # Search Bar
        search_frame = tk.Frame(self.tab_roster, bg=BG_CARD)
        search_frame.pack(fill=tk.X, pady=(0, 6))

        search_icon = tk.Label(search_frame, text="🔍", bg=BG_CARD, fg=TEXT_MUTED)
        search_icon.pack(side=tk.LEFT, padx=(0, 4))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_changed)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                                insertbackground=TEXT_WHITE, relief=tk.FLAT, font=("Segoe UI", 9))
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Filter Pills: All / Confirmed / Pending
        filters_frame = tk.Frame(self.tab_roster, bg=BG_CARD)
        filters_frame.pack(fill=tk.X, pady=(0, 6))

        self.filter_btns = {}
        for mode, title in [("all", "All Staff"), ("confirmed", "Confirmed (✓)"), ("pending", "Pending (⏳)")]:
            btn = tk.Button(filters_frame, text=title, font=("Segoe UI", 8),
                            bg=ACCENT_BLUE if mode == "all" else BG_CARD_LIGHT,
                            fg=TEXT_WHITE, relief=tk.FLAT, padx=8, pady=2, cursor="hand2",
                            command=lambda m=mode: self._set_roster_filter(m))
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self.filter_btns[mode] = btn

        # Scrollable Employee List Container
        list_container = tk.Frame(self.tab_roster, bg=BG_CARD)
        list_container.pack(fill=tk.BOTH, expand=True)

        self.roster_canvas = tk.Canvas(list_container, bg=BG_CARD, highlightthickness=0)
        roster_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.roster_canvas.yview)
        self.roster_scroll_frame = tk.Frame(self.roster_canvas, bg=BG_CARD)

        self.roster_scroll_frame.bind(
            "<Configure>",
            lambda e: self.roster_canvas.configure(scrollregion=self.roster_canvas.bbox("all"))
        )
        self.roster_canvas.create_window((0, 0), window=self.roster_scroll_frame, anchor="nw", width=440)
        self.roster_canvas.configure(yscrollcommand=roster_scrollbar.set)

        self.roster_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        roster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_roster_filter(self, mode: str):
        self.roster_filter_mode = mode
        for m, btn in self.filter_btns.items():
            btn.configure(bg=ACCENT_BLUE if m == mode else BG_CARD_LIGHT)
        self.refresh_roster()

    def _on_search_changed(self, *args):
        self.roster_search_query = self.search_var.get().strip().lower()
        self.refresh_roster()

    def refresh_roster(self):
        """Re-render staff cards in roster panel."""
        for widget in self.roster_scroll_frame.winfo_children():
            widget.destroy()

        if self.pipeline is None or self.pipeline.db is None:
            return

        try:
            employees = self.pipeline.db.get_employees_with_attendance_status()
        except Exception as e:
            logger.error(f"Error fetching employees: {e}")
            return

        total_count = len(employees)
        present_count = sum(1 for e in employees if e.get("attended_today"))
        self.roster_meta_lbl.configure(text=f"{total_count} Staff Registered")
        self.roster_present_lbl.configure(text=f"{present_count} Present Today")

        # Filter
        filtered = []
        for emp in employees:
            name = emp.get("name", "").lower()
            emp_id = emp.get("id", "").lower()
            if self.roster_search_query and (self.roster_search_query not in name and self.roster_search_query not in emp_id):
                continue

            attended = emp.get("attended_today", False)
            if self.roster_filter_mode == "confirmed" and not attended:
                continue
            if self.roster_filter_mode == "pending" and attended:
                continue

            filtered.append(emp)

        if not filtered:
            empty_lbl = tk.Label(self.roster_scroll_frame, text="No staff matching filter.", font=("Segoe UI", 9, "italic"),
                                 bg=BG_CARD, fg=TEXT_MUTED, pady=20)
            empty_lbl.pack()
            return

        for emp in filtered:
            self._render_staff_card(emp)

    def _render_staff_card(self, emp: Dict[str, Any]):
        card = tk.Frame(self.roster_scroll_frame, bg=BG_CARD_LIGHT, padx=8, pady=6, highlightbackground=BORDER_COLOR, highlightthickness=1)
        card.pack(fill=tk.X, pady=3)

        # Avatar thumbnail
        avatar_lbl = tk.Label(card, bg=BG_CARD_LIGHT)
        avatar_lbl.pack(side=tk.LEFT, padx=(0, 8))
        self._load_avatar(emp.get("photo_path", ""), avatar_lbl)

        # Info Box
        info = tk.Frame(card, bg=BG_CARD_LIGHT)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        name_lbl = tk.Label(info, text=emp.get("name", "Unknown"), font=("Segoe UI", 9, "bold"), bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
        name_lbl.pack(anchor="w")

        id_dept = f"ID: {emp.get('id', '')}  •  {emp.get('department', 'Staff')}"
        sub_lbl = tk.Label(info, text=id_dept, font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_MUTED)
        sub_lbl.pack(anchor="w")

        # Status badge
        status_box = tk.Frame(card, bg=BG_CARD_LIGHT)
        status_box.pack(side=tk.RIGHT, padx=(4, 0))

        attended = emp.get("attended_today", False)
        if attended:
            time_str = emp.get("last_attendance_time", "")
            t_display = time_str.split("T")[-1][:5] if "T" in time_str else time_str[:5]
            punches = emp.get("punches_today", 1)
            badge = tk.Label(status_box, text=f"✓ CONFIRMED\n{t_display} ({punches}x)", font=("Segoe UI", 7, "bold"),
                             bg="#166534", fg=TEXT_WHITE, padx=6, pady=2, justify=tk.CENTER)
            badge.pack()
        else:
            badge = tk.Label(status_box, text="⏳ PENDING", font=("Segoe UI", 7, "bold"),
                             bg="#475569", fg=TEXT_MUTED, padx=6, pady=4)
            badge.pack()

        # Delete Action Button
        del_btn = tk.Button(status_box, text="🗑", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=ACCENT_RED,
                            relief=tk.FLAT, cursor="hand2", command=lambda eid=emp["id"], ename=emp["name"]: self.delete_employee(eid, ename))
        del_btn.pack(pady=(2, 0))

    def _load_avatar(self, photo_path: str, label_widget: tk.Label):
        """Loads and caches thumbnail image for staff card."""
        if not photo_path or not os.path.exists(photo_path):
            # Placeholder text avatar
            label_widget.configure(text="👤", font=("Segoe UI", 16), fg=TEXT_MUTED, width=3, height=2)
            return

        if photo_path in self.avatar_cache:
            label_widget.configure(image=self.avatar_cache[photo_path], text="")
            return

        try:
            pil_img = Image.open(photo_path).convert("RGB")
            pil_img = pil_img.resize((42, 42), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.avatar_cache[photo_path] = tk_img
            label_widget.configure(image=tk_img, text="")
        except Exception:
            label_widget.configure(text="👤", font=("Segoe UI", 16), fg=TEXT_MUTED, width=3, height=2)

    def delete_employee(self, emp_id: str, emp_name: str):
        """Confirm and delete an enrolled employee."""
        confirm = messagebox.askyesno(
            "Confirm Delete Employee",
            f"Are you sure you want to delete employee '{emp_name}' (ID: {emp_id})?\nThis will remove their embeddings and reference data.",
            icon="warning"
        )
        if not confirm:
            return

        success = self.pipeline.delete_employee(emp_id)
        if success:
            messagebox.showinfo("Success", f"Employee '{emp_name}' deleted.")
            self.refresh_roster()
            self.update_kpi_metrics()
        else:
            messagebox.showerror("Error", f"Failed to delete employee {emp_id}.")

    # --------------------------------------------------------------------------
    # Right Tab 2: Interactive Embedding Verification Tester
    # --------------------------------------------------------------------------
    def _build_verifier_tab(self):
        desc = tk.Label(self.tab_verifier,
                        text="Compare two 128-d SFace face embeddings from disk or audit records to test match confidence and Euclidean/Cosine similarity.",
                        font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED, wraplength=430, justify=tk.LEFT)
        desc.pack(anchor="w", pady=(0, 10))

        # Input A
        grp_a = tk.LabelFrame(self.tab_verifier, text="Embedding A (Reference)", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=TEXT_WHITE, padx=8, pady=6)
        grp_a.pack(fill=tk.X, pady=(0, 8))

        self.emb_a_combo = ttk.Combobox(grp_a, state="readonly", font=("Segoe UI", 8))
        self.emb_a_combo.pack(fill=tk.X, pady=(0, 4))

        btn_browse_a = tk.Button(grp_a, text="📁 Browse Custom File (.npy / .json)...", font=("Segoe UI", 8),
                                 bg=BG_CARD_LIGHT, fg=TEXT_WHITE, relief=tk.FLAT, cursor="hand2", command=lambda: self._browse_embedding_file(self.emb_a_combo))
        btn_browse_a.pack(anchor="e")

        # Input B
        grp_b = tk.LabelFrame(self.tab_verifier, text="Embedding B (Probe)", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=TEXT_WHITE, padx=8, pady=6)
        grp_b.pack(fill=tk.X, pady=(0, 10))

        self.emb_b_combo = ttk.Combobox(grp_b, state="readonly", font=("Segoe UI", 8))
        self.emb_b_combo.pack(fill=tk.X, pady=(0, 4))

        btn_browse_b = tk.Button(grp_b, text="📁 Browse Custom File (.npy / .json)...", font=("Segoe UI", 8),
                                 bg=BG_CARD_LIGHT, fg=TEXT_WHITE, relief=tk.FLAT, cursor="hand2", command=lambda: self._browse_embedding_file(self.emb_b_combo))
        btn_browse_b.pack(anchor="e")

        # Run Test Button
        run_btn = tk.Button(self.tab_verifier, text="⚡ Run Cosine Verification", font=("Segoe UI", 10, "bold"),
                            bg=ACCENT_BLUE, fg=TEXT_WHITE, activebackground=ACCENT_HOVER, activeforeground=TEXT_WHITE,
                            relief=tk.FLAT, pady=6, cursor="hand2", command=self.run_embedding_verification)
        run_btn.pack(fill=tk.X, pady=(0, 10))

        # Result Box
        self.verify_result_card = tk.Frame(self.tab_verifier, bg=BG_CARD_LIGHT, padx=12, pady=10,
                                           highlightbackground=BORDER_COLOR, highlightthickness=1)
        self.verify_result_card.pack(fill=tk.BOTH, expand=True)

        self.verdict_title = tk.Label(self.verify_result_card, text="READY FOR VERIFICATION", font=("Segoe UI", 12, "bold"),
                                      bg=BG_CARD_LIGHT, fg=TEXT_MUTED)
        self.verdict_title.pack(anchor="w")

        self.verdict_sim = tk.Label(self.verify_result_card, text="Similarity: -", font=("Segoe UI", 14, "bold"),
                                    bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
        self.verdict_sim.pack(anchor="w", pady=(4, 8))

        self.verdict_cosine = tk.Label(self.verify_result_card, text="Cosine Similarity: -", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
        self.verdict_cosine.pack(anchor="w")

        self.verdict_l2 = tk.Label(self.verify_result_card, text="Euclidean L2 Dist: -", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
        self.verdict_l2.pack(anchor="w")

        thresh_info = f"Threshold Benchmark: >= {COSINE_SIMILARITY_THRESHOLD:.3f} (SFace Std)"
        self.verdict_thresh = tk.Label(self.verify_result_card, text=thresh_info, font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_MUTED)
        self.verdict_thresh.pack(anchor="w", pady=(4, 0))

        self.verdict_conf = tk.Label(self.verify_result_card, text="Confidence Level: -", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
        self.verdict_conf.pack(anchor="w")

        self.refresh_saved_embeddings_dropdowns()

    def _browse_embedding_file(self, combo: ttk.Combobox):
        path = filedialog.askopenfilename(
            title="Select Face Embedding File",
            initialdir=str(EMBEDDINGS_DIR),
            filetypes=[("Embedding Files", "*.npy *.json"), ("All Files", "*.*")]
        )
        if path:
            current_vals = list(combo["values"])
            if path not in current_vals:
                current_vals.insert(0, path)
                combo["values"] = current_vals
            combo.set(path)

    def refresh_saved_embeddings_dropdowns(self):
        """Populate comboboxes with available embeddings from data/embeddings/."""
        if not self.pipeline:
            return

        try:
            embeddings_list = self.pipeline.storage.list_saved_embeddings()
            choices = []
            for item in embeddings_list:
                cam = item.get("camera_id", "CAM")
                ts = item.get("created_at", "").replace("T", " ")[:19]
                file_path = item.get("embedding_npy") or item.get("embedding_path", "")
                if file_path:
                    choices.append(f"{ts} [{cam}] -> {Path(file_path).name}|{file_path}")

            display_vals = [c.split("|")[0] for c in choices]
            self.emb_choices_map = {c.split("|")[0]: c.split("|")[1] for c in choices}

            self.emb_a_combo["values"] = display_vals
            self.emb_b_combo["values"] = display_vals

            if len(display_vals) > 0:
                self.emb_a_combo.set(display_vals[0])
            if len(display_vals) > 1:
                self.emb_b_combo.set(display_vals[1])
        except Exception as e:
            logger.error(f"Error refreshing embedding choices: {e}")

    def run_embedding_verification(self):
        val_a = self.emb_a_combo.get()
        val_b = self.emb_b_combo.get()

        if not val_a or not val_b:
            messagebox.showwarning("Incomplete Selection", "Please select both Embedding A and Embedding B.")
            return

        path_a = self.emb_choices_map.get(val_a, val_a)
        path_b = self.emb_choices_map.get(val_b, val_b)

        try:
            emb1 = self.pipeline.storage.load_embedding(path_a)
            emb2 = self.pipeline.storage.load_embedding(path_b)

            sim = float(self.pipeline.recognizer.compute_similarity(emb1, emb2))
            l2_dist = float(self.pipeline.recognizer.compute_l2_distance(emb1, emb2))
            is_match = sim >= COSINE_SIMILARITY_THRESHOLD
            conf_percent = round(max(0.0, min(100.0, sim * 100)), 1)

            if is_match:
                self.verdict_title.configure(text="✓ MATCH VERIFIED", fg=ACCENT_GREEN)
                self.verify_result_card.configure(highlightbackground=ACCENT_GREEN)
            else:
                self.verdict_title.configure(text="✗ NO MATCH (DIFFERENT PERSON)", fg=ACCENT_RED)
                self.verify_result_card.configure(highlightbackground=ACCENT_RED)

            self.verdict_sim.configure(text=f"Similarity: {sim:.4f}")
            self.verdict_cosine.configure(text=f"Cosine Similarity: {sim:.4f}")
            self.verdict_l2.configure(text=f"Euclidean (L2) Dist: {l2_dist:.4f}")
            self.verdict_conf.configure(text=f"Confidence Level: {conf_percent}%")

        except Exception as e:
            messagebox.showerror("Verification Failed", f"Could not compare embeddings:\n{e}")

    # --------------------------------------------------------------------------
    # Lower Section: Multi-Tab Data Explorer
    # --------------------------------------------------------------------------
    def _build_lower_data_tabs(self):
        bottom_frame = tk.Frame(self.container, bg=BG_MAIN, padx=16, pady=6)
        bottom_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        notebook = ttk.Notebook(bottom_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Embeddings & Face Chips Gallery
        self.tab_gallery = tk.Frame(notebook, bg=BG_CARD, padx=8, pady=8)
        notebook.add(self.tab_gallery, text="🗂 Stored Face Embeddings & Crops")
        self._build_embeddings_gallery_tab()

        # Tab 2: Attendance Events Table
        self.tab_attendance = tk.Frame(notebook, bg=BG_CARD, padx=8, pady=8)
        notebook.add(self.tab_attendance, text="📋 Verified Attendance Events")
        self._build_attendance_table_tab()

        # Tab 3: Quality Audit Logs Table
        self.tab_quality = tk.Frame(notebook, bg=BG_CARD, padx=8, pady=8)
        notebook.add(self.tab_quality, text="📊 Quality Audit Logs")
        self._build_quality_table_tab()

        # Tab 4: System Logs Console
        self.tab_logs = tk.Frame(notebook, bg=BG_CARD, padx=8, pady=8)
        notebook.add(self.tab_logs, text="💻 System Terminal & Logs")
        self._build_logs_console_tab()

    def _build_embeddings_gallery_tab(self):
        toolbar = tk.Frame(self.tab_gallery, bg=BG_CARD)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        sub = tk.Label(toolbar, text="Aligned 112x112 face chips and 128-d vectors stored in data/embeddings/ & data/faces/",
                       font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED)
        sub.pack(side=tk.LEFT)

        ref_btn = tk.Button(toolbar, text="🔄 Refresh Gallery", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=self.load_embeddings_gallery)
        ref_btn.pack(side=tk.RIGHT)

        # Gallery Canvas with Scrollbar
        gal_container = tk.Frame(self.tab_gallery, bg=BG_CARD)
        gal_container.pack(fill=tk.BOTH, expand=True)

        self.gallery_canvas = tk.Canvas(gal_container, bg=BG_CARD, highlightthickness=0)
        gal_scrollbar = ttk.Scrollbar(gal_container, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.gallery_scroll_frame = tk.Frame(self.gallery_canvas, bg=BG_CARD)

        self.gallery_scroll_frame.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        self.gallery_canvas.create_window((0, 0), window=self.gallery_scroll_frame, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=gal_scrollbar.set)

        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def load_embeddings_gallery(self):
        for widget in self.gallery_scroll_frame.winfo_children():
            widget.destroy()

        if not self.pipeline:
            return

        try:
            records = self.pipeline.storage.list_saved_embeddings()[:24]
        except Exception as e:
            logger.error(f"Error loading gallery: {e}")
            return

        if not records:
            empty_lbl = tk.Label(self.gallery_scroll_frame, text="No embeddings stored yet. Live faces passing quality will appear here.",
                                 font=("Segoe UI", 9, "italic"), bg=BG_CARD, fg=TEXT_MUTED, pady=20)
            empty_lbl.pack()
            return

        # Render chips in a grid (columns of 6)
        cols = 6
        for idx, rec in enumerate(records):
            r = idx // cols
            c = idx % cols

            chip_card = tk.Frame(self.gallery_scroll_frame, bg=BG_CARD_LIGHT, padx=6, pady=6, highlightbackground=BORDER_COLOR, highlightthickness=1)
            chip_card.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            # Aligned image thumbnail
            img_path = rec.get("aligned_image")
            img_lbl = tk.Label(chip_card, bg=BG_CARD_LIGHT)
            img_lbl.pack()

            if img_path and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).resize((70, 70), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.photo_preview_cache[img_path] = tk_img
                    img_lbl.configure(image=tk_img)
                except Exception:
                    img_lbl.configure(text="[Face Chip]", fg=TEXT_MUTED)
            else:
                img_lbl.configure(text="[No Image]", fg=TEXT_MUTED)

            # Metadata info
            created = rec.get("created_at", "").replace("T", " ")[:16]
            time_lbl = tk.Label(chip_card, text=created, font=("Segoe UI", 7), bg=BG_CARD_LIGHT, fg=TEXT_MUTED)
            time_lbl.pack(pady=(2, 0))

            q_metrics = rec.get("quality_metrics", {})
            blur = q_metrics.get("blur_score", 0.0)
            blur_lbl = tk.Label(chip_card, text=f"Blur: {blur:.1f}", font=("Segoe UI", 7, "bold"), bg=BG_CARD_LIGHT, fg=ACCENT_GREEN if blur >= 50 else ACCENT_AMBER)
            blur_lbl.pack()

            # Test in verifier button
            test_btn = tk.Button(chip_card, text="Verify", font=("Segoe UI", 7), bg=ACCENT_BLUE, fg=TEXT_WHITE,
                                 relief=tk.FLAT, padx=4, pady=1, cursor="hand2", command=lambda p=rec.get("embedding_npy"): self._set_probe_verifier(p))
            test_btn.pack(pady=(2, 0))

    def _set_probe_verifier(self, path: Optional[str]):
        if path:
            self.emb_b_combo.set(path)
            self.right_notebook.select(self.tab_verifier)

    # --------------------------------------------------------------------------
    # Tab 2: Attendance Table
    # --------------------------------------------------------------------------
    def _build_attendance_table_tab(self):
        toolbar = tk.Frame(self.tab_attendance, bg=BG_CARD)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        export_btn = tk.Button(toolbar, text="💾 Export CSV", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                               relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self._export_treeview_to_csv(self.att_tree, "attendance_records.csv"))
        export_btn.pack(side=tk.LEFT, padx=(0, 6))

        ref_btn = tk.Button(toolbar, text="🔄 Refresh", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=self.refresh_attendance_table)
        ref_btn.pack(side=tk.LEFT)

        cols = ("id", "name", "emp_id", "timestamp", "camera", "confidence", "status")
        self.att_tree = ttk.Treeview(self.tab_attendance, columns=cols, show="headings", height=8)

        self.att_tree.heading("id", text="ID")
        self.att_tree.heading("name", text="Employee Name")
        self.att_tree.heading("emp_id", text="Employee ID")
        self.att_tree.heading("timestamp", text="Timestamp")
        self.att_tree.heading("camera", text="Camera")
        self.att_tree.heading("confidence", text="Confidence")
        self.att_tree.heading("status", text="Status")

        self.att_tree.column("id", width=50, anchor="center")
        self.att_tree.column("name", width=160)
        self.att_tree.column("emp_id", width=100)
        self.att_tree.column("timestamp", width=180)
        self.att_tree.column("camera", width=120)
        self.att_tree.column("confidence", width=90, anchor="center")
        self.att_tree.column("status", width=100, anchor="center")

        tree_scroll = ttk.Scrollbar(self.tab_attendance, orient=tk.VERTICAL, command=self.att_tree.yview)
        self.att_tree.configure(yscrollcommand=tree_scroll.set)

        self.att_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_attendance_table(self):
        for item in self.att_tree.get_children():
            self.att_tree.delete(item)

        if not self.pipeline or not self.pipeline.db:
            return

        try:
            records = self.pipeline.db.get_recent_attendance(limit=100)
            for r in records:
                conf = f"{float(r.get('confidence', 0.0)):.3f}"
                self.att_tree.insert("", tk.END, values=(
                    r.get("id"),
                    r.get("employee_name"),
                    r.get("employee_id"),
                    r.get("timestamp", "").replace("T", " "),
                    r.get("camera_id"),
                    conf,
                    r.get("status")
                ))
        except Exception as e:
            logger.error(f"Error loading attendance records: {e}")

    # --------------------------------------------------------------------------
    # Tab 3: Quality Audit Table
    # --------------------------------------------------------------------------
    def _build_quality_table_tab(self):
        toolbar = tk.Frame(self.tab_quality, bg=BG_CARD)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        export_btn = tk.Button(toolbar, text="💾 Export CSV", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                               relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self._export_treeview_to_csv(self.qual_tree, "quality_captures.csv"))
        export_btn.pack(side=tk.LEFT, padx=(0, 6))

        ref_btn = tk.Button(toolbar, text="🔄 Refresh", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=self.refresh_quality_table)
        ref_btn.pack(side=tk.LEFT)

        cols = ("time", "camera", "dims", "blur", "roll", "yaw", "status", "match")
        self.qual_tree = ttk.Treeview(self.tab_quality, columns=cols, show="headings", height=8)

        self.qual_tree.heading("time", text="Timestamp")
        self.qual_tree.heading("camera", text="Camera")
        self.qual_tree.heading("dims", text="Dimensions (WxH)")
        self.qual_tree.heading("blur", text="Blur Score")
        self.qual_tree.heading("roll", text="Roll Angle")
        self.qual_tree.heading("yaw", text="Yaw Ratio")
        self.qual_tree.heading("status", text="Quality Status")
        self.qual_tree.heading("match", text="Identity / Reason")

        self.qual_tree.column("time", width=150)
        self.qual_tree.column("camera", width=100)
        self.qual_tree.column("dims", width=110, anchor="center")
        self.qual_tree.column("blur", width=90, anchor="center")
        self.qual_tree.column("roll", width=80, anchor="center")
        self.qual_tree.column("yaw", width=80, anchor="center")
        self.qual_tree.column("status", width=90, anchor="center")
        self.qual_tree.column("match", width=180)

        tree_scroll = ttk.Scrollbar(self.tab_quality, orient=tk.VERTICAL, command=self.qual_tree.yview)
        self.qual_tree.configure(yscrollcommand=tree_scroll.set)

        self.qual_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_quality_table(self):
        for item in self.qual_tree.get_children():
            self.qual_tree.delete(item)

        if not self.pipeline or not self.pipeline.db:
            return

        try:
            captures = self.pipeline.db.get_recent_captures(limit=100)
            for c in captures:
                passed = "✓ PASSED" if c.get("passed_quality") == 1 else "✗ REJECTED"
                dims = f"{int(c.get('bbox_w', 0))}x{int(c.get('bbox_h', 0))}"
                blur = f"{float(c.get('blur_score', 0)):.1f}"
                roll = f"{float(c.get('roll_angle', 0)):.1f}°"
                yaw = f"{float(c.get('yaw_ratio', 0)):.2f}"
                reason = c.get("rejection_reason") or c.get("match_name", "UNKNOWN")

                self.qual_tree.insert("", tk.END, values=(
                    c.get("timestamp", "").replace("T", " "),
                    c.get("camera_id"),
                    dims,
                    blur,
                    roll,
                    yaw,
                    passed,
                    reason
                ))
        except Exception as e:
            logger.error(f"Error loading quality records: {e}")

    def _export_treeview_to_csv(self, tree: ttk.Treeview, default_name: str):
        path = filedialog.asksaveasfilename(
            title="Export Table as CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = [tree.heading(col)["text"] for col in tree["columns"]]
                writer.writerow(headers)
                for row_id in tree.get_children():
                    writer.writerow(tree.item(row_id)["values"])
            messagebox.showinfo("Export Successful", f"Saved records to {path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write CSV:\n{e}")

    # --------------------------------------------------------------------------
    # Tab 4: System Logs Console
    # --------------------------------------------------------------------------
    def _build_logs_console_tab(self):
        toolbar = tk.Frame(self.tab_logs, bg=BG_CARD)
        toolbar.pack(fill=tk.X, pady=(0, 4))

        clear_btn = tk.Button(toolbar, text="Clear Console", font=("Segoe UI", 8), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                              relief=tk.FLAT, padx=6, pady=2, cursor="hand2", command=self._clear_logs)
        clear_btn.pack(side=tk.RIGHT)

        self.log_text = ScrolledText(self.tab_logs, bg="#050811", fg="#a5f3fc", insertbackground=TEXT_WHITE,
                                     font=("Consolas", 9), relief=tk.FLAT, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _clear_logs(self):
        self.log_text.delete("1.0", tk.END)

    # --------------------------------------------------------------------------
    # Background Processing & Live Video Stream
    # --------------------------------------------------------------------------
    def _start_processing_thread(self):
        """Dedicated background thread for sampling and processing pipeline frames."""
        def worker():
            logger.info("Background Face Detection & Recognition Worker started.")
            while self.is_running:
                if self.stream_paused:
                    time.sleep(0.1)
                    continue

                if self.pipeline and self.pipeline.stream:
                    # 1. Check for sample frame to process
                    has_sample, sample_frame, _ = self.pipeline.stream.get_sampled_frame()
                    if has_sample and sample_frame is not None:
                        try:
                            annotated, _ = self.pipeline.process_frame(
                                sample_frame,
                                save_passed_embeddings=SAVE_LIVE_CAPTURES_TO_DISK
                            )
                            self.latest_display_frame = annotated
                        except Exception as e:
                            logger.error(f"Error in pipeline process_frame: {e}")

                time.sleep(0.01)

        self.proc_thread = threading.Thread(target=worker, daemon=True, name="FacePipelineWorker")
        self.proc_thread.start()

    def _schedule_periodic_tasks(self):
        """Periodic timers for GUI refresh."""
        self._update_video_canvas()
        self._drain_logs()
        self._periodic_metrics_refresh()

        # Initial data loading
        self.root.after(500, self.refresh_roster)
        self.root.after(800, self.refresh_attendance_table)
        self.root.after(1000, self.refresh_quality_table)
        self.root.after(1200, self.load_embeddings_gallery)

    def _update_video_canvas(self):
        """Renders live video frame on Tkinter canvas at ~30 FPS."""
        if not self.is_running:
            return

        if not self.stream_paused and self.pipeline and self.pipeline.stream:
            is_connected = self.pipeline.stream.is_connected
            if is_connected:
                self.status_pill.configure(text="● LIVE (Connected)", bg="#14532d", fg="#86efac")
            else:
                self.status_pill.configure(text="● Connecting RTSP...", bg="#78350f", fg="#fde047")

            # Prefer latest annotated frame, fallback to raw frame
            frame_to_show = self.latest_display_frame
            if frame_to_show is None:
                is_fresh, raw_frame, _ = self.pipeline.stream.get_latest_frame()
                if is_fresh and raw_frame is not None:
                    frame_to_show = raw_frame

            if frame_to_show is not None:
                c_width = self.video_canvas.winfo_width()
                c_height = self.video_canvas.winfo_height()

                if c_width > 100 and c_height > 100:
                    # Calculate aspect fit (16:9)
                    h, w = frame_to_show.shape[:2]
                    scale = min(c_width / w, c_height / h)
                    nw = int(w * scale)
                    nh = int(h * scale)

                    resized = cv2.resize(frame_to_show, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    tk_img = ImageTk.PhotoImage(pil_img)

                    self.video_canvas.delete("all")
                    pos_x = (c_width - nw) // 2
                    pos_y = (c_height - nh) // 2
                    self.video_canvas.create_image(pos_x, pos_y, image=tk_img, anchor="nw")
                    self._canvas_img_ref = tk_img
            else:
                # Render connecting placeholder
                c_width = self.video_canvas.winfo_width()
                c_height = self.video_canvas.winfo_height()
                if c_width > 100 and c_height > 100:
                    self.video_canvas.delete("all")
                    self.video_canvas.create_text(
                        c_width // 2, c_height // 2,
                        text="Connecting to RTSP Stream...\n(Please ensure Camera is online on LAN)",
                        fill="#fde047", font=("Segoe UI", 12, "bold"), justify=tk.CENTER
                    )

        self.root.after(33, self._update_video_canvas)

    def _drain_logs(self):
        """Transfers log messages from queue to ScrolledText widget."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
            except queue.Empty:
                break
        self.root.after(100, self._drain_logs)

    def _periodic_metrics_refresh(self):
        """Refreshes KPI metrics cards every 2.5 seconds."""
        if self.is_running:
            self.update_kpi_metrics()
            self.root.after(2500, self._periodic_metrics_refresh)

    def update_kpi_metrics(self):
        if not self.pipeline or not self.pipeline.db:
            return

        try:
            stats = self.pipeline.db.get_stats()
            enrolled_count = len(self.pipeline.enrolled_gallery)

            self.kpi_cards["captures"].configure(text=f"{stats.get('total_captures', 0):,}")
            self.kpi_cards["quality"].configure(text=f"{stats.get('quality_pass_rate', 0.0):.1f}%")
            self.kpi_cards["enrolled"].configure(text=f"{enrolled_count} Staff")
            self.kpi_cards["attendance"].configure(text=f"{stats.get('total_attendance', 0)} Today")
        except Exception as e:
            logger.error(f"Error updating KPI cards: {e}")

    def toggle_stream_pause(self):
        self.stream_paused = not self.stream_paused
        if self.stream_paused:
            self.pause_btn.configure(text="▶ Resume", bg=ACCENT_BLUE)
            self.status_pill.configure(text="● PAUSED", bg="#78350f", fg="#fde047")
        else:
            self.pause_btn.configure(text="⏸ Pause", bg=BG_CARD_LIGHT)

    def save_snapshot(self):
        if self.latest_display_frame is None and self.pipeline and self.pipeline.stream:
            _, frame, _ = self.pipeline.stream.get_latest_frame()
            self.latest_display_frame = frame

        if self.latest_display_frame is not None:
            captures_dir = PROJECT_ROOT / "captures"
            captures_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            snap_path = captures_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(snap_path), self.latest_display_frame)
            messagebox.showinfo("Snapshot Saved", f"Saved live snapshot to:\n{snap_path}")
        else:
            messagebox.showwarning("No Frame", "No video frame currently available to capture.")

    def reload_gallery(self):
        if self.pipeline:
            self.pipeline.reload_enrolled_gallery()
            self.refresh_roster()
            self.refresh_saved_embeddings_dropdowns()
            self.update_kpi_metrics()
            messagebox.showinfo("Gallery Reloaded", f"Successfully reloaded {len(self.pipeline.enrolled_gallery)} employee profiles.")

    # --------------------------------------------------------------------------
    # Settings & Camera Source Dialog
    # --------------------------------------------------------------------------
    def open_settings_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Camera Stream Configuration")
        dialog.geometry("600x340")
        dialog.configure(bg=BG_CARD)
        dialog.transient(self.root)
        dialog.grab_set()

        lbl = tk.Label(dialog, text="Configure Camera RTSP Stream / Source", font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
        lbl.pack(anchor="w", padx=16, pady=(16, 8))

        sub = tk.Label(dialog, text="Enter IP Camera RTSP URL, or integer (0) for USB Webcam, or local video path.",
                       font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED)
        sub.pack(anchor="w", padx=16, pady=(0, 12))

        # URL Input
        form = tk.Frame(dialog, bg=BG_CARD, padx=16)
        form.pack(fill=tk.X)

        tk.Label(form, text="Camera Stream URL / Index:", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w", pady=(0, 4))
        url_entry = tk.Entry(form, font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT)
        url_entry.insert(0, self.current_rtsp_url)
        url_entry.pack(fill=tk.X, pady=(0, 10))

        # Camera ID
        tk.Label(form, text="Camera Identifier Tag:", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w", pady=(0, 4))
        cam_id_entry = tk.Entry(form, font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT)
        cam_id_entry.insert(0, CAMERA_ID)
        cam_id_entry.pack(fill=tk.X, pady=(0, 16))

        # Buttons
        btn_box = tk.Frame(dialog, bg=BG_CARD, padx=16)
        btn_box.pack(fill=tk.X, side=tk.BOTTOM, pady=16)

        def save_and_reconnect():
            new_url = url_entry.get().strip()
            if not new_url:
                messagebox.showerror("Error", "Stream URL cannot be empty.", parent=dialog)
                return

            self.current_rtsp_url = new_url
            if self.pipeline and self.pipeline.stream:
                logger.info(f"Reconnecting stream to: {new_url}")
                self.pipeline.stream.stop()
                self.pipeline.stream.rtsp_url = new_url
                self.pipeline.stream.start()

            dialog.destroy()
            messagebox.showinfo("Stream Reconnected", f"Camera stream pointed to:\n{new_url}")

        save_btn = tk.Button(btn_box, text="Save & Connect", font=("Segoe UI", 9, "bold"), bg=ACCENT_BLUE, fg=TEXT_WHITE,
                             relief=tk.FLAT, padx=12, pady=5, cursor="hand2", command=save_and_reconnect)
        save_btn.pack(side=tk.RIGHT, padx=(6, 0))

        cancel_btn = tk.Button(btn_box, text="Cancel", font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                               relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT)

    # --------------------------------------------------------------------------
    # Register Staff Modal / Dialog (Full 3-Photo Enrollment)
    # --------------------------------------------------------------------------
    def open_enroll_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Staff Member — Multi-Angle SFace")
        dialog.geometry("720x680")
        dialog.minsize(640, 500)
        dialog.configure(bg=BG_CARD)
        dialog.transient(self.root)
        dialog.grab_set()

        # 1. Header Frame (Top)
        header_frame = tk.Frame(dialog, bg=BG_CARD, padx=16, pady=12)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        h_title = tk.Label(header_frame, text="Register New Staff Member", font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=TEXT_WHITE)
        h_title.pack(anchor="w")

        h_sub = tk.Label(header_frame, text="Upload 3 portrait photos to extract multi-angle SFace embeddings into Staffs/ folder",
                         font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_MUTED)
        h_sub.pack(anchor="w")

        # 2. Modal Bottom Bar (Pinned permanently to the bottom first — NEVER pushed off-screen!)
        bottom_bar = tk.Frame(dialog, bg=BG_CARD, padx=16, pady=12, highlightbackground=BORDER_COLOR, highlightthickness=1)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Status text for live progress updates
        status_label = tk.Label(bottom_bar, text="", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=ACCENT_BLUE)
        status_label.pack(side=tk.LEFT, padx=(0, 10))

        # 3. Middle Scrollable Container
        scroll_container = tk.Frame(dialog, bg=BG_CARD)
        scroll_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_container, bg=BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=canvas.yview)
        form_scroll = tk.Frame(canvas, bg=BG_CARD, padx=16, pady=4)

        form_scroll.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=form_scroll, anchor="nw", width=680)

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
        dialog.bind("<MouseWheel>", _on_mousewheel)

        # Form fields inside form_scroll
        row1 = tk.Frame(form_scroll, bg=BG_CARD)
        row1.pack(fill=tk.X, pady=(0, 8))

        # Staff ID
        col1 = tk.Frame(row1, bg=BG_CARD)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        tk.Label(col1, text="Staff ID / Code *", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")
        id_var = tk.StringVar()
        id_entry = tk.Entry(col1, textvariable=id_var, font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT)
        id_entry.pack(fill=tk.X, pady=(2, 0))

        # Staff Name
        col2 = tk.Frame(row1, bg=BG_CARD)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(col2, text="Staff Full Name *", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")
        name_var = tk.StringVar()
        name_entry = tk.Entry(col2, textvariable=name_var, font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, pady=(2, 0))

        # Department
        row2 = tk.Frame(form_scroll, bg=BG_CARD)
        row2.pack(fill=tk.X, pady=(0, 8))

        tk.Label(row2, text="Department / Role:", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")
        dept_var = tk.StringVar(value="Staff")
        dept_entry = tk.Entry(row2, textvariable=dept_var, font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT)
        dept_entry.pack(fill=tk.X, pady=(2, 0))

        # Destination Folder Preview
        folder_box = tk.Frame(form_scroll, bg=BG_CARD_LIGHT, padx=10, pady=8, highlightbackground=BORDER_COLOR, highlightthickness=1)
        folder_box.pack(fill=tk.X, pady=(4, 10))

        folder_preview_lbl = tk.Label(folder_box, text="📂 Target Archive: Attendence System\\Staffs\\staffid_name",
                                      font=("Segoe UI", 8, "bold"), bg=BG_CARD_LIGHT, fg=ACCENT_BLUE)
        folder_preview_lbl.pack(anchor="w")

        def update_folder_label(*args):
            clean_id = id_var.get().strip().replace(" ", "_") or "staffid"
            clean_name = name_var.get().strip().replace(" ", "_") or "name"
            folder_preview_lbl.configure(text=f"📂 Target Archive: Attendence System\\Staffs\\{clean_id}_{clean_name}")

        id_var.trace_add("write", update_folder_label)
        name_var.trace_add("write", update_folder_label)

        # 3 Photo Slots
        photos_frame = tk.LabelFrame(form_scroll, text="Upload 3 Portrait Photos (Frontal + Left + Right)",
                                     font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_WHITE, padx=10, pady=8)
        photos_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        selected_photos: Dict[int, Optional[str]] = {1: None, 2: None, 3: None}
        photo_labels: Dict[int, tk.Label] = {}

        slots_row = tk.Frame(photos_frame, bg=BG_CARD)
        slots_row.pack(fill=tk.BOTH, expand=True)

        slot_titles = {
            1: "Photo 1 (Frontal)",
            2: "Photo 2 (Slight Left)",
            3: "Photo 3 (Slight Right)"
        }

        for slot_num in [1, 2, 3]:
            slot_box = tk.Frame(slots_row, bg=BG_CARD_LIGHT, padx=6, pady=6, highlightbackground=BORDER_COLOR, highlightthickness=1)
            slot_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if slot_num == 1 else 6, 0))

            s_title = tk.Label(slot_box, text=slot_titles[slot_num], font=("Segoe UI", 8, "bold"), bg=BG_CARD_LIGHT, fg=TEXT_WHITE)
            s_title.pack(anchor="n")

            img_preview = tk.Label(slot_box, text="📷\nClick Browse", font=("Segoe UI", 8), bg="#0f172a", fg=TEXT_MUTED,
                                   width=14, height=5, relief=tk.FLAT)
            img_preview.pack(pady=4)
            photo_labels[slot_num] = img_preview

            def make_browse_cmd(s=slot_num):
                def browse():
                    p = filedialog.askopenfilename(
                        title=f"Select {slot_titles[s]}",
                        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
                    )
                    if p:
                        selected_photos[s] = p
                        try:
                            pil_img = Image.open(p).convert("RGB").resize((95, 95), Image.Resampling.LANCZOS)
                            tk_img = ImageTk.PhotoImage(pil_img)
                            self.photo_preview_cache[f"slot_{s}"] = tk_img
                            photo_labels[s].configure(image=tk_img, text="", width=95, height=95)
                        except Exception:
                            photo_labels[s].configure(text="Selected")
                return browse

            btn_browse = tk.Button(slot_box, text="Choose File...", font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_WHITE,
                                   relief=tk.FLAT, cursor="hand2", command=make_browse_cmd(slot_num))
            btn_browse.pack()

        # Alternative: Capture from live stream
        cam_snap_frame = tk.Frame(form_scroll, bg=BG_CARD)
        cam_snap_frame.pack(fill=tk.X, pady=(0, 10))

        def capture_from_live():
            if self.pipeline and self.pipeline.stream:
                is_fresh, frame, _ = self.pipeline.stream.get_latest_frame()
                if is_fresh and frame is not None:
                    temp_p = PROJECT_ROOT / "data" / f"temp_cam_slot1_{int(time.time())}.jpg"
                    cv2.imwrite(str(temp_p), frame)
                    selected_photos[1] = str(temp_p)

                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((95, 95), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.photo_preview_cache["slot_1"] = tk_img
                    photo_labels[1].configure(image=tk_img, text="", width=95, height=95)
                    messagebox.showinfo("Captured", "Captured live camera frame into Photo Slot 1 (Frontal).", parent=dialog)
                else:
                    messagebox.showwarning("No Stream", "Could not capture frame from RTSP stream.", parent=dialog)

        snap_cam_btn = tk.Button(cam_snap_frame, text="📸 Grab Live Camera Frame into Photo 1", font=("Segoe UI", 8),
                                 bg=BG_CARD_LIGHT, fg=TEXT_WHITE, relief=tk.FLAT, padx=8, pady=4, cursor="hand2", command=capture_from_live)
        snap_cam_btn.pack(anchor="w")

        # Bottom Bar Buttons (Pinned & Always Visible)
        def submit_enrollment():
            staff_id = id_var.get().strip()
            name = name_var.get().strip()
            dept = dept_var.get().strip()

            if not staff_id or not name:
                messagebox.showerror("Validation Error", "Staff ID and Full Name are required.", parent=dialog)
                return

            # Read selected photos
            photos_payload = []
            for s_num in [1, 2, 3]:
                p_path = selected_photos.get(s_num)
                if p_path and os.path.exists(p_path):
                    try:
                        with open(p_path, "rb") as f:
                            photos_payload.append((Path(p_path).name, f.read()))
                    except Exception as e:
                        logger.warning(f"Could not read selected photo {s_num}: {e}")

            if not photos_payload:
                messagebox.showerror("No Photos Selected", "Please select or capture at least 1 portrait photo (preferably 3).", parent=dialog)
                return

            submit_btn.configure(text="Enrolling Staff...", state=tk.DISABLED)
            status_label.configure(text="Starting facial landmark analysis...")
            dialog.update()

            def update_progress(msg_text: str):
                self.root.after(0, lambda: status_label.configure(text=msg_text))

            def async_enroll():
                try:
                    success, msg, info = self.pipeline.enroll_staff_from_photos(
                        staff_id=staff_id,
                        name=name,
                        department=dept,
                        email="",
                        photos=photos_payload,
                        progress_callback=update_progress
                    )
                except Exception as err:
                    logger.exception(f"Unexpected error in enroll_staff_from_photos: {err}")
                    success = False
                    msg = f"System Error during enrollment:\n{err}"

                def on_done():
                    submit_btn.configure(text="Confirm & Register Staff", state=tk.NORMAL)
                    status_label.configure(text="")
                    if success:
                        try:
                            dialog.grab_release()
                        except Exception:
                            pass
                        dialog.destroy()
                        messagebox.showinfo("Registration Successful", msg, parent=self.root)
                        self.refresh_roster()
                        self.update_kpi_metrics()
                        self.load_embeddings_gallery()
                    else:
                        messagebox.showerror("Registration Failed", msg, parent=dialog)

                self.root.after(0, on_done)

            threading.Thread(target=async_enroll, daemon=True).start()

        submit_btn = tk.Button(bottom_bar, text="Confirm & Register Staff", font=("Segoe UI", 10, "bold"),
                               bg=ACCENT_BLUE, fg=TEXT_WHITE, activebackground=ACCENT_HOVER, activeforeground=TEXT_WHITE,
                               relief=tk.FLAT, padx=14, pady=6, cursor="hand2", command=submit_enrollment)
        submit_btn.pack(side=tk.RIGHT, padx=(6, 0))

        cancel_btn = tk.Button(bottom_bar, text="Cancel", font=("Segoe UI", 9), bg=BG_CARD_LIGHT, fg=TEXT_WHITE,
                               relief=tk.FLAT, padx=12, pady=6, cursor="hand2", command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT)

    # --------------------------------------------------------------------------
    # Window Close Handler
    # --------------------------------------------------------------------------
    def _on_close(self):
        logger.info("Closing VisionGuard Native GUI...")
        self.is_running = False
        if self.pipeline and self.pipeline.stream:
            self.pipeline.stream.stop()
        if not getattr(self, "is_embedded", False):
            self.root.destroy()


def launch_gui():
    """Entry point for standalone execution."""
    # Force reliable TCP transport for RTSP streams
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["PYTHONUNBUFFERED"] = "1"

    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
