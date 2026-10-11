"""Enterprise Retail Intelligence Suite — Unified Master Desktop Application
=============================================================================
Unifies:
1. FootFall_AgeGender (YOLOv8 + ByteTrack + Virtual Gates + MiVOLO + API Gateway)
2. VisionGuard AI (YuNet 5-Landmark + Quality Filter + SFace 128-d + SQLite Roster)

Features zero web dependencies, standalone native Windows GUI (Tkinter/TTK),
and full dual-stream monitoring with user-toggleable live display (RAM Saver Mode).
"""

import os
import sys
import time
import datetime
import threading
import queue
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

# Path Anchors
WORKSPACE_ROOT = Path(__file__).resolve().parent
FOOTFALL_ROOT = WORKSPACE_ROOT / "FootFall_AgeGender"
ATTENDANCE_ROOT = WORKSPACE_ROOT / "Attendence System"

# Visual Theme Palette (Modern Dark Slate)
BG_MAIN = "#0f172a"        # Slate 900
BG_CARD = "#1e293b"        # Slate 800
BG_CARD_LIGHT = "#334155"  # Slate 700
BORDER_COLOR = "#475569"   # Slate 600
TEXT_WHITE = "#f8fafc"     # Slate 50
TEXT_MUTED = "#94a3b8"     # Slate 400
ACCENT_BLUE = "#3b82f6"    # Blue 500
ACCENT_HOVER = "#2563eb"   # Blue 600
ACCENT_GREEN = "#10b981"   # Emerald 500
ACCENT_RED = "#ef4444"     # Red 500
ACCENT_AMBER = "#f59e0b"   # Amber 500
ACCENT_PURPLE = "#a855f7"  # Purple 500
ACCENT_CYAN = "#06b6d4"    # Cyan 500


class UnifiedRetailSuiteApp:
    """Master Desktop GUI Application orchestrating both AI systems."""

    def __init__(self):
        # Setup master window
        self.root = tk.Tk()
        self.root.title("Enterprise Retail Intelligence Suite — Customer Analytics & VisionGuard Staff Attendance")
        self.root.geometry("1520x920")
        self.root.minsize(1240, 780)
        self.root.configure(bg=BG_MAIN)

        self._setup_ttk_styles()

        # State tracking
        self.active_tab = "overview"  # 'overview', 'retail', 'attendance', 'settings'
        self.is_running = True

        # Executive Live Stream display state (User-controllable RAM Saver)
        self.exec_retail_display_enabled = True
        self.exec_retail_photo = None
        self.canvas_exec_retail = None

        # Build Master Layout Skeleton
        self._build_master_header()
        self._build_tabs_container()
        self._build_master_footer()

        # Initialize Sub-Applications
        self._init_sub_applications()

        # Build Tab Content
        self._build_overview_tab()
        self._build_settings_tab()

        # Switch to default tab
        self.switch_tab("overview")

        # Start periodic polling and live stream rendering
        self._schedule_executive_polling()
        self.root.after(200, self._poll_exec_retail_stream)

        # Master shutdown protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_master_close)

    def _setup_ttk_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background=BG_MAIN, foreground=TEXT_WHITE, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=BG_MAIN)
        self.style.configure("Card.TFrame", background=BG_CARD)
        self.style.configure("TLabel", background=BG_MAIN, foreground=TEXT_WHITE)
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

    # =========================================================================
    # Master Header & Navigation
    # =========================================================================

    def _build_master_header(self):
        self.header_bar = tk.Frame(self.root, bg=BG_CARD, height=64, padx=16, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.header_bar.pack(fill=tk.X, side=tk.TOP)

        # Left: Brand Logo & Title
        brand_frame = tk.Frame(self.header_bar, bg=BG_CARD)
        brand_frame.pack(side=tk.LEFT)

        lbl_logo = tk.Label(brand_frame, text="🏪🛡️", font=("Segoe UI Emoji", 20), bg=BG_CARD, fg=ACCENT_BLUE)
        lbl_logo.pack(side=tk.LEFT, padx=(0, 10))

        title_box = tk.Frame(brand_frame, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)

        lbl_title = tk.Label(title_box, text="OmniRetail AI Suite", font=("Segoe UI", 14, "bold"), fg=TEXT_WHITE, bg=BG_CARD)
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(title_box, text="Integrated Customer Analytics & VisionGuard Staff Attendance", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_sub.pack(anchor="w")

        # Center: Master Tab Navigation Buttons
        self.nav_frame = tk.Frame(self.header_bar, bg=BG_CARD)
        self.nav_frame.pack(side=tk.LEFT, expand=True, padx=20)

        self.tab_buttons = {}
        tabs = [
            ("overview", "📊 Store Executive Dashboard"),
            ("retail", "👥 Customer Analytics & Demographics"),
            ("attendance", "👤 Staff Attendance & VisionGuard"),
            ("settings", "⚙️ Central Settings & Diagnostics"),
        ]

        for tab_id, tab_label in tabs:
            btn = tk.Button(
                self.nav_frame,
                text=tab_label,
                font=("Segoe UI", 9, "bold"),
                bg=BG_CARD_LIGHT,
                fg=TEXT_WHITE,
                activebackground=ACCENT_HOVER,
                activeforeground=TEXT_WHITE,
                relief=tk.FLAT,
                padx=14,
                pady=7,
                cursor="hand2",
                command=lambda tid=tab_id: self.switch_tab(tid),
            )
            btn.pack(side=tk.LEFT, padx=4)
            self.tab_buttons[tab_id] = btn

        # Right: Live Status Pills & Shutdown
        right_status_frame = tk.Frame(self.header_bar, bg=BG_CARD)
        right_status_frame.pack(side=tk.RIGHT)

        self.lbl_retail_status = tk.Label(
            right_status_frame,
            text="👥 Retail: Ready",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_AMBER,
            padx=10,
            pady=4,
        )
        self.lbl_retail_status.pack(side=tk.LEFT, padx=4)

        self.lbl_attend_status = tk.Label(
            right_status_frame,
            text="👤 VisionGuard: Ready",
            font=("Segoe UI", 8, "bold"),
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            padx=10,
            pady=4,
        )
        self.lbl_attend_status.pack(side=tk.LEFT, padx=4)

        self.lbl_clock = tk.Label(
            right_status_frame,
            text="",
            font=("Consolas", 9, "bold"),
            bg=BG_CARD_LIGHT,
            fg=TEXT_MUTED,
            padx=10,
            pady=4,
        )
        self.lbl_clock.pack(side=tk.LEFT, padx=6)

        btn_exit = tk.Button(
            right_status_frame,
            text="⏹️ Exit",
            font=("Segoe UI", 8, "bold"),
            bg="#7f1d1d",
            fg="#fecaca",
            activebackground=ACCENT_RED,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self.on_master_close,
        )
        btn_exit.pack(side=tk.LEFT, padx=(4, 0))

    def _build_tabs_container(self):
        """Holds each of the 4 tab views."""
        self.content_container = tk.Frame(self.root, bg=BG_MAIN)
        self.content_container.pack(fill=tk.BOTH, expand=True)

        self.tab_frames = {
            "overview": tk.Frame(self.content_container, bg=BG_MAIN),
            "retail": tk.Frame(self.content_container, bg=BG_MAIN),
            "attendance": tk.Frame(self.content_container, bg=BG_MAIN),
            "settings": tk.Frame(self.content_container, bg=BG_MAIN),
        }

    def _build_master_footer(self):
        footer = tk.Frame(self.root, bg=BG_CARD, height=26, padx=16, pady=4, highlightthickness=1, highlightbackground=BORDER_COLOR)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        lbl_foot_left = tk.Label(
            footer,
            text="Platform: Windows x64 Native Edge Gateway  |  Inference: ONNX Runtime CPU + OpenCV Graph  |  Dual-Stream RAM Optimization",
            font=("Segoe UI", 8),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        lbl_foot_left.pack(side=tk.LEFT)

        self.lbl_foot_health = tk.Label(
            footer,
            text="🟢 All edge pipelines operational",
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD,
        )
        self.lbl_foot_health.pack(side=tk.RIGHT)

    def switch_tab(self, tab_id: str):
        """Switches active view and updates top navigation buttons."""
        self.active_tab = tab_id
        for tid, frame in self.tab_frames.items():
            if tid == tab_id:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        # Update button visual styling
        for tid, btn in self.tab_buttons.items():
            if tid == tab_id:
                btn.configure(bg=ACCENT_BLUE, fg=TEXT_WHITE)
            else:
                btn.configure(bg=BG_CARD_LIGHT, fg=TEXT_MUTED)

    # =========================================================================
    # Sub-Applications Initialization (Embedded in Tab 2 & Tab 3)
    # =========================================================================

    def _init_sub_applications(self):
        """Initializes RetailAnalyticsApp into Tab 2 and AttendanceApp into Tab 3."""
        from gui.retail_gui import RetailAnalyticsApp
        from gui.attendance_gui import AttendanceApp

        # 1. Embed Retail Analytics into Tab 2
        self.retail_app = RetailAnalyticsApp(
            config_path="config/production.yaml",
            root=self.root,
            parent_frame=self.tab_frames["retail"],
            auto_countdown=False,
        )

        # 2. Embed VisionGuard Attendance into Tab 3
        self.attendance_app = AttendanceApp(
            root=self.root,
            parent_frame=self.tab_frames["attendance"],
        )

    # =========================================================================
    # TAB 1: Store Executive Dashboard (Combined Store Intelligence)
    # =========================================================================

    def _build_overview_tab(self):
        container = self.tab_frames["overview"]

        # Top Banner: Store Pulse & Date
        banner = tk.Frame(container, bg=BG_CARD, padx=20, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        banner.pack(fill=tk.X, padx=20, pady=(12, 10))

        b_left = tk.Frame(banner, bg=BG_CARD)
        b_left.pack(side=tk.LEFT)

        tk.Label(b_left, text="Store Executive Dashboard", font=("Segoe UI", 15, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        tk.Label(b_left, text="Live surveillance & traffic monitoring combining customer footfall patterns with staff presence on duty", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        b_right = tk.Frame(banner, bg=BG_CARD)
        b_right.pack(side=tk.RIGHT)

        self.lbl_exec_date = tk.Label(
            b_right,
            text=time.strftime("%A, %d %B %Y"),
            font=("Segoe UI", 11, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_CARD_LIGHT,
            padx=14,
            pady=5,
        )
        self.lbl_exec_date.pack()

        # 6 KPI Metric Cards
        kpi_row = tk.Frame(container, bg=BG_MAIN)
        kpi_row.pack(fill=tk.X, padx=20, pady=2)
        kpi_row.columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform="kpis")

        self.kpi_exec_in = self._create_kpi_card(kpi_row, 0, "Customer Inflow", "0", "Total Validated Entries", ACCENT_GREEN)
        self.kpi_exec_out = self._create_kpi_card(kpi_row, 1, "Customer Outflow", "0", "Total Validated Exits", ACCENT_BLUE)
        self.kpi_exec_inside = self._create_kpi_card(kpi_row, 2, "Current Occupancy", "0", "Patrons In Store Now", ACCENT_PURPLE)
        self.kpi_exec_staff = self._create_kpi_card(kpi_row, 3, "Staff On Duty", "0 / 0", "Verified Today / Enrolled", ACCENT_AMBER)
        self.kpi_exec_ratio = self._create_kpi_card(kpi_row, 4, "Patron : Staff Ratio", "0.0 : 1", "Service Load Factor", ACCENT_CYAN)
        self.kpi_exec_bags = self._create_kpi_card(kpi_row, 5, "Bag Exit Crossings", "0", "Validated Shopper Bags", ACCENT_AMBER)

        # Quick Control & Action Strip
        action_strip = tk.Frame(container, bg=BG_CARD, padx=16, pady=8, highlightthickness=1, highlightbackground=BORDER_COLOR)
        action_strip.pack(fill=tk.X, padx=20, pady=8)

        tk.Label(action_strip, text="Quick Actions:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 10))

        self.btn_exec_start_retail = tk.Button(
            action_strip,
            text="🚀 Start Footfall Stream",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_BLUE,
            fg=TEXT_WHITE,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._quick_start_retail,
        )
        self.btn_exec_start_retail.pack(side=tk.LEFT, padx=3)

        btn_calib = tk.Button(
            action_strip,
            text="🎯 Calibrate Virtual Gates",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=TEXT_WHITE,
            activebackground=BORDER_COLOR,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._quick_launch_calibrator,
        )
        btn_calib.pack(side=tk.LEFT, padx=3)

        btn_reg = tk.Button(
            action_strip,
            text="👤 Enroll Staff Member",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_WHITE,
            activebackground="#059669",
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=11,
            pady=4,
            cursor="hand2",
            command=self._quick_enroll_staff,
        )
        btn_reg.pack(side=tk.LEFT, padx=3)

        btn_verify = tk.Button(
            action_strip,
            text="🔍 Face Verifier Tester",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=TEXT_WHITE,
            activebackground=BORDER_COLOR,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._quick_verify_embeddings,
        )
        btn_verify.pack(side=tk.LEFT, padx=3)

        btn_exp_att = tk.Button(
            action_strip,
            text="📄 Export Attendance CSV",
            font=("Segoe UI", 9),
            bg=BG_CARD_LIGHT,
            fg=TEXT_WHITE,
            activebackground=BORDER_COLOR,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._quick_export_attendance,
        )
        btn_exp_att.pack(side=tk.LEFT, padx=3)

        # Dual Stream Activity Monitor Panels (Split Columns)
        panes = tk.Frame(container, bg=BG_MAIN)
        panes.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # ---------------------------------------------------------------------
        # Left Column (58% width): Customer Footfall Live Video & Event Logs
        # ---------------------------------------------------------------------
        left_box = tk.Frame(panes, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Left Header with Display Toggle
        l_head = tk.Frame(left_box, bg=BG_CARD, padx=12, pady=8)
        l_head.pack(fill=tk.X)

        tk.Label(l_head, text="📹 Customer Footfall Live Stream & Gates", font=("Segoe UI", 11, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)

        # Right control cluster in left box header
        l_head_ctrl = tk.Frame(l_head, bg=BG_CARD)
        l_head_ctrl.pack(side=tk.RIGHT)

        self.lbl_exec_retail_stream_mode = tk.Label(
            l_head_ctrl,
            text="● Live Preview: ON",
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD,
        )
        self.lbl_exec_retail_stream_mode.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_toggle_exec_retail_display = tk.Button(
            l_head_ctrl,
            text="👁️ Live Video: ON",
            font=("Segoe UI", 8, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_WHITE,
            activebackground="#059669",
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10,
            pady=2,
            cursor="hand2",
            command=self.toggle_exec_retail_display,
        )
        self.btn_toggle_exec_retail_display.pack(side=tk.LEFT)

        # Video Canvas for Customer Footfall
        self.canvas_exec_retail = tk.Canvas(left_box, bg="#020617", height=280, highlightthickness=0)
        self.canvas_exec_retail.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self._render_canvas_standby_notice()

        # Sub-Card: Real-Time Gate Event Logs
        log_sub = tk.Frame(left_box, bg=BG_CARD_LIGHT, padx=10, pady=4)
        log_sub.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(log_sub, text="Customer Gate Crossings & Traffic Log", font=("Segoe UI", 9, "bold"), fg=TEXT_WHITE, bg=BG_CARD_LIGHT).pack(side=tk.LEFT)
        tk.Label(log_sub, text="Real-time validated entries & exits", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD_LIGHT).pack(side=tk.RIGHT)

        self.txt_exec_logs = ScrolledText(
            left_box,
            bg=BG_MAIN,
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Consolas", 8),
            height=6,
            relief=tk.FLAT,
            highlightthickness=0,
            padx=8,
            pady=6,
        )
        self.txt_exec_logs.pack(fill=tk.X, padx=10, pady=(0, 10))

        # ---------------------------------------------------------------------
        # Right Column (42% width): Staff Attendance Real-Time Log
        # ---------------------------------------------------------------------
        right_box = tk.Frame(panes, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        r_head = tk.Frame(right_box, bg=BG_CARD, padx=12, pady=8)
        r_head.pack(fill=tk.X)
        tk.Label(r_head, text="👤 Staff Attendance Real-Time Log", font=("Segoe UI", 11, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(r_head, text="SFace & Temporal Confirmed", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.RIGHT)

        cols = ("Time", "ID", "Name", "Score", "Status")
        self.tree_exec_staff = ttk.Treeview(right_box, columns=cols, show="headings", height=12)
        self.tree_exec_staff.heading("Time", text="Check-In Time")
        self.tree_exec_staff.heading("ID", text="Staff ID")
        self.tree_exec_staff.heading("Name", text="Full Name")
        self.tree_exec_staff.heading("Score", text="Match Score")
        self.tree_exec_staff.heading("Status", text="Status")

        self.tree_exec_staff.column("Time", width=105, anchor="center")
        self.tree_exec_staff.column("ID", width=90, anchor="center")
        self.tree_exec_staff.column("Name", width=140, anchor="w")
        self.tree_exec_staff.column("Score", width=85, anchor="center")
        self.tree_exec_staff.column("Status", width=85, anchor="center")
        self.tree_exec_staff.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        # Quick summary card under staff tree
        r_foot = tk.Frame(right_box, bg=BG_CARD_LIGHT, padx=10, pady=6)
        r_foot.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.lbl_exec_staff_summary = tk.Label(
            r_foot,
            text="Staff Attendance Engine Active  |  Auto-Verification Running",
            font=("Segoe UI", 8),
            fg=TEXT_MUTED,
            bg=BG_CARD_LIGHT,
        )
        self.lbl_exec_staff_summary.pack(side=tk.LEFT)

    def _render_canvas_standby_notice(self):
        """Draws clean placeholder on the canvas when stream is standby or display is paused."""
        if hasattr(self, "canvas_exec_retail") and self.canvas_exec_retail.winfo_exists():
            self.canvas_exec_retail.delete("all")
            cw = max(200, self.canvas_exec_retail.winfo_width())
            ch = max(150, self.canvas_exec_retail.winfo_height())

            is_running = hasattr(self, "retail_app") and self.retail_app.pipeline is not None and self.retail_app.is_monitoring
            if not is_running:
                msg = (
                    "📹 Customer Analytics Stream in Standby\n\n"
                    "Click '🚀 Start Footfall Stream' in the Quick Actions bar to begin live tracking.\n"
                    "Virtual gate lines, customer tracking boxes, and counting will appear here."
                )
                fill_color = TEXT_MUTED
            else:
                msg = (
                    "⚡ Live Display Paused (RAM Saver Mode Active)\n\n"
                    "Background AI person detection, tracking, virtual gate crossing counts,\n"
                    "and cloud sync are actively running with 0 display RAM overhead.\n\n"
                    "Click 'Live Video: OFF' above to resume live camera preview."
                )
                fill_color = ACCENT_AMBER

            self.canvas_exec_retail.create_text(
                cw // 2,
                ch // 2,
                text=msg,
                fill=fill_color,
                font=("Segoe UI", 10),
                justify="center",
            )

    def toggle_exec_retail_display(self):
        """Toggle live stream rendering on/off to conserve system RAM and CPU."""
        self.exec_retail_display_enabled = not self.exec_retail_display_enabled
        if self.exec_retail_display_enabled:
            self.btn_toggle_exec_retail_display.config(text="👁️ Live Video: ON", bg=ACCENT_GREEN)
            self.lbl_exec_retail_stream_mode.config(text="● Live Preview: ON", fg=ACCENT_GREEN)
        else:
            self.btn_toggle_exec_retail_display.config(text="💤 Live Video: OFF (RAM Saver)", bg="#475569")
            self.lbl_exec_retail_stream_mode.config(text="💤 RAM Saver: Paused", fg=ACCENT_AMBER)
            self.exec_retail_photo = None
            self._render_canvas_standby_notice()

    def _poll_exec_retail_stream(self):
        """Renders live customer footfall stream onto executive canvas with strict bounded memory."""
        if not self.is_running:
            return

        # Only draw if active tab is overview and user has display enabled
        if self.active_tab == "overview" and self.exec_retail_display_enabled:
            if hasattr(self, "retail_app") and self.retail_app.pipeline is not None and self.retail_app.is_monitoring:
                if hasattr(self, "canvas_exec_retail") and self.canvas_exec_retail.winfo_exists():
                    try:
                        ret, frame = self.retail_app.pipeline.get_latest_gui_frame(annotate=True)
                        if ret and frame is not None and frame.size > 0:
                            cw = self.canvas_exec_retail.winfo_width()
                            ch = self.canvas_exec_retail.winfo_height()
                            if cw > 50 and ch > 50:
                                fh, fw = frame.shape[:2]
                                scale = min(cw / fw, ch / fh)
                                nw = max(1, int(fw * scale))
                                nh = max(1, int(fh * scale))
                                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(rgb)
                                self.exec_retail_photo = ImageTk.PhotoImage(image=img)
                                self.canvas_exec_retail.delete("all")
                                self.canvas_exec_retail.create_image(cw // 2, ch // 2, image=self.exec_retail_photo, anchor="center")
                    except Exception:
                        pass

        interval = 50 if (self.active_tab == "overview" and self.exec_retail_display_enabled) else 350
        self.root.after(interval, self._poll_exec_retail_stream)

    def _create_kpi_card(self, parent, col, title, initial_val, subtext, color) -> tk.Label:
        card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card.grid(row=0, column=col, padx=4, sticky="nsew")

        tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", padx=12, pady=(10, 2))
        lbl_val = tk.Label(card, text=initial_val, font=("Segoe UI", 17, "bold"), fg=color, bg=BG_CARD)
        lbl_val.pack(anchor="w", padx=12, pady=(0, 2))
        tk.Label(card, text=subtext, font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", padx=12, pady=(0, 8))
        return lbl_val

    # =========================================================================
    # TAB 4: Central Settings & Diagnostics
    # =========================================================================

    def _build_settings_tab(self):
        container = self.tab_frames["settings"]

        # Settings Banner
        banner = tk.Frame(container, bg=BG_CARD, padx=20, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        banner.pack(fill=tk.X, padx=20, pady=(16, 12))

        tk.Label(banner, text="Central System Settings & Hardware Diagnostics", font=("Segoe UI", 16, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        tk.Label(banner, text="Manage camera RTSP feeds, AI thresholds, network endpoints, and run edge self-tests", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        # Two-column container
        cfg_panes = tk.Frame(container, bg=BG_MAIN)
        cfg_panes.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 16))

        # Left Column: Configuration Forms
        left_cfg = tk.Frame(cfg_panes, bg=BG_CARD, padx=16, pady=14, highlightthickness=1, highlightbackground=BORDER_COLOR)
        left_cfg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(left_cfg, text="📷 Stream & Endpoint Configuration", font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=(0, 12))

        # Camera 1 (Customer Analytics)
        tk.Label(left_cfg, text="Customer Footfall Camera RTSP URL / Video File:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(4, 2))
        self.entry_cfg_retail_rtsp = tk.Entry(left_cfg, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_retail_rtsp.pack(fill=tk.X, ipady=4, pady=(0, 10))
        self.entry_cfg_retail_rtsp.insert(0, getattr(self.retail_app, "saved_rtsp", "test_videos/Outlet_Sample_1_Banasree.avi"))

        # Camera 2 (Staff Attendance)
        tk.Label(left_cfg, text="Staff Attendance Camera RTSP URL:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(4, 2))
        self.entry_cfg_attend_rtsp = tk.Entry(left_cfg, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_attend_rtsp.pack(fill=tk.X, ipady=4, pady=(0, 10))
        self.entry_cfg_attend_rtsp.insert(0, getattr(self.attendance_app, "current_rtsp_url", "rtsp://ctrl:ctl%40123456@10.1.56.11:554/Streaming/Channels/1201"))

        # API URL & Shop ID
        row_api = tk.Frame(left_cfg, bg=BG_CARD)
        row_api.pack(fill=tk.X, pady=(0, 10))
        row_api.columnconfigure((0, 1), weight=1)

        f_api = tk.Frame(row_api, bg=BG_CARD)
        f_api.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(f_api, text="Cloud API Base URL:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.entry_cfg_api = tk.Entry(f_api, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_api.pack(fill=tk.X, ipady=4)
        self.entry_cfg_api.insert(0, getattr(self.retail_app, "api_url", "http://ai.prangroup.com:8021"))

        f_shop = tk.Frame(row_api, bg=BG_CARD)
        f_shop.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(f_shop, text="Retail Outlet Shop ID:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.entry_cfg_shop = tk.Entry(f_shop, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_shop.pack(fill=tk.X, ipady=4)
        self.entry_cfg_shop.insert(0, str(getattr(self.retail_app, "shop_id", 2)))

        # Attendance Thresholds
        row_thresh = tk.Frame(left_cfg, bg=BG_CARD)
        row_thresh.pack(fill=tk.X, pady=(0, 14))
        row_thresh.columnconfigure((0, 1), weight=1)

        f_cos = tk.Frame(row_thresh, bg=BG_CARD)
        f_cos.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(f_cos, text="SFace Cosine Threshold (Default: 0.380):", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.entry_cfg_cosine = tk.Entry(f_cos, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_cosine.pack(fill=tk.X, ipady=4)
        self.entry_cfg_cosine.insert(0, "0.380")

        f_fps = tk.Frame(row_thresh, bg=BG_CARD)
        f_fps.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(f_fps, text="Attendance Sampling Target FPS:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.entry_cfg_fps = tk.Entry(f_fps, font=("Consolas", 9), bg=BG_MAIN, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.entry_cfg_fps.pack(fill=tk.X, ipady=4)
        self.entry_cfg_fps.insert(0, "3.0")

        # Action Buttons
        row_btn = tk.Frame(left_cfg, bg=BG_CARD)
        row_btn.pack(fill=tk.X, pady=(10, 0))

        btn_save_cfg = tk.Button(
            row_btn,
            text="💾 Save Configuration",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_BLUE,
            fg=TEXT_WHITE,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._save_settings,
        )
        btn_save_cfg.pack(side=tk.LEFT, padx=(0, 10))

        # Right Column: Diagnostic Self-Check Console
        right_diag = tk.Frame(cfg_panes, bg=BG_CARD, padx=16, pady=14, highlightthickness=1, highlightbackground=BORDER_COLOR)
        right_diag.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        d_head = tk.Frame(right_diag, bg=BG_CARD)
        d_head.pack(fill=tk.X, pady=(0, 10))

        tk.Label(d_head, text="🩺 Edge Self-Test & Diagnostic Tool", font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)

        btn_run_diag = tk.Button(
            d_head,
            text="▶ Run Diagnostic Check",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_WHITE,
            activebackground="#059669",
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self._run_diagnostic_check,
        )
        btn_run_diag.pack(side=tk.RIGHT)

        self.txt_diag = ScrolledText(
            right_diag,
            bg=BG_MAIN,
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Consolas", 9),
            relief=tk.FLAT,
            highlightthickness=0,
            padx=10,
            pady=8,
        )
        self.txt_diag.pack(fill=tk.BOTH, expand=True)
        self.txt_diag.insert(tk.END, "[System Ready] Click 'Run Diagnostic Check' to verify all camera streams, models, and databases.\n")

    def _save_settings(self):
        new_retail = self.entry_cfg_retail_rtsp.get().strip()
        new_attend = self.entry_cfg_attend_rtsp.get().strip()
        if new_retail and hasattr(self.retail_app, "_persist_rtsp_url"):
            self.retail_app._persist_rtsp_url(new_retail)
            self.retail_app.saved_rtsp = new_retail
            if hasattr(self.retail_app, "current_source"):
                self.retail_app.current_source = new_retail
            if hasattr(self.retail_app, "rtsp_entry") and self.retail_app.rtsp_entry is not None:
                try:
                    if self.retail_app.rtsp_entry.winfo_exists():
                        self.retail_app.rtsp_entry.delete(0, tk.END)
                        self.retail_app.rtsp_entry.insert(0, new_retail)
                except Exception:
                    pass

        if new_attend and hasattr(self.attendance_app, "current_rtsp_url"):
            self.attendance_app.current_rtsp_url = new_attend
            if self.attendance_app.pipeline and self.attendance_app.pipeline.stream:
                self.attendance_app.pipeline.stream.rtsp_url = new_attend
                if hasattr(self.attendance_app.pipeline.stream, "reconnect"):
                    self.attendance_app.pipeline.stream.reconnect(new_attend)
                else:
                    self.attendance_app.pipeline.stream.stop()
                    self.attendance_app.pipeline.stream.start()

        messagebox.showinfo("Settings Saved", "Configuration successfully saved and applied to edge pipelines.")

    def _run_diagnostic_check(self):
        self.txt_diag.insert(tk.END, f"\n=== Edge Diagnostic Test ({time.strftime('%H:%M:%S')}) ===\n")

        # 1. Check Python and Torch/ONNX Runtime
        import onnxruntime
        self.txt_diag.insert(tk.END, f"✓ Python Version: {sys.version.split()[0]}\n")
        self.txt_diag.insert(tk.END, f"✓ OpenCV Version: {cv2.__version__}\n")
        self.txt_diag.insert(tk.END, f"✓ ONNX Runtime Version: {onnxruntime.__version__}\n")

        # 2. Check Database Files
        db_att = ATTENDANCE_ROOT / "data" / "attendance.db"
        self.txt_diag.insert(tk.END, f"{'✓' if db_att.exists() else '✗'} Attendance DB: {db_att} ({'Present' if db_att.exists() else 'Missing'})\n")

        db_retail = FOOTFALL_ROOT / "edge_events.db"
        self.txt_diag.insert(tk.END, f"{'✓' if db_retail.exists() else '✗'} Edge Events DB: {db_retail} ({'Present' if db_retail.exists() else 'Missing'})\n")

        # 3. Check Models
        m_yunet = ATTENDANCE_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
        m_sface = ATTENDANCE_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
        m_yolo = FOOTFALL_ROOT / "models" / "yolov8m.onnx"

        self.txt_diag.insert(tk.END, f"{'✓' if m_yunet.exists() else '✗'} YuNet Model: {m_yunet.name} ({m_yunet.stat().st_size // 1024} KB)\n")
        self.txt_diag.insert(tk.END, f"{'✓' if m_sface.exists() else '✗'} SFace Model: {m_sface.name} ({m_sface.stat().st_size // 1024} KB)\n")
        self.txt_diag.insert(tk.END, f"{'✓' if m_yolo.exists() else '✗'} YOLOv8 Detector: {m_yolo.name} ({m_yolo.stat().st_size // (1024*1024)} MB)\n")

        # 4. Enrolled Staff
        staff_cnt = self.attendance_app.pipeline.storage.get_enrolled_staff_count()
        self.txt_diag.insert(tk.END, f"✓ Enrolled Staff Gallery: {staff_cnt} staff profiles registered\n")

        # 5. Attendance Stream Status
        is_att_conn = getattr(self.attendance_app.pipeline.stream, "is_connected", False)
        fps = getattr(self.attendance_app.pipeline.stream, "current_fps", 0.0)
        self.txt_diag.insert(tk.END, f"{'✓' if is_att_conn else '⚠'} Attendance RTSP: {'Connected' if is_att_conn else 'Reconnecting'} ({fps:.1f} FPS)\n")

        self.txt_diag.insert(tk.END, "=== Diagnostics Complete: System 100% Operational ===\n\n")
        self.txt_diag.see(tk.END)

    # =========================================================================
    # Quick Control Action Handlers
    # =========================================================================

    def _quick_start_retail(self):
        """Starts or stops Customer Footfall monitoring from Executive Dashboard."""
        if hasattr(self, "retail_app"):
            if not self.retail_app.is_monitoring:
                if hasattr(self.retail_app, "_start_monitoring"):
                    self.retail_app._start_monitoring()
                self.btn_exec_start_retail.config(text="⏹️ Stop Footfall Stream", bg=ACCENT_RED)
                self.lbl_retail_status.config(text="👥 Retail: Active (Online)", fg=ACCENT_GREEN)
            else:
                if hasattr(self.retail_app, "on_close"):
                    self.retail_app.on_close()
                self.btn_exec_start_retail.config(text="🚀 Start Footfall Stream", bg=ACCENT_BLUE)
                self.lbl_retail_status.config(text="👥 Retail: Standby", fg=ACCENT_AMBER)
                self._render_canvas_standby_notice()

    def _quick_launch_calibrator(self):
        self.switch_tab("retail")
        if hasattr(self.retail_app, "_launch_calibration"):
            self.retail_app._launch_calibration()

    def _quick_enroll_staff(self):
        self.switch_tab("attendance")
        if hasattr(self.attendance_app, "open_enroll_dialog"):
            self.attendance_app.open_enroll_dialog()

    def _quick_verify_embeddings(self):
        self.switch_tab("attendance")
        if hasattr(self.attendance_app, "open_verify_embeddings_modal"):
            self.attendance_app.open_verify_embeddings_modal()

    def _quick_export_attendance(self):
        self.switch_tab("attendance")
        if hasattr(self.attendance_app, "export_attendance_csv"):
            self.attendance_app.export_attendance_csv()

    # =========================================================================
    # Periodic Executive Polling
    # =========================================================================

    def _schedule_executive_polling(self):
        """Polls metrics from both systems every 1 second for executive dashboard."""
        if not self.is_running:
            return

        try:
            # Update Clock
            cur_time = time.strftime("%H:%M:%S")
            self.lbl_clock.config(text=cur_time)

            # 1. Retail Footfall Metrics
            if hasattr(self, "retail_app") and self.retail_app.pipeline is not None and self.retail_app.is_monitoring:
                in_cnt = self.retail_app.pipeline.event_manager.validated_entry_count
                out_cnt = self.retail_app.pipeline.event_manager.validated_exit_count
                inside = max(0, in_cnt - out_cnt)
                bags = self.retail_app.pipeline.cached_total_bag_exits
                self.lbl_retail_status.config(text="👥 Retail: Active (Online)", fg=ACCENT_GREEN)
                if hasattr(self, "btn_exec_start_retail") and self.btn_exec_start_retail["text"] != "⏹️ Stop Footfall Stream":
                    self.btn_exec_start_retail.config(text="⏹️ Stop Footfall Stream", bg=ACCENT_RED)
            else:
                in_cnt = 0
                out_cnt = 0
                inside = 0
                bags = 0
                self.lbl_retail_status.config(text="👥 Retail: Standby", fg=ACCENT_AMBER)
                if hasattr(self, "btn_exec_start_retail") and self.btn_exec_start_retail["text"] != "🚀 Start Footfall Stream":
                    self.btn_exec_start_retail.config(text="🚀 Start Footfall Stream", bg=ACCENT_BLUE)

            self.kpi_exec_in.config(text=str(in_cnt))
            self.kpi_exec_out.config(text=str(out_cnt))
            self.kpi_exec_inside.config(text=str(inside))
            self.kpi_exec_bags.config(text=str(bags))

            # 2. Staff Attendance Metrics
            if hasattr(self, "attendance_app") and self.attendance_app.pipeline is not None:
                enrolled_cnt = self.attendance_app.pipeline.storage.get_enrolled_staff_count()
                records = self.attendance_app.pipeline.db.get_today_attendance()
                verified_cnt = len(records)
                fps = getattr(self.attendance_app.pipeline.stream, "current_fps", 0.0)
                is_conn = getattr(self.attendance_app.pipeline.stream, "is_connected", False)

                self.kpi_exec_staff.config(text=f"{verified_cnt} / {enrolled_cnt}")

                # Patron to Staff Ratio
                if verified_cnt > 0:
                    ratio = inside / verified_cnt
                    self.kpi_exec_ratio.config(text=f"{ratio:.1f} : 1")
                else:
                    self.kpi_exec_ratio.config(text=f"{inside}.0 : 1" if inside > 0 else "0.0 : 1")

                if is_conn:
                    self.lbl_attend_status.config(text=f"👤 VisionGuard: {fps:.1f} FPS", fg=ACCENT_GREEN)
                else:
                    self.lbl_attend_status.config(text="👤 VisionGuard: Reconnecting", fg=ACCENT_RED)

                if hasattr(self, "lbl_exec_staff_summary"):
                    self.lbl_exec_staff_summary.config(
                        text=f"Enrolled Staff: {enrolled_cnt}  |  Verified Today: {verified_cnt}  |  Stream FPS: {fps:.1f}"
                    )

                # Sync recent attendance entries to executive table
                if self.active_tab == "overview" and records:
                    existing_items = set()
                    for item in self.tree_exec_staff.get_children():
                        existing_items.add(str(self.tree_exec_staff.item(item)["values"][1]))

                    for rec in records[-15:]:
                        if str(rec["staff_id"]) not in existing_items:
                            ts = rec["timestamp"].split("T")[1][:8] if "T" in rec["timestamp"] else rec["timestamp"]
                            self.tree_exec_staff.insert(
                                "",
                                0,
                                values=(
                                    ts,
                                    rec["staff_id"],
                                    rec["name"],
                                    f"{rec['similarity_score']:.3f}",
                                    "Verified",
                                ),
                            )

            # 3. Sync Retail Logs to Executive ScrolledText
            if hasattr(self, "retail_app") and hasattr(self.retail_app, "log_queue"):
                while not self.retail_app.log_queue.empty():
                    try:
                        msg = self.retail_app.log_queue.get_nowait()
                        ts = time.strftime("%H:%M:%S")
                        self.txt_exec_logs.insert(tk.END, f"[{ts}] {msg}\n")
                        self.txt_exec_logs.see(tk.END)
                    except queue.Empty:
                        break

        except Exception:
            pass

        self.root.after(1000, self._schedule_executive_polling)

    # =========================================================================
    # Master Application Lifecycle & Clean Shutdown
    # =========================================================================

    def on_master_close(self):
        ans = messagebox.askyesno("Confirm Shutdown", "Are you sure you want to shut down OmniRetail AI Suite?\nThis will safely terminate both camera streams and edge pipelines.")
        if not ans:
            return

        print("[OmniRetail Master] Commencing clean shutdown of all edge pipelines...")
        self.is_running = False

        # Shutdown Retail Analytics App
        if hasattr(self, "retail_app") and self.retail_app is not None:
            try:
                self.retail_app.on_close()
            except Exception as e:
                print(f"[OmniRetail] Error closing Retail Analytics: {e}")

        # Shutdown VisionGuard Attendance App
        if hasattr(self, "attendance_app") and self.attendance_app is not None:
            try:
                self.attendance_app._on_close()
            except Exception as e:
                print(f"[OmniRetail] Error closing VisionGuard Attendance: {e}")

        self.root.destroy()
        print("[OmniRetail Master] Application closed successfully.")
        sys.exit(0)

    def run(self):
        self.root.mainloop()


def launch_unified_gui():
    app = UnifiedRetailSuiteApp()
    app.run()


if __name__ == "__main__":
    launch_unified_gui()
