"""Windows Focus Guard: local webcam + active-tab distraction monitor.

The heavy work (camera capture, MediaPipe Face Mesh, phone detection, screen
grab) runs on background threads in ``vision_worker``; this module keeps a light
Tk loop that reads the latest results, fuses them through the decision tree in
``distraction_model``, smooths the outcome, and raises alerts.
"""
from __future__ import annotations

import csv
import ctypes
import os
import threading
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import sounddevice as sd
from PIL import Image, ImageTk

import session_chart
import settings as settings_store
from distraction_model import DISTRACTED, FOCUSED, HeuristicTree, describe_model, load_model
from session_logger import SessionLogger
from vision_worker import ScreenWorker, VisionWorker

try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = mouse = None

APP_TITLE = "Focus Guard — local only"
APP_DIR = Path(__file__).resolve().parent
HISTORY_PATH = APP_DIR / "session_history.csv"

# --- palette -------------------------------------------------------------
BG = "#0f1420"        # window background
CARD = "#1a2230"      # raised panel
FIELD = "#232d3d"     # input background
INK = "#e8edf4"       # primary text
MUTE = "#8b98ab"      # secondary text
GREEN = "#35b17e"
AMBER = "#e0a13a"
RED = "#e0574a"
BANNER = {
    "idle": "#2c3a4f",
    "good": "#1f7a4e",
    "warn": "#8a6320",
    "bad": "#a03a30",
    "pause": "#3a4a63",
}


def foreground_title() -> str:
    """Read the active Windows title; it is never stored."""
    window = ctypes.windll.user32.GetForegroundWindow()
    size = ctypes.windll.user32.GetWindowTextLengthW(window)
    text = ctypes.create_unicode_buffer(size + 1)
    ctypes.windll.user32.GetWindowTextW(window, text, size + 1)
    return text.value


class FocusGuard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.prefs = settings_store.load()
        self.title(APP_TITLE)
        self.geometry(self._sane_geometry(self.prefs.get("geometry")))
        self.minsize(900, 830)
        self.configure(padx=22, pady=20, bg=BG)
        self.columnconfigure(0, weight=1)
        self._settings_win = None
        self._gaze_hold_until = 0.0

        self.target = tk.StringVar(value=self.prefs["target"])
        self.grace = tk.IntVar(value=self.prefs["grace"])
        self.camera_index = tk.IntVar(value=self.prefs["camera_index"])
        self.alarm_volume = tk.DoubleVar(value=self.prefs["alarm_volume"])
        self.alarm_file = tk.StringVar(value=self.prefs["alarm_file"])
        self.preview_enabled = tk.BooleanVar(value=self.prefs["preview_enabled"])
        self.camera_enabled = tk.BooleanVar(value=self.prefs["camera_enabled"])
        self.realert_seconds = tk.IntVar(value=self.prefs["realert_seconds"])
        self.sensitivity = tk.DoubleVar(value=self.prefs["sensitivity"])
        self.calibration_seconds = tk.IntVar(value=self.prefs["calibration_seconds"])
        self.break_minutes = tk.IntVar(value=self.prefs["break_minutes"])
        self.save_summaries = tk.BooleanVar(value=False)
        self.log_label = tk.StringVar(value="focused")

        self.running = False
        self.focused = self.off_task = 0.0
        self.alerts = 0
        self.score = 0.0
        self.current_streak = self.best_streak = 0.0
        self.reason_counts: Counter = Counter()
        self.off_started = None
        self.distraction_started = None
        self.distraction_alert_sent = False
        self.last_alert_time = 0.0
        self.paused_until = 0.0
        self.last_tick = time.monotonic()
        self.task_tab_active = False
        self._last_cal_seq = 0

        self.alarm_playing = False
        self.visual_alert_until = 0.0
        self.visual_alert_on = False

        self.current_title = foreground_title()
        self.current_title_started = time.monotonic()
        self.title_switch_times: deque = deque()
        self.last_keyboard_activity = time.monotonic()
        self.last_mouse_activity = time.monotonic()
        self.input_listeners: list = []

        self.model = load_model()
        self.model_backend = tk.StringVar(value=f"Model: {describe_model(self.model)}")
        # A trained tree gets a second opinion from the heuristic tree every
        # tick so a session summary can report how often they agree — a
        # cheap ablation of whether learning from data actually changed
        # anything versus the hand-tuned thresholds.
        self.compare_model = None if isinstance(self.model, HeuristicTree) else HeuristicTree()
        self.model_compared = 0
        self.model_agree = 0
        self.session_start = time.monotonic()
        self.score_history: list = []
        self.alert_events: list = []
        self.resting_pose_samples: deque = deque(maxlen=400)
        self.drift_prompts = 0
        self._drift_notice = ""
        self._last_drift_prompt = 0.0
        self.logger = SessionLogger()
        self.log_label.trace_add("write", lambda *_: self.logger.set_label(self.log_label.get()))

        self.vision = VisionWorker()
        self.screen = ScreenWorker()

        self.metrics = {
            "Head rotation": tk.StringVar(value="--"),
            "Eye openness": tk.StringVar(value="--"),
            "Looking away": tk.StringVar(value="0"),
            "Tab switches/min": tk.StringVar(value="0"),
            "Keyboard inactivity": tk.StringVar(value="0 sec"),
            "Mouse inactivity": tk.StringVar(value="0 sec"),
            "Time on current tab": tk.StringVar(value="0 sec"),
            "Non-task tab active": tk.StringVar(value="0"),
            "Model state": tk.StringVar(value="--"),
            "Data logging": tk.StringVar(value="off"),
        }
        self.video_image = self.screen_image = None

        self.make_ui()
        self.bind_shortcuts()
        self.start_input_tracking()
        self.wire_settings_traces()

        self.vision.start()
        self.screen.start()
        self.vision.set_camera_index(self._num(self.camera_index, 0))
        self.vision.set_sensitivity(self._num(self.sensitivity, 1.0))
        if self.camera_enabled.get():
            self.vision.enable()
        if self.preview_enabled.get():
            self.screen.enable()

        if not str(self.prefs.get("geometry") or "").strip():
            self.update_idletasks()
            width, height = 1040, 850
            x = max(0, (self.winfo_screenwidth() - width) // 2)
            y = max(0, (self.winfo_screenheight() - height) // 3)
            self.geometry(f"{width}x{height}+{x}+{y}")

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(150, self.pump)

    # ------------------------------------------------------------------ UI
    def _init_style(self) -> None:
        style = ttk.Style(self)
        self.style = style
        style.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", "#33415a")

        style.configure(".", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Muted.TLabel", background=BG, foreground=MUTE)
        style.configure("H1.TLabel", background=BG, foreground=INK, font=("Segoe UI", 21, "bold"))
        style.configure("Card.TLabel", background=CARD, foreground=INK)
        style.configure("CardMuted.TLabel", background=CARD, foreground=MUTE)
        style.configure("Section.TLabel", background=CARD, foreground=MUTE, font=("Segoe UI", 9, "bold"))

        for name in ("TButton", "Ghost.TButton"):
            style.configure(name, background="#2a3546", foreground=INK, bordercolor="#2a3546",
                            focuscolor="", padding=(13, 7))
            style.map(name, background=[("active", "#33415a"), ("disabled", "#1f2734")],
                      foreground=[("disabled", "#5a6678")])
        style.configure("Accent.TButton", background=GREEN, foreground="#08130d",
                        padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#3ec78f"), ("disabled", "#2a4a3c")],
                  foreground=[("disabled", "#6f8a7d")])

        style.configure("TCheckbutton", background=CARD, foreground=INK, focuscolor="",
                        indicatorbackground=FIELD, indicatorforeground="#08130d")
        style.map("TCheckbutton",
                  background=[("active", CARD)],
                  indicatorbackground=[("selected", GREEN), ("active", FIELD)])
        style.configure("TEntry", fieldbackground=FIELD, foreground=INK, insertcolor=INK,
                        bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD, padding=7)
        for name in ("TSpinbox", "TCombobox"):
            style.configure(name, fieldbackground=FIELD, foreground=INK, insertcolor=INK,
                            bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD, arrowsize=13, padding=5)
            style.map(name, fieldbackground=[("readonly", FIELD)], foreground=[("readonly", INK)])
        style.configure("Horizontal.TScale", background=CARD, troughcolor=FIELD)
        style.configure("Meter.Horizontal.TProgressbar", troughcolor=FIELD, bordercolor=FIELD,
                        background=GREEN, lightcolor=GREEN, darkcolor=GREEN, thickness=16)

    def _card(self, row: int, **grid) -> ttk.Frame:
        frame = ttk.Frame(self, style="Card.TFrame", padding=(18, 15))
        frame.grid(row=row, column=0, sticky=grid.pop("sticky", "ew"), pady=grid.pop("pady", (12, 0)), **grid)
        return frame

    def make_ui(self) -> None:
        self._init_style()

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Focus Guard", style="H1.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Local, on-device distraction monitor — nothing leaves your machine",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(header, text="⚙  Settings", style="Ghost.TButton",
                   command=self.open_settings).grid(row=0, column=1, rowspan=2, sticky="e")

        setup = self._card(1)
        setup.columnconfigure(1, weight=1)
        ttk.Label(setup, text="TASK", style="Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w")
        ttk.Label(setup, text="Tab / window title contains", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        ttk.Entry(setup, textvariable=self.target).grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Label(setup, text="Alert after", style="Card.TLabel").grid(row=1, column=2, sticky="e", padx=(16, 6), pady=(10, 0))
        ttk.Spinbox(setup, from_=1, to=60, textvariable=self.grace, width=4).grid(row=1, column=3, pady=(10, 0))
        ttk.Label(setup, text="sec", style="Card.TLabel").grid(row=1, column=4, sticky="w", padx=(6, 0), pady=(10, 0))
        ttk.Label(setup, text="e.g. “Research Essay” from a Google-Docs tab title",
                  style="CardMuted.TLabel").grid(row=2, column=1, columnspan=4, sticky="w", pady=(6, 0))

        self.status = tk.Label(self, text="Enter a task-title phrase, then press Start (or F5).",
                               bg=BANNER["idle"], fg="white", anchor="w", padx=18, pady=15,
                               font=("Segoe UI", 14, "bold"))
        self.status.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        meter = ttk.Frame(self)
        meter.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        meter.columnconfigure(1, weight=1)
        ttk.Label(meter, text="Distraction", style="Muted.TLabel").grid(row=0, column=0, padx=(0, 12))
        self.score_bar = ttk.Progressbar(meter, style="Meter.Horizontal.TProgressbar", maximum=100)
        self.score_bar.grid(row=0, column=1, sticky="ew")
        self.score_text = ttk.Label(meter, text="0.00", style="Muted.TLabel", width=5, anchor="e")
        self.score_text.grid(row=0, column=2, padx=(12, 0))

        previews = self._card(4, sticky="nsew")
        self.rowconfigure(4, weight=1)
        previews.columnconfigure((0, 1), weight=1)
        previews.rowconfigure(0, weight=1)
        self.camera_label = tk.Label(previews, text="Webcam preview", bg="#10151f", fg=MUTE)
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.screen_label = tk.Label(previews, text="Screen preview", bg="#10151f", fg=MUTE)
        self.screen_label.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.gaze_status = ttk.Label(previews, text="Camera: enable the webcam in Settings, then calibrate (F7).",
                                     style="CardMuted.TLabel")
        self.gaze_status.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.health = ttk.Label(previews, text="", style="CardMuted.TLabel", anchor="e")
        self.health.grid(row=1, column=1, sticky="e", pady=(10, 0))

        action = self._card(5)
        action.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(action, text="Start focus session", style="Accent.TButton", command=self.toggle)
        self.start_button.grid(row=0, column=0)
        self.calibrate_button = ttk.Button(action, text="Calibrate gaze", command=self.start_calibration, state="disabled")
        self.calibrate_button.grid(row=0, column=1, padx=(10, 0))
        break_group = ttk.Frame(action, style="Card.TFrame")
        break_group.grid(row=0, column=2, padx=(10, 0))
        self.pause_button = ttk.Button(break_group, text="Take a break", command=self.toggle_pause)
        self.pause_button.pack(side="left")
        ttk.Spinbox(break_group, from_=1, to=120, textvariable=self.break_minutes, width=3).pack(
            side="left", padx=(6, 3))
        ttk.Label(break_group, text="min", style="Card.TLabel").pack(side="left")
        self.stats = ttk.Label(action, text="Focused 0:00    Off-task 0:00    Alerts 0    Streak 0:00", style="Card.TLabel")
        self.stats.grid(row=0, column=3, sticky="e")

        toggle_row = ttk.Frame(self)
        toggle_row.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        self.signals_toggle = ttk.Label(toggle_row, text="▸  Live signals", style="Muted.TLabel", cursor="hand2")
        self.signals_toggle.pack(side="left")
        self.signals_toggle.bind("<Button-1>", lambda _e: self.toggle_signals())

        self.signals_body = ttk.Frame(self, style="Card.TFrame", padding=(18, 14))
        self.signals_body.columnconfigure((1, 3, 5), weight=1)
        for index, (name, value) in enumerate(self.metrics.items()):
            r, c = divmod(index, 3)
            c *= 2
            ttk.Label(self.signals_body, text=name, style="CardMuted.TLabel").grid(
                row=r, column=c, sticky="w", padx=(0 if c == 0 else 22, 8), pady=3)
            ttk.Label(self.signals_body, textvariable=value, style="Card.TLabel").grid(
                row=r, column=c + 1, sticky="w", pady=3)
        ttk.Label(self.signals_body, textvariable=self.model_backend, style="CardMuted.TLabel").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(10, 0))

    # ------------------------------------------------------------------ settings dialog
    def open_settings(self) -> None:
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_set()
            return

        win = tk.Toplevel(self)
        self._settings_win = win
        win.title("Focus Guard — Settings")
        win.configure(bg=BG, padx=20, pady=18)
        win.transient(self)
        win.resizable(False, False)
        win.columnconfigure(0, weight=1)
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        def section(row: int, title: str) -> ttk.Frame:
            frame = ttk.Frame(win, style="Card.TFrame", padding=(16, 14))
            frame.grid(row=row, column=0, sticky="ew", pady=(0 if row == 0 else 12, 0))
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=title, style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
            return frame

        cam = section(0, "CAMERA")
        ttk.Checkbutton(cam, text="Use webcam for gaze & phone detection", variable=self.camera_enabled).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(cam, text="Camera number", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(cam, from_=0, to=3, textvariable=self.camera_index, width=4).grid(row=2, column=1, sticky="w", pady=(10, 0))
        ttk.Checkbutton(cam, text="Show live screen preview", variable=self.preview_enabled).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(cam, text="Sensitivity", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(cam, from_=0.6, to=1.6, variable=self.sensitivity, orient="horizontal").grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(cam, text="Calibration length", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(cam, from_=5, to=30, textvariable=self.calibration_seconds, width=4).grid(row=5, column=1, sticky="w", pady=(10, 0))
        ttk.Label(cam, text="seconds", style="CardMuted.TLabel").grid(row=5, column=2, sticky="w", pady=(10, 0))

        alr = section(1, "ALERTS")
        ttk.Label(alr, text="Re-alert every", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(alr, from_=0, to=120, textvariable=self.realert_seconds, width=5).grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Label(alr, text="seconds  (0 = alert once per episode)", style="CardMuted.TLabel").grid(
            row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Label(alr, text="Alarm volume", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(alr, from_=0, to=1, variable=self.alarm_volume, orient="horizontal").grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        tones = ttk.Frame(alr, style="Card.TFrame")
        tones.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(tones, text="Upload tone", command=self.choose_alarm_file).pack(side="left")
        ttk.Button(tones, text="Default tone", command=self.clear_alarm_file).pack(side="left", padx=(8, 0))
        ttk.Button(tones, text="Test alert", command=self.trigger_alert).pack(side="left", padx=(8, 0))

        dat = section(2, "DATA & MODEL")
        ttk.Label(dat, textvariable=self.model_backend, style="CardMuted.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(dat, text="Logging label", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(dat, textvariable=self.log_label, values=["focused", "distracted"], width=12,
                     state="readonly").grid(row=2, column=1, sticky="w", pady=(10, 0))
        self.log_button = ttk.Button(dat, text="Stop logging" if self.logger.active else "Start logging",
                                     command=self.toggle_logging)
        self.log_button.grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(10, 0))
        ttk.Checkbutton(dat, text="Save session summaries to session_history.csv",
                        variable=self.save_summaries).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(dat, style="CardMuted.TLabel", wraplength=460, justify="left",
                  text=("No webcam video, screenshots, or window titles are saved or sent anywhere. "
                        "settings.json stores your preferences; feature logs and session summaries are written "
                        "only when you turn them on.")
                  ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Button(win, text="Done", command=win.destroy).grid(row=3, column=0, sticky="e", pady=(16, 0))

        win.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw, ph = self.winfo_width(), self.winfo_height()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry(f"+{max(0, px + (pw - w) // 2)}+{max(0, py + (ph - h) // 3)}")

    def toggle_signals(self) -> None:
        if self.signals_body.winfo_ismapped():
            self.signals_body.grid_forget()
            self.signals_toggle.configure(text="▸  Live signals")
        else:
            self.signals_body.grid(row=7, column=0, sticky="ew", pady=(8, 0))
            self.signals_toggle.configure(text="▾  Live signals")

    def _update_log_button(self) -> None:
        button = getattr(self, "log_button", None)
        if button is not None:
            try:
                button.configure(text="Stop logging" if self.logger.active else "Start logging")
            except tk.TclError:
                pass

    def _reset_meter(self) -> None:
        self.score = 0.0
        self.score_bar["value"] = 0
        self.score_text.configure(text="0.00")
        self.style.configure("Meter.Horizontal.TProgressbar", background=GREEN, lightcolor=GREEN, darkcolor=GREEN)

    def bind_shortcuts(self) -> None:
        self.bind("<F5>", lambda _event: self.toggle())
        self.bind("<F6>", lambda _event: self.toggle_pause())
        self.bind("<F7>", lambda _event: self.start_calibration())
        self.bind("<F8>", lambda _event: self.flip_log_label())

    def wire_settings_traces(self) -> None:
        self.camera_enabled.trace_add("write", lambda *_: self._apply_camera_toggle())
        self.preview_enabled.trace_add("write", lambda *_: (
            self.screen.enable() if self.preview_enabled.get() else self.screen.disable()))
        self.camera_index.trace_add("write", lambda *_: self._safe(
            lambda: self.vision.set_camera_index(self._num(self.camera_index, 0))))
        self.sensitivity.trace_add("write", lambda *_: self._safe(
            lambda: self.vision.set_sensitivity(self._num(self.sensitivity, 1.0))))

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _sane_geometry(value: str | None) -> str:
        """Restore a saved window geometry only if it looks usable."""
        try:
            size = (value or "").split("+")[0]
            width, height = (int(part) for part in size.split("x"))
            if width >= 900 and height >= 830:
                return value
        except (ValueError, AttributeError):
            pass
        return "1040x850"

    @staticmethod
    def _num(var, default):
        try:
            return var.get()
        except (tk.TclError, ValueError):
            return default

    @staticmethod
    def _safe(func) -> None:
        try:
            func()
        except (tk.TclError, ValueError):
            pass

    def _apply_camera_toggle(self) -> None:
        if self.camera_enabled.get():
            self.vision.enable()
        else:
            self.vision.disable()

    @staticmethod
    def show_time(value: float) -> str:
        return f"{int(value) // 60}:{int(value) % 60:02d}"

    def set_status(self, text: str, color: str) -> None:
        if time.monotonic() < self.visual_alert_until and self.visual_alert_on:
            color = "#d18a00"
        self.status.configure(text=text, bg=color)

    # ------------------------------------------------------------ main loop
    def pump(self) -> None:
        now = time.monotonic()
        try:
            self.render_previews()
            self.update_health(now)
            if self.running:
                self.session_tick(now)
        finally:
            self.after(90, self.pump)

    @staticmethod
    def _fit_to_label(frame: np.ndarray, label: tk.Label) -> np.ndarray:
        """Scale a frame to fill the label's current on-screen size.

        The workers cap their source resolution for performance, but the
        label itself grows with the window (e.g. on maximize), so the
        frame is rescaled here every tick to match whatever room is
        actually available instead of staying pinned at its source size.
        """
        box_w, box_h = label.winfo_width(), label.winfo_height()
        if box_w <= 1 or box_h <= 1:
            return frame
        src_h, src_w = frame.shape[:2]
        scale = min(box_w / src_w, box_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        if (new_w, new_h) == (src_w, src_h):
            return frame
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)

    def render_previews(self) -> None:
        res = self.vision.latest()
        if self.camera_enabled.get() and res.frame_rgb is not None:
            frame = self._fit_to_label(res.frame_rgb, self.camera_label)
            self.video_image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.camera_label.configure(image=self.video_image, text="")
        else:
            self.video_image = None
            self.camera_label.configure(
                image="", text="Webcam disabled" if not self.camera_enabled.get()
                else (res.camera_message or "Starting webcam…"))

        arr, message = self.screen.latest()
        if self.preview_enabled.get() and arr is not None:
            frame = self._fit_to_label(arr, self.screen_label)
            self.screen_image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.screen_label.configure(image=self.screen_image, text="")
        else:
            self.screen_image = None
            self.screen_label.configure(image="", text=message or "Screen preview disabled")

    def update_health(self, now: float) -> None:
        res = self.vision.latest()

        if res.calibration_seq != self._last_cal_seq:
            self._last_cal_seq = res.calibration_seq
            self._gaze_hold_until = time.monotonic() + 2.5
            if self.vision.calibrated():
                self.gaze_status.configure(text="Camera: calibration complete — neutral pose saved.", foreground=GREEN)
                self.save_settings()
            else:
                self.gaze_status.configure(text="Camera: calibration failed — keep your face visible and retry (F7).", foreground=RED)

        cam_on = self.camera_enabled.get()
        self.calibrate_button.configure(
            state="normal" if (cam_on and res.camera_ok and not res.calibrating) else "disabled")

        bits = [f"cam {res.fps:.0f} fps" if (cam_on and res.camera_ok) else (res.camera_message if cam_on else "cam off")]
        if self.preview_enabled.get():
            bits.append("screen on")
        if self.logger.active:
            bits.append(f"log {self.logger.rows}")
        if now < self.paused_until:
            bits.append(f"break {self.show_time(self.paused_until - now)}")
        if self._drift_notice and now - self._last_drift_prompt < 120:
            bits.append(self._drift_notice)
        self.health.configure(text="   ·   ".join(bit for bit in bits if bit))

        if res.gaze_text and time.monotonic() >= self._gaze_hold_until:
            self.gaze_status.configure(text=res.gaze_text, foreground=res.gaze_color)

        self.metrics["Head rotation"].set("--" if res.head_rotation_deg is None else f"{res.head_rotation_deg:.0f} deg")
        self.metrics["Eye openness"].set("--" if res.eye_openness_value is None else f"{res.eye_openness_value:.2f}")
        self.metrics["Looking away"].set("1" if res.looking_away else "0")

    def session_tick(self, now: float) -> None:
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now
        grace = self._num(self.grace, 3)

        title = foreground_title()
        phrase = self.target.get().strip().casefold()
        allowed = (bool(phrase) and phrase in title.casefold()) or APP_TITLE.casefold() in title.casefold()
        self.task_tab_active = allowed
        self.update_tab_metrics(title, allowed, now)
        self.update_input_metrics(now)
        if allowed:
            self.focused += dt
            self.off_started = None
        else:
            self.off_task += dt
            self.off_started = self.off_started or now

        res = self.vision.latest()
        features = self.collect_features(now, res)
        if self.logger.active:
            self.logger.log(features)
            self.metrics["Data logging"].set(f"{self.logger.rows} rows · {self.logger.label}")

        state, _confidence, reason = self.model.predict(features)
        if self.compare_model is not None:
            alt_state, _, _ = self.compare_model.predict(features)
            self.model_compared += 1
            self.model_agree += 1 if alt_state == state else 0

        target_score = 1.0 if state == DISTRACTED else 0.0
        rate = 0.45 if target_score >= self.score else 0.18
        self.score = min(1.0, max(0.0, self.score + (target_score - self.score) * rate))
        self.score_bar["value"] = self.score * 100
        self.score_text.configure(text=f"{self.score:.2f}")
        meter_color = GREEN if self.score < 0.4 else (AMBER if self.score < 0.7 else RED)
        self.style.configure("Meter.Horizontal.TProgressbar",
                             background=meter_color, lightcolor=meter_color, darkcolor=meter_color)
        distracted = self.score >= 0.55
        self.score_history.append((now - self.session_start, self.score))

        self.metrics["Model state"].set(
            f"{reason} · score {self.score:.2f}" if state == DISTRACTED else f"focused · score {self.score:.2f}")

        if features.get("calibrated") and features.get("face_visible") and not res.looking_away and state == FOCUSED:
            self.resting_pose_samples.append((features["head_pitch"], features["head_yaw"]))
        self.check_calibration_drift(now, features)

        self.handle_alerts(now, distracted, reason, grace)
        self.update_status_and_stats(now, allowed, title, distracted, reason, grace, dt)

    def check_calibration_drift(self, now: float, features: dict) -> None:
        """Flag a stale calibration baseline from resting-pose readings.

        Samples pitch/yaw only from ticks the model itself called "focused"
        with a visible, gaze-tracked face — i.e. moments the user was
        presumably facing the screen. If the median of those readings
        drifts far from the calibrated neutral pose (0,0 by construction),
        the baseline probably no longer matches how the user is sitting
        (camera bumped, chair moved), which would make the look-away
        thresholds unreliable without the user knowing.
        """
        if not features.get("calibrated") or len(self.resting_pose_samples) < 60:
            return
        if now - self._last_drift_prompt < 180:
            return
        pitches = sorted(abs(p) for p, _ in self.resting_pose_samples)
        yaws = sorted(abs(y) for _, y in self.resting_pose_samples)
        median_pitch = pitches[len(pitches) // 2]
        median_yaw = yaws[len(yaws) // 2]
        if median_pitch > 8.0 or median_yaw > 8.0:
            self.drift_prompts += 1
            self._last_drift_prompt = now
            self._drift_notice = "posture drift — recalibrate (F7)"

    def collect_features(self, now: float, res) -> dict:
        face = res.features if res else {}
        return {
            "task_tab_active": 1.0 if self.task_tab_active else 0.0,
            "tab_switches_per_min": float(len(self.title_switch_times)),
            "time_on_current_tab": now - self.current_title_started,
            "keyboard_inactivity": now - self.last_keyboard_activity,
            "mouse_inactivity": now - self.last_mouse_activity,
            "calibrated": float(face.get("calibrated", 0.0)),
            "face_visible": float(face.get("face_visible", 0.0)),
            "head_pitch": float(face.get("head_pitch", 0.0)),
            "head_yaw": float(face.get("head_yaw", 0.0)),
            "iris_delta": float(face.get("iris_delta", 0.0)),
            "eye_openness": float(face.get("eye_openness", 0.0)),
            "phone_present": 1.0 if (res and res.phone_present) else 0.0,
        }

    def handle_alerts(self, now: float, distracted: bool, reason: str, grace: float) -> None:
        if not distracted:
            self.distraction_started = None
            self.distraction_alert_sent = False
            return
        self.distraction_started = self.distraction_started or now
        if now - self.distraction_started < grace or now < self.paused_until:
            return
        realert = self._num(self.realert_seconds, 0)
        first = not self.distraction_alert_sent
        repeat = realert > 0 and (now - self.last_alert_time) >= realert
        if (first or repeat) and (now - self.last_alert_time) >= 4:
            self.trigger_alert()
            self.distraction_alert_sent = True
            self.last_alert_time = now
            self.alerts += 1
            self.reason_counts[reason] += 1
            self.alert_events.append((now - self.session_start, reason))

    def update_status_and_stats(self, now, allowed, title, distracted, reason, grace, dt) -> None:
        if allowed and not distracted:
            self.current_streak += dt
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0.0

        if now < self.paused_until:
            left = self.paused_until - now
            self.set_status(f"On a break — {int(left // 60)}:{int(left % 60):02d} left (F6 to end)", BANNER["pause"])
        elif distracted and self.distraction_started and now - self.distraction_started >= grace:
            self.set_status(f"DISTRACTED — {reason}. Back to: {self.target.get() or 'your task'}", BANNER["bad"])
        elif distracted:
            self.set_status(f"Checking — {reason}…", BANNER["warn"])
        elif allowed:
            self.set_status(f"FOCUSED — {title[:88]}", BANNER["good"])
        else:
            self.set_status(f"OFF TASK — expected tab: {self.target.get()}", BANNER["warn"])

        self.stats.configure(text=(
            f"Focused {self.show_time(self.focused)}    Off-task {self.show_time(self.off_task)}    "
            f"Alerts {self.alerts}    Streak {self.show_time(self.best_streak)}"))

    # ------------------------------------------------------------ tab / input
    def update_tab_metrics(self, title: str, allowed: bool, now: float) -> None:
        if title != self.current_title:
            self.current_title = title
            self.current_title_started = now
            self.title_switch_times.append(now)
        while self.title_switch_times and now - self.title_switch_times[0] > 60:
            self.title_switch_times.popleft()
        self.metrics["Tab switches/min"].set(str(len(self.title_switch_times)))
        self.metrics["Time on current tab"].set(f"{int(now - self.current_title_started)} sec")
        self.metrics["Non-task tab active"].set("0" if allowed else "1")

    def update_input_metrics(self, now: float) -> None:
        if keyboard is not None and mouse is not None:
            self.metrics["Keyboard inactivity"].set(f"{int(now - self.last_keyboard_activity)} sec")
            self.metrics["Mouse inactivity"].set(f"{int(now - self.last_mouse_activity)} sec")

    def start_input_tracking(self) -> None:
        if keyboard is None or mouse is None:
            self.metrics["Keyboard inactivity"].set("unavailable")
            self.metrics["Mouse inactivity"].set("unavailable")
            return

        def on_keyboard_activity(*_):
            self.last_keyboard_activity = time.monotonic()

        def on_mouse_activity(*_):
            self.last_mouse_activity = time.monotonic()

        try:
            keyboard_listener = keyboard.Listener(on_press=on_keyboard_activity)
            mouse_listener = mouse.Listener(on_move=on_mouse_activity, on_click=on_mouse_activity, on_scroll=on_mouse_activity)
            keyboard_listener.start()
            mouse_listener.start()
            self.input_listeners.extend((keyboard_listener, mouse_listener))
        except Exception:
            self.metrics["Keyboard inactivity"].set("unavailable")
            self.metrics["Mouse inactivity"].set("unavailable")

    # ------------------------------------------------------------ session control
    def toggle(self) -> None:
        if self.running:
            self.stop_session()
            return
        if not self.target.get().strip():
            self.set_status("Enter words that appear in your task tab/window title.", BANNER["bad"])
            return
        self.running = True
        self.focused = self.off_task = 0.0
        self.alerts = 0
        self.score = 0.0
        self.current_streak = self.best_streak = 0.0
        self.reason_counts.clear()
        self.off_started = self.distraction_started = None
        self.distraction_alert_sent = False
        self.last_alert_time = 0.0
        self.paused_until = 0.0
        self.pause_button.configure(text="Take a break")
        self.current_title = foreground_title()
        self.current_title_started = time.monotonic()
        self.title_switch_times.clear()
        self.last_tick = time.monotonic()
        self.session_start = time.monotonic()
        self.score_history = []
        self.alert_events = []
        self.model_compared = 0
        self.model_agree = 0
        self.resting_pose_samples.clear()
        self.drift_prompts = 0
        self._drift_notice = ""
        self._last_drift_prompt = 0.0
        self._reset_meter()
        if self.camera_enabled.get():
            self.vision.enable()
        self.start_button.configure(text="Stop session", style="Ghost.TButton")
        self.set_status("Focus session started — calibrate the camera for gaze detection (F7).", BANNER["idle"])
        self.save_settings()

    def stop_session(self) -> None:
        if not self.running:
            return
        self.running = False
        total = self.focused + self.off_task
        summary = self.session_summary(total)
        self.start_button.configure(text="Start focus session", style="Accent.TButton")
        self._reset_meter()
        self.set_status("Session ended — no video or titles were saved.", BANNER["idle"])
        self.save_settings()
        if self.save_summaries.get() and total >= 5:
            self.append_history(total)
            chart_path = APP_DIR / "session_charts" / f"session_{time.strftime('%Y%m%d_%H%M%S')}.png"
            if session_chart.save_chart(chart_path, self.score_history, self.alert_events):
                summary += f"\nChart saved:  session_charts/{chart_path.name}"
        if total >= 20:
            messagebox.showinfo("Focus session summary", summary)

    def session_summary(self, total: float) -> str:
        pct = (100 * self.focused / total) if total else 0.0
        causes = ", ".join(f"{name} ×{count}" for name, count in self.reason_counts.most_common()) or "none"
        extra_lines = []
        if self.model_compared:
            agree_pct = 100 * self.model_agree / self.model_compared
            extra_lines.append(f"Trained vs. heuristic agreement:  {agree_pct:.0f}% (n={self.model_compared})")
        if self.drift_prompts:
            extra_lines.append(f"Posture-drift prompts:  {self.drift_prompts}")
        extra = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
        return (
            f"Session length:  {self.show_time(total)}\n"
            f"Focused:  {self.show_time(self.focused)}  ({pct:.0f}%)\n"
            f"Off-task:  {self.show_time(self.off_task)}\n"
            f"Alerts:  {self.alerts}\n"
            f"Longest focus streak:  {self.show_time(self.best_streak)}\n"
            f"Alert causes:  {causes}"
            f"{extra}"
        )

    def append_history(self, total: float) -> None:
        try:
            exists = HISTORY_PATH.exists()
            with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not exists:
                    writer.writerow(["ended", "seconds", "focused_seconds", "off_task_seconds",
                                     "alerts", "best_streak_seconds", "causes"])
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M"), f"{total:.0f}", f"{self.focused:.0f}",
                    f"{self.off_task:.0f}", self.alerts, f"{self.best_streak:.0f}",
                    "; ".join(f"{name}:{count}" for name, count in self.reason_counts.most_common()),
                ])
        except OSError:
            pass

    def toggle_pause(self) -> None:
        if time.monotonic() < self.paused_until:
            self.paused_until = 0.0
            self.pause_button.configure(text="Take a break")
            self.set_status("Break ended — alerts resumed.", BANNER["idle"])
        else:
            minutes = self._num(self.break_minutes, 5)
            self.paused_until = time.monotonic() + max(1, minutes) * 60
            self.pause_button.configure(text="End break")

    def start_calibration(self) -> None:
        if not self.camera_enabled.get():
            self.gaze_status.configure(text="Camera: turn on the webcam in Settings first.", foreground=RED)
            return
        if not self.vision.latest().camera_ok:
            self.gaze_status.configure(text="Camera: waiting for the webcam to start…", foreground=RED)
            return
        self.vision.request_calibration(self._num(self.calibration_seconds, 10))
        self.calibrate_button.configure(state="disabled")
        self._gaze_hold_until = time.monotonic() + 2.0
        self.gaze_status.configure(text="Camera: calibrating — look normally at your screen.", foreground=AMBER)

    def flip_log_label(self) -> None:
        self.log_label.set("distracted" if self.log_label.get() == "focused" else "focused")

    # ------------------------------------------------------------ logging
    def toggle_logging(self) -> None:
        if self.logger.active:
            self.logger.stop()
            self._update_log_button()
            self.metrics["Data logging"].set("off")
            self.set_status(f"Logging stopped — {self.logger.rows} rows written.", BANNER["idle"])
            return
        path = filedialog.asksaveasfilename(
            title="Save labelled session data", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialfile="focus_session.csv")
        if not path:
            return
        try:
            self.logger.start(path, self.log_label.get())
        except OSError as error:
            self.set_status(f"Could not open log file: {error}", BANNER["bad"])
            return
        self._update_log_button()
        self.set_status(
            f"Logging to {Path(path).name} as ‘{self.log_label.get()}’. "
            "Rows are recorded while a focus session runs; press F8 to flip the label.", BANNER["idle"])

    # ------------------------------------------------------------ alarm
    def choose_alarm_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose alarm tone",
            filetypes=[("Audio files", "*.wav *.mp3"), ("WAV audio", "*.wav"), ("MP3 audio", "*.mp3"), ("All files", "*.*")])
        if path:
            self.alarm_file.set(path)
            self.set_status(f"Custom alarm tone selected: {Path(path).name}", BANNER["idle"])
            self.save_settings()

    def clear_alarm_file(self) -> None:
        self.alarm_file.set("")
        self.set_status("Using default alarm tone.", BANNER["idle"])
        self.save_settings()

    def play_alarm(self) -> None:
        if self.alarm_playing:
            return
        volume = max(0.0, min(1.0, float(self._num(self.alarm_volume, 0.25))))
        audio_path = self.alarm_file.get()

        def alarm_worker() -> None:
            self.alarm_playing = True
            try:
                if volume <= 0:
                    return
                if audio_path:
                    try:
                        self.play_uploaded_alarm(Path(audio_path), volume)
                        return
                    except Exception:
                        audio, sample_rate = self.default_alarm_audio()
                else:
                    audio, sample_rate = self.default_alarm_audio()
                sd.play(np.clip(audio * volume, -1, 1), sample_rate)
                sd.wait()
            finally:
                self.alarm_playing = False

        threading.Thread(target=alarm_worker, daemon=True).start()

    @staticmethod
    def default_alarm_audio():
        sample_rate = 44100
        parts = []
        for frequency in (440, 330, 440):
            duration = .12
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            tone = np.sin(2 * np.pi * frequency * t) * .35
            fade = np.linspace(0, 1, min(400, len(tone)))
            tone[:len(fade)] *= fade
            tone[-len(fade):] *= fade[::-1]
            parts.append(tone.astype(np.float32))
            parts.append(np.zeros(int(sample_rate * .08), dtype=np.float32))
        return np.concatenate(parts), sample_rate

    @staticmethod
    def play_uploaded_alarm(path: Path, volume: float) -> None:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        started = time.monotonic()
        while pygame.mixer.music.get_busy() and time.monotonic() - started < 4:
            time.sleep(.05)
        pygame.mixer.music.stop()

    def trigger_alert(self) -> None:
        self.play_alarm()
        self.flash_window()
        self.visual_alert_until = time.monotonic() + 2.5
        self.visual_alert_on = True
        self.title("DISTRACTED - Focus Guard")
        self.after(250, self.pulse_visual_alert)
        self.after(2500, lambda: self.title(APP_TITLE))

    def pulse_visual_alert(self) -> None:
        if time.monotonic() >= self.visual_alert_until:
            self.visual_alert_on = False
            return
        self.visual_alert_on = not self.visual_alert_on
        self.after(250, self.pulse_visual_alert)

    def flash_window(self) -> None:
        try:
            class FlashInfo(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint),
                    ("hwnd", ctypes.c_void_p),
                    ("dwFlags", ctypes.c_uint),
                    ("uCount", ctypes.c_uint),
                    ("dwTimeout", ctypes.c_uint),
                ]

            info = FlashInfo(ctypes.sizeof(FlashInfo), self.winfo_id(), 15, 4, 0)
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))
        except Exception:
            pass

    # ------------------------------------------------------------ settings / shutdown
    def save_settings(self) -> None:
        settings_store.save({
            "target": self.target.get(),
            "grace": self._num(self.grace, 3),
            "camera_index": self._num(self.camera_index, 0),
            "alarm_volume": round(float(self._num(self.alarm_volume, 0.25)), 3),
            "alarm_file": self.alarm_file.get(),
            "preview_enabled": bool(self.preview_enabled.get()),
            "camera_enabled": bool(self.camera_enabled.get()),
            "realert_seconds": self._num(self.realert_seconds, 20),
            "sensitivity": round(float(self._num(self.sensitivity, 1.0)), 2),
            "calibration_seconds": self._num(self.calibration_seconds, 10),
            "break_minutes": self._num(self.break_minutes, 5),
            "geometry": self.geometry(),
        })

    def close(self) -> None:
        self.save_settings()
        self.logger.stop()
        self.running = False
        for worker in (self.vision, self.screen):
            try:
                worker.stop()
            except Exception:
                pass
        for worker in (self.vision, self.screen):
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
        for listener in self.input_listeners:
            try:
                listener.stop()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    FocusGuard().mainloop()
