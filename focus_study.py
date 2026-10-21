"""Timed work task for the "does real-time feedback actually help?" study.

This answers the project's biggest question, not "does the detector work?"
but "do people focus better WITH it?". Each participant does two equal
sessions of mental arithmetic, one with Focus Guard alerting and one
without, and this script measures both the same way, independently of the
detector (so the detector can't grade its own homework):

  - problems attempted and answered correctly
  - correct answers per minute
  - seconds spent outside this task window, and how many times they left
  - a 1-10 self-rating of how focused they felt

Protocol (keep it identical for every participant):
  1. Phone on the desk, face up, notifications on. A browser open behind the
     task window. Don't tell them to avoid it; real distractions are the point.
  2. Counterbalance the order so practice/fatigue doesn't favour one condition:
       odd-numbered participants (P01, P03, ...):  alerts first, then no_alerts
       even-numbered participants (P02, P04, ...): no_alerts first, then alerts
  3. "alerts" condition: run main.py, set the task phrase to "Focus Study",
     calibrate (F7), start the session (F5), then start this script.
     "no_alerts" condition: don't run Focus Guard at all.
  4. Short break (2-3 min) between the two sessions.

Usage:
    python focus_study.py --participant P01 --condition alerts
    python focus_study.py --participant P01 --condition no_alerts --minutes 10

Results are appended to study/results.csv (one row per session). Analyse with
analyze_study.py. Only these numbers are saved; nothing about the screen or camera.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import random
import time
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

WINDOW_TITLE = "Focus Study Task"
DEFAULT_OUT = Path(__file__).resolve().parent / "study" / "results.csv"
COLUMNS = ("finished", "participant", "condition", "minutes", "attempted", "correct",
           "accuracy", "correct_per_min", "seconds_away", "times_left", "self_rating")

BG, CARD, INK, MUTE, GREEN, RED = "#0f1420", "#1a2230", "#e8edf4", "#8b98ab", "#35b17e", "#e0574a"


def make_problem(rng: random.Random):
    """Medium-difficulty mental arithmetic, kept to one consistent mix of operations."""
    kind = rng.choice(("add", "sub", "mul"))
    if kind == "add":
        a, b = rng.randint(23, 89), rng.randint(23, 89)
        return f"{a} + {b}", a + b
    if kind == "sub":
        a, b = rng.randint(50, 99), rng.randint(12, 49)
        return f"{a} − {b}", a - b
    a, b = rng.randint(6, 19), rng.randint(3, 9)
    return f"{a} × {b}", a * b


class StudyTask(tk.Tk):
    def __init__(self, participant: str, condition: str, minutes: float, out: Path) -> None:
        super().__init__()
        self.participant, self.condition, self.minutes, self.out = participant, condition, minutes, out
        # Same participant + condition always gets the same problem list, but the two
        # conditions get different (equally hard) lists, so nobody repeats answers.
        self.rng = random.Random(f"{participant}-{condition}")
        self.attempted = self.correct = self.times_left = 0
        self.seconds_away = 0.0
        self.was_foreground = True
        self.started = self.ends = None
        self.last_poll = time.monotonic()

        self.title(WINDOW_TITLE)
        self.configure(bg=BG, padx=40, pady=30)
        self.geometry("720x480")
        self.problem = tk.Label(self, text="", font=("Segoe UI", 54, "bold"), fg=INK, bg=BG)
        self.problem.pack(pady=(30, 16))
        self.answer = tk.Entry(self, font=("Segoe UI", 32), width=8, justify="center",
                               bg=CARD, fg=INK, insertbackground=INK, relief="flat")
        self.answer.pack()
        self.feedback = tk.Label(self, text="", font=("Segoe UI", 16), fg=MUTE, bg=BG)
        self.feedback.pack(pady=(12, 0))
        self.clock = tk.Label(self, text="", font=("Segoe UI", 14), fg=MUTE, bg=BG)
        self.clock.pack(side="bottom")

        self.problem.configure(text="Press Enter to start", font=("Segoe UI", 32, "bold"))
        self.feedback.configure(text=f"{minutes:g} minutes of arithmetic. Type each answer and press Enter. "
                                     "Answer as many as you can.", wraplength=600)
        self.answer.bind("<Return>", self.on_enter)
        self.answer.focus_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # finish via the timer only

    def on_enter(self, _event=None) -> None:
        if self.started is None:
            self.started = time.monotonic()
            self.ends = self.started + self.minutes * 60
            self.last_poll = self.started
            self.problem.configure(font=("Segoe UI", 54, "bold"))
            self.feedback.configure(text="")
            self.next_problem()
            self.after(250, self.tick)
            return
        text = self.answer.get().strip().replace("−", "-")
        if not text:
            return
        self.attempted += 1
        try:
            right = int(text) == self.expected
        except ValueError:
            right = False
        self.correct += right
        self.feedback.configure(text="✓" if right else f"✗  ({self.expected})", fg=GREEN if right else RED)
        self.next_problem()

    def next_problem(self) -> None:
        text, self.expected = make_problem(self.rng)
        self.problem.configure(text=text)
        self.answer.delete(0, "end")

    def is_foreground(self) -> bool:
        try:
            return ctypes.windll.user32.GetForegroundWindow() == int(self.wm_frame(), 16)
        except Exception:
            return True

    def tick(self) -> None:
        now = time.monotonic()
        foreground = self.is_foreground()
        if not foreground:
            self.seconds_away += now - self.last_poll
            if self.was_foreground:
                self.times_left += 1
        self.was_foreground = foreground
        self.last_poll = now

        left = self.ends - now
        if left <= 0:
            self.finish()
            return
        self.clock.configure(text=f"{int(left // 60)}:{int(left % 60):02d} left")
        self.after(250, self.tick)

    def finish(self) -> None:
        self.answer.configure(state="disabled")
        self.problem.configure(text="Time's up!", font=("Segoe UI", 40, "bold"))
        self.feedback.configure(text="", fg=MUTE)
        self.clock.configure(text="")
        self.lift()
        self.focus_force()
        rating = simpledialog.askinteger(
            "One quick question", "How focused did you feel during this session?\n"
            "1 = not at all, 10 = completely focused", parent=self, minvalue=1, maxvalue=10)
        self.save(rating)
        self.feedback.configure(
            text=f"Saved. {self.correct} correct out of {self.attempted}. You can close this window.")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def save(self, rating) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.out.exists() or self.out.stat().st_size == 0
        with self.out.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(COLUMNS)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M"), self.participant, self.condition, f"{self.minutes:g}",
                self.attempted, self.correct,
                f"{self.correct / self.attempted:.3f}" if self.attempted else "0",
                f"{self.correct / self.minutes:.2f}", f"{self.seconds_away:.1f}", self.times_left,
                rating if rating is not None else "",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", required=True, help="anonymous ID like P01 (no real names)")
    parser.add_argument("--condition", required=True, choices=("alerts", "no_alerts"))
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    StudyTask(args.participant.strip(), args.condition, args.minutes, args.out).mainloop()


if __name__ == "__main__":
    main()
