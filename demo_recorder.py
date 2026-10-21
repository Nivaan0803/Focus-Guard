"""Opt-in, local-only recording of booth demo runs (F9 demo view).

The demo view itself is read-only: it draws what the model already decided.
This module adds an optional paper trail for a science-fair booth, so a run
with a visitor leaves behind evidence instead of just a memory:

* ``demo_logs/demo_<timestamp>.csv`` - one row every ``PERIOD`` seconds with
  the feature vector the model saw, its verdict, the smoothed distraction
  score and the reason it named.
* ``demo_logs/demo_index.csv`` - one roll-up row per demo (length, tick count,
  share of distracted ticks, alerts fired, most common reason).

Off by default, exactly like ``session_logger``: it only runs while the
"Record booth demos" box in Settings is ticked. No webcam frames, no
screenshots and no window titles are written - the same twelve numbers the
decision tree sees, nothing more.

The per-tick ``label`` column is always ``unlabelled``: a visitor's demo has
no ground truth, so this data is for describing demos, not for training. The
explicit label keeps ``train_model.py`` from silently reading these rows as
"focused" if they are ever globbed in by accident.
"""
from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from distraction_model import FEATURE_NAMES

DEMO_DIR = Path(__file__).resolve().parent / "demo_logs"
INDEX_PATH = DEMO_DIR / "demo_index.csv"

#: Seconds between recorded rows. The app ticks ~11x a second, which is more
#: resolution than a booth run needs and would bloat a day of demos.
PERIOD = 0.5

COLUMNS = ("elapsed", *FEATURE_NAMES, "verdict", "score", "reason", "label")
INDEX_COLUMNS = ("started", "seconds", "ticks", "distracted_ticks", "distracted_pct",
                 "alerts", "top_reason", "file")


class DemoRecorder:
    def __init__(self) -> None:
        self._handle = None
        self._writer = None
        self.path: Optional[Path] = None
        self.rows = 0
        self._started_wall = ""
        self._last_write = 0.0
        self._last_elapsed = 0.0
        self._distracted = 0
        self._reasons: Counter = Counter()
        self._alerts_first: Optional[int] = None
        self._alerts_last = 0

    @property
    def active(self) -> bool:
        return self._handle is not None

    def start(self) -> Optional[Path]:
        """Open a fresh per-demo CSV. Returns its path, or None if it failed."""
        self.stop()
        try:
            DEMO_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target = DEMO_DIR / f"demo_{stamp}.csv"
            self._handle = target.open("w", newline="", encoding="utf-8")
        except OSError:
            self._handle = None
            return None
        self._writer = csv.writer(self._handle)
        self._writer.writerow(COLUMNS)
        self.path = target
        self.rows = 0
        self._started_wall = time.strftime("%Y-%m-%d %H:%M")
        self._last_write = 0.0
        self._last_elapsed = 0.0
        self._distracted = 0
        self._reasons.clear()
        self._alerts_first = None
        self._alerts_last = 0
        return target

    def log(self, elapsed: float, features: Dict[str, float], verdict: str,
            score: float, reason: str, alerts: int) -> None:
        """Append one row, at most once per ``PERIOD`` seconds of session time."""
        if not self._writer:
            return
        if self._alerts_first is None:
            self._alerts_first = alerts
        self._alerts_last = alerts
        if self.rows and elapsed - self._last_write < PERIOD:
            return
        self._last_write = elapsed
        self._last_elapsed = elapsed
        try:
            self._writer.writerow([
                f"{elapsed:.2f}",
                *[f"{float(features.get(name, 0.0)):.4f}" for name in FEATURE_NAMES],
                verdict,
                f"{score:.3f}",
                reason or "",
                "unlabelled",
            ])
        except (OSError, ValueError):
            return
        self.rows += 1
        if verdict != "focused":
            self._distracted += 1
            if reason:
                self._reasons[reason] += 1

    def stop(self) -> Optional[Path]:
        """Close the per-demo CSV and append its roll-up row to the index."""
        if not self._handle:
            return None
        path = self.path
        try:
            self._handle.flush()
            self._handle.close()
        except Exception:
            pass
        self._handle = self._writer = None
        if self.rows:
            self._append_index(path)
        elif path is not None:
            try:  # a demo opened and closed without a single tick - no evidence
                path.unlink()
            except OSError:
                pass
            return None
        return path

    def _append_index(self, path: Optional[Path]) -> None:
        alerts = max(0, self._alerts_last - (self._alerts_first or 0))
        pct = 100.0 * self._distracted / self.rows if self.rows else 0.0
        top = self._reasons.most_common(1)
        try:
            fresh = not INDEX_PATH.exists() or INDEX_PATH.stat().st_size == 0
            with INDEX_PATH.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if fresh:
                    writer.writerow(INDEX_COLUMNS)
                writer.writerow([
                    self._started_wall,
                    f"{self._last_elapsed:.0f}",
                    self.rows,
                    self._distracted,
                    f"{pct:.0f}",
                    alerts,
                    top[0][0] if top else "none",
                    path.name if path else "",
                ])
        except OSError:
            pass
