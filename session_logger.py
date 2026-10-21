"""Opt-in, local-only CSV logging of Focus Guard feature vectors.

Disabled by default. When the user clicks "Start logging" they pick a file and
a label; every tick of a running focus session then appends one row. Like the
other writers (``demo_recorder`` for booth runs, ``session_history.csv`` for
summaries) it only runs on an explicit user action. Use the logs with
``train_model.py`` to fit a custom decision tree.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

from distraction_model import FEATURE_NAMES

COLUMNS = ("elapsed", *FEATURE_NAMES, "label")


class SessionLogger:
    def __init__(self) -> None:
        self._handle = None
        self._writer = None
        self.path: Optional[Path] = None
        self.label = "focused"
        self.rows = 0
        self._t0 = 0.0

    @property
    def active(self) -> bool:
        return self._handle is not None

    def start(self, path: str, label: str) -> None:
        self.stop()
        target = Path(path)
        fresh = not target.exists() or target.stat().st_size == 0
        self._handle = target.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if fresh:
            self._writer.writerow(COLUMNS)
        self.path = target
        self.label = label or "focused"
        self.rows = 0
        self._t0 = time.monotonic()

    def set_label(self, label: str) -> None:
        self.label = label or "focused"

    def log(self, features: Dict[str, float]) -> None:
        """Append one row, stamped with seconds since logging started.

        Elapsed seconds rather than a Unix epoch: the epoch is a ten-digit
        number that spreadsheets render as ``1.79E+09``, and nothing needs
        the absolute wall-clock time - every plot is against time-into-run.
        """
        if not self._writer:
            return
        self._writer.writerow([
            f"{time.monotonic() - self._t0:.2f}",
            *[f"{float(features.get(name, 0.0)):.4f}" for name in FEATURE_NAMES],
            self.label,
        ])
        self.rows += 1

    def stop(self) -> None:
        if self._handle:
            try:
                self._handle.flush()
                self._handle.close()
            except Exception:
                pass
        self._handle = self._writer = None
