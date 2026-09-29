"""Opt-in, local-only CSV logging of Focus Guard feature vectors.

Disabled by default. When the user clicks "Start logging" they pick a file and
a label; every tick of a running focus session then appends one row. This is
the only part of the app that writes analysis data to disk, and only on an
explicit user action. Use the logs with ``train_model.py`` to fit a custom
decision tree.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

from distraction_model import FEATURE_NAMES

COLUMNS = ("timestamp", *FEATURE_NAMES, "label")


class SessionLogger:
    def __init__(self) -> None:
        self._handle = None
        self._writer = None
        self.path: Optional[Path] = None
        self.label = "focused"
        self.rows = 0

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

    def set_label(self, label: str) -> None:
        self.label = label or "focused"

    def log(self, features: Dict[str, float]) -> None:
        if not self._writer:
            return
        self._writer.writerow([
            f"{time.time():.3f}",
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
