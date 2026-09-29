"""Local preference persistence for Focus Guard.

Stores UI preferences (task phrase, timings, camera index, volume, window size)
in ``settings.json`` next to the app. This holds *your settings only* — never
webcam frames, screenshots, or window titles.
"""
from __future__ import annotations

import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

DEFAULTS = {
    "target": "",
    "grace": 3,
    "camera_index": 0,
    "alarm_volume": 0.25,
    "alarm_file": "",
    "preview_enabled": True,
    "camera_enabled": True,
    "realert_seconds": 20,
    "sensitivity": 1.0,
    "calibration_seconds": 10,
    "break_minutes": 5,
    "geometry": "",
}


def load() -> dict:
    """Return stored settings merged over the defaults; never raises."""
    values = dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        for key in DEFAULTS:
            if key in stored and isinstance(stored[key], type(DEFAULTS[key])):
                values[key] = stored[key]
    except Exception:
        pass
    return values


def save(values: dict) -> None:
    """Write the given settings, keeping only known keys; never raises."""
    try:
        payload = {key: values.get(key, DEFAULTS[key]) for key in DEFAULTS}
        SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
