"""Unified distraction classifier for Focus Guard.

The app produces several weak signals every tick (head pose, gaze/iris offset,
eye openness, active-tab context, keyboard/mouse inactivity, phone presence).
This module fuses them into a single ``focused`` / ``distracted`` decision plus
a short human-readable reason.

Two interchangeable back ends:

* ``SklearnTree`` - a trained ``DecisionTreeClassifier`` loaded from
  ``models/distraction_tree.joblib`` (produced by ``train_model.py``).
* ``HeuristicTree`` - a small transparent decision tree used whenever
  scikit-learn or a trained model file is missing, using the shared
  ``LOOK_AWAY_LIMITS`` so it agrees with the camera overlay.

Only ``load_model`` touches the disk, and only to read an existing model file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

FEATURE_NAMES: Tuple[str, ...] = (
    "task_tab_active",
    "tab_switches_per_min",
    "time_on_current_tab",
    "keyboard_inactivity",
    "mouse_inactivity",
    "calibrated",
    "face_visible",
    "head_pitch",
    "head_yaw",
    "iris_delta",
    "eye_openness",
    "phone_present",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "distraction_tree.joblib"

# Head pitch (deg), head yaw (deg), iris offset beyond which the user counts as
# looking away. Deliberately lenient: small head turns while still reading the
# screen also shift the iris the other way, so tight limits flag normal motion.
LOOK_AWAY_LIMITS: Tuple[float, float, float] = (28.0, 35.0, 0.25)

FOCUSED = "focused"
DISTRACTED = "distracted"

Prediction = Tuple[str, float, str]

FEATURE_LABELS: Dict[str, str] = {
    "task_tab_active": "task tab active",
    "tab_switches_per_min": "tab switches/min",
    "time_on_current_tab": "time on current tab",
    "keyboard_inactivity": "keyboard idle (s)",
    "mouse_inactivity": "mouse idle (s)",
    "calibrated": "gaze calibrated",
    "face_visible": "face visible",
    "head_pitch": "head pitch (deg)",
    "head_yaw": "head yaw (deg)",
    "iris_delta": "iris offset",
    "eye_openness": "eye openness",
    "phone_present": "phone present",
}


def _decision_reason(model, row: List[float], feature_names: Tuple[str, ...]) -> str:
    """Name the exact split that decided a trained tree's verdict.

    Walks the path ``row`` took through ``model``'s tree and reports the last
    internal node before the leaf — the split that actually separated this
    row from the other class. This is what makes a trained tree's alerts
    legible instead of a bare "trained model" label.
    """
    try:
        tree = model.tree_
        leaf_id = int(model.apply([row])[0])
        path = model.decision_path([row])
        node_ids = path.indices[path.indptr[0]:path.indptr[1]]
        for node_id in reversed(node_ids):
            if node_id == leaf_id:
                continue
            feature_index = tree.feature[node_id]
            if feature_index < 0:  # a leaf can appear mid-path only if malformed
                continue
            name = feature_names[feature_index]
            threshold = tree.threshold[node_id]
            value = row[feature_index]
            direction = "<=" if value <= threshold else ">"
            label = FEATURE_LABELS.get(name, name)
            return f"{label} {value:.2f} {direction} {threshold:.2f}"
    except Exception:
        pass
    return "trained model"


def feature_row(features: Dict[str, float]) -> List[float]:
    """Order a feature dict into the fixed ``FEATURE_NAMES`` vector."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


class HeuristicTree:
    """Readable fallback tree; mirrors the app's original threshold logic."""

    name = "built-in heuristic tree"

    def predict(self, features: Dict[str, float]) -> Prediction:
        if features.get("phone_present"):
            return DISTRACTED, 0.90, "phone in view"

        if not features.get("task_tab_active", 1.0):
            return DISTRACTED, 0.80, "non-task window active"

        if features.get("calibrated"):
            if not features.get("face_visible"):
                return DISTRACTED, 0.70, "face not visible"
            pitch = abs(features.get("head_pitch", 0.0))
            yaw = abs(features.get("head_yaw", 0.0))
            iris = abs(features.get("iris_delta", 0.0))
            pitch_limit, yaw_limit, iris_limit = LOOK_AWAY_LIMITS
            if pitch > pitch_limit or yaw > yaw_limit or iris > iris_limit:
                return DISTRACTED, 0.75, "looking away from screen"

        if features.get("tab_switches_per_min", 0.0) >= 15:
            return DISTRACTED, 0.55, "rapid tab switching"

        return FOCUSED, 0.65, "on task"


class SklearnTree:
    """Wraps a trained ``DecisionTreeClassifier`` and its feature order."""

    name = "trained decision tree"

    def __init__(self, model, feature_names) -> None:
        self._model = model
        self._feature_names = tuple(feature_names)

    def predict(self, features: Dict[str, float]) -> Prediction:
        row = [float(features.get(name, 0.0)) for name in self._feature_names]
        label = str(self._model.predict([row])[0])
        try:
            confidence = float(max(self._model.predict_proba([row])[0]))
        except Exception:
            confidence = 1.0
        reason = _decision_reason(self._model, row, self._feature_names) if label == DISTRACTED else "on task"
        return label, confidence, reason


def load_model() -> object:
    """Return a trained tree if one is on disk, else the heuristic fallback."""
    try:
        import joblib  # optional dependency
    except Exception:
        return HeuristicTree()
    if not MODEL_PATH.is_file():
        return HeuristicTree()
    try:
        bundle = joblib.load(MODEL_PATH)
        return SklearnTree(bundle["model"], bundle["feature_names"])
    except Exception:
        return HeuristicTree()


def describe_model(model: object) -> str:
    return getattr(model, "name", type(model).__name__)
