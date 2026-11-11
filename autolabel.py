"""Label feature rows from window/input telemetry, for camera-only training.

WHY THIS EXISTS
---------------
Hand-labelling is the bottleneck. ``main.py`` gets ground truth from an
operator flipping a focused/distracted toggle (F8), which works at a booth and
not at all for a remote participant. Thought probes (``web/lib/labels.js``)
give real labels but only ~10 moments per session. This script labels every
tick, with nobody in the loop.

The obvious way to do that is circular and worthless: if you label rows by
"head yaw > 22 degrees = distracted" and then train a model on head yaw, the
model just re-learns the rule that made the labels, scores ~99%, and has
discovered nothing. ``distraction_model.py``'s HeuristicTree must stay a
BASELINE TO BEAT, never a label source.

The way out is to take labels from a sensor the model cannot see. The app
records two independent channels:

    camera    face_visible, calibrated, head_pitch, head_yaw, iris_delta,
              eye_openness, phone_present      <- the FEATURES
    telemetry task_tab_active, tab_switches_per_min, time_on_current_tab,
              keyboard_inactivity, mouse_inactivity   <- the LABELS

Labelling from telemetry and training on camera is cross-modal weak
supervision. It is not circular, because no column is ever on both sides, and
the question it answers is a real one: *can the webcam alone recover what the
window telemetry knows?* That is exactly the ``camera_only`` baseline sitting
empty in ``web/data/validation.json``.

HONESTY NOTES
-------------
These labels are SILVER, not gold. "Switched to another window" is a confident
proxy for disengagement; "stayed in the window" is a weak proxy for attention,
because someone can stare straight through the screen while mind-wandering.
Expect the focused class to be noisier than the distracted one, and report
final numbers against probe labels (gold), not against these.

Published webcam mind-wandering detection reaches F1 ~0.40 on the positive
class. A model scoring 95% accuracy here has almost certainly found the class
imbalance, not the signal. Use --balance and read the F1, not the accuracy.

USAGE
-----
    python autolabel.py "focus_session*.csv" -o models/autolabelled.csv
    python train_model.py models/autolabelled.csv

The output keeps ``session_logger.py``'s 14 columns exactly, so train_model.py
consumes it unmodified.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from typing import Dict, List, Optional, Sequence

from distraction_model import FEATURE_NAMES

FOCUSED = "focused"
DISTRACTED = "distracted"

# The split that makes this non-circular. Nothing may appear in both.
CAMERA_FEATURES = (
    "calibrated",
    "face_visible",
    "head_pitch",
    "head_yaw",
    "iris_delta",
    "eye_openness",
    "phone_present",      # camera-derived: EfficientDet runs on the webcam frame
)
TELEMETRY_FEATURES = (
    "task_tab_active",
    "tab_switches_per_min",
    "time_on_current_tab",
    "keyboard_inactivity",
    "mouse_inactivity",
)

COLUMNS = ("elapsed", *FEATURE_NAMES, "label")


def validate_split() -> None:
    """Fail loudly if the two sides ever overlap or stop covering the features.

    This is the single assumption the whole method rests on, and it would be
    silently broken by someone moving one column. A leaked telemetry feature
    would not raise anything downstream - it would just quietly inflate the
    score and make the result meaningless.
    """
    camera, telemetry = set(CAMERA_FEATURES), set(TELEMETRY_FEATURES)
    overlap = camera & telemetry
    if overlap:
        sys.exit(f"LEAK: {sorted(overlap)} is both a feature and a label source.")
    missing = set(FEATURE_NAMES) - camera - telemetry
    if missing:
        sys.exit(f"Unassigned feature(s): {sorted(missing)}. Put each in exactly one group.")
    unknown = (camera | telemetry) - set(FEATURE_NAMES)
    if unknown:
        sys.exit(f"Unknown feature(s): {sorted(unknown)}.")


# --------------------------------------------------------------- loading --

def load_session(path: str) -> List[Dict[str, float]]:
    """One CSV is one session. Rows missing any feature are skipped, matching
    train_model.py:52-55 rather than inventing values for them."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            try:
                row = {name: float(record[name]) for name in FEATURE_NAMES}
                row["elapsed"] = float(record["elapsed"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["elapsed"])
    return rows


# -------------------------------------------------------------- labelling --

def _runs(rows: Sequence[Dict[str, float]], gap: float):
    """Split into maximal runs of constant on/off state.

    A time gap larger than ``gap`` starts a new run even when the state did not
    change: the rows either side are not contiguous evidence, and treating them
    as one long run would credit a duration that was never observed.
    """
    if not rows:
        return
    start = 0
    for i in range(1, len(rows) + 1):
        ended = i == len(rows)
        if not ended:
            same = bool(rows[i]["task_tab_active"]) == bool(rows[start]["task_tab_active"])
            contiguous = rows[i]["elapsed"] - rows[i - 1]["elapsed"] <= gap
            if same and contiguous:
                continue
        yield start, i, bool(rows[start]["task_tab_active"])
        start = i


def label_session(
    rows: Sequence[Dict[str, float]],
    min_away: float,
    min_on: float,
    guard: float,
    gap: float,
    switch_rate: float,
) -> List[Optional[str]]:
    """Return one label per row: FOCUSED, DISTRACTED, or None (drop it).

    Two deliberate conservatisms:

    * A run must last long enough to mean something. One stray tick of lost
      focus is a misclick, not disengagement.
    * Rows within ``guard`` seconds of a run boundary are dropped. The camera
      lags the window event - a head is still turned toward the screen for a
      moment after the alt-tab, and already turning back before the click. The
      boundary is precisely where the label is least trustworthy, so labelling
      it teaches the model noise.

    The two interact, and it is worth knowing how. A run only survives the
    guard if it is longer than ``2 * guard``, so the effective minimum is
    ``max(min_away, 2 * guard)``. At the shipped defaults those are both 2.0 s,
    which means lowering ``min_away`` alone changes nothing until it is also
    below twice the guard. Raise ``min_away`` to make it bind on its own.
    """
    labels: List[Optional[str]] = [None] * len(rows)

    for start, end, on_task in _runs(rows, gap):
        duration = rows[end - 1]["elapsed"] - rows[start]["elapsed"]
        verdict = None
        if not on_task and duration >= min_away:
            verdict = DISTRACTED
        elif on_task and duration >= min_on:
            verdict = FOCUSED
        if verdict is None:
            continue

        run_start, run_end = rows[start]["elapsed"], rows[end - 1]["elapsed"]
        for i in range(start, end):
            t = rows[i]["elapsed"]
            if t - run_start < guard or run_end - t < guard:
                continue
            labels[i] = verdict

    # Optional second labelling function. Rapid tab churn is disengagement even
    # when each individual window is the task one. Off by default: it overlaps
    # the first rule and makes the label harder to interpret.
    if switch_rate > 0:
        for i, row in enumerate(rows):
            if row["tab_switches_per_min"] >= switch_rate:
                labels[i] = DISTRACTED

    return labels


# ---------------------------------------------------------------- output --

def blank_telemetry(row: Dict[str, float]) -> Dict[str, float]:
    """Zero the telemetry columns in the emitted row.

    The columns stay present so train_model.py's positional read still works,
    but they are constant, and sklearn cannot split on a constant feature. That
    makes camera-only training structural rather than a promise - there is no
    flag anyone can forget to pass.
    """
    out = dict(row)
    for name in TELEMETRY_FEATURES:
        out[name] = 0.0
    return out


def write_rows(path: str, rows: Sequence[Dict[str, float]], labels: Sequence[str],
               keep_telemetry: bool) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row, label in zip(rows, labels):
            emit = row if keep_telemetry else blank_telemetry(row)
            writer.writerow(
                [f"{row['elapsed']:.2f}"]
                + [f"{emit[name]:.4f}" for name in FEATURE_NAMES]
                + [label]
            )


def balance(rows: List[Dict[str, float]], labels: List[str]):
    """Downsample the majority class by taking an evenly spaced subset.

    Evenly spaced, not random: it keeps the retained rows spread across the
    whole session instead of clustering, and it is reproducible without a seed.
    """
    by_label: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)
    if len(by_label) < 2:
        return rows, labels

    smallest = min(len(v) for v in by_label.values())
    keep = set()
    for indices in by_label.values():
        step = len(indices) / smallest
        keep.update(indices[int(i * step)] for i in range(smallest))
    ordered = sorted(keep)
    return [rows[i] for i in ordered], [labels[i] for i in ordered]


# ------------------------------------------------------------------ main --

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive focused/distracted labels from window telemetry so a "
                    "camera-only model can be trained without hand annotation.")
    parser.add_argument("patterns", nargs="+",
                        help="CSV paths or globs; each file is treated as one session")
    parser.add_argument("-o", "--out", default="models/autolabelled.csv")
    parser.add_argument("--min-away", type=float, default=2.0,
                        help="seconds off-task before a run counts as distracted (default 2)")
    parser.add_argument("--min-on", type=float, default=5.0,
                        help="seconds on-task before a run counts as focused (default 5)")
    parser.add_argument("--guard", type=float, default=1.0,
                        help="seconds dropped either side of a transition (default 1)")
    parser.add_argument("--gap", type=float, default=1.0,
                        help="a time gap larger than this breaks a run (default 1)")
    parser.add_argument("--switch-rate", type=float, default=0.0,
                        help="also mark rows with at least this many tab switches/min "
                             "as distracted (0 disables, the default)")
    parser.add_argument("--balance", action="store_true",
                        help="downsample the majority class to equal size")
    parser.add_argument("--keep-telemetry", action="store_true",
                        help="DIAGNOSTIC ONLY: emit real telemetry values. Training on "
                             "the result is circular and the score is meaningless.")
    args = parser.parse_args()

    validate_split()

    paths: List[str] = []
    for pattern in args.patterns:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    all_rows: List[Dict[str, float]] = []
    all_labels: List[str] = []
    per_file = []

    for path in paths:
        try:
            rows = load_session(path)
        except OSError as err:
            print(f"  skipped {path}: {err}")
            continue
        if not rows:
            per_file.append((path, 0, 0, 0))
            continue

        labels = label_session(rows, args.min_away, args.min_on,
                               args.guard, args.gap, args.switch_rate)
        kept = [(r, l) for r, l in zip(rows, labels) if l is not None]
        n_d = sum(1 for _, l in kept if l == DISTRACTED)
        per_file.append((path, len(rows), len(kept), n_d))
        all_rows.extend(r for r, _ in kept)
        all_labels.extend(l for _, l in kept)

    if not all_rows:
        sys.exit("No rows could be labelled. Every run was shorter than the minimums, "
                 "or the files had no usable feature columns.")

    if args.balance:
        all_rows, all_labels = balance(all_rows, all_labels)

    write_rows(args.out, all_rows, all_labels, args.keep_telemetry)

    # ---- report ----
    print(f"\n{'file':<44}{'rows':>8}{'labelled':>10}{'distracted':>12}")
    for path, total, kept, n_d in per_file:
        name = path if len(path) <= 43 else "..." + path[-40:]
        print(f"{name:<44}{total:>8}{kept:>10}{n_d:>12}")

    n = len(all_labels)
    n_d = sum(1 for l in all_labels if l == DISTRACTED)
    print(f"\nWrote {n} rows to {args.out}")
    print(f"  distracted {n_d} ({n_d / n:.1%})   focused {n - n_d} ({(n - n_d) / n:.1%})")

    if args.keep_telemetry:
        print("\n  !! --keep-telemetry: the label source is still in the features.")
        print("     Training on this file is CIRCULAR. Diagnostics only.")
    else:
        print(f"  telemetry columns zeroed: {', '.join(TELEMETRY_FEATURES)}")
        print("  the tree can only learn from the camera columns.")

    minority = min(n_d, n - n_d)
    if minority == 0:
        print("\n  !! Only one class present. train_model.py will refuse this file.")
    elif minority / n < 0.15:
        print(f"\n  !! Minority class is {minority / n:.1%}. Accuracy will look high for "
              f"the wrong reason -\n     read F1 on the distracted class, or pass --balance.")

    print(f"\nNext:  python train_model.py {args.out}")
    print("Report F1 on the distracted class. Published webcam work sits near 0.40;")
    print("a much higher number usually means leakage or imbalance, not success.")


if __name__ == "__main__":
    main()
