"""Guided, precisely-timed trial runner for the SEFH project's data tables.

This does NOT replace running real trials — you still have to be the one in
front of the camera, doing what the script says, live. What it removes is the
two biggest sources of error in hand-labeling a live main.py session: your own
reaction-time lag on a "press F8 now" ground-truth marker, and manually
reading the status banner to record what the app said. Instead:

  - a scripted sequence drives the ground truth (console countdown tells you
    what to do and for how long) instead of a hand-pressed key, so transition
    timestamps are exact
  - every ~125ms tick is classified live with the same model the real app
    uses (distraction_model.load_model())
  - the same score-smoothing + 3-second grace-period logic main.py's
    session_tick uses is replicated here, so you get both an instant
    per-frame verdict AND a "would this have actually fired an alert" verdict
  - the full raw feature vector is logged too, in case you want to compute
    something the summary doesn't

One run = one condition = one CSV, with a summary printed at the end that
maps straight onto a row in Data Table 2, 3 or 4 of the project doc. Feed the
same CSVs into evaluate_model.py for the sklearn-grade precision/recall
numbers and the poster-ready bar chart.

Data Table 5 (real-time feedback vs. self-monitoring) is a real work-session
comparison, not a scripted sequence — run the full app for that one.

Usage:
    python run_trial.py --condition angle_15 --camera 0 \\
        --script "screen:10:focused,phone:6:distracted,screen:10:focused,side:6:distracted,screen:10:focused" \\
        --out logs/angle_15.csv

Run it once per row you need (angle_0, angle_15, ...; bright, dim, back_lit;
bare_face, glasses, hair, facial_hair; ...), changing only --condition,
--script and --out each time.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from distraction_model import DISTRACTED, FEATURE_NAMES, FOCUSED, load_model
from vision_worker import VisionWorker

POLL_HZ = 8
CALIBRATION_SECONDS = 8

# Mirrors main.py's session_tick smoothing exactly, so "system_alert" reflects
# what the live app would actually have alerted on, not just a raw per-frame guess.
SCORE_UP_RATE = 0.45
SCORE_DOWN_RATE = 0.18
ALERT_THRESHOLD = 0.55
GRACE_SECONDS = 3.0


def parse_script(text: str):
    phases = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            sys.exit(f"Bad --script segment {chunk!r}; expected label:seconds:focused|distracted")
        label, seconds, truth = parts
        truth = truth.strip().lower()
        if truth not in (FOCUSED, DISTRACTED):
            sys.exit(f"Bad ground truth {truth!r} in {chunk!r}; use 'focused' or 'distracted'")
        try:
            seconds = float(seconds)
        except ValueError:
            sys.exit(f"Bad duration {seconds!r} in {chunk!r}")
        if seconds <= 0:
            sys.exit(f"Duration must be positive in {chunk!r}")
        phases.append((label.strip(), seconds, truth))
    if not phases:
        sys.exit("Empty --script")
    return phases


def wait_for_camera(vision: VisionWorker, timeout: float = 15.0) -> None:
    print("Starting camera...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if vision.latest().camera_ok:
            print("Camera ready.")
            return
        time.sleep(0.2)
    sys.exit("Camera never came up -- check --camera index and that no other app is using it.")


def run_calibration(vision: VisionWorker) -> None:
    print(f"\nCalibrating -- look normally at the screen for {CALIBRATION_SECONDS}s...")
    vision.request_calibration(CALIBRATION_SECONDS)
    deadline = time.monotonic() + CALIBRATION_SECONDS + 3.0
    while time.monotonic() < deadline:
        res = vision.latest()
        if not res.calibrating and res.calibration_result:
            break
        time.sleep(0.2)
    if not vision.calibrated():
        sys.exit("Calibration failed -- keep your face in frame and try again.")
    print("Calibrated.\n")


def collect_features(res) -> dict:
    face = res.features or {}
    return {
        "task_tab_active": 1.0,
        "tab_switches_per_min": 0.0,
        "time_on_current_tab": 999.0,
        "keyboard_inactivity": 0.0,
        "mouse_inactivity": 0.0,
        "calibrated": float(face.get("calibrated", 0.0)),
        "face_visible": float(face.get("face_visible", 1.0 if res.face_visible else 0.0)),
        "head_pitch": float(face.get("head_pitch", 0.0)),
        "head_yaw": float(face.get("head_yaw", 0.0)),
        "iris_delta": float(face.get("iris_delta", 0.0)),
        "eye_openness": float(face.get("eye_openness", 0.0)),
        "phone_present": 1.0 if res.phone_present else 0.0,
    }


def summarize(path: Path, condition: str) -> None:
    total = correct_instant = correct_alert = false_positives = focused_ticks = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != condition:
                continue
            total += 1
            truth = row["ground_truth"]
            if truth == FOCUSED:
                focused_ticks += 1
                if row["system_alert"] == DISTRACTED:
                    false_positives += 1
            correct_instant += row["system_instant"] == truth
            correct_alert += row["system_alert"] == truth
    if total == 0:
        print("No rows for this condition to summarize.")
        return
    print(f"\n--- Summary for '{condition}' ({total} ticks) ---")
    print(f"Instant-frame accuracy:  {correct_instant / total * 100:.1f}%")
    print(f"After-grace accuracy:    {correct_alert / total * 100:.1f}%  (what the app would actually alert on)")
    if focused_ticks:
        print(f"False-positive rate:     {false_positives / focused_ticks * 100:.1f}%  "
              f"({false_positives}/{focused_ticks} focused ticks wrongly alerted)")
    print("\nCopy the relevant number(s) into this condition's row in the project doc's Data Table.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, help="name for this run, e.g. angle_15, dim_light, glasses")
    parser.add_argument("--script", required=True, help="label:seconds:focused|distracted, comma-separated")
    parser.add_argument("--out", required=True, help="CSV path to append to, e.g. logs/angle_15.csv")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    phases = parse_script(args.script)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vision = VisionWorker()
    vision.start()
    vision.set_camera_index(args.camera)
    vision.enable()
    try:
        wait_for_camera(vision)
        run_calibration(vision)

        model = load_model()
        print(f"Using model: {getattr(model, 'name', type(model).__name__)}\n")

        columns = ["timestamp", "condition", "phase", "ground_truth", "system_instant",
                   "system_alert", "score", *FEATURE_NAMES]
        rows_written = 0
        score = 0.0
        distraction_started = None
        is_fresh = not out_path.exists() or out_path.stat().st_size == 0

        with out_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_fresh:
                writer.writerow(columns)

            for label, seconds, truth in phases:
                print(f">>> {label.upper()} for {seconds:.0f}s -- ground truth = {truth}")
                phase_end = time.monotonic() + seconds
                while time.monotonic() < phase_end:
                    remaining = phase_end - time.monotonic()
                    print(f"\r    {max(0.0, remaining):4.1f}s left  ", end="", flush=True)

                    res = vision.latest()
                    features = collect_features(res)
                    state, _confidence, _reason = model.predict(features)

                    target = 1.0 if state == DISTRACTED else 0.0
                    rate = SCORE_UP_RATE if target >= score else SCORE_DOWN_RATE
                    score = min(1.0, max(0.0, score + (target - score) * rate))
                    distracted_smoothed = score >= ALERT_THRESHOLD

                    now = time.monotonic()
                    distraction_started = (distraction_started or now) if distracted_smoothed else None
                    would_alert = bool(distraction_started and now - distraction_started >= GRACE_SECONDS)

                    writer.writerow([
                        f"{time.time():.3f}", args.condition, label, truth, state,
                        DISTRACTED if would_alert else FOCUSED,
                        f"{score:.3f}",
                        *[f"{features[name]:.4f}" for name in FEATURE_NAMES],
                    ])
                    rows_written += 1
                    time.sleep(1.0 / POLL_HZ)
                print()
    finally:
        vision.stop()

    print(f"\nWrote {rows_written} rows to {out_path}")
    summarize(out_path, args.condition)


if __name__ == "__main__":
    main()
