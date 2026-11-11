"""Check that the browser's head-pose scale matches the desktop's.

Usage:
    python compare_pose.py                 # webcam, 300 frames, live preview
    python compare_pose.py --frames 600
    python compare_pose.py --video clip.mp4
    python compare_pose.py --no-display    # headless

WHAT THIS RESOLVES
------------------
The two implementations compute head pose differently:

  desktop  vision_worker.py:52-60   cv2.solvePnP over six points -> Rodrigues
  browser  web/lib/features.js:51   MediaPipe's facial transformation matrix

web/lib/features.js says why that is allowed - every consumer uses pitch/yaw as
a DELTA from the calibrated neutral pose, so a constant offset between the two
conventions cancels - and then says the part that is not automatic:

    "The scale does not automatically cancel."

That matters because distraction_model.py tests against FIXED thresholds:
pitch > 18, yaw > 22. If MediaPipe's degrees run 1.3x OpenCV's for the same
real head turn, then 18 in the browser is a different amount of head movement
than 18 on the desktop, and the deployed detector fires at a different point
from the one the Python write-up describes.

The 78 parity checks in /selftest cannot catch this. They feed numbers straight
into the tree, proving it agrees given identical inputs - not that the browser
PRODUCES identical inputs from the same face. Nothing tests the camera-to-angle
step. This does.

HOW
---
MediaPipe's Python Tasks API can emit the same transformation matrix the
browser gets, so both estimators run on the SAME frames. No second camera, no
lighting difference, no timing skew - the only thing that varies is the maths.

Turn your head slowly left and right, then up and down, through the range the
detector cares about (roughly +/-30 degrees). Hold still a moment at the
extremes. The slope is meaningless if you only ever look straight ahead.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
# The browser is served this exact file, so comparing against it is comparing
# against what participants actually run.
FACE_MODEL_PATH = APP_DIR / "web" / "models" / "face_landmarker.task"


def pose_from_matrix(matrix) -> tuple[float, float] | None:
    """Python port of poseFromMatrix in web/lib/features.js.

    The JS reads a column-major Float32Array with m(r, c) = matrix[c*4 + r],
    so m(2, 0) is row 2 / column 0 - the same element as M[2, 0] here.
    """
    M = np.asarray(matrix, dtype=float).reshape(4, 4)
    r20 = min(1.0, max(-1.0, float(M[2, 0])))
    pitch = math.degrees(math.atan2(-float(M[2, 1]), float(M[2, 2])))
    yaw = math.degrees(math.asin(r20))
    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        return None
    return pitch, yaw


def fit(desktop, browser):
    """Least squares browser = slope * desktop + intercept.

    The intercept is the convention offset, which calibration cancels. The
    slope is the thing that does not, and the whole reason for this script.
    """
    d = np.asarray(desktop, float)
    b = np.asarray(browser, float)
    if len(d) < 10 or np.ptp(d) < 5.0:
        return None
    slope, intercept = np.polyfit(d, b, 1)
    predicted = slope * d + intercept
    ss_res = float(np.sum((b - predicted) ** 2))
    ss_tot = float(np.sum((b - np.mean(b)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2, float(np.ptp(d))


def report(name, stats, threshold):
    print(f"\n== {name} ==")
    if stats is None:
        print("   not enough movement to measure. Turn your head further and re-run;")
        print("   a slope fitted to a few degrees of jitter is noise.")
        return None
    slope, intercept, r2, spread = stats
    print(f"   range covered      : {spread:.1f} degrees (desktop)")
    print(f"   slope  browser/desktop : {slope:.3f}")
    print(f"   intercept (offset)     : {intercept:+.2f} deg   <- calibration cancels this")
    print(f"   R^2                    : {r2:.4f}")
    off = abs(slope - 1.0)
    if r2 < 0.9:
        print("   !! R^2 is low. The two are not tracking each other linearly;")
        print("      a single scale factor will not fix it. Check the landmarks are stable.")
    if off < 0.05:
        print(f"   -> scales agree. The {threshold:g} degree threshold means the same")
        print("      thing in both, and needs no change.")
    else:
        suggested = threshold * slope
        print(f"   -> scales DIFFER by {off * 100:.0f}%.")
        print(f"      A real head turn that reads {threshold:g} on the desktop reads")
        print(f"      {threshold * slope:.1f} in the browser.")
        print(f"      To fire at the same point, the browser threshold should be "
              f"~{suggested:.0f}, not {threshold:g}.")
    return slope


def main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=Path, default=None,
                    help="read a video file instead of the webcam")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--chart", type=Path, default=APP_DIR / "study" / "pose_comparison.png")
    args = ap.parse_args(argv)

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as err:
        sys.exit(f"Needs opencv-python and mediapipe: {err}")

    try:
        from vision_worker import estimate_pose
    except Exception as err:            # noqa: BLE001 - any import failure is fatal here
        sys.exit(f"Could not import estimate_pose from vision_worker.py: {err}")

    if not FACE_MODEL_PATH.is_file():
        sys.exit(f"Missing {FACE_MODEL_PATH}")

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        # The browser's input. Without this the comparison cannot be made.
        output_facial_transformation_matrixes=True,
    )

    source = str(args.video) if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"Could not open {'video ' + str(args.video) if args.video else f'camera {args.camera}'}")

    pitch_d, pitch_b, yaw_d, yaw_b = [], [], [], []
    frames = 0
    used = 0

    print("Turn your head slowly left/right, then up/down, roughly +/-30 degrees.")
    print("Hold still at the extremes. Press q in the preview window to stop early.\n")

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(frames * 33))

            if result.face_landmarks and result.facial_transformation_matrixes:
                h, w = rgb.shape[:2]
                # The desktop path, using the real function from vision_worker.
                desktop = estimate_pose(result.face_landmarks[0], w, h)
                # The browser path, using the matrix MediaPipe hands the browser.
                browser = pose_from_matrix(result.facial_transformation_matrixes[0])
                if desktop and browser:
                    used += 1
                    pitch_d.append(desktop[0]); yaw_d.append(desktop[1])
                    pitch_b.append(browser[0]); yaw_b.append(browser[1])
                    if not args.no_display:
                        cv2.putText(frame, f"desktop  p{desktop[0]:+6.1f} y{desktop[1]:+6.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"browser  p{browser[0]:+6.1f} y{browser[1]:+6.1f}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        cv2.putText(frame, f"{used}/{args.frames}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if not args.no_display:
                cv2.imshow("pose comparison - q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    capture.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"{used} usable frames out of {frames}")
    if used < 10:
        sys.exit("Too few frames with a detected face to conclude anything.")

    # 18 and 22 are the thresholds in distraction_model.py:114-133.
    slope_p = report("PITCH (threshold 18)", fit(pitch_d, pitch_b), 18.0)
    slope_y = report("YAW (threshold 22)", fit(yaw_d, yaw_b), 22.0)

    print("\nThe intercept does not matter: both sides subtract the calibrated")
    print("neutral pose, so a constant offset cancels. Only the slope does.")

    _chart(args.chart, pitch_d, pitch_b, yaw_d, yaw_b, slope_p, slope_y)


def _chart(path: Path, pitch_d, pitch_b, yaw_d, yaw_b, slope_p, slope_y) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (d, b, name, slope) in zip(axes, [
            (pitch_d, pitch_b, "pitch", slope_p), (yaw_d, yaw_b, "yaw", slope_y)]):
        ax.scatter(d, b, s=9, alpha=0.5, color="#2a78d6", label="frames")
        lo, hi = min(d), max(d)
        ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--",
                linewidth=1, label="perfect agreement")
        if slope:
            intercept = float(np.mean(b)) - slope * float(np.mean(d))
            ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                    color="#b00020", linewidth=1.6, label=f"fit (slope {slope:.2f})")
        ax.set_xlabel(f"desktop {name} (solvePnP, degrees)")
        ax.set_ylabel(f"browser {name} (matrix, degrees)")
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same frames, both estimators - a slope of 1 means the thresholds transfer")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
