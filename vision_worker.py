"""Background webcam + screen workers for Focus Guard.

Camera capture, MediaPipe Face Mesh, and the EfficientDet phone detector run on
``VisionWorker``'s own thread; the desktop grab runs on ``ScreenWorker``'s. The
Tk UI only ever reads the most recent published result, so inference never
blocks the event loop. Nothing is written to disk — each frame is overwritten
in memory on the next loop.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageGrab

from distraction_model import LOOK_AWAY_LIMITS

APP_DIR = Path(__file__).resolve().parent
PHONE_MODEL = APP_DIR / "models" / "efficientdet_lite0.tflite"
PHONE_MODEL_RELATIVE = Path("models") / "efficientdet_lite0.tflite"

POSE_POINTS = (1, 152, 33, 263, 61, 291)
FACE_MODEL = np.array([(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26),
                       (43.3, 32.7, -26), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], np.float64)

PREVIEW_MAX = (960, 540)
PREVIEW_SIZE = PREVIEW_MAX
DEFAULT_THRESHOLDS = LOOK_AWAY_LIMITS
POSE_SMOOTHING = 0.35  # EMA weight of the newest frame; damps solvePnP jitter

# Phone detection: a raw hit must clear PHONE_MIN_SCORE, look phone-sized
# (fingers or a hand right at the lens fill most of the frame), and repeat in
# PHONE_CONFIRM_HITS of the last PHONE_WINDOW checks before it counts.
PHONE_MIN_SCORE = 0.45
PHONE_MAX_FRAME_FRACTION = (0.45, 0.65)  # max box width, height as share of frame
PHONE_CHECK_INTERVAL = 0.35
PHONE_WINDOW = 4
PHONE_CONFIRM_HITS = 3


def fit_preview(image_rgb: np.ndarray) -> np.ndarray:
    """Resize to fit inside PREVIEW_MAX without changing the aspect ratio.

    This is only a cap on the source resolution handed to the UI thread —
    the UI itself scales the frame again to match however big the preview
    widget currently is, so a maximized window still fills up.
    """
    height, width = image_rgb.shape[:2]
    scale = min(PREVIEW_MAX[0] / width, PREVIEW_MAX[1] / height, 1.0)
    return cv2.resize(image_rgb, (max(1, int(width * scale)), max(1, int(height * scale))))


def estimate_pose(landmarks, width: int, height: int):
    """Estimate pitch/yaw from Face Mesh landmarks in degrees."""
    image_points = np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in POSE_POINTS], np.float64)
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], np.float64)
    ok, rotation, _ = cv2.solvePnP(FACE_MODEL, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    return math.degrees(math.atan2(-matrix[2, 1], matrix[2, 2])), math.degrees(math.asin(matrix[2, 0]))


def iris_position(landmarks):
    """Return the averaged normalised iris offset within the eyes."""
    def position(iris, left, right):
        span = landmarks[right].x - landmarks[left].x
        return (landmarks[iris].x - (landmarks[left].x + landmarks[right].x) / 2) / span if abs(span) > .001 else 0
    return (position(468, 33, 133) + position(473, 362, 263)) / 2


def eye_openness(landmarks) -> float:
    """Estimate eye openness from vertical/horizontal eyelid ratios (0..1)."""
    def distance(first, second):
        dx = landmarks[first].x - landmarks[second].x
        dy = landmarks[first].y - landmarks[second].y
        return math.hypot(dx, dy)

    left = distance(159, 145) / max(distance(33, 133), .001)
    right = distance(386, 374) / max(distance(362, 263), .001)
    return min(1.0, max(0.0, ((left + right) / 2) / .32))


_chdir_lock = threading.Lock()


def create_phone_detector():
    """Create the MediaPipe phone detector, working around Windows path issues."""
    if not PHONE_MODEL.is_file():
        raise FileNotFoundError(f"Missing phone model: {PHONE_MODEL}")

    options = dict(score_threshold=PHONE_MIN_SCORE, category_allowlist=["cell phone"], max_results=3)
    try:
        return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_buffer=PHONE_MODEL.read_bytes()), **options))
    except Exception as buffer_error:
        with _chdir_lock:
            cwd = Path.cwd()
            try:
                os.chdir(APP_DIR)
                return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
                    base_options=python.BaseOptions(model_asset_path=str(PHONE_MODEL_RELATIVE)), **options))
            except Exception as path_error:
                raise RuntimeError(
                    f"Unable to load phone detector. buffer error: {buffer_error}; path error: {path_error}"
                ) from path_error
            finally:
                os.chdir(cwd)


def plausible_phone(detections, width: int, height: int):
    """Return the best phone-sized detection box, or None."""
    max_w, max_h = PHONE_MAX_FRAME_FRACTION
    for detection in sorted(detections, key=lambda d: d.categories[0].score, reverse=True):
        box = detection.bounding_box
        if box.width <= max_w * width and box.height <= max_h * height:
            return box.origin_x, box.origin_y, box.width, box.height
    return None


def open_camera(index: int):
    """Open the selected camera with the two common Windows backends."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, _ = camera.read()
            if ok:
                return camera
        camera.release()
    return None


@dataclass
class VisionResult:
    """Snapshot the UI thread reads; the worker replaces it wholesale each loop."""
    frame_rgb: Optional[np.ndarray] = None
    fps: float = 0.0
    camera_ok: bool = False
    camera_message: str = "Camera off"
    face_visible: bool = False
    phone_present: bool = False
    looking_away: bool = False
    features: dict = field(default_factory=dict)
    gaze_text: str = ""
    gaze_color: str = "#a9b6c8"
    head_rotation_deg: Optional[float] = None
    eye_openness_value: Optional[float] = None
    calibrating: bool = False
    calibration_remaining: float = 0.0
    calibration_seq: int = 0
    calibration_result: str = ""  # "", "done", "failed"


class VisionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="VisionWorker")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._want = threading.Event()
        self._result = VisionResult()
        self._camera_index = 0
        self._sensitivity = 1.0
        self._calibration_seconds = 10
        self._calibrate_deadline: Optional[float] = None
        self._samples: list = []
        self._neutral_pose: Optional[tuple] = None
        self._neutral_iris: Optional[float] = None
        self._thresholds = DEFAULT_THRESHOLDS
        self._calibration_seq = 0

    # ---------- controls, called from the UI thread ----------
    def enable(self) -> None:
        self._want.set()

    def disable(self) -> None:
        self._want.clear()

    def stop(self) -> None:
        self._stop.set()

    def set_camera_index(self, index: int) -> None:
        with self._lock:
            self._camera_index = int(index)

    def set_sensitivity(self, value: float) -> None:
        with self._lock:
            self._sensitivity = max(0.4, min(1.8, float(value)))

    def request_calibration(self, seconds: int) -> None:
        with self._lock:
            self._calibration_seconds = max(3, int(seconds))
            self._calibrate_deadline = time.monotonic() + self._calibration_seconds
            self._samples = []
            self._neutral_pose = None
            self._neutral_iris = None

    def clear_calibration(self) -> None:
        with self._lock:
            self._calibrate_deadline = None
            self._samples = []
            self._neutral_pose = None
            self._neutral_iris = None

    def calibrated(self) -> bool:
        with self._lock:
            return self._neutral_pose is not None

    def latest(self) -> VisionResult:
        with self._lock:
            return self._result

    def _publish(self, result: VisionResult) -> None:
        with self._lock:
            self._result = result

    # ---------- worker thread ----------
    def run(self) -> None:
        cap = mesh = detector = None
        open_index = None
        next_retry = 0.0
        last_phone = 0.0
        phone_box = None
        phone_hits: deque = deque(maxlen=PHONE_WINDOW)
        smooth_pose = smooth_iris = None
        stamps: deque = deque(maxlen=30)

        while not self._stop.is_set():
            try:
                if not self._want.is_set():
                    cap, mesh, detector, open_index = self._teardown(cap, mesh, detector)
                    phone_box = smooth_pose = smooth_iris = None
                    phone_hits.clear()
                    self._publish(VisionResult(camera_message="Camera off"))
                    time.sleep(0.15)
                    continue

                with self._lock:
                    cam_index = self._camera_index
                    sensitivity = max(0.4, self._sensitivity)
                    neutral_pose = self._neutral_pose
                    neutral_iris = self._neutral_iris
                    thresholds = self._thresholds
                    calibrating = self._calibrate_deadline is not None
                    cal_remaining = max(0.0, self._calibrate_deadline - time.monotonic()) if calibrating else 0.0

                if cap is None or open_index != cam_index:
                    if cap is not None:
                        cap.release()
                        cap = None
                    if time.monotonic() < next_retry and open_index == cam_index:
                        time.sleep(0.1)
                        continue
                    cap = open_camera(cam_index)
                    open_index = cam_index
                    if cap is None:
                        next_retry = time.monotonic() + 2.0
                        self._publish(VisionResult(camera_message=f"Camera {cam_index} unavailable — retrying…"))
                        time.sleep(0.4)
                        continue
                    if mesh is None:
                        mesh = mp.solutions.face_mesh.FaceMesh(
                            max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=.55, min_tracking_confidence=.55)
                    if detector is None:
                        try:
                            detector = create_phone_detector()
                        except Exception:
                            detector = None

                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    open_index = None
                    next_retry = time.monotonic() + 1.5
                    self._publish(VisionResult(camera_message="Camera returned no frame — reconnecting…"))
                    time.sleep(0.3)
                    continue

                loop_start = time.monotonic()
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mesh_result = mesh.process(rgb) if mesh else None

                result = VisionResult(camera_ok=True, camera_message="")
                features = {"face_visible": 0.0, "calibrated": 1.0 if neutral_pose is not None else 0.0}

                if mesh_result and mesh_result.multi_face_landmarks:
                    landmarks = mesh_result.multi_face_landmarks[0].landmark
                    pose = estimate_pose(landmarks, width, height)
                    iris = iris_position(landmarks)
                    if pose is not None:
                        smooth_pose = pose if smooth_pose is None else tuple(
                            old + POSE_SMOOTHING * (new - old) for old, new in zip(smooth_pose, pose))
                        pose = smooth_pose
                    smooth_iris = iris if smooth_iris is None else smooth_iris + POSE_SMOOTHING * (iris - smooth_iris)
                    iris = smooth_iris
                    openness = eye_openness(landmarks)
                    result.face_visible = True
                    result.eye_openness_value = openness

                    neutral_pose, neutral_iris, thresholds, cal_event, calibrating, cal_remaining = \
                        self._advance_calibration(pose, iris)
                    result.calibration_result = cal_event
                    with self._lock:
                        result.calibration_seq = self._calibration_seq

                    head_dp = head_dy = iris_delta = 0.0
                    if neutral_pose is not None and pose is not None:
                        head_dp = pose[0] - neutral_pose[0]
                        head_dy = pose[1] - neutral_pose[1]
                    if neutral_iris is not None:
                        iris_delta = iris - neutral_iris

                    features.update({
                        "face_visible": 1.0,
                        "calibrated": 1.0 if neutral_pose is not None else 0.0,
                        "head_pitch": head_dp,
                        "head_yaw": head_dy,
                        "iris_delta": iris_delta,
                        "eye_openness": openness,
                    })
                    if neutral_pose is not None:
                        result.head_rotation_deg = max(abs(head_dp), abs(head_dy))
                    elif pose is not None:
                        result.head_rotation_deg = max(abs(pose[0]), abs(pose[1]))

                    pitch_t, yaw_t, iris_t = thresholds
                    looking_away = neutral_pose is not None and (
                        abs(head_dp) > pitch_t / sensitivity
                        or abs(head_dy) > yaw_t / sensitivity
                        or abs(iris_delta) > iris_t / sensitivity)
                    result.looking_away = looking_away

                    if calibrating:
                        result.calibrating = True
                        result.calibration_remaining = cal_remaining
                        result.gaze_text = f"Camera: calibrating — look normally at your screen: {cal_remaining:.1f}s"
                        result.gaze_color = "#f0b44d"
                    elif neutral_pose is None:
                        result.gaze_text = "Camera: press ‘Calibrate gaze’ (F7) while looking normally at your screen."
                        result.gaze_color = "#f0b44d"
                    elif looking_away:
                        result.gaze_text = f"Camera: looking away  head {head_dp:+.0f}°/{head_dy:+.0f}°"
                        result.gaze_color = "#ff6666"
                    else:
                        result.gaze_text = f"Camera: screen-facing  head {head_dp:+.0f}°/{head_dy:+.0f}°"
                        result.gaze_color = "#72dd91"

                    for point in landmarks[::12]:
                        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 1, (70, 230, 90), -1)
                    tag = "Gaze: looking away" if looking_away else "Face Mesh: detected"
                    cv2.putText(frame, tag, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (70, 230, 90), 2)
                else:
                    smooth_pose = smooth_iris = None
                    _, _, _, cal_event, calibrating, _ = self._advance_calibration(None, None)
                    result.calibration_result = cal_event
                    with self._lock:
                        result.calibration_seq = self._calibration_seq
                    result.calibrating = calibrating
                    if calibrating:
                        result.gaze_text = "Camera: face not visible — keep your face in view to calibrate."
                    else:
                        result.gaze_text = "Camera: no face detected."
                    result.gaze_color = "#ff6666"
                    cv2.putText(frame, "Face Mesh: no face detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (60, 60, 255), 2)

                now = time.monotonic()
                if detector is not None and now - last_phone >= PHONE_CHECK_INTERVAL:
                    last_phone = now
                    try:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detections = detector.detect(mp_image).detections
                    except Exception:
                        detections = None
                    candidate = plausible_phone(detections or [], width, height)
                    phone_hits.append(candidate is not None)
                    if sum(phone_hits) >= PHONE_CONFIRM_HITS:
                        phone_box = candidate or phone_box
                    else:
                        phone_box = None

                result.phone_present = phone_box is not None
                if phone_box is not None:
                    x, y, bw, bh = phone_box
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 60, 255), 3)
                    cv2.putText(frame, "PHONE DETECTED", (x, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 60, 255), 2)
                    if result.gaze_color != "#ff6666":
                        result.gaze_text = "Camera: phone in view"
                        result.gaze_color = "#ff6666"

                features["phone_present"] = 1.0 if phone_box is not None else 0.0
                result.features = features
                result.frame_rgb = fit_preview(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                stamps.append(time.monotonic())
                if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                    result.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])

                self._publish(result)
                time.sleep(max(0.0, (1 / 30) - (time.monotonic() - loop_start)))
            except Exception as error:  # keep the thread alive through transient failures
                self._publish(VisionResult(camera_message=f"Vision error: {type(error).__name__}"))
                time.sleep(0.5)

        self._teardown(cap, mesh, detector)

    def _advance_calibration(self, pose, iris):
        """Collect a calibration sample and finalise the baseline when time is up."""
        cal_event = ""
        with self._lock:
            deadline = self._calibrate_deadline
            if deadline is not None:
                if pose is not None:
                    self._samples.append((pose[0], pose[1], iris))
                if time.monotonic() >= deadline:
                    if len(self._samples) >= 25:
                        s = np.array(self._samples)
                        self._neutral_pose = (float(np.median(s[:, 0])), float(np.median(s[:, 1])))
                        self._neutral_iris = float(np.median(s[:, 2]))
                        self._thresholds = (
                            float(min(40.0, 3.5 * np.std(s[:, 0]) + DEFAULT_THRESHOLDS[0])),
                            float(min(48.0, 3.5 * np.std(s[:, 1]) + DEFAULT_THRESHOLDS[1])),
                            float(min(0.35, 5.0 * np.std(s[:, 2]) + DEFAULT_THRESHOLDS[2])),
                        )
                        cal_event = "done"
                    else:
                        cal_event = "failed"
                    self._calibrate_deadline = None
                    self._samples = []
                    self._calibration_seq += 1
            calibrating = self._calibrate_deadline is not None
            remaining = max(0.0, self._calibrate_deadline - time.monotonic()) if calibrating else 0.0
            return self._neutral_pose, self._neutral_iris, self._thresholds, cal_event, calibrating, remaining

    @staticmethod
    def _teardown(cap, mesh, detector):
        for resource in (detector, mesh, cap):
            try:
                if resource is not None:
                    (resource.close if hasattr(resource, "close") else resource.release)()
            except Exception:
                pass
        return None, None, None, None


class ScreenWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="ScreenWorker")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._want = threading.Event()
        self._image: Optional[np.ndarray] = None
        self._message = "Screen preview starting…"
        self._interval = 0.5

    def enable(self) -> None:
        self._want.set()

    def disable(self) -> None:
        self._want.clear()

    def stop(self) -> None:
        self._stop.set()

    def latest(self):
        with self._lock:
            return self._image, self._message

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._want.is_set():
                with self._lock:
                    self._image, self._message = None, "Screen preview disabled"
                time.sleep(0.2)
                continue
            try:
                grab = ImageGrab.grab()
                grab.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
                arr = np.asarray(grab.convert("RGB"))
                with self._lock:
                    self._image, self._message = arr, ""
            except Exception as error:
                with self._lock:
                    self._image, self._message = None, f"Screen preview unavailable: {type(error).__name__}"
            time.sleep(self._interval)
