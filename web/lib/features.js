/* Landmark math ported from vision_worker.py.
 *
 * The Python worker flips the frame horizontally (vision_worker.py:271
 * `cv2.flip(frame, 1)`) before anything touches it, so the browser mirrors the
 * video into a canvas before inference and these functions see the same
 * geometry the desktop app does.
 */

// vision_worker.py:37
export const DEFAULT_THRESHOLDS = Object.freeze([18.0, 22.0, 0.18]);

// vision_worker.py:31 — the six solvePnP correspondence points.
export const POSE_POINTS = Object.freeze([1, 152, 33, 263, 61, 291]);

/* vision_worker.py:63-68 — averaged normalised iris offset within the eyes.
 * Landmarks 468 / 473 are the iris centres, present only because the Python
 * side runs Face Mesh with refine_landmarks; the web `face_landmarker.task`
 * bundle ships the same 478-point topology, so the indices carry over. */
export function irisPosition(lm) {
  const position = (iris, left, right) => {
    const span = lm[right].x - lm[left].x;
    if (Math.abs(span) <= 0.001) return 0;
    return (lm[iris].x - (lm[left].x + lm[right].x) / 2) / span;
  };
  return (position(468, 33, 133) + position(473, 362, 263)) / 2;
}

/* vision_worker.py:71-80 — eyelid vertical/horizontal ratio, normalised by
 * 0.32 and clamped to 0..1. */
export function eyeOpenness(lm) {
  const distance = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
  const left = distance(159, 145) / Math.max(distance(33, 133), 0.001);
  const right = distance(386, 374) / Math.max(distance(362, 263), 0.001);
  return Math.min(1, Math.max(0, ((left + right) / 2) / 0.32));
}

/* Head pose.
 *
 * vision_worker.py:52-60 runs cv2.solvePnP over POSE_POINTS and converts the
 * Rodrigues vector to pitch/yaw. There is no solvePnP in the browser, so we
 * take the 4x4 facial transformation matrix MediaPipe can emit directly and
 * pull the same two Euler angles out of its rotation block.
 *
 * This is viable only because every consumer uses pitch/yaw as a DELTA from the
 * calibrated neutral pose (vision_worker.py:293-297), so a constant offset
 * between the two conventions cancels. The scale does not automatically cancel
 * — see the parity check in the plan's verification section.
 *
 * `matrix` is the column-major Float32Array(16) MediaPipe returns.
 */
export function poseFromMatrix(matrix) {
  if (!matrix || matrix.length < 16) return null;
  const m = (r, c) => matrix[c * 4 + r];      // column-major -> (row, col)

  // Mirrors the Rodrigues->Euler extraction at vision_worker.py:59-60:
  //   pitch = atan2(-R[2,1], R[2,2]);  yaw = asin(R[2,0])
  const r20 = Math.min(1, Math.max(-1, m(2, 0)));
  const pitch = Math.atan2(-m(2, 1), m(2, 2)) * 180 / Math.PI;
  const yaw = Math.asin(r20) * 180 / Math.PI;
  if (!Number.isFinite(pitch) || !Number.isFinite(yaw)) return null;
  return [pitch, yaw];
}

/* Assemble the per-frame camera features, matching the dict built at
 * vision_worker.py:277-307. Returns the camera half of the feature vector plus
 * the derived display values; the tab/input half is added by the caller. */
export function cameraFeatures({ landmarks, matrix, calibration, sensitivity = 1.0 }) {
  const out = {
    face_visible: 0.0,
    calibrated: calibration && calibration.neutralPose ? 1.0 : 0.0,
    head_pitch: 0.0,
    head_yaw: 0.0,
    iris_delta: 0.0,
    eye_openness: 0.0,
  };
  const extra = { pose: null, iris: null, lookingAway: false, headRotationDeg: null };

  if (!landmarks || !landmarks.length) return { features: out, ...extra };

  const pose = poseFromMatrix(matrix);
  const iris = irisPosition(landmarks);
  const openness = eyeOpenness(landmarks);

  out.face_visible = 1.0;
  out.eye_openness = openness;
  extra.pose = pose;
  extra.iris = iris;

  const neutralPose = calibration ? calibration.neutralPose : null;
  const neutralIris = calibration ? calibration.neutralIris : null;

  if (neutralPose && pose) {
    out.head_pitch = pose[0] - neutralPose[0];
    out.head_yaw = pose[1] - neutralPose[1];
  }
  if (neutralIris !== null && neutralIris !== undefined) {
    out.iris_delta = iris - neutralIris;
  }
  out.calibrated = neutralPose ? 1.0 : 0.0;

  // vision_worker.py:308-312
  if (neutralPose) {
    extra.headRotationDeg = Math.max(Math.abs(out.head_pitch), Math.abs(out.head_yaw));
  } else if (pose) {
    extra.headRotationDeg = Math.max(Math.abs(pose[0]), Math.abs(pose[1]));
  }

  // vision_worker.py:314-319 — note this uses the CALIBRATED thresholds scaled
  // by sensitivity, which is a separate signal from the heuristic tree's own
  // fixed 18/22/0.18 comparison in distraction_model.py.
  const [pitchT, yawT, irisT] = (calibration && calibration.thresholds) || DEFAULT_THRESHOLDS;
  extra.lookingAway = Boolean(neutralPose) && (
    Math.abs(out.head_pitch) > pitchT / sensitivity ||
    Math.abs(out.head_yaw) > yawT / sensitivity ||
    Math.abs(out.iris_delta) > irisT / sensitivity
  );

  return { features: out, ...extra };
}
