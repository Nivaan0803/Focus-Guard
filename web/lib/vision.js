/* Camera + MediaPipe Tasks for Web setup.
 *
 * Pinned to @mediapipe/tasks-vision 0.10.21 — the same MediaPipe version as
 * `mediapipe==0.10.21` in requirements.txt, so the browser runs the same
 * Face Mesh and EfficientDet graphs as the desktop app.
 *
 * Both model files are self-hosted under models/ (the .tflite is a byte copy of
 * the repo's own models/efficientdet_lite0.tflite) so a flaky CDN cannot take
 * a participant session down mid-study.
 */

const MP_VERSION = "0.10.21";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

export const PHONE_SCORE_THRESHOLD = 0.20;   // vision_worker.py:91

let _mp = null;
async function mediapipe() {
  if (!_mp) _mp = await import(/* @vite-ignore */ `${MP_BASE}/vision_bundle.mjs`);
  return _mp;
}

export function isSupported() {
  return Boolean(
    navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia &&
    typeof WebAssembly === "object"
  );
}

/* Wraps getUserMedia with the error cases that actually happen with a real
 * participant, so the caller can show a sentence instead of a stack trace. */
export async function openCamera(video, { width = 1280, height = 720 } = {}) {
  if (!isSupported()) {
    throw new Error("This browser cannot use a camera. Try desktop Chrome, Edge, or Safari.");
  }
  if (!window.isSecureContext) {
    throw new Error("The camera needs a secure connection (https:// or localhost).");
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: width }, height: { ideal: height }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    const name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      throw new Error("Camera permission was blocked. Allow camera access in your browser's address bar, then reload.");
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      throw new Error("No camera was found on this device.");
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      throw new Error("The camera is in use by another app. Close it (Zoom, Teams, Meet) and reload.");
    }
    throw new Error(`Could not start the camera: ${name || err}`);
  }
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  // Metadata can lag play() on Safari; videoWidth stays 0 until it lands.
  if (!video.videoWidth) {
    await new Promise((resolve) => {
      const done = () => resolve();
      video.addEventListener("loadedmetadata", done, { once: true });
      setTimeout(done, 3000);
    });
  }
  return stream;
}

export function closeCamera(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try { track.stop(); } catch { /* already gone */ }
  }
}

export async function createFaceLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await mediapipe();
  const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/face_landmarker.task" },
    runningMode: "VIDEO",
    numFaces: 1,
    // The 478-point topology with iris landmarks 468/473, matching the Python
    // side's refine_landmarks. features.js depends on these indices.
    outputFaceBlendshapes: false,
    // Supplies the 4x4 we read pitch/yaw from in features.js:poseFromMatrix.
    outputFacialTransformationMatrixes: true,
  });
}

export async function createPhoneDetector() {
  const { ObjectDetector, FilesetResolver } = await mediapipe();
  const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  // Same options as vision_worker.py:91
  return ObjectDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/efficientdet_lite0.tflite" },
    runningMode: "VIDEO",
    scoreThreshold: PHONE_SCORE_THRESHOLD,
    categoryAllowlist: ["cell phone"],
    maxResults: 1,
  });
}

/* Mirror the frame into a canvas before inference.
 *
 * vision_worker.py:271 does `cv2.flip(frame, 1)` first, so every landmark the
 * Python model sees is mirrored. Doing the same here keeps iris_delta's sign
 * and the head-yaw direction consistent between the two implementations —
 * which matters if a tree trained on desktop logs is ever run in the browser.
 */
export function mirrorToCanvas(video, canvas) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return null;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  return ctx;
}
