/* Booth demo: the live detector, read-only.
 *
 * Records nothing. There is deliberately no logging, no upload, and no storage
 * here, so a visitor trying the demo is not a research participant and needs no
 * consent form. If you ever add recording, this page must go behind the gate.
 */

import * as vision from "../lib/vision.js";
import { cameraFeatures, poseFromMatrix, irisPosition } from "../lib/features.js";
import { Calibration } from "../lib/calibrate.js";
import { loadModel, describeModel, DISTRACTED } from "../lib/tree.js";
import { ScoreTracker } from "../lib/score.js";
import { beep, unlock } from "../lib/beep.js";

const $ = (id) => document.getElementById(id);
const TICK_MS = 90;

const camEl = $("cam");
const workCanvas = $("work");
const overlay = $("overlay");

const state = {
  stream: null, face: null, phone: null, model: null,
  calibration: new Calibration(),
  score: new ScoreTracker({ grace: 3, realertSeconds: 0 }),
  timer: null,
  lastKey: performance.now() / 1000,
  lastPointer: performance.now() / 1000,
  tabSwitches: [],
  currentTabStarted: performance.now() / 1000,
  phoneBox: null,
};

const now = () => performance.now() / 1000;

addEventListener("keydown", () => { state.lastKey = now(); }, { passive: true });
addEventListener("pointermove", () => { state.lastPointer = now(); }, { passive: true });
addEventListener("visibilitychange", () => {
  state.tabSwitches.push(now());
  state.currentTabStarted = now();
});

function drawOverlay(landmarks, box) {
  const w = workCanvas.width, h = workCanvas.height;
  if (!w || !h) return;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  if (landmarks) {
    // Every 12th point, matching the desktop preview's density.
    ctx.fillStyle = "rgba(114, 221, 145, .85)";
    for (let i = 0; i < landmarks.length; i += 12) {
      ctx.fillRect(landmarks[i].x * w - 1, landmarks[i].y * h - 1, 2.5, 2.5);
    }
  }
  if (box) {
    ctx.strokeStyle = "#e0574a";
    ctx.lineWidth = Math.max(2, w / 300);
    ctx.strokeRect(box.originX, box.originY, box.width, box.height);
    ctx.fillStyle = "#e0574a";
    ctx.font = `${Math.max(14, w / 40)}px system-ui, sans-serif`;
    ctx.fillText("PHONE DETECTED", box.originX, Math.max(18, box.originY - 6));
  }
}

function tick() {
  if (!vision.mirrorToCanvas(camEl, workCanvas)) return;
  const t = now();
  const stamp = Math.round(t * 1000);

  let landmarks = null, matrix = null, box = null, phonePresent = 0;
  try {
    const fr = state.face.detectForVideo(workCanvas, stamp);
    if (fr && fr.faceLandmarks && fr.faceLandmarks.length) {
      landmarks = fr.faceLandmarks[0];
      const m = fr.facialTransformationMatrixes;
      if (m && m.length) matrix = m[0].data;
    }
  } catch { /* a dropped frame must not stop the demo */ }
  try {
    const pr = state.phone.detectForVideo(workCanvas, stamp);
    if (pr && pr.detections && pr.detections.length) {
      phonePresent = 1;
      box = pr.detections[0].boundingBox;
    }
  } catch { /* ditto */ }

  let cam;
  if (landmarks) {
    state.calibration.advance(poseFromMatrix(matrix), irisPosition(landmarks));
    cam = cameraFeatures({ landmarks, matrix, calibration: state.calibration }).features;
  } else {
    state.calibration.advance(null, null);
    cam = { face_visible: 0, calibrated: state.calibration.calibrated ? 1 : 0,
            head_pitch: 0, head_yaw: 0, iris_delta: 0, eye_openness: 0 };
  }

  state.tabSwitches = state.tabSwitches.filter((s) => t - s <= 60);
  const onTask = document.hasFocus() && !document.hidden;

  const features = {
    task_tab_active: onTask ? 1 : 0,
    tab_switches_per_min: state.tabSwitches.length,
    time_on_current_tab: t - state.currentTabStarted,
    keyboard_inactivity: t - state.lastKey,
    mouse_inactivity: t - state.lastPointer,
    ...cam,
    phone_present: phonePresent,
  };

  const [verdict, , reason] = state.model.predict(features);
  state.score.update(verdict, t, 0);
  if (state.score.shouldAlert(t, reason)) beep();

  drawOverlay(landmarks, box);

  const cal = state.calibration;
  const banner = $("verdict");
  if (cal.calibrating) {
    banner.className = "banner warn";
    banner.textContent = `CALIBRATING — look at your screen: ${cal.remaining.toFixed(1)}s`;
  } else if (!cal.calibrated) {
    banner.className = "banner warn";
    banner.textContent = "NOT CALIBRATED — press Recalibrate";
  } else {
    const b = state.score.bannerState(t, { taskTabActive: onTask });
    banner.className = `banner ${b}`;
    banner.textContent = b === "bad" ? `DISTRACTED — ${reason}`
      : b === "warn" ? `CHECKING — ${reason}`
      : "FOCUSED";
  }

  $("meter").style.width = `${(state.score.score * 100).toFixed(0)}%`;
  $("meter").style.background = state.score.meterColor;

  $("s-pitch").textContent = `${features.head_pitch >= 0 ? "+" : ""}${features.head_pitch.toFixed(0)}°`;
  $("s-yaw").textContent = `${features.head_yaw >= 0 ? "+" : ""}${features.head_yaw.toFixed(0)}°`;
  $("s-iris").textContent = features.iris_delta.toFixed(2);
  $("s-open").textContent = features.eye_openness.toFixed(2);
  $("s-face").textContent = features.face_visible ? "visible" : "not visible";
  $("s-phone").textContent = phonePresent ? "detected" : "none";
  $("s-tab").textContent = onTask ? "focused" : "away";
  $("s-score").textContent = state.score.score.toFixed(2);
}

async function start() {
  const err = $("err");
  err.textContent = "";
  if (!vision.isSupported()) {
    err.textContent = "This browser cannot run the demo. Try an up-to-date Chrome, Edge, Firefox, or Safari.";
    return;
  }
  $("start").disabled = true;
  $("start").textContent = "Loading models…";
  unlock();   // must be inside the click, or the first alert is silent

  try {
    state.stream = await vision.openCamera(camEl);
    [state.face, state.phone, state.model] = await Promise.all([
      vision.createFaceLandmarker(),
      vision.createPhoneDetector(),
      loadModel(),
    ]);
  } catch (e) {
    err.textContent = e.message;
    $("start").disabled = false;
    $("start").textContent = "Try again";
    return;
  }

  $("intro").classList.add("hidden");
  $("live").classList.remove("hidden");
  $("model").textContent = `Model: ${describeModel(state.model)}.`;
  state.calibration.request(10);
  state.timer = setInterval(tick, TICK_MS);
}

function stop() {
  clearInterval(state.timer);
  state.timer = null;
  vision.closeCamera(state.stream);
  try { state.face?.close(); state.phone?.close(); } catch { /* already gone */ }
  $("live").classList.add("hidden");
  $("intro").classList.remove("hidden");
  $("start").disabled = false;
  $("start").textContent = "Start the demo";
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("recal").addEventListener("click", () => state.calibration.request(10));
addEventListener("pagehide", () => vision.closeCamera(state.stream));
