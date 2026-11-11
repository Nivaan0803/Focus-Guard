/* Participant session flow.
 *
 *   consent gate  ->  setup  ->  [calibrate]  ->  arithmetic task  ->  rating  ->  upload
 *
 * Ports focus_study.py. The two behavioural measures it exists to produce,
 * `seconds_away` and `times_left`, come from focus_study.py:124-137, where
 * `is_foreground()` compares GetForegroundWindow() against the task window's
 * OWN handle. The browser equivalent of that exact question is
 * `document.hasFocus() && !document.hidden`, polled on the same 250 ms tick.
 *
 * Protocol fidelity: focus_study.py:22 defines the no_alerts condition as
 * "don't run Focus Guard at all", so the camera is NOT started in that
 * condition. Running it silently would put a camera in both arms of a
 * counterbalanced comparison and change what is being compared. That costs us
 * feature rows from half the sessions; the comparison is worth more.
 */

import { gate, consentConfig } from "../lib/consent.js";
import { PROVENANCE } from "../lib/labels.js";
import * as vision from "../lib/vision.js";
import { cameraFeatures, poseFromMatrix, irisPosition } from "../lib/features.js";
import { Calibration } from "../lib/calibrate.js";
import { loadModel, describeModel, DISTRACTED } from "../lib/tree.js";
import { ScoreTracker } from "../lib/score.js";
import { beep } from "../lib/beep.js";
import {
  submit, flush, downloadCSV, featureRowFor, studyRowFor,
  STUDY_COLUMNS, FEATURE_COLUMNS,
} from "../lib/upload.js";

const app = document.getElementById("app");
const camEl = document.getElementById("cam");
const workCanvas = document.getElementById("work");

const DEFAULT_MINUTES = 10;
const POLL_MS = 250;   // focus_study.py:145
const TICK_MS = 90;    // main.py:496

const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const now = () => performance.now() / 1000;

function view(html) {
  app.innerHTML = `<div class="wrap" style="padding-top:28px;padding-bottom:48px">${html}</div>`;
  return (sel) => app.querySelector(sel);
}

/* ---------------------------------------------------------------- setup -- */

function setup(receipt) {
  return new Promise((resolve) => {
    const params = new URLSearchParams(location.search);
    const presetCondition = params.get("condition");
    const presetMinutes = Number(params.get("minutes")) || DEFAULT_MINUTES;

    const $ = view(`
      <h1>Session setup</h1>
      <p class="muted">Participant <span class="mono">${esc(receipt.participant)}</span></p>
      <div class="card stack">
        <div>
          <label for="cond">Which session is this?</label>
          <select id="cond">
            <option value="">Choose&hellip;</option>
            <option value="alerts">With FocusGuard alerts</option>
            <option value="no_alerts">Without alerts</option>
          </select>
          <p class="small muted" style="margin:8px 0 0">
            Counterbalanced, as in the protocol: odd-numbered participants do
            <strong>alerts first</strong>, even-numbered do <strong>no alerts first</strong>.
          </p>
        </div>
        <div>
          <label for="mins">Minutes</label>
          <input id="mins" type="number" min="1" max="30" step="1" value="${esc(presetMinutes)}">
        </div>
      </div>
      <div class="card">
        <p class="small muted" id="note"></p>
        <button class="primary" id="go" disabled>Continue</button>
      </div>
    `);

    const cond = $("#cond"), mins = $("#mins"), go = $("#go"), note = $("#note");
    if (presetCondition === "alerts" || presetCondition === "no_alerts") {
      cond.value = presetCondition;
    }

    function refresh() {
      go.disabled = !cond.value;
      note.textContent = cond.value === "alerts"
        ? "Your webcam will be used for this session. You will calibrate first, then the task begins."
        : cond.value === "no_alerts"
          ? "The webcam stays off for this session. You will go straight to the task."
          : "";
    }
    cond.addEventListener("change", refresh);
    refresh();

    go.addEventListener("click", () => resolve({
      participant: receipt.participant,
      condition: cond.value,
      minutes: Math.max(1, Math.min(30, Number(mins.value) || DEFAULT_MINUTES)),
    }));
  });
}

/* ---------------------------------------------------------- the detector -- */

/* Owns camera, models, calibration and scoring for an "alerts" session.
 * Produces one feature row per tick in the exact desktop column order. */
class Detector {
  constructor() {
    this.stream = null;
    this.face = null;
    this.phone = null;
    this.calibration = new Calibration();
    this.model = null;
    this.score = new ScoreTracker({ grace: 3, realertSeconds: 0 });
    this.rows = [];
    this.lastFeatures = null;
    this.lastReason = "";
    this.running = false;
    this.t0 = 0;

    // Browser substitutes for the desktop's OS-level signals. Named here so
    // the honesty table on the showcase page and this code cannot drift.
    this.lastKey = now();
    this.lastPointer = now();
    this.tabSwitches = [];
    this.currentTabStarted = now();

    this._onKey = () => { this.lastKey = now(); };
    this._onPointer = () => { this.lastPointer = now(); };
    this._onVisibility = () => {
      this.tabSwitches.push(now());
      this.currentTabStarted = now();
    };
  }

  async start() {
    this.stream = await vision.openCamera(camEl);
    camEl.classList.remove("hidden");
    [this.face, this.phone, this.model] = await Promise.all([
      vision.createFaceLandmarker(),
      vision.createPhoneDetector(),
      loadModel(),
    ]);
    addEventListener("keydown", this._onKey, { passive: true });
    addEventListener("pointermove", this._onPointer, { passive: true });
    addEventListener("visibilitychange", this._onVisibility);
    this.running = true;
  }

  stop() {
    this.running = false;
    removeEventListener("keydown", this._onKey);
    removeEventListener("pointermove", this._onPointer);
    removeEventListener("visibilitychange", this._onVisibility);
    vision.closeCamera(this.stream);
    camEl.classList.add("hidden");
    try { this.face?.close(); this.phone?.close(); } catch { /* already gone */ }
  }

  /* One tick. Returns a snapshot for the caller to display. */
  tick(label) {
    const t = now();
    if (!vision.mirrorToCanvas(camEl, workCanvas)) return null;
    const stamp = Math.round(t * 1000);

    let landmarks = null, matrix = null, phonePresent = 0;
    try {
      const fr = this.face.detectForVideo(workCanvas, stamp);
      if (fr && fr.faceLandmarks && fr.faceLandmarks.length) {
        landmarks = fr.faceLandmarks[0];
        const m = fr.facialTransformationMatrixes;
        if (m && m.length) matrix = m[0].data;
      }
    } catch { /* a dropped frame must not kill the session */ }
    try {
      const pr = this.phone.detectForVideo(workCanvas, stamp);
      phonePresent = pr && pr.detections && pr.detections.length ? 1 : 0;
    } catch { /* ditto */ }

    // Same order as vision_worker.py:280-307: read the raw pose/iris, let the
    // calibration window consume them, then build the features against the
    // (possibly just-finalised) neutral baseline.
    let cam;
    if (landmarks) {
      this.calibration.advance(poseFromMatrix(matrix), irisPosition(landmarks));
      cam = cameraFeatures({ landmarks, matrix, calibration: this.calibration }).features;
    } else {
      this.calibration.advance(null, null);
      cam = { face_visible: 0, calibrated: this.calibration.calibrated ? 1 : 0,
              head_pitch: 0, head_yaw: 0, iris_delta: 0, eye_openness: 0 };
    }

    // Keep only the last 60 s of switches, matching the desktop's per-minute rate.
    this.tabSwitches = this.tabSwitches.filter((s) => t - s <= 60);
    const onTask = document.hasFocus() && !document.hidden;

    const features = {
      task_tab_active: onTask ? 1 : 0,
      tab_switches_per_min: this.tabSwitches.length,
      time_on_current_tab: t - this.currentTabStarted,
      keyboard_inactivity: t - this.lastKey,
      mouse_inactivity: t - this.lastPointer,
      ...cam,
      phone_present: phonePresent,
    };

    const [state, , reason] = this.model.predict(features);
    this.score.update(state, t, this.t0);
    this.lastFeatures = features;
    this.lastReason = state === DISTRACTED ? reason : "";
    this.rows.push(featureRowFor(t - this.t0, features, label));

    const fired = this.score.shouldAlert(t, reason);
    if (fired) beep();

    return {
      state, reason, fired,
      score: this.score.score,
      banner: this.score.bannerState(t, { taskTabActive: onTask }),
      calibrated: this.calibration.calibrated,
    };
  }
}

/* ------------------------------------------------------------ calibrate -- */

function calibrate(detector) {
  return new Promise((resolve) => {
    const $ = view(`
      <h1>Calibrate</h1>
      <div class="card">
        <p>Sit the way you normally would and <strong>look at the screen, not at the
           camera</strong>. This learns your resting head position so the detector
           can tell a real look-away from the way you naturally sit.</p>
        <p class="banner" id="cal">Press start when you are ready.</p>
        <div class="row" style="margin-top:14px">
          <button class="primary" id="begin">Start calibration</button>
          <button id="skip">Skip</button>
        </div>
      </div>
    `);
    const cal = $("#cal"), begin = $("#begin"), skip = $("#skip");

    let timer = null;
    const finish = () => { clearInterval(timer); resolve(); };

    begin.addEventListener("click", () => {
      begin.disabled = true;
      skip.disabled = true;
      detector.calibration.request(10);
      cal.className = "banner warn";
      timer = setInterval(() => {
        detector.tick(PROVENANCE.CALIBRATION);
        const c = detector.calibration;
        if (c.calibrating) {
          cal.textContent = `Calibrating — keep working normally: ${c.remaining.toFixed(1)}s`;
          return;
        }
        if (c.lastEvent === "failed" || !c.calibrated) {
          cal.className = "banner bad";
          cal.textContent = "Could not see your face for long enough. Check your lighting and try again.";
          begin.disabled = false;
          skip.disabled = false;
          begin.textContent = "Try again";
          clearInterval(timer);
          return;
        }
        cal.className = "banner good";
        cal.textContent = "Calibrated. Starting the task…";
        setTimeout(finish, 900);
        clearInterval(timer);
      }, TICK_MS);
    });

    skip.addEventListener("click", finish);
  });
}

/* ----------------------------------------------------------------- task -- */

/* Which task module to run. A URL parameter rather than a constant so a
 * mid-study regression to arithmetic is a link, not a git revert - and so the
 * task_type column can ever hold more than one value. */
const TASKS = {
  sart: () => import("./sart-task.js"),
  arithmetic: () => import("./arithmetic-task.js"),
};
const DEFAULT_TASK = "sart";

function taskName() {
  const want = new URLSearchParams(location.search).get("task");
  return Object.prototype.hasOwnProperty.call(TASKS, want) ? want : DEFAULT_TASK;
}

/* The task harness.
 *
 * Owns the clock, both timers, the away-poll and the detector, and knows
 * nothing about what the participant is actually doing. A task module owns
 * its own stimulus area and its own notion of a correct response (see
 * arithmetic-task.js for the contract).
 *
 * Previously all of this was fused into one function alongside the arithmetic
 * input handling, which is why swapping the task meant rewriting the timers.
 */
function task(session, detector) {
  return new Promise((resolve) => {
    const withCam = Boolean(detector);
    const $ = view(`
      <div id="task">
        <p class="banner ${withCam ? "" : "hidden"}" id="banner">Focused</p>
        <div id="taskroot"></div>
        <div class="meter ${withCam ? "" : "hidden"}" style="max-width:320px;margin:18px auto 0">
          <span id="meter"></span>
        </div>
        <p id="clock"></p>
      </div>
    `);

    const clockEl = $("#clock");
    const bannerEl = $("#banner"), meterEl = $("#meter");
    const taskRoot = $("#taskroot");

    let started = null, ends = null, lastPoll = null;
    let timesLeft = 0, secondsAway = 0;
    let wasForeground = true;
    let pollTimer = null, tickTimer = null;
    let active = null;

    const finish = () => {
      clearInterval(pollTimer);
      clearInterval(tickTimer);
      active.stop();
      clockEl.textContent = "";
      resolve({
        // Task-shaped figures (attempted/correct for arithmetic, the SART
        // aggregates for SART) come from the task; everything else is the
        // harness's own.
        ...active.summary(),
        trials: active.trials,
        timesLeft, secondsAway,
        rows: detector ? detector.rows : [],
        alerts: detector ? detector.score.alerts : 0,
        modelName: detector ? describeModel(detector.model) : "",
      });
    };

    // focus_study.py:130-137
    const poll = () => {
      const t = now();
      const foreground = document.hasFocus() && !document.hidden;
      if (!foreground) {
        secondsAway += t - lastPoll;
        if (wasForeground) timesLeft += 1;
      }
      wasForeground = foreground;
      lastPoll = t;

      const left = ends - t;
      if (left <= 0) return finish();
      clockEl.textContent = `${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, "0")} left`;
    };

    /* The task calls this when the participant has signalled they are ready.
     * The clock starts here and nowhere else, so every task agrees on what
     * t0 means. */
    const begin = () => {
      if (started !== null) return;
      started = now();
      ends = started + session.minutes * 60;
      lastPoll = started;
      if (detector) detector.t0 = started;

      active.start(started);
      pollTimer = setInterval(poll, POLL_MS);

      if (detector) {
        tickTimer = setInterval(() => {
          // PROVENANCE.TASK, not session.condition. The condition is already
          // on the session row; writing it here left `label` holding no
          // information and would have had train_model.py learn to predict
          // which arm a session was. See lib/labels.js.
          const snap = detector.tick(PROVENANCE.TASK);
          if (!snap) return;
          bannerEl.className = `banner ${snap.banner}`;
          bannerEl.textContent = snap.reason && snap.banner !== "good"
            ? snap.reason
            : "Focused";
          meterEl.style.width = `${(snap.score * 100).toFixed(0)}%`;
          meterEl.style.background = detector.score.meterColor;
        }, TICK_MS);
      }
    };

    TASKS[taskName()]().then(({ createTask }) => {
      active = createTask({
        root: taskRoot,
        participant: session.participant,
        condition: session.condition,
        minutes: session.minutes,
        onBegin: begin,
      });
    });
  });
}

/* --------------------------------------------------------------- rating -- */

function rating() {
  return new Promise((resolve) => {
    const buttons = Array.from({ length: 10 }, (_, i) =>
      `<button type="button" data-r="${i + 1}" style="min-width:48px">${i + 1}</button>`).join(" ");
    view(`
      <h1>One quick question</h1>
      <div class="card">
        <p>How focused did you feel during this session?<br>
           <span class="muted small">1 = not at all, 10 = completely focused</span></p>
        <div class="row" style="margin-top:12px">${buttons}</div>
      </div>
    `);
    app.querySelectorAll("[data-r]").forEach((b) => {
      b.addEventListener("click", () => resolve(Number(b.dataset.r)));
    });
  });
}

/* ----------------------------------------------------------------- done -- */

async function done(session, result, selfRating) {
  const finished = new Date().toISOString().replace("T", " ").slice(0, 16);
  const studyRow = studyRowFor({
    finished,
    participant: session.participant,
    condition: session.condition,
    minutes: session.minutes,
    attempted: result.attempted,
    correct: result.correct,
    secondsAway: result.secondsAway,
    timesLeft: result.timesLeft,
    selfRating,
  });

  const payload = {
    id: `${session.participant}-${session.condition}-${Date.now()}`,
    kind: "session",
    participant: session.participant,
    condition: session.condition,
    finished,
    study: studyRow,
    features: result.rows,
    meta: {
      alerts: result.alerts,
      model: result.modelName,
      userAgent: navigator.userAgent,
      source: "web",
    },
  };

  // The consent PDF never leaves the participant's device - only a
  // de-identified receipt is uploaded - but the study records require the
  // original signed form, so ask for it here. Non-fatal if the config will not
  // load: the results still upload, the participant just gets generic wording.
  let cfg = null;
  try {
    cfg = await consentConfig();
  } catch {
    /* the gate already loaded this once; a failure here is not worth blocking on */
  }

  const pdfName = cfg && cfg.version
    ? `consent_${session.participant}_${cfg.version}.pdf`
    : null;
  const contact = (cfg && cfg.sponsorContact) || null;
  // encodeURI, not encodeURIComponent: the latter escapes "@" to "%40", which
  // RFC 6068 does not want in a mailto addr-spec and some clients mishandle.
  const mailto = contact
    ? `mailto:${encodeURI(contact)}`
      + `?subject=${encodeURIComponent(`FocusGuard signed consent - ${session.participant}`)}`
    : null;

  const consentReturn = `
    <div class="card">
      <h3>Please send your signed consent form</h3>
      <p class="small">When you signed, a copy
         ${pdfName ? `named <span class="mono">${esc(pdfName)}</span>` : ""}
         was downloaded to this device. It was <strong>not</strong> uploaded with
         your results, and the study records need that original signed form.</p>
      <p class="small">Please email it to
         ${mailto
           ? `<a href="${esc(mailto)}">${esc(contact)}</a>`
           : "the researcher"}.</p>
      <p class="small muted">Look in your Downloads folder if you cannot find it.</p>
    </div>`;

  const $ = view(`
    <h1>Thank you</h1>
    <div class="card">
      <p><strong>${esc(result.correct)}</strong> correct out of
         <strong>${esc(result.attempted)}</strong>.</p>
      <p class="muted small">You left the task window ${esc(result.timesLeft)} time(s),
         for ${esc(result.secondsAway.toFixed(1))} seconds in total.</p>
      <p id="status" class="muted">Sending&hellip;</p>
    </div>
    ${consentReturn}
    <div class="card">
      <h3>Keep a copy</h3>
      <p class="small muted">If anything went wrong with the upload, download these
         and send them to the researcher.</p>
      <div class="row">
        <button id="dlstudy">Download session CSV</button>
        <button id="dlfeat" ${result.rows.length ? "" : "disabled"}>Download detector CSV</button>
      </div>
    </div>
    <div class="card">
      <button class="primary" id="again">Start the other session</button>
    </div>
  `);

  const status = $("#status");
  const { uploaded } = await submit(payload);
  status.textContent = uploaded
    ? "Sent to the researcher. You can close this tab."
    : "Could not reach the server, so this session is saved on this device and will "
      + "be sent automatically next time it opens. Please also download the files below.";
  status.className = uploaded ? "muted" : "banner warn";

  $("#dlstudy").addEventListener("click", () => downloadCSV(
    `results_${session.participant}_${session.condition}.csv`, STUDY_COLUMNS, [studyRow]));
  $("#dlfeat").addEventListener("click", () => downloadCSV(
    `${session.participant.toLowerCase()}_${session.condition}.csv`, FEATURE_COLUMNS, result.rows));
  $("#again").addEventListener("click", () => { location.search = ""; });
}

/* ----------------------------------------------------------------- main -- */

async function main() {
  if (!vision.isSupported()) {
    view(`<div class="card"><h2>This browser will not work</h2>
      <p>The study needs a browser that can use a camera and run WebAssembly.
         Please use an up-to-date Chrome, Edge, Firefox, or Safari on a laptop
         or desktop.</p></div>`);
    return;
  }

  flush();   // retry anything stranded by an earlier session

  const receipt = await gate(app);
  const session = await setup(receipt);

  let detector = null;
  if (session.condition === "alerts") {
    detector = new Detector();
    view(`<div class="card"><p class="muted">Starting the camera and loading the
      models&hellip; this can take a few seconds the first time.</p></div>`);
    try {
      await detector.start();
    } catch (err) {
      view(`<div class="card"><h2>Camera problem</h2>
        <p>${esc(err.message)}</p>
        <p class="small muted">Tell the researcher. You can reload to try again.</p>
        </div>`);
      return;
    }
    await calibrate(detector);
  }

  const result = await task(session, detector);
  if (detector) detector.stop();
  const selfRating = await rating();
  await done(session, result, selfRating);
}

main();
