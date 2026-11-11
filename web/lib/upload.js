/* Buffer results in memory, post once at the end, and never lose a session.
 *
 * A participant on school wifi must not lose ten minutes of work to a dropped
 * request, so every payload is written to localStorage before the POST is
 * attempted and only removed once the server acknowledges it. Anything left
 * over is retried on the next page load, and the participant can always export
 * a CSV by hand as a last resort.
 */

// One definition of "off-task", shared with the training export.
import { mapProbe, DISTRACTED } from "./labels.js";

const QUEUE_KEY = "focusguard.pending";

function readQueue() {
  try {
    return JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
  } catch {
    return [];
  }
}

function writeQueue(queue) {
  try {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
  } catch {
    /* private mode, or quota - the in-memory copy is still live */
  }
}

export function pendingCount() {
  return readQueue().length;
}

async function post(payload) {
  const res = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  return res.json().catch(() => ({}));
}

/* Returns { uploaded: boolean }. A false means it is safely queued, not lost. */
export async function submit(payload) {
  const queue = readQueue();
  queue.push(payload);
  writeQueue(queue);

  try {
    await post(payload);
  } catch {
    return { uploaded: false };
  }

  // Drop just this payload; a concurrent tab may have added others.
  const after = readQueue().filter((p) => p.id !== payload.id);
  writeQueue(after);
  return { uploaded: true };
}

export async function flush() {
  const queue = readQueue();
  if (!queue.length) return { sent: 0, left: 0 };
  const left = [];
  let sent = 0;
  for (const payload of queue) {
    try {
      await post(payload);
      sent += 1;
    } catch {
      left.push(payload);
    }
  }
  writeQueue(left);
  return { sent, left: left.length };
}

/* ---- CSV fallback -------------------------------------------------------
 * Column orders below are byte-compatible with the desktop writers so an
 * exported file drops straight into the existing Python analysis scripts.
 */

// focus_study.py:45-46
export const STUDY_COLUMNS = [
  "finished", "participant", "condition", "minutes", "attempted", "correct",
  "accuracy", "correct_per_min", "seconds_away", "times_left", "self_rating",
];

// session_logger.py:18 - ("elapsed", *FEATURE_NAMES, "label")
export const FEATURE_COLUMNS = [
  "elapsed", "task_tab_active", "tab_switches_per_min", "time_on_current_tab",
  "keyboard_inactivity", "mouse_inactivity", "calibrated", "face_visible",
  "head_pitch", "head_yaw", "iris_delta", "eye_openness", "phone_present",
  "label",
];

function csvCell(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCSV(columns, rows) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((c) => csvCell(row[c])).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}

export function downloadCSV(filename, columns, rows) {
  const blob = new Blob([toCSV(columns, rows)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/* Formatting helpers that reproduce the desktop writers exactly. */

// session_logger.py:60-62 - elapsed 2dp, every feature 4dp
export function featureRowFor(elapsed, features, label) {
  const row = { elapsed: elapsed.toFixed(2), label };
  for (const name of FEATURE_COLUMNS.slice(1, -1)) {
    row[name] = Number(features[name] ?? 0).toFixed(4);
  }
  return row;
}

// focus_study.py:169-175 - accuracy 3dp, correct_per_min 2dp, seconds_away 1dp
export function studyRowFor(s) {
  /* A SART session has no notion of "attempted" or "correct". Those four
   * columns stay in the row so the shape still matches focus_study.py:45-46,
   * but they are left EMPTY rather than filled with a zero, a NaN, or a SART
   * analogue.
   *
   * Reusing them would be the worst option: analyze_study.py:68 does a bare
   * float() on whatever is in the column, so a commission rate parked in
   * `accuracy` would be pooled with real arithmetic accuracies and nothing
   * would complain. Empty means NULL on the way into Postgres, and task_type
   * is what separates the two populations.
   *
   * Arithmetic output is unchanged: attempted/correct of 0 are still finite,
   * so a zero-score session formats exactly as it always did.
   */
  const scored = Number.isFinite(s.attempted) && Number.isFinite(s.correct);
  return {
    finished: s.finished,
    participant: s.participant,
    condition: s.condition,
    minutes: String(s.minutes),
    attempted: scored ? s.attempted : "",
    correct: scored ? s.correct : "",
    accuracy: scored ? (s.attempted ? (s.correct / s.attempted).toFixed(3) : "0") : "",
    correct_per_min: scored ? (s.correct / s.minutes).toFixed(2) : "",
    seconds_away: s.secondsAway.toFixed(1),
    times_left: s.timesLeft,
    self_rating: s.selfRating ?? "",
  };
}

/* ---- SART ---------------------------------------------------------------
 * None of these share a column name with the arithmetic measures, on purpose.
 */

export const SART_COLUMNS = [
  "finished", "participant", "condition", "minutes",
  "sart_trials", "sart_targets", "sart_go_trials", "sart_dropped_trials",
  "sart_accuracy", "commission_errors", "commission_rate",
  "omission_errors", "omission_rate", "rt_mean_ms", "rt_sd_ms", "rt_cv",
  "probes_shown", "probes_answered", "mw_rate",
  "seconds_away", "times_left", "self_rating", "wall_seconds", "probe_seconds",
];

export const TRIAL_COLUMNS = [
  "trial", "onset", "digit", "is_target", "responded", "rt_ms", "outcome",
  "extra_presses",
];

export const PROBE_COLUMNS = [
  "probe_index", "elapsed", "response", "latency_ms", "trials_before",
];

const fixed = (v, places) => (Number.isFinite(v) ? v.toFixed(places) : "");

/* Probe-reported off-task rate.
 *
 * Numerator is every answered probe that lib/labels.js maps to `distracted`,
 * so there is one definition of "off-task" in the codebase rather than two
 * that can drift. Denominator is every ANSWERED probe, which means a
 * `task_related` answer counts as not-off-task rather than being discarded -
 * the participant was still engaged with the task, just not performing it.
 */
export function probeSummary(probes) {
  const list = probes || [];
  const answered = list.filter((p) => p && p.response);
  const offTask = answered.filter((p) => mapProbe(p.response) === DISTRACTED);
  return {
    probes_shown: list.length,
    probes_answered: answered.length,
    mw_rate: answered.length ? offTask.length / answered.length : null,
  };
}

export function sartRowFor(s) {
  const a = s.aggregates || {};
  const p = probeSummary(s.probes);
  return {
    finished: s.finished,
    participant: s.participant,
    condition: s.condition,
    minutes: String(s.minutes),
    sart_trials: a.sart_trials ?? "",
    sart_targets: a.sart_targets ?? "",
    sart_go_trials: a.sart_go_trials ?? "",
    sart_dropped_trials: a.sart_dropped_trials ?? "",
    sart_accuracy: fixed(a.sart_accuracy, 4),
    commission_errors: a.commission_errors ?? "",
    commission_rate: fixed(a.commission_rate, 4),
    omission_errors: a.omission_errors ?? "",
    omission_rate: fixed(a.omission_rate, 4),
    rt_mean_ms: fixed(a.rt_mean_ms, 1),
    rt_sd_ms: fixed(a.rt_sd_ms, 1),
    rt_cv: fixed(a.rt_cv, 4),
    probes_shown: p.probes_shown,
    probes_answered: p.probes_answered,
    mw_rate: fixed(p.mw_rate, 4),
    seconds_away: s.secondsAway.toFixed(1),
    times_left: s.timesLeft,
    self_rating: s.selfRating ?? "",
    wall_seconds: fixed(s.wallSeconds, 1),
    probe_seconds: fixed(s.probeSeconds, 1),
  };
}

export function trialRowsFor(trials) {
  return (trials || []).map((t) => ({
    trial: t.trial,
    onset: Number(t.onset).toFixed(3),
    digit: t.digit,
    is_target: t.is_target ? 1 : 0,
    responded: t.responded ? 1 : 0,
    rt_ms: fixed(t.rt_ms, 1),
    outcome: t.outcome,
    extra_presses: t.extra_presses ?? 0,
  }));
}

export function probeRowsFor(probes) {
  return (probes || []).map((p) => ({
    probe_index: p.probe_index,
    elapsed: Number(p.elapsed).toFixed(2),
    response: p.response ?? "",
    latency_ms: fixed(p.latency_ms, 1),
    trials_before: p.trials_before ?? "",
  }));
}
