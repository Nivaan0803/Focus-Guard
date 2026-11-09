/* Buffer results in memory, post once at the end, and never lose a session.
 *
 * A participant on school wifi must not lose ten minutes of work to a dropped
 * request, so every payload is written to localStorage before the POST is
 * attempted and only removed once the server acknowledges it. Anything left
 * over is retried on the next page load, and the participant can always export
 * a CSV by hand as a last resort.
 */

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
  return {
    finished: s.finished,
    participant: s.participant,
    condition: s.condition,
    minutes: String(s.minutes),
    attempted: s.attempted,
    correct: s.correct,
    accuracy: s.attempted ? (s.correct / s.attempted).toFixed(3) : "0",
    correct_per_min: (s.correct / s.minutes).toFixed(2),
    seconds_away: s.secondsAway.toFixed(1),
    times_left: s.timesLeft,
    self_rating: s.selfRating ?? "",
  };
}
