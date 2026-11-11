/* Turns thought-probe answers into focused/distracted labels for feature rows.
 *
 * THE PROBLEM THIS SOLVES. Until probes existed, the web study wrote the
 * experimental condition into the `label` column (flow.js passed
 * session.condition into Detector.tick), so every task row read "alerts".
 * Handed to train_model.py that trains a classifier for *which condition a
 * session was*, not for whether anyone was distracted. The desktop app had
 * real ground truth - an operator flipping a focused/distracted toggle with
 * F8 (main.py:380-381) - but a remote participant has no operator. A probe is
 * the participant acting as their own annotator.
 *
 * WHY DERIVATION LIVES HERE, NOT IN SQL OR THE CLIENT. The window length and
 * the task_related mapping are research parameters, not facts: you will want
 * to re-cut them after seeing the data. Deriving at export time from raw,
 * untouched probe and feature rows makes retuning a re-download instead of a
 * re-collection. Both api/export.js and selftest.html import THIS file, so the
 * parity checks exercise the production path rather than a copy. Do not add a
 * SQL view or a client-side duplicate of these rules - three implementations
 * of the same rule drift silently, and the drift would be invisible in the
 * data it produces.
 *
 * Pure functions only: no DOM, no fetch, no Date. Safe in a serverless
 * function and in the browser.
 */

export const FOCUSED = "focused";
export const DISTRACTED = "distracted";

/* The four probe answers. `task_related` is deliberately absent from the
 * mapping: see TASK_RELATED below. */
export const PROBE_TO_LABEL = {
  on_task: FOCUSED,
  mind_wandering: DISTRACTED,
  external: DISTRACTED,
};

export const PROBE_RESPONSES = ["on_task", "task_related", "mind_wandering", "external"];

/* "Thinking about the task, but not doing it" is the ambiguous middle of the
 * attention literature - engaged with the material, not performing. Forcing it
 * either way injects label noise into a two-class tree, so the default is to
 * drop those windows. Exposed as a parameter so a sensitivity analysis costs
 * one more export rather than another round of data collection. */
export const TASK_RELATED = { DROP: "drop", FOCUSED, DISTRACTED };

export const DEFAULT_WINDOW_SECONDS = 15;

/* Provenance values written into feature_rows.label by the study flow. Only
 * TASK rows describe a participant doing the task, so only they can carry a
 * verdict. */
export const PROVENANCE = { TASK: "task", PROBE: "probe", CALIBRATION: "calibration" };

export function mapProbe(response, taskRelated = TASK_RELATED.DROP) {
  if (response === "task_related") {
    return taskRelated === TASK_RELATED.DROP ? null : taskRelated;
  }
  return PROBE_TO_LABEL[response] ?? null;
}

const num = (v) => (typeof v === "number" ? v : Number(v));

/* The probe a row belongs to is the first ANSWERED probe at or after it: a
 * probe asks about the window that just elapsed, so rows look forward.
 *
 * Note it does NOT fall through to a later probe when the nearest one maps to
 * null. A dropped `task_related` window means "we do not know what this window
 * was", and attributing a later probe's verdict to it would be inventing data.
 */
export function probeForRow(rowElapsed, sortedProbes, windowSeconds) {
  for (const probe of sortedProbes) {
    const at = num(probe.elapsed);
    if (at < rowElapsed) continue;
    return at - rowElapsed <= windowSeconds ? probe : null;
  }
  return null;
}

/* rows:   feature rows, each { elapsed, label, ...the 12 features }
 * probes: [{ probe_index, elapsed, response }], unanswered -> response null
 *
 * Returns { rows, counts }. `rows` are shallow copies with `label` replaced by
 * the derived verdict, already filtered to only the labelled ones - exactly
 * what /api/export?kind=training emits, in the same 14 columns
 * session_logger.py:18 writes. `counts` is for diagnostics; never put it in
 * the CSV, or the byte-compatibility with the desktop writer dies.
 */
export function deriveLabels(rows, probes, options = {}) {
  const windowSeconds = options.windowSeconds ?? DEFAULT_WINDOW_SECONDS;
  const taskRelated = options.taskRelated ?? TASK_RELATED.DROP;

  const answered = (probes || [])
    .filter((p) => p && p.response)
    .map((p) => ({ ...p, elapsed: num(p.elapsed) }))
    .sort((a, b) => a.elapsed - b.elapsed);

  const counts = {
    total: rows.length,
    labelled: 0,
    focused: 0,
    distracted: 0,
    skipped_not_task: 0,
    skipped_no_probe: 0,
    skipped_ambiguous: 0,
  };

  const out = [];
  for (const row of rows) {
    if (row.label !== PROVENANCE.TASK) {
      counts.skipped_not_task += 1;
      continue;
    }

    const probe = probeForRow(num(row.elapsed), answered, windowSeconds);
    if (!probe) {
      counts.skipped_no_probe += 1;
      continue;
    }

    const verdict = mapProbe(probe.response, taskRelated);
    if (!verdict) {
      counts.skipped_ambiguous += 1;
      continue;
    }

    counts.labelled += 1;
    counts[verdict] += 1;
    out.push({ ...row, label: verdict });
  }

  return { rows: out, counts };
}
