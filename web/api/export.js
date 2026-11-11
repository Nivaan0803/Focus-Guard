/* GET /api/export - token-protected CSV export.
 *
 * Emits files that are byte-compatible with the desktop writers, so exported
 * data drops straight into the existing Python analysis scripts with no edits:
 *
 *   ?kind=study                     -> focus_study.py:45-46  -> analyze_study.py
 *   ?kind=features&participant=P01  -> session_logger.py:18  -> train_model.py
 *   ?kind=training                  -> session_logger.py:18  -> train_model.py
 *   ?kind=study_sart                -> the SART session measures
 *   ?kind=trials                    -> one row per presented digit
 *   ?kind=probes                    -> one row per thought probe
 *   ?kind=consent                   -> the consent registry, for the SRC
 *   ?kind=participants              -> what has been collected so far
 *
 * Decimal places are reproduced exactly (accuracy 3dp, correct_per_min 2dp,
 * seconds_away 1dp, elapsed 2dp, every feature 4dp), because a diff against a
 * desktop-produced CSV is the cheapest way to prove the two paths agree.
 *
 * TWO DEFAULTS THAT WILL SURPRISE YOU IF YOU DO NOT READ THIS.
 *
 * `kind=study` returns ARITHMETIC sessions only. A SART session has no
 * `attempted` or `correct`, so its row would be mostly empty and
 * analyze_study.py:68 would float() its way into nonsense. SART measures live
 * in `kind=study_sart`. Pass ?task_type=sart or ?task_type=all to override.
 *
 * `kind=training` returns `label_only` sessions only. Labels from an `alerts`
 * session were collected while the system was actively beeping at the
 * participant - the very intervention the study measures - so pooling them
 * into training data confounds it. Pass ?condition=all if you accept that and
 * want the volume. An empty file means you have not run a label_only session
 * yet; it is not a bug.
 */

import { isConfigured, sql, notConfigured, authorized } from "./_db.js";
import { deriveLabels, DEFAULT_WINDOW_SECONDS, TASK_RELATED } from "../lib/labels.js";

// focus_study.py:45-46
const STUDY_COLUMNS = [
  "finished", "participant", "condition", "minutes", "attempted", "correct",
  "accuracy", "correct_per_min", "seconds_away", "times_left", "self_rating",
];

// session_logger.py:18 - ("elapsed", *FEATURE_NAMES, "label")
const FEATURE_COLUMNS = [
  "elapsed", "task_tab_active", "tab_switches_per_min", "time_on_current_tab",
  "keyboard_inactivity", "mouse_inactivity", "calibrated", "face_visible",
  "head_pitch", "head_yaw", "iris_delta", "eye_openness", "phone_present",
  "label",
];

const SART_COLUMNS = [
  "finished", "participant", "condition", "minutes",
  "sart_trials", "sart_targets", "sart_go_trials", "sart_dropped_trials",
  "sart_accuracy", "commission_errors", "commission_rate",
  "omission_errors", "omission_rate", "rt_mean_ms", "rt_sd_ms", "rt_cv",
  "probes_shown", "probes_answered", "mw_rate",
  "seconds_away", "times_left", "self_rating", "wall_seconds", "probe_seconds",
];

const TRIAL_COLUMNS = [
  "participant", "condition", "session_id", "trial", "onset", "digit",
  "is_target", "responded", "rt_ms", "outcome", "extra_presses",
];

const PROBE_COLUMNS = [
  "participant", "condition", "session_id", "probe_index", "elapsed",
  "response", "latency_ms", "trials_before",
];

function cell(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csv(columns, rows) {
  return [columns.join(","), ...rows.map((r) => columns.map((c) => cell(r[c])).join(","))]
    .join("\r\n") + "\r\n";
}

const fixed = (v, places) => (v === null || v === undefined ? "" : Number(v).toFixed(places));
// `minutes` is written by focus_study.py with %g, so 10 stays "10", not "10.0".
const g = (v) => (v === null || v === undefined ? "" : String(Number(v)));
const bool01 = (v) => (v === null || v === undefined ? "" : (v ? 1 : 0));

function send(res, filename, text) {
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
  res.setHeader("Cache-Control", "no-store");
  return res.status(200).send(text);
}

/* Group rows by session_id, preserving the order the query returned. */
function bySession(rows) {
  const map = new Map();
  for (const r of rows) {
    if (!map.has(r.session_id)) map.set(r.session_id, []);
    map.get(r.session_id).push(r);
  }
  return map;
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ ok: false, error: "method_not_allowed" });
  }
  if (!authorized(req)) {
    // 401 whether the token is wrong or unset: this returns human-participants
    // data and should not confirm anything to an unauthenticated caller.
    res.setHeader("WWW-Authenticate", "Bearer");
    return res.status(401).json({ ok: false, error: "unauthorized" });
  }
  if (!isConfigured()) return notConfigured(res);

  const kind = String(req.query.kind || "study");
  const participant = req.query.participant ? String(req.query.participant).toUpperCase() : null;
  const taskType = req.query.task_type ? String(req.query.task_type) : null;
  const condition = req.query.condition ? String(req.query.condition) : null;

  try {
    const q = await sql();

    if (kind === "study") {
      // Arithmetic by default - see the header comment.
      const want = taskType || "arithmetic";
      const { rows } = await q.query(
        `SELECT finished, participant, condition, minutes, attempted, correct,
                accuracy, correct_per_min, seconds_away, times_left, self_rating
         FROM sessions
         WHERE ($1 = 'all' OR task_type = $1)
           AND ($2::text IS NULL OR participant = $2)
         ORDER BY participant, condition, finished`,
        [want, participant]);
      const out = rows.map((r) => ({
        finished: r.finished,
        participant: r.participant,
        condition: r.condition,
        minutes: g(r.minutes),
        attempted: r.attempted,
        correct: r.correct,
        accuracy: fixed(r.accuracy, 3),
        correct_per_min: fixed(r.correct_per_min, 2),
        seconds_away: fixed(r.seconds_away, 1),
        times_left: r.times_left,
        self_rating: r.self_rating ?? "",
      }));
      return send(res, "results.csv", csv(STUDY_COLUMNS, out));
    }

    if (kind === "study_sart") {
      const { rows } = await q.query(
        `SELECT * FROM sessions
         WHERE task_type = 'sart'
           AND ($1::text IS NULL OR participant = $1)
         ORDER BY participant, condition, finished`,
        [participant]);
      const out = rows.map((r) => ({
        finished: r.finished,
        participant: r.participant,
        condition: r.condition,
        minutes: g(r.minutes),
        sart_trials: r.sart_trials,
        sart_targets: r.sart_targets,
        sart_go_trials: r.sart_go_trials,
        sart_dropped_trials: r.sart_dropped_trials,
        sart_accuracy: fixed(r.sart_accuracy, 4),
        commission_errors: r.commission_errors,
        commission_rate: fixed(r.commission_rate, 4),
        omission_errors: r.omission_errors,
        omission_rate: fixed(r.omission_rate, 4),
        rt_mean_ms: fixed(r.rt_mean_ms, 1),
        rt_sd_ms: fixed(r.rt_sd_ms, 1),
        rt_cv: fixed(r.rt_cv, 4),
        probes_shown: r.probes_shown,
        probes_answered: r.probes_answered,
        mw_rate: fixed(r.mw_rate, 4),
        seconds_away: fixed(r.seconds_away, 1),
        times_left: r.times_left,
        self_rating: r.self_rating ?? "",
        wall_seconds: fixed(r.wall_seconds, 1),
        probe_seconds: fixed(r.probe_seconds, 1),
      }));
      return send(res, "results_sart.csv", csv(SART_COLUMNS, out));
    }

    if (kind === "features") {
      const { rows } = await q.query(
        `SELECT f.* FROM feature_rows f
         JOIN sessions s ON s.id = f.session_id
         WHERE ($1::text IS NULL OR s.participant = $1)
         ORDER BY f.session_id, f.elapsed`,
        [participant]);
      const out = rows.map((r) => {
        const row = { elapsed: fixed(r.elapsed, 2), label: r.label || "unlabelled" };
        for (const name of FEATURE_COLUMNS.slice(1, -1)) row[name] = fixed(r[name], 4);
        return row;
      });
      const name = participant ? `${participant.toLowerCase()}_features.csv` : "all_features.csv";
      return send(res, name, csv(FEATURE_COLUMNS, out));
    }

    /* The point of the whole redesign: feature rows carrying REAL
     * focused/distracted labels, in exactly the 14 columns
     * session_logger.py:18 writes, so train_model.py consumes them unmodified.
     *
     * Derivation happens here rather than in SQL or on the client because the
     * window length and the task_related mapping are research parameters, not
     * facts. Re-cutting them is a re-download instead of a re-collection, and
     * lib/labels.js is imported by the parity self-test too, so these rules
     * are checked against the same code that runs in production. */
    if (kind === "training") {
      const wantCondition = condition || "label_only";
      const wantTask = taskType || "all";
      const windowSeconds = Math.min(60, Math.max(1,
        Number(req.query.window) || DEFAULT_WINDOW_SECONDS));
      const taskRelated = [TASK_RELATED.DROP, TASK_RELATED.FOCUSED, TASK_RELATED.DISTRACTED]
        .includes(req.query.task_related) ? req.query.task_related : TASK_RELATED.DROP;

      const filters = [wantCondition, wantTask, participant];
      const where = `WHERE ($1 = 'all' OR s.condition = $1)
                       AND ($2 = 'all' OR s.task_type = $2)
                       AND ($3::text IS NULL OR s.participant = $3)`;

      const { rows: featureRows } = await q.query(
        `SELECT f.* FROM feature_rows f
         JOIN sessions s ON s.id = f.session_id
         ${where}
         ORDER BY f.session_id, f.elapsed`, filters);
      const { rows: probeRows } = await q.query(
        `SELECT p.* FROM probe_responses p
         JOIN sessions s ON s.id = p.session_id
         ${where}
         ORDER BY p.session_id, p.elapsed`, filters);

      const probesFor = bySession(probeRows);
      const out = [];
      for (const [sessionId, rows] of bySession(featureRows)) {
        const { rows: labelled } = deriveLabels(
          rows, probesFor.get(sessionId) || [], { windowSeconds, taskRelated });
        for (const r of labelled) {
          const row = { elapsed: fixed(r.elapsed, 2), label: r.label };
          for (const name of FEATURE_COLUMNS.slice(1, -1)) row[name] = fixed(r[name], 4);
          out.push(row);
        }
      }
      return send(res, "training.csv", csv(FEATURE_COLUMNS, out));
    }

    if (kind === "trials") {
      const { rows } = await q.query(
        `SELECT s.participant, s.condition, t.*
         FROM sart_trials t
         JOIN sessions s ON s.id = t.session_id
         WHERE ($1::text IS NULL OR s.participant = $1)
         ORDER BY t.session_id, t.trial`,
        [participant]);
      const out = rows.map((r) => ({
        participant: r.participant,
        condition: r.condition,
        session_id: r.session_id,
        trial: r.trial,
        onset: fixed(r.onset, 3),
        digit: r.digit,
        is_target: bool01(r.is_target),
        responded: bool01(r.responded),
        rt_ms: fixed(r.rt_ms, 1),
        outcome: r.outcome,
        extra_presses: r.extra_presses,
      }));
      return send(res, "sart_trials.csv", csv(TRIAL_COLUMNS, out));
    }

    if (kind === "probes") {
      const { rows } = await q.query(
        `SELECT s.participant, s.condition, p.*
         FROM probe_responses p
         JOIN sessions s ON s.id = p.session_id
         WHERE ($1::text IS NULL OR s.participant = $1)
         ORDER BY p.session_id, p.probe_index`,
        [participant]);
      const out = rows.map((r) => ({
        participant: r.participant,
        condition: r.condition,
        session_id: r.session_id,
        probe_index: r.probe_index,
        elapsed: fixed(r.elapsed, 2),
        response: r.response ?? "",
        latency_ms: fixed(r.latency_ms, 1),
        trials_before: r.trials_before ?? "",
      }));
      return send(res, "probe_responses.csv", csv(PROBE_COLUMNS, out));
    }

    if (kind === "consent") {
      const { rows } = await q`
        SELECT participant, form_version, signed_at, is_minor, signature_hash,
               statements, pdf_saved, received_at
        FROM consent_receipts ORDER BY received_at`;
      const cols = ["participant", "form_version", "signed_at", "is_minor",
                    "signature_hash", "statements", "pdf_saved", "received_at"];
      return send(res, "consent_receipts.csv", csv(cols, rows));
    }

    /* Deliberately shows the label_only count and the probe total: an empty
     * kind=training file is almost always "no label_only sessions yet", and
     * this is where you find that out. */
    if (kind === "participants") {
      const { rows } = await q`
        SELECT s.participant,
               COUNT(*)::int                                            AS sessions,
               COUNT(*) FILTER (WHERE s.task_type = 'sart')::int         AS sart_sessions,
               COUNT(*) FILTER (WHERE s.condition = 'alerts')::int       AS alerts_sessions,
               COUNT(*) FILTER (WHERE s.condition = 'no_alerts')::int    AS no_alerts_sessions,
               COUNT(*) FILTER (WHERE s.condition = 'label_only')::int   AS label_only_sessions,
               COALESCE(SUM(fr.n), 0)::int                               AS feature_rows,
               COALESCE(SUM(tr.n), 0)::int                               AS trials,
               COALESCE(SUM(pr.n), 0)::int                               AS probes,
               COALESCE(SUM(pr.answered), 0)::int                        AS probes_answered
        FROM sessions s
        LEFT JOIN (
          SELECT session_id, COUNT(*)::int AS n FROM feature_rows GROUP BY session_id
        ) fr ON fr.session_id = s.id
        LEFT JOIN (
          SELECT session_id, COUNT(*)::int AS n FROM sart_trials GROUP BY session_id
        ) tr ON tr.session_id = s.id
        LEFT JOIN (
          SELECT session_id, COUNT(*)::int AS n,
                 COUNT(*) FILTER (WHERE response IS NOT NULL)::int AS answered
          FROM probe_responses GROUP BY session_id
        ) pr ON pr.session_id = s.id
        GROUP BY s.participant ORDER BY s.participant`;
      return res.status(200).json({ ok: true, participants: rows });
    }

    return res.status(400).json({ ok: false, error: "unknown_kind" });
  } catch (err) {
    console.error("export failed", err);
    return res.status(500).json({ ok: false, error: "server_error" });
  }
}
