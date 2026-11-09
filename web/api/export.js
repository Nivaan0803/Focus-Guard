/* GET /api/export - token-protected CSV export.
 *
 * Emits files that are byte-compatible with the desktop writers, so exported
 * data drops straight into the existing Python analysis scripts with no edits:
 *
 *   ?kind=study                     -> focus_study.py:45-46  -> analyze_study.py
 *   ?kind=features&participant=P01  -> session_logger.py:18  -> train_model.py
 *   ?kind=consent                   -> the consent registry, for the SRC
 *   ?kind=participants              -> what has been collected so far
 *
 * Decimal places are reproduced exactly (accuracy 3dp, correct_per_min 2dp,
 * seconds_away 1dp, elapsed 2dp, every feature 4dp), because a diff against a
 * desktop-produced CSV is the cheapest way to prove the two paths agree.
 */

import { isConfigured, sql, notConfigured, authorized } from "./_db.js";

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

function send(res, filename, text) {
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
  res.setHeader("Cache-Control", "no-store");
  return res.status(200).send(text);
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

  try {
    const q = await sql();

    if (kind === "study") {
      const { rows } = await q`
        SELECT finished, participant, condition, minutes, attempted, correct,
               accuracy, correct_per_min, seconds_away, times_left, self_rating
        FROM sessions ORDER BY participant, condition, finished`;
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

    if (kind === "features") {
      const { rows } = participant
        ? await q`
            SELECT f.* FROM feature_rows f
            JOIN sessions s ON s.id = f.session_id
            WHERE s.participant = ${participant}
            ORDER BY f.session_id, f.elapsed`
        : await q`
            SELECT f.* FROM feature_rows f
            ORDER BY f.session_id, f.elapsed`;
      const out = rows.map((r) => {
        const row = { elapsed: fixed(r.elapsed, 2), label: r.label || "unlabelled" };
        for (const name of FEATURE_COLUMNS.slice(1, -1)) row[name] = fixed(r[name], 4);
        return row;
      });
      const name = participant ? `${participant.toLowerCase()}_features.csv` : "all_features.csv";
      return send(res, name, csv(FEATURE_COLUMNS, out));
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

    if (kind === "participants") {
      const { rows } = await q`
        SELECT s.participant,
               COUNT(*)::int                                        AS sessions,
               COUNT(*) FILTER (WHERE s.condition = 'alerts')::int   AS alerts_sessions,
               COUNT(*) FILTER (WHERE s.condition = 'no_alerts')::int AS no_alerts_sessions,
               COALESCE(SUM(fr.n), 0)::int                           AS feature_rows
        FROM sessions s
        LEFT JOIN (
          SELECT session_id, COUNT(*)::int AS n FROM feature_rows GROUP BY session_id
        ) fr ON fr.session_id = s.id
        GROUP BY s.participant ORDER BY s.participant`;
      return res.status(200).json({ ok: true, participants: rows });
    }

    return res.status(400).json({ ok: false, error: "unknown_kind" });
  } catch (err) {
    console.error("export failed", err);
    return res.status(500).json({ ok: false, error: "server_error" });
  }
}
