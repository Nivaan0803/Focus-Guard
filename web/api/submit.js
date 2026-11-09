/* POST /api/submit - one completed study session.
 *
 * Accepts the study row plus the detector's feature rows. Idempotent on the
 * client-generated `id`, so a retry after a flaky upload cannot create a
 * duplicate session.
 */

import { isConfigured, sql, notConfigured, readJsonBody, validParticipant, num } from "./_db.js";

const MAX_FEATURE_ROWS = 20000;   // ~30 min at 90 ms; guards against a runaway client
const CHUNK = 500;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method_not_allowed" });
  }
  if (!isConfigured()) return notConfigured(res);

  const body = readJsonBody(req);
  if (!body || !body.id || !body.study) {
    return res.status(400).json({ ok: false, error: "bad_payload" });
  }
  if (!validParticipant(body.participant)) {
    return res.status(400).json({ ok: false, error: "bad_participant_code" });
  }
  if (body.condition !== "alerts" && body.condition !== "no_alerts") {
    return res.status(400).json({ ok: false, error: "bad_condition" });
  }

  const s = body.study;
  const meta = body.meta || {};
  const rows = Array.isArray(body.features) ? body.features.slice(0, MAX_FEATURE_ROWS) : [];

  try {
    const q = await sql();

    const inserted = await q`
      INSERT INTO sessions (
        id, participant, condition, finished, minutes, attempted, correct,
        accuracy, correct_per_min, seconds_away, times_left, self_rating,
        alerts, model, user_agent
      ) VALUES (
        ${body.id}, ${body.participant}, ${body.condition}, ${body.finished || s.finished},
        ${num(s.minutes)}, ${num(s.attempted)}, ${num(s.correct)},
        ${num(s.accuracy)}, ${num(s.correct_per_min)}, ${num(s.seconds_away)},
        ${num(s.times_left)}, ${num(s.self_rating)},
        ${num(meta.alerts)}, ${meta.model || null}, ${(meta.userAgent || "").slice(0, 400)}
      )
      ON CONFLICT (id) DO NOTHING
      RETURNING id`;

    // A retry of an already-stored session: acknowledge without re-inserting
    // its feature rows.
    if (inserted.rowCount === 0) {
      return res.status(200).json({ ok: true, duplicate: true, id: body.id });
    }

    for (let i = 0; i < rows.length; i += CHUNK) {
      const chunk = rows.slice(i, i + CHUNK);
      await Promise.all(chunk.map((r) => q`
        INSERT INTO feature_rows (
          session_id, elapsed, task_tab_active, tab_switches_per_min,
          time_on_current_tab, keyboard_inactivity, mouse_inactivity,
          calibrated, face_visible, head_pitch, head_yaw, iris_delta,
          eye_openness, phone_present, label
        ) VALUES (
          ${body.id}, ${num(r.elapsed)}, ${num(r.task_tab_active)},
          ${num(r.tab_switches_per_min)}, ${num(r.time_on_current_tab)},
          ${num(r.keyboard_inactivity)}, ${num(r.mouse_inactivity)},
          ${num(r.calibrated)}, ${num(r.face_visible)}, ${num(r.head_pitch)},
          ${num(r.head_yaw)}, ${num(r.iris_delta)}, ${num(r.eye_openness)},
          ${num(r.phone_present)}, ${r.label || "unlabelled"}
        )`));
    }

    return res.status(200).json({ ok: true, id: body.id, featureRows: rows.length });
  } catch (err) {
    console.error("submit failed", err);
    return res.status(500).json({ ok: false, error: "server_error" });
  }
}
