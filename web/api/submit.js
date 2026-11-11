/* POST /api/submit - one completed study session.
 *
 * Accepts the study row, the detector's feature rows, and - for a SART
 * session - its per-trial records and probe answers. Idempotent on the
 * client-generated `id`, so a retry after a flaky upload cannot create a
 * duplicate session.
 */

import { isConfigured, sql, notConfigured, readJsonBody, validParticipant, num } from "./_db.js";

const MAX_FEATURE_ROWS = 20000;   // ~30 min at 90 ms; guards against a runaway client
const MAX_TRIALS = 2000;          // ~38 min at 1150 ms, same reasoning
const MAX_PROBES = 50;            // ~9-10 per 10 min session
const CHUNK = 500;

/* `label_only` is the third session type: camera on, alerts suppressed,
 * probes on, sitting outside the counterbalanced A/B. It exists so training
 * labels are not collected exclusively under the intervention being measured.
 * See docs/sart-redesign.md s2.1. */
const CONDITIONS = new Set(["alerts", "no_alerts", "label_only"]);
const TASK_TYPES = new Set(["arithmetic", "sart"]);

const OUTCOMES = new Set([
  "hit", "correct_omission", "commission_error", "omission_error", "dropped",
]);
const PROBE_RESPONSES = new Set([
  "on_task", "task_related", "mind_wandering", "external",
]);

const bool = (v) => v === true || v === 1 || v === "1";

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
  if (!CONDITIONS.has(body.condition)) {
    return res.status(400).json({ ok: false, error: "bad_condition" });
  }

  // A client that predates the SART work sends no task_type at all, and those
  // sessions were arithmetic by definition - the same assumption the column
  // default encodes.
  const taskType = body.task_type || "arithmetic";
  if (!TASK_TYPES.has(taskType)) {
    return res.status(400).json({ ok: false, error: "bad_task_type" });
  }

  const s = body.study;
  const a = body.sart || {};
  const meta = body.meta || {};
  const rows = Array.isArray(body.features) ? body.features.slice(0, MAX_FEATURE_ROWS) : [];
  const trials = Array.isArray(body.trials) ? body.trials.slice(0, MAX_TRIALS) : [];
  const probes = Array.isArray(body.probes) ? body.probes.slice(0, MAX_PROBES) : [];

  /* Validate before touching the database, so a client bug is a 400 naming
   * the problem rather than a half-written session. These are loud on purpose:
   * coercing an unrecognised outcome to `dropped` would silently discard a
   * trial, and coercing an unrecognised probe answer to NULL would silently
   * discard the only ground truth this study has. The client keeps the payload
   * queued either way, and the CSV fallback is still available. */
  for (const t of trials) {
    if (!OUTCOMES.has(t.outcome)) {
      return res.status(400).json({ ok: false, error: "bad_trial_outcome", value: String(t.outcome).slice(0, 40) });
    }
  }
  for (const p of probes) {
    if (p.response !== null && p.response !== "" && !PROBE_RESPONSES.has(p.response)) {
      return res.status(400).json({ ok: false, error: "bad_probe_response", value: String(p.response).slice(0, 40) });
    }
  }

  try {
    const q = await sql();

    const inserted = await q`
      INSERT INTO sessions (
        id, participant, condition, finished, minutes, attempted, correct,
        accuracy, correct_per_min, seconds_away, times_left, self_rating,
        alerts, model, user_agent,
        task_type, task_version, wall_seconds, probe_seconds,
        sart_trials, sart_targets, sart_go_trials, sart_dropped_trials,
        sart_accuracy, commission_errors, commission_rate,
        omission_errors, omission_rate, rt_mean_ms, rt_sd_ms, rt_cv,
        probes_shown, probes_answered, mw_rate
      ) VALUES (
        ${body.id}, ${body.participant}, ${body.condition}, ${body.finished || s.finished},
        ${num(s.minutes)}, ${num(s.attempted)}, ${num(s.correct)},
        ${num(s.accuracy)}, ${num(s.correct_per_min)}, ${num(s.seconds_away)},
        ${num(s.times_left)}, ${num(s.self_rating)},
        ${num(meta.alerts)}, ${meta.model || null}, ${(meta.userAgent || "").slice(0, 400)},
        ${taskType}, ${body.task_version || null},
        ${num(body.wall_seconds)}, ${num(body.probe_seconds)},
        ${num(a.sart_trials)}, ${num(a.sart_targets)}, ${num(a.sart_go_trials)},
        ${num(a.sart_dropped_trials)}, ${num(a.sart_accuracy)},
        ${num(a.commission_errors)}, ${num(a.commission_rate)},
        ${num(a.omission_errors)}, ${num(a.omission_rate)},
        ${num(a.rt_mean_ms)}, ${num(a.rt_sd_ms)}, ${num(a.rt_cv)},
        ${num(a.probes_shown)}, ${num(a.probes_answered)}, ${num(a.mw_rate)}
      )
      ON CONFLICT (id) DO NOTHING
      RETURNING id`;

    // A retry of an already-stored session: acknowledge without re-inserting
    // its child rows. The UNIQUE constraints on sart_trials and
    // probe_responses make that safe even if this guard is ever disturbed.
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

    for (let i = 0; i < trials.length; i += CHUNK) {
      const chunk = trials.slice(i, i + CHUNK);
      await Promise.all(chunk.map((t) => q`
        INSERT INTO sart_trials (
          session_id, trial, onset, digit, is_target, responded, rt_ms,
          outcome, extra_presses
        ) VALUES (
          ${body.id}, ${num(t.trial)}, ${num(t.onset)}, ${num(t.digit)},
          ${bool(t.is_target)}, ${bool(t.responded)}, ${num(t.rt_ms)},
          ${t.outcome}, ${num(t.extra_presses) ?? 0}
        )
        ON CONFLICT (session_id, trial) DO NOTHING`));
    }

    for (const p of probes) {
      await q`
        INSERT INTO probe_responses (
          session_id, probe_index, elapsed, response, latency_ms, trials_before
        ) VALUES (
          ${body.id}, ${num(p.probe_index)}, ${num(p.elapsed)},
          ${p.response || null}, ${num(p.latency_ms)}, ${num(p.trials_before)}
        )
        ON CONFLICT (session_id, probe_index) DO NOTHING`;
    }

    return res.status(200).json({
      ok: true,
      id: body.id,
      taskType,
      featureRows: rows.length,
      trials: trials.length,
      probes: probes.length,
    });
  } catch (err) {
    console.error("submit failed", err);
    return res.status(500).json({ ok: false, error: "server_error" });
  }
}
