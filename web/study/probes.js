/* Thought probes: periodic self-reports of where attention actually was.
 *
 * This is the only source of ground truth the web study has. The detector's
 * own verdict cannot label its own training data, and a remote participant
 * has no operator to flip the focused/distracted toggle that main.py relies
 * on. Asking the participant makes them their own annotator, which is how
 * attention research has gotten labels since Giambra: you interrupt, and you
 * ask before the memory decays.
 *
 * That last point drives the design. A probe asks about the window that just
 * elapsed, so it must arrive unannounced - a predictable probe would have
 * people watching for it, which is itself a form of on-task vigilance and
 * would bias every answer toward "on task".
 *
 * probeSchedule is pure and seeded, so a participant's probe times are
 * reproducible and selftest.html can check the spacing rules without a
 * browser. The card below is the only part that touches the DOM.
 */

import { randInt } from "./rng.js";
import { PROBE_RESPONSES } from "../lib/labels.js";

export const QUESTION = "Where was your attention just now?";

/* Order matters: it runs from fully on-task to fully off-task, so the
 * participant reads it as a scale rather than four unrelated options.
 * `value` must match PROBE_RESPONSES in lib/labels.js, which maps them to
 * focused/distracted. */
export const CHOICES = [
  { value: "on_task", label: "On the task" },
  { value: "task_related", label: "Thinking about the task, but not doing it" },
  { value: "mind_wandering", label: "Mind-wandering — thinking about something unrelated" },
  { value: "external", label: "Distracted by something around me" },
];

export const MIN_GAP_S = 45;
export const MAX_GAP_S = 75;    // mean 60; see docs/sart-redesign.md s5.1
export const TAIL_GUARD_S = 30;

/* Sanity: the card cannot offer a response that lib/labels.js has never heard
 * of, or the answer would silently drop out of every training export. */
const known = new Set(PROBE_RESPONSES);
for (const choice of CHOICES) {
  if (!known.has(choice.value)) {
    throw new Error(`probe choice "${choice.value}" is not in PROBE_RESPONSES`);
  }
}

/* Probe times in seconds from task start.
 *
 * Gaps are drawn uniformly from [MIN_GAP_S, MAX_GAP_S] rather than fixed at
 * 60, so the participant cannot anticipate one. The first gap is drawn the
 * same way, which also satisfies "no probe in the opening 45 s" - the task
 * needs a chance to become monotonous before the question means anything.
 *
 * Nothing is scheduled within TAIL_GUARD_S of the end: a probe at 9:55 has
 * almost no preceding window to report on, and only delays the finish.
 */
export function probeSchedule(rand, minutes, options = {}) {
  const minGap = options.minGap ?? MIN_GAP_S;
  const maxGap = options.maxGap ?? MAX_GAP_S;
  const tailGuard = options.tailGuard ?? TAIL_GUARD_S;
  const horizon = minutes * 60 - tailGuard;

  const times = [];
  let t = 0;
  for (;;) {
    t += randInt(rand, minGap, maxGap);
    if (t > horizon) break;
    times.push(t);
  }
  return times;
}

/* Renders the card and resolves once an option is chosen.
 *
 * Returns { response, latencyMs }. The caller is responsible for having
 * suspended the task and stopped the clock first - this function deliberately
 * knows nothing about either.
 */
export function askProbe(root, nowMs) {
  return new Promise((resolve) => {
    const openedAt = nowMs();

    root.innerHTML = `
      <div class="probe" role="group" aria-labelledby="probe-q">
        <p id="probe-q" class="probe-q">${QUESTION}</p>
        <div class="probe-choices">
          ${CHOICES.map((c, i) => `
            <button type="button" class="probe-choice" data-value="${c.value}">
              <span class="probe-key">${i + 1}</span>
              <span>${c.label}</span>
            </button>`).join("")}
        </div>
        <p class="muted small">Answer honestly — there is no right answer, and
           nobody is judged on it. Press 1-4 or click.</p>
      </div>
    `;

    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      removeEventListener("keydown", onKey, true);
      resolve({ response: value, latencyMs: nowMs() - openedAt });
    };

    root.querySelectorAll(".probe-choice").forEach((button) => {
      button.addEventListener("click", () => finish(button.dataset.value));
    });

    function onKey(ev) {
      const n = Number(ev.key);
      if (!Number.isInteger(n) || n < 1 || n > CHOICES.length) return;
      ev.preventDefault();
      ev.stopPropagation();
      finish(CHOICES[n - 1].value);
    }

    // Capture phase: the task's own keydown listener is still attached, and
    // this must win. SPACE stays inert because the task is suspended, so a
    // reflexive press cannot be swallowed into a probe answer.
    addEventListener("keydown", onKey, true);

    root.querySelector(".probe-choice")?.focus();
  });
}
