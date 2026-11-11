/* The SART task, behind the task seam. DOM, timing and keyboard only -
 * the sequence, the outcome table and every aggregate live in sart.js, which
 * stays pure so selftest.html can exercise them without a browser.
 *
 * TIMING. Each trial's onset is computed as `index * TRIAL_MS` from the
 * task's own clock, and a single ~20 ms driver advances state when that
 * boundary passes. Chaining two setTimeouts per trial would accumulate drift
 * across ~460 trials and silently stretch the session.
 *
 * THE TASK CLOCK IS NOT WALL TIME. A thought probe suspends the task, and the
 * paused interval is subtracted, so trial onsets stay continuous in task-time
 * across an interruption. Without that the whole schedule would shift by the
 * length of every probe.
 *
 * THROTTLING. Background tabs throttle timers to >=1 s. That means the
 * participant was away, which the harness's away-poll already records. If the
 * driver simply resumed, every skipped trial would score as an omission error
 * and omission_rate would become a second, worse copy of seconds_away. So any
 * trial whose window was not watched continuously is marked `dropped` and
 * excluded from every aggregate.
 */

import { seeded } from "./rng.js";
import {
  buildSequence, isTarget, classify,
  summarize, OUTCOME,
  STIMULUS_MS, MASK_MS, TRIAL_MS, TASK_TYPE, TASK_VERSION,
} from "./sart.js";

const DRIVER_MS = 20;

/* A step gap longer than this means the timer was throttled or the thread
 * blocked. Comfortably above the 20 ms driver and below one 1150 ms trial, so
 * ordinary jitter never trips it and a backgrounded tab always does. */
const THROTTLE_GAP_MS = 400;

/* Probes pause the task, so wall time exceeds minutes*60. Generate enough
 * trials that a session cannot run off the end of the sequence. */
const TRIAL_MARGIN = 1.3;

export { TASK_TYPE, TASK_VERSION };

export function createTask({ root, participant, condition, minutes, onBegin }) {
  const total = Math.ceil((minutes * 60 * 1000 / TRIAL_MS) * TRIAL_MARGIN);
  const digits = buildSequence(seeded(`${participant}-${condition}-sart`), total);

  root.innerHTML = `
    <p id="stimulus" class="sart-idle">Press SPACE to start</p>
    <p id="sart-help" class="muted"></p>
  `;
  const stimulusEl = root.querySelector("#stimulus");
  const helpEl = root.querySelector("#sart-help");
  helpEl.textContent =
    "Press SPACE for every digit. Do NOT press for the digit 3. " +
    "Go as fast as you can while still withholding on 3.";

  const trials = [];

  let begun = false;
  let accepting = false;
  let driver = null;

  let t0 = 0;              // ms, from performance.now()
  let pausedMs = 0;        // total time suspended, subtracted from task time
  let pauseStartedAt = null;
  let lastStepAt = 0;      // task-time of the previous driver step

  let index = -1;          // trial currently on screen
  let observed = false;    // was this trial's window watched continuously?
  let shownAt = null;      // task-time the digit actually painted
  let responded = false;
  let rtMs = null;
  let extraPresses = 0;
  let painted = -1;        // which phase is currently rendered

  const nowMs = () => performance.now();
  const taskTime = () => nowMs() - t0 - pausedMs;

  function finalize(i, wasObserved) {
    if (i < 0 || i >= digits.length) return;
    const target = isTarget(digits[i]);
    const outcome = wasObserved
      ? classify(target, responded)
      : OUTCOME.DROPPED;
    trials.push({
      trial: i,
      onset: (i * TRIAL_MS) / 1000,
      digit: digits[i],
      is_target: target,
      responded: wasObserved ? responded : false,
      rt_ms: wasObserved && responded ? rtMs : null,
      outcome,
      extra_presses: wasObserved ? extraPresses : 0,
    });
  }

  function openTrial(i, wasObserved) {
    index = i;
    observed = wasObserved;
    shownAt = null;
    responded = false;
    rtMs = null;
    extraPresses = 0;
    painted = -1;
  }

  function paint(elapsed) {
    if (index < 0 || index >= digits.length) return;
    const phase = elapsed - index * TRIAL_MS;
    const showDigit = phase < STIMULUS_MS;
    const want = showDigit ? 1 : 0;
    if (want === painted) return;
    painted = want;

    if (showDigit) {
      stimulusEl.textContent = String(digits[index]);
      stimulusEl.className = "sart-digit";
      if (shownAt === null) shownAt = elapsed;
    } else {
      stimulusEl.textContent = "+";
      stimulusEl.className = "sart-mask";
    }
  }

  function step() {
    const elapsed = taskTime();
    const gap = elapsed - lastStepAt;
    lastStepAt = elapsed;
    const interrupted = gap > THROTTLE_GAP_MS;

    const want = Math.floor(elapsed / TRIAL_MS);

    if (want !== index) {
      // Close the trial that was on screen. An interrupted step means its
      // window was not watched to the end, so it cannot be scored.
      finalize(index, observed && !interrupted);
      // Trials between the two indices never appeared at all.
      for (let k = index + 1; k < want; k++) {
        openTrial(k, false);
        finalize(k, false);
      }
      if (want < digits.length) {
        openTrial(want, !interrupted);
      } else {
        index = digits.length;   // sequence exhausted; harness ends on the clock
        stimulusEl.textContent = "+";
        stimulusEl.className = "sart-mask";
        return;
      }
    } else if (interrupted) {
      observed = false;
    }

    paint(elapsed);
  }

  function onKey(ev) {
    if (ev.code !== "Space") return;
    // Space scrolls the page by default, which would move the stimulus.
    ev.preventDefault();
    if (ev.repeat) return;        // a held key is one intention, not many

    if (!begun) {
      begun = true;
      onBegin();
      return;
    }
    if (!accepting || index < 0 || index >= digits.length) return;

    if (responded) {
      extraPresses += 1;          // counted, but the first press is the response
      return;
    }
    responded = true;
    // Measured from when the digit actually painted, not its nominal onset,
    // so a late frame is not charged to the participant.
    rtMs = taskTime() - (shownAt ?? index * TRIAL_MS);
  }

  addEventListener("keydown", onKey);

  return {
    trials,

    /* How long until the current trial ends, so the harness can hold a probe
     * until a boundary. Interrupting mid-stimulus would cost that trial (it
     * is marked dropped, never scored on a partial window) and show the card
     * over a half-read digit. Bounded by one trial, so the probe is never
     * delayed by more than ~1.2 s. */
    msToBoundary() {
      if (index < 0 || index >= digits.length) return 0;
      return Math.max(0, (index + 1) * TRIAL_MS - taskTime());
    },

    start() {
      accepting = true;
      t0 = nowMs();
      pausedMs = 0;
      lastStepAt = 0;
      helpEl.textContent = "SPACE for every digit except 3.";
      openTrial(0, true);
      paint(0);
      driver = setInterval(step, DRIVER_MS);
    },

    suspend() {
      accepting = false;
      clearInterval(driver);
      driver = null;
      pauseStartedAt = nowMs();
      observed = false;           // this trial's window is broken by the pause
      stimulusEl.textContent = "";
      stimulusEl.className = "sart-idle";
      painted = -1;
    },

    resume() {
      if (pauseStartedAt !== null) {
        pausedMs += nowMs() - pauseStartedAt;
        pauseStartedAt = null;
      }
      accepting = true;
      lastStepAt = taskTime();    // do not read the pause as a throttle gap
      driver = setInterval(step, DRIVER_MS);
    },

    stop() {
      accepting = false;
      clearInterval(driver);
      driver = null;
      removeEventListener("keydown", onKey);
      // The trial on screen was cut off by the clock, not completed. One
      // partial window out of ~460 is immaterial; scoring it would not be.
      finalize(index, false);
      index = digits.length;
      stimulusEl.textContent = "Time's up!";
      stimulusEl.className = "sart-idle";
      helpEl.textContent = "";
    },

    summary() {
      return summarize(trials);
    },
  };
}
