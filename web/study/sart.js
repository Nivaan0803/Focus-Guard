/* SART (Sustained Attention to Response Task) - pure logic, no DOM.
 *
 * Digits 1-9 appear one at a time. The participant presses SPACE for every
 * digit EXCEPT the target (3), which they must withhold. Pressing on a target
 * is a COMMISSION ERROR, the standard operational marker of an attention
 * lapse: the response has become automatic and attention has gone elsewhere.
 *
 * Why this replaced mental arithmetic: arithmetic keeps people engaged, which
 * is the opposite of the state the detector is meant to catch. The SART is
 * deliberately monotonous, so mind-wandering happens on its own, and at
 * ~1.15 s per trial a 10-minute session yields ~460 scored trials instead of
 * ~40 problems - each one carrying a reaction time.
 *
 * Everything here is deterministic given a seeded generator from rng.js, and
 * selftest.html pins the sequence, the outcome table and every aggregate
 * formula. Keep this file free of DOM and timers so it stays testable.
 */

import { shuffle } from "./rng.js";

export const TARGET_DIGIT = 3;
export const DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9];
export const BLOCK_SIZE = DIGITS.length;      // 9 -> an exact 1-in-9 target rate

export const STIMULUS_MS = 250;
export const MASK_MS = 900;
export const TRIAL_MS = STIMULUS_MS + MASK_MS;   // 1150

export const TASK_TYPE = "sart";
export const TASK_VERSION = "sart-1.0";

export const OUTCOME = {
  HIT: "hit",                            // non-target, responded - correct
  CORRECT_OMISSION: "correct_omission",  // target, withheld - correct
  COMMISSION_ERROR: "commission_error",  // target, responded - THE lapse marker
  OMISSION_ERROR: "omission_error",      // non-target, missed
  DROPPED: "dropped",                    // tab throttled; excluded from everything
};

/* ------------------------------------------------------------- sequence -- */

/* Each block is a shuffled permutation of 1..9, which by construction holds
 * exactly one target. That fixes the target rate at 1/9 = 11.1% (matching
 * Robertson's 25/225) and equalises digit frequency, instead of the clumping
 * and per-session variance you get sampling i.i.d. at p=0.11. A noisy target
 * count would make commission_rate incomparable between participants.
 *
 * Two adjacent targets across a block boundary are broken up: a target is a
 * withholding trial, and back-to-back withholds change the task's character.
 */
export function buildSequence(rand, count) {
  const out = [];
  let previousWasTarget = false;

  while (out.length < count) {
    const block = shuffle(rand, DIGITS.slice());

    if (previousWasTarget && block[0] === TARGET_DIGIT) {
      // block[0] is the only target, so the midpoint is guaranteed not to be
      // one. A single deterministic swap - never a reshuffle loop, which
      // would consume an unpredictable number of draws and break determinism.
      const mid = Math.floor(BLOCK_SIZE / 2);
      [block[0], block[mid]] = [block[mid], block[0]];
    }

    for (const digit of block) {
      if (out.length >= count) break;
      out.push(digit);
    }
    previousWasTarget = out[out.length - 1] === TARGET_DIGIT;
  }

  return out;
}

export function isTarget(digit) {
  return digit === TARGET_DIGIT;
}

/* -------------------------------------------------------------- outcomes -- */

export function classify(target, responded) {
  if (target) return responded ? OUTCOME.COMMISSION_ERROR : OUTCOME.CORRECT_OMISSION;
  return responded ? OUTCOME.HIT : OUTCOME.OMISSION_ERROR;
}

/* ------------------------------------------------------------ aggregates -- */

function mean(values) {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/* SAMPLE standard deviation (n-1). These are a sample of the participant's
 * responses, not the population of every response they could make, and the
 * n-1 convention is what the literature reports. selftest.html pins this
 * against a NumPy-generated value so the convention cannot drift. */
function sampleSd(values) {
  if (values.length < 2) return null;
  const m = mean(values);
  const ss = values.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return Math.sqrt(ss / (values.length - 1));
}

const ratio = (num, den) => (den > 0 ? num / den : null);

/* `trials` are the per-trial records built by the task harness:
 *   { trial, onset, digit, is_target, responded, rt_ms, outcome, extra_presses }
 *
 * Dropped trials are excluded from every figure below. They happen when the
 * tab was throttled - which means the participant was away, something
 * seconds_away already records. Scoring them as omissions would turn
 * omission_rate into a second, worse copy of that column.
 */
export function summarize(trials) {
  const valid = trials.filter((t) => t.outcome !== OUTCOME.DROPPED);
  const dropped = trials.length - valid.length;

  const targets = valid.filter((t) => t.is_target);
  const goTrials = valid.filter((t) => !t.is_target);

  const count = (outcome) => valid.filter((t) => t.outcome === outcome).length;
  const commissionErrors = count(OUTCOME.COMMISSION_ERROR);
  const omissionErrors = count(OUTCOME.OMISSION_ERROR);
  const hits = count(OUTCOME.HIT);
  const correctOmissions = count(OUTCOME.CORRECT_OMISSION);

  // RT over hits only - standard practice. A commission error's RT reflects a
  // different process (failure to inhibit), and mixing them muddies rt_cv.
  const rts = valid
    .filter((t) => t.outcome === OUTCOME.HIT && typeof t.rt_ms === "number")
    .map((t) => t.rt_ms);

  const rtMean = mean(rts);
  const rtSd = sampleSd(rts);

  return {
    task_type: TASK_TYPE,
    task_version: TASK_VERSION,
    sart_trials: valid.length,
    sart_targets: targets.length,
    sart_go_trials: goTrials.length,
    sart_dropped_trials: dropped,
    sart_accuracy: ratio(hits + correctOmissions, valid.length),
    commission_errors: commissionErrors,
    commission_rate: ratio(commissionErrors, targets.length),
    omission_errors: omissionErrors,
    omission_rate: ratio(omissionErrors, goTrials.length),
    rt_mean_ms: rtMean,
    rt_sd_ms: rtSd,
    // Coefficient of variation: how erratic the responses are, independent of
    // how fast. A validated marker of mind-wandering, and continuous rather
    // than the binary a correct/incorrect task gives you.
    rt_cv: rtMean && rtSd !== null ? rtSd / rtMean : null,
  };
}
