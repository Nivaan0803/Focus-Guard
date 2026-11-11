/* Mental-arithmetic task generator, ported from focus_study.py:51-61.
 *
 * Same operations and same ranges, so difficulty matches the desktop version:
 *   add  23-89 + 23-89
 *   sub  50-99 - 12-49
 *   mul  6-19  x 3-9
 *
 * focus_study.py:70 seeds Python's Mersenne Twister with
 * `f"{participant}-{condition}"`. The JavaScript stand-in lives in rng.js;
 * see that file for why the number sequences differ from Python's. Do not pool
 * browser-collected and desktop-collected rows as though participants saw the
 * same problems.
 *
 * The RNG moved to rng.js so sart.js can share it. selftest.html pins this
 * generator's first ten problems for seed `P01-alerts`, so that extraction is
 * provably behaviour-preserving.
 */

import { seeded, randInt } from "./rng.js";

export class ProblemSource {
  constructor(participant, condition) {
    this.rand = seeded(`${participant}-${condition}`);
  }

  // Inclusive on both ends, matching Python's random.randint.
  _int(lo, hi) {
    return randInt(this.rand, lo, hi);
  }

  next() {
    const kind = ["add", "sub", "mul"][this._int(0, 2)];
    if (kind === "add") {
      const a = this._int(23, 89);
      const b = this._int(23, 89);
      return { text: `${a} + ${b}`, answer: a + b };
    }
    if (kind === "sub") {
      const a = this._int(50, 99);
      const b = this._int(12, 49);
      return { text: `${a} − ${b}`, answer: a - b };
    }
    const a = this._int(6, 19);
    const b = this._int(3, 9);
    return { text: `${a} × ${b}`, answer: a * b };
  }
}
