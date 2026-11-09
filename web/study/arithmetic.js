/* Mental-arithmetic task generator, ported from focus_study.py:51-61.
 *
 * Same operations and same ranges, so difficulty matches the desktop version:
 *   add  23-89 + 23-89
 *   sub  50-99 - 12-49
 *   mul  6-19  x 3-9
 *
 * focus_study.py:70 seeds Python's Mersenne Twister with
 * `f"{participant}-{condition}"`. JavaScript has no Mersenne Twister, so this
 * uses mulberry32 over a hashed seed. That preserves the DESIGN properties the
 * study relies on - deterministic per participant+condition, different between
 * the two conditions, identical difficulty distribution - but NOT the identical
 * number sequence. Do not pool browser-collected and desktop-collected rows as
 * though participants saw the same problems.
 */

function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class ProblemSource {
  constructor(participant, condition) {
    this.rand = mulberry32(xmur3(`${participant}-${condition}`)());
  }

  // Inclusive on both ends, matching Python's random.randint.
  _int(lo, hi) {
    return lo + Math.floor(this.rand() * (hi - lo + 1));
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
      return { text: `${a} \u2212 ${b}`, answer: a - b };
    }
    const a = this._int(6, 19);
    const b = this._int(3, 9);
    return { text: `${a} \u00d7 ${b}`, answer: a * b };
  }
}
