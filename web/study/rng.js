/* Seeded pseudo-random number generation, shared by every task module.
 *
 * Extracted verbatim from arithmetic.js, where it was written to stand in for
 * Python's Mersenne Twister (focus_study.py:70 seeds with
 * `f"{participant}-{condition}"`). JavaScript has no Mersenne Twister, so this
 * is mulberry32 over an xmur3-hashed seed. That preserves the DESIGN
 * properties the study relies on - deterministic per seed string, different
 * between conditions, uniform - but NOT Python's number sequence. Browser and
 * desktop runs are not item-for-item comparable and must not be pooled.
 *
 * selftest.html pins the output of both `seeded` consumers, so any change here
 * is caught rather than silently reshuffling every participant's stimuli.
 */

export function xmur3(str) {
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

export function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* The one entry point task modules should use: a seed string in, a
 * `() => [0, 1)` generator out. */
export function seeded(seedString) {
  return mulberry32(xmur3(seedString)());
}

/* Inclusive on both ends, matching Python's random.randint. */
export function randInt(rand, lo, hi) {
  return lo + Math.floor(rand() * (hi - lo + 1));
}

/* Fisher-Yates, in place, returning the same array. Walks downward so each
 * index is drawn exactly once - the ascending variant is subtly biased. */
export function shuffle(rand, array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = randInt(rand, 0, i);
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}
