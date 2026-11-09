/* Alert tone. Replaces the desktop app's pygame/sounddevice alarm.
 *
 * The AudioContext must be created or resumed inside a user gesture or the
 * browser's autoplay policy leaves it suspended and the first alert is silent.
 * unlock() is therefore called from the consent "I agree" click, long before
 * any alert can fire.
 */

let ctx = null;

export function unlock() {
  try {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC();
    }
    if (ctx.state === "suspended") ctx.resume();
    return true;
  } catch {
    return false;
  }
}

export function ready() { return Boolean(ctx) && ctx.state === "running"; }

/* Two short rising tones — distinct from a notification chime so it reads as
 * "come back" rather than "you have mail". */
export function beep({ volume = 0.35 } = {}) {
  if (!ctx) unlock();
  if (!ctx) return;
  if (ctx.state === "suspended") ctx.resume();

  const now = ctx.currentTime;
  const gain = ctx.createGain();
  gain.connect(ctx.destination);
  gain.gain.setValueAtTime(0, now);

  [[0, 660], [0.18, 880]].forEach(([offset, freq]) => {
    const t = now + offset;
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, t);
    osc.connect(gain);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(volume, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    osc.start(t);
    osc.stop(t + 0.18);
  });
}
