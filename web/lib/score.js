/* Leaky distraction score and the alert state machine.
 * Ported from main.py:605-613 (smoothing) and main.py:670-683 (handle_alerts).
 */

export const RISE_RATE = 0.45;        // main.py:606
export const FALL_RATE = 0.18;
export const ALERT_AT = 0.55;         // main.py:613
export const MIN_ALERT_GAP = 4;       // main.py:681, seconds
export const METER_AMBER = 0.4;       // main.py:610
export const METER_RED = 0.7;

export class ScoreTracker {
  constructor({ grace = 3, realertSeconds = 0 } = {}) {
    this.grace = grace;
    this.realertSeconds = realertSeconds;
    this.reset();
  }

  reset() {
    this.score = 0;
    this.history = [];               // [{t, score}] seconds since session start
    this.alerts = 0;
    this.alertEvents = [];           // [{t, reason}]
    this.distractionStarted = null;
    this.alertSent = false;
    this.lastAlertTime = -Infinity;
    this.pausedUntil = 0;
  }

  /* main.py:605-614. `state` is "focused" | "distracted", `now` in seconds. */
  update(state, now, sessionStart) {
    const target = state === "distracted" ? 1 : 0;
    const rate = target >= this.score ? RISE_RATE : FALL_RATE;
    this.score = Math.min(1, Math.max(0, this.score + (target - this.score) * rate));
    this.history.push({ t: now - sessionStart, score: this.score });
    return this.score;
  }

  get distracted() { return this.score >= ALERT_AT; }

  get meterColor() {
    if (this.score < METER_AMBER) return "var(--green)";
    if (this.score < METER_RED) return "var(--amber)";
    return "var(--red)";
  }

  pause(seconds, now) { this.pausedUntil = now + seconds; }
  isPaused(now) { return now < this.pausedUntil; }

  /* main.py:670-683. Returns true on the tick an alert should fire. */
  shouldAlert(now, reason) {
    if (!this.distracted) {
      this.distractionStarted = null;
      this.alertSent = false;
      return false;
    }
    this.distractionStarted = this.distractionStarted ?? now;
    if (now - this.distractionStarted < this.grace || now < this.pausedUntil) return false;

    const first = !this.alertSent;
    const repeat = this.realertSeconds > 0 &&
      (now - this.lastAlertTime) >= this.realertSeconds;
    if ((first || repeat) && (now - this.lastAlertTime) >= MIN_ALERT_GAP) {
      this.lastAlertTime = now;
      this.alertSent = true;
      this.alerts += 1;
      this.alertEvents.push({ t: now, reason });
      return true;
    }
    return false;
  }

  /* Banner state, mirroring main.py:689-711. */
  bannerState(now, { taskTabActive = true } = {}) {
    if (now < this.pausedUntil) return "pause";
    if (this.distracted && this.distractionStarted !== null &&
        now - this.distractionStarted >= this.grace) return "bad";
    if (this.distracted) return "warn";
    if (!taskTabActive) return "warn";
    return "good";
  }
}
