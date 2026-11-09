/* Gaze calibration ported from vision_worker.py:390-416 (_advance_calibration).
 *
 * Collects a baseline while the participant works normally, then sets the
 * neutral pose from the MEDIAN and the look-away thresholds from the SPREAD, so
 * someone who sits very still gets tighter thresholds than someone who fidgets.
 */

import { DEFAULT_THRESHOLDS } from "./features.js";

const MIN_SAMPLES = 25;   // vision_worker.py:400

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function std(xs) {
  // numpy.std default is the population deviation (ddof=0) — match it.
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
  return Math.sqrt(v);
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class Calibration {
  constructor() {
    this.neutralPose = null;     // [pitch, yaw]
    this.neutralIris = null;
    this.thresholds = [...DEFAULT_THRESHOLDS];
    this.seconds = 10;           // vision_worker.py:156
    this.deadline = null;
    this.samples = [];
    this.seq = 0;
    this.lastEvent = "";         // "", "done", "failed"
  }

  get calibrated() { return this.neutralPose !== null; }
  get calibrating() { return this.deadline !== null; }

  get remaining() {
    return this.calibrating ? Math.max(0, (this.deadline - performance.now()) / 1000) : 0;
  }

  // vision_worker.py:182-188
  request(seconds = this.seconds) {
    this.seconds = Math.max(3, Math.floor(seconds));
    this.deadline = performance.now() + this.seconds * 1000;
    this.samples = [];
    this.neutralPose = null;
    this.neutralIris = null;
  }

  // vision_worker.py:190-195
  clear() {
    this.deadline = null;
    this.samples = [];
    this.neutralPose = null;
    this.neutralIris = null;
    this.thresholds = [...DEFAULT_THRESHOLDS];
  }

  /* Feed one frame. `pose` is [pitch, yaw] or null; `iris` a number or null.
   * Returns "" normally, or "done"/"failed" on the frame the window closes. */
  advance(pose, iris) {
    this.lastEvent = "";
    if (this.deadline === null) return "";

    if (pose) this.samples.push([pose[0], pose[1], iris ?? 0]);

    if (performance.now() < this.deadline) return "";

    if (this.samples.length >= MIN_SAMPLES) {
      const pitches = this.samples.map((s) => s[0]);
      const yaws = this.samples.map((s) => s[1]);
      const irises = this.samples.map((s) => s[2]);

      this.neutralPose = [median(pitches), median(yaws)];
      this.neutralIris = median(irises);
      // vision_worker.py:404-408
      this.thresholds = [
        clamp(3.5 * std(pitches) + 12.0, 12.0, 25.0),
        clamp(3.5 * std(yaws) + 15.0, 15.0, 30.0),
        clamp(5.0 * std(irises) + 0.11, 0.11, 0.25),
      ];
      this.lastEvent = "done";
    } else {
      this.lastEvent = "failed";
    }

    this.deadline = null;
    this.samples = [];
    this.seq += 1;
    return this.lastEvent;
  }
}
