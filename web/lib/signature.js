/* Signature pad: a canvas you sign with a finger, stylus, or mouse.
 *
 * Pointer events cover all three with one code path. The canvas is sized to its
 * CSS box times devicePixelRatio so a signature drawn on a phone is not a
 * blurry upscale when it lands in the PDF.
 */

export class SignaturePad {
  constructor(canvas, { color = "#111827", width = 2.2 } = {}) {
    this.canvas = canvas;
    this.color = color;
    this.lineWidth = width;
    this.strokes = [];
    this._current = null;
    this._dirty = false;

    this.resize();
    this._bind();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    this._redraw();
  }

  _point(ev) {
    const r = this.canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  _bind() {
    const down = (ev) => {
      ev.preventDefault();
      this.canvas.setPointerCapture?.(ev.pointerId);
      this._current = [this._point(ev)];
      this.strokes.push(this._current);
      this._dirty = true;
      this._redraw();
    };
    const move = (ev) => {
      if (!this._current) return;
      ev.preventDefault();
      // coalesced events keep fast strokes smooth on high-rate styluses
      const pts = ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev];
      for (const p of (pts.length ? pts : [ev])) this._current.push(this._point(p));
      this._redraw();
    };
    const up = (ev) => {
      if (!this._current) return;
      ev.preventDefault();
      this._current = null;
      this.canvas.dispatchEvent(new CustomEvent("signature:change"));
    };

    this.canvas.addEventListener("pointerdown", down);
    this.canvas.addEventListener("pointermove", move);
    this.canvas.addEventListener("pointerup", up);
    this.canvas.addEventListener("pointercancel", up);
    this.canvas.addEventListener("pointerleave", up);
    // Stop the page from scrolling while a finger is drawing.
    this.canvas.style.touchAction = "none";

    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
  }

  _redraw() {
    const ctx = this.canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    ctx.clearRect(0, 0, this.canvas.width / dpr, this.canvas.height / dpr);
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.lineWidth;
    for (const stroke of this.strokes) {
      if (stroke.length < 2) {
        if (stroke.length === 1) {
          ctx.beginPath();
          ctx.arc(stroke[0].x, stroke[0].y, this.lineWidth / 2, 0, Math.PI * 2);
          ctx.fillStyle = this.color;
          ctx.fill();
        }
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (let i = 1; i < stroke.length; i++) ctx.lineTo(stroke[i].x, stroke[i].y);
      ctx.stroke();
    }
  }

  clear() {
    this.strokes = [];
    this._current = null;
    this._dirty = false;
    this._redraw();
    this.canvas.dispatchEvent(new CustomEvent("signature:change"));
  }

  /* A couple of stray dots is not a signature. Require some actual ink. */
  get isSigned() {
    const points = this.strokes.reduce((n, s) => n + s.length, 0);
    if (points < 8) return false;
    let length = 0;
    for (const s of this.strokes) {
      for (let i = 1; i < s.length; i++) {
        length += Math.hypot(s[i].x - s[i - 1].x, s[i].y - s[i - 1].y);
      }
    }
    return length > 40;
  }

  /* Trimmed PNG on a transparent background, ready to stamp into the PDF. */
  toPNG() {
    if (!this.isSigned) return null;
    return this.canvas.toDataURL("image/png");
  }

  destroy() { window.removeEventListener("resize", this._onResize); }
}
