/* The order-entry work task, behind the task seam (contract: see
 * arithmetic-task.js). Pure logic lives in dataentry.js.
 *
 * The instructions matter as much as the form. They tell the participant,
 * explicitly, that ordinary off-task behaviour is fine - checking a phone,
 * opening another tab. Without that permission the study measures how well
 * people perform "being a good participant", which is the survey bias the
 * SART sessions ran into.
 */

import { SlipSource, FIELDS, checkEntry, TASK_TYPE, TASK_VERSION } from "./dataentry.js";

export { TASK_TYPE, TASK_VERSION };

/* Probe spacing for this task. A question every minute is itself a reminder
 * that attention is being watched, which undoes the naturalistic framing, so
 * this task asks roughly every two minutes instead. Fewer labels per session
 * is the price. */
export const PROBE_OPTIONS = { minGap: 90, maxGap: 150, tailGuard: 45 };

const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

export function createTask({ root, participant, condition, minutes, onBegin }) {
  root.innerHTML = `
    <div class="de-intro" id="de-intro">
      <h2 style="margin-top:0">Order entry</h2>
      <p>For the next ${esc(minutes)} minutes you'll be doing a small office job:
         copying paper order slips into the order system. Each slip appears on
         the left; type it into the form and press <strong>Enter</strong> to file it.</p>
      <p>Work at a normal, comfortable pace, the way you would on a real
         assignment. It is not a speed test and there is no target number.</p>
      <p><strong>Use your computer the way you usually do while working.</strong>
         If you would normally glance at your phone, reply to a message, or open
         another tab, that's fine - nobody is penalised for it. We are interested
         in how people actually work, not in how hard they can force themselves
         to stare at one screen.</p>
      <p class="muted small">Now and then the page will pause to ask one quick question.</p>
      <button class="primary" id="de-start">Start working</button>
    </div>
    <div class="de-work" id="de-work" hidden>
      <div class="de-slip" id="de-slip" aria-label="Order slip"></div>
      <form class="de-form" id="de-form" autocomplete="off">
        ${FIELDS.map((f) => `
          <label for="de-${f.key}">${f.label}</label>
          <input id="de-${f.key}" name="${f.key}" type="text" autocomplete="off"
                 autocorrect="off" autocapitalize="off" spellcheck="false">`).join("")}
        <button class="primary" type="submit">File order (Enter)</button>
        <p class="muted small" id="de-count"></p>
      </form>
    </div>
  `;

  const intro = root.querySelector("#de-intro");
  const work = root.querySelector("#de-work");
  const slipEl = root.querySelector("#de-slip");
  const form = root.querySelector("#de-form");
  const countEl = root.querySelector("#de-count");
  const inputs = FIELDS.map((f) => root.querySelector(`#de-${f.key}`));

  const source = new SlipSource(participant, condition);
  let slip = null;
  let attempted = 0;
  let correct = 0;
  let accepting = false;

  const showSlip = () => {
    slip = source.next();
    slipEl.innerHTML = `
      <div class="de-slip-head"><span>ORDER SLIP</span><span class="mono">No. ${slip.slip}</span></div>
      <dl>
        <dt>Customer</dt><dd>${esc(slip.customer)}</dd>
        <dt>Date</dt><dd class="mono">${esc(slip.date)}</dd>
        <dt>Item</dt><dd><span class="mono">${esc(slip.item)}</span><br>
            <span class="muted small">${esc(slip.description)}</span></dd>
        <dt>Qty</dt><dd class="mono">${esc(slip.qty)}</dd>
        <dt>Unit price</dt><dd class="mono">$${esc(slip.price)}</dd>
      </dl>`;
    inputs.forEach((i) => { i.value = ""; });
    inputs[0].focus();
  };

  // Enter in any field files the order; a blank form is ignored so a stray
  // Enter cannot count as an attempt.
  const onSubmit = (ev) => {
    ev.preventDefault();
    if (!accepting) return;
    const entry = Object.fromEntries(inputs.map((i) => [i.name, i.value]));
    if (Object.values(entry).every((v) => !v.trim())) return;
    attempted += 1;
    if (!checkEntry(slip, entry).length) correct += 1;
    // Deliberately no right/wrong feedback: real data entry gives none, and a
    // running score turns the job back into a test.
    countEl.textContent = `${attempted} order${attempted === 1 ? "" : "s"} filed`;
    showSlip();
  };

  root.querySelector("#de-start").addEventListener("click", () => {
    intro.hidden = true;
    work.hidden = false;
    onBegin();
  });
  form.addEventListener("submit", onSubmit);

  const setDisabled = (d) => inputs.forEach((i) => { i.disabled = d; });

  return {
    trials: [],   // no per-item timing data, like arithmetic
    msToBoundary() { return 0; },

    start() {
      accepting = true;
      showSlip();
    },
    suspend() {
      accepting = false;
      setDisabled(true);
    },
    resume() {
      accepting = true;
      setDisabled(false);
      // Back to the first empty field, so a half-typed slip carries on.
      (inputs.find((i) => !i.value) || inputs[0]).focus();
    },
    stop() {
      accepting = false;
      setDisabled(true);
      form.removeEventListener("submit", onSubmit);
      slipEl.innerHTML = `<p style="font-weight:600">Time's up - thanks!</p>`;
    },
    summary() {
      return { task_type: TASK_TYPE, task_version: TASK_VERSION, attempted, correct };
    },
  };
}
