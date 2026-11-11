/* The mental-arithmetic task, behind the task seam.
 *
 * Split from arithmetic.js so that file stays pure: selftest.html imports
 * ProblemSource and runs it in a plain module with no DOM, and the parity
 * check that pins the problem sequence has to keep working. Everything here
 * touches the document; everything there is arithmetic.
 *
 * The same split applies to sart.js / sart-task.js. Logic is testable, DOM is
 * not, so the line goes between them.
 *
 * Contract (see docs/sart-redesign.md s3.1). The harness in flow.js owns the
 * clock, both timers, the away-poll and the detector; a task owns its own
 * stimulus area and its own notion of a correct response, and nothing else.
 */

import { ProblemSource } from "./arithmetic.js";

export const TASK_TYPE = "arithmetic";
export const TASK_VERSION = "arithmetic-1.0";

export function createTask({ root, participant, condition, minutes, onBegin }) {
  root.innerHTML = `
    <p id="problem">Press Enter to start</p>
    <input id="answer" type="text" inputmode="numeric" autocomplete="off"
           autocorrect="off" spellcheck="false" aria-label="Your answer">
    <p id="feedback" class="muted"></p>
  `;

  const problemEl = root.querySelector("#problem");
  const answerEl = root.querySelector("#answer");
  const feedbackEl = root.querySelector("#feedback");

  feedbackEl.textContent =
    `${minutes} minutes of arithmetic. Type each answer and press Enter. ` +
    "Answer as many as you can.";

  const source = new ProblemSource(participant, condition);
  let expected = null;
  let attempted = 0;
  let correct = 0;
  let begun = false;
  let accepting = false;

  const nextProblem = () => {
    const p = source.next();
    expected = p.answer;
    problemEl.textContent = p.text;
    answerEl.value = "";
  };

  const onKey = (ev) => {
    if (ev.key !== "Enter") return;
    ev.preventDefault();

    // The first Enter is the start gate. The harness decides what that means
    // for the clock; the task only reports that the participant is ready.
    if (!begun) {
      begun = true;
      onBegin();
      return;
    }
    if (!accepting) return;

    const text = answerEl.value.trim().replace("−", "-");
    if (!text) return;
    attempted += 1;
    const right = Number.parseInt(text, 10) === expected;
    if (right) correct += 1;
    feedbackEl.textContent = right ? "✓" : `✗  (${expected})`;
    feedbackEl.className = right ? "right" : "wrong";
    nextProblem();
  };

  // The participant may click away; keep the caret where the task is. Held in
  // a named reference so stop() can detach it - the original installed this
  // anonymously and it outlived the session.
  const keepCaret = () => { if (!answerEl.disabled) answerEl.focus(); };

  answerEl.addEventListener("keydown", onKey);
  addEventListener("focus", keepCaret);
  answerEl.focus();

  return {
    trials: [],   // arithmetic records no per-item data; SART does

    start() {
      accepting = true;
      feedbackEl.textContent = "";
      feedbackEl.className = "muted";
      nextProblem();
    },

    suspend() {
      accepting = false;
      answerEl.disabled = true;
    },

    resume() {
      accepting = true;
      answerEl.disabled = false;
      answerEl.focus();
    },

    stop() {
      accepting = false;
      answerEl.disabled = true;
      answerEl.removeEventListener("keydown", onKey);
      removeEventListener("focus", keepCaret);
      problemEl.textContent = "Time's up!";
      feedbackEl.textContent = "";
      feedbackEl.className = "muted";
    },

    summary() {
      return {
        task_type: TASK_TYPE,
        task_version: TASK_VERSION,
        attempted,
        correct,
      };
    },
  };
}
