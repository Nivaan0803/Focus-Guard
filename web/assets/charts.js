/* Renders data/validation.json.
 *
 * Every value in that file starts null. Anything still null renders as
 * "not yet measured" rather than a placeholder number, because an unsourced
 * figure on a public science-fair page is the one failure worth engineering
 * against. There is deliberately no code path that invents a value.
 */

const fmtPct = (v) => (v === null || v === undefined ? null : `${(v * 100).toFixed(1)}%`);

function pending(label) {
  return `<tr><th>${label}</th><td class="muted">not yet measured</td></tr>`;
}

function row(label, value, suffix = "") {
  if (value === null || value === undefined) return pending(label);
  return `<tr><th>${label}</th><td class="mono">${value}${suffix}</td></tr>`;
}

function detectorTable(d) {
  const rows = [
    row("Out-of-fold accuracy", fmtPct(d.out_of_fold_accuracy)),
    row("Rows", d.n_rows),
    row("Sessions", d.n_sessions),
  ];
  const c = d.confusion || {};
  const haveConfusion = ["tp", "fp", "fn", "tn"].every((k) => c[k] !== null && c[k] !== undefined);
  rows.push(haveConfusion
    ? `<tr><th>Confusion (TP/FP/FN/TN)</th><td class="mono">${c.tp} / ${c.fp} / ${c.fn} / ${c.tn}</td></tr>`
    : pending("Confusion matrix"));
  return `<table>${rows.join("")}</table>`;
}

function lopoTable(list) {
  if (!list || !list.length) {
    return `<p class="muted">Not yet measured. Run
      <span class="mono">python train_model.py p01="logs/p01_*.csv" p02="logs/p02_*.csv"</span>
      and paste the per-person numbers into <span class="mono">data/validation.json</span>.</p>`;
  }
  const body = list.map((p) => `<tr>
      <th>${p.person}</th>
      <td class="mono">${fmtPct(p.accuracy) ?? "&mdash;"}</td>
      <td class="mono">${fmtPct(p.precision) ?? "&mdash;"}</td>
      <td class="mono">${fmtPct(p.recall) ?? "&mdash;"}</td>
    </tr>`).join("");
  return `<table>
    <tr><th>Held-out person</th><th>Accuracy</th><th>Precision</th><th>Recall</th></tr>
    ${body}</table>`;
}

function baselinesTable(b) {
  const labels = {
    full_model: "Full model (all signals)",
    phone_only: "Phone only",
    tab_only: "Tab only",
    camera_only: "Camera only",
  };
  const rows = Object.keys(labels).map((k) => row(labels[k], fmtPct(b[k])));
  return `<table>${rows.join("")}</table>`;
}

function studyTable(s) {
  if (!s.measures || !s.measures.length) {
    return `<p class="muted">Not yet measured. Collect paired sessions, then run
      <span class="mono">python analyze_study.py</span> and paste the mean change
      and p-value for each measure into <span class="mono">data/validation.json</span>.</p>`;
  }
  const body = s.measures.map((m) => `<tr>
      <th>${m.name}</th>
      <td class="mono">${m.mean_change ?? "&mdash;"}</td>
      <td class="mono">${m.p_value ?? "&mdash;"}</td>
    </tr>`).join("");
  return `<table>
    <tr><th>Measure</th><th>Mean change (alerts &minus; no alerts)</th><th>p</th></tr>
    ${body}
    </table><p class="small muted">n = ${s.n_participants ?? "not yet measured"} participants.</p>`;
}

export async function renderValidation(root) {
  let data;
  try {
    const res = await fetch("data/validation.json", { cache: "no-cache" });
    data = await res.json();
  } catch {
    root.innerHTML = `<p class="muted">Results could not be loaded.</p>`;
    return;
  }

  root.innerHTML = `
    <div class="card">
      <h3>Detector</h3>
      ${detectorTable(data.detector || {})}
    </div>
    <div class="card">
      <h3>Does it work on people it has never seen?</h3>
      <p class="small muted">Leave-one-person-out: train on everyone except one
        person, score that person, repeat. Ordinary k-fold is optimistic here,
        because neighbouring ticks from one session are near-duplicates.</p>
      ${lopoTable(data.leave_one_person_out)}
    </div>
    <div class="card">
      <h3>Does each signal earn its place?</h3>
      <p class="small muted">If the full model does not beat every single-signal
        baseline, combining signals is not justified.</p>
      ${baselinesTable(data.baselines || {})}
    </div>
    <div class="card">
      <h3>Does real-time feedback actually help?</h3>
      <p class="small muted">Each participant is compared with themselves, across
        a session with alerts and a session without.</p>
      ${studyTable(data.study || {})}
    </div>
    <p class="small muted">${data.updated ? `Last updated ${data.updated}.`
      : "No results have been published yet."}</p>`;
}
