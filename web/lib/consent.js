/* The consent + signature gate.
 *
 * This is a HARD BLOCK: it is the first thing /study renders, and the camera is
 * never requested and no data is ever buffered until gate() resolves.
 *
 * Privacy design, which is deliberate and should not be "simplified" later.
 * A signature is identifying data: a legal name plus a handwriting sample,
 * often a minor's. The study dataset is otherwise de-identified. So the signed
 * PDF is built and saved LOCALLY (lib/pdf.js) and only a non-identifying
 * RECEIPT is uploaded: participant code, form version, timestamp, and a
 * SHA-256 hash binding the signature to them. That proves to a reviewer that
 * P01 consented to form version X at a given time, without putting names or
 * signature images in a cloud database.
 *
 * ISEF note: the International Rules require the researcher to keep the
 * original signed form. The downloaded PDF is that original, so run the study
 * on the researcher's device (a booth laptop) where the download lands with
 * the researcher, or collect the PDFs from remote participants by hand.
 */

import { SignaturePad } from "./signature.js";
import { buildSignedPDF, downloadBlob } from "./pdf.js";
import { unlock as unlockAudio } from "./beep.js";

const STORE_KEY = "focusguard.consent";
const QUEUE_KEY = "focusguard.consent.queue";
const ID_PATTERN = /^[A-Z]{1,3}\d{1,4}$/; // P01, P123, ABC9 - never a name

const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

async function sha256Hex(text) {
  try {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    // crypto.subtle is unavailable over plain http on a LAN address; the
    // receipt is still valid, it just carries no hash.
    return "";
  }
}

export function storedConsent() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function clearConsent() {
  try {
    localStorage.removeItem(STORE_KEY);
  } catch {
    /* private mode */
  }
}

async function postReceipt(receipt) {
  try {
    const res = await fetch("/api/consent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(receipt),
      keepalive: true,
    });
    if (res.ok) return true;
  } catch {
    /* offline, or running under a plain static server with no /api */
  }
  try {
    const queue = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
    queue.push(receipt);
    localStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
  } catch {
    /* private mode */
  }
  return false;
}

/* Retry anything stranded by an earlier offline session. */
export async function flushConsentQueue() {
  let queue;
  try {
    queue = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
  } catch {
    return;
  }
  if (!queue.length) return;
  const left = [];
  for (const receipt of queue) {
    try {
      const res = await fetch("/api/consent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(receipt),
      });
      if (!res.ok) left.push(receipt);
    } catch {
      left.push(receipt);
    }
  }
  try {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(left));
  } catch {
    /* private mode */
  }
}

async function loadConfig() {
  const res = await fetch("consent/consent.json", { cache: "no-cache" });
  if (!res.ok) throw new Error("Could not load the consent form configuration.");
  return res.json();
}

function template(cfg) {
  const summary = cfg.summary.map((s) => `<li>${esc(s)}</li>`).join("");
  const statements = cfg.statements
    .map(
      (s, i) => `
    <div class="check">
      <input type="checkbox" id="st${i}" data-statement>
      <label for="st${i}">${esc(s)}</label>
    </div>`,
    )
    .join("");

  const pdfBlock =
    cfg.pdfPresent && cfg.pdf
      ? `
    <div class="card">
      <h3>The full consent form</h3>
      <iframe src="${esc(cfg.pdf)}#view=FitH" title="Informed consent form"
              style="width:100%;height:clamp(320px,60vh,560px);border:1px solid var(--line);border-radius:10px;background:#fff"></iframe>
      <p class="small" style="margin-top:10px">
        <a href="${esc(cfg.pdf)}" download>Download the form as a PDF</a>
        &mdash; some phones cannot display a PDF inline.
      </p>
    </div>`
      : `
    <div class="card">
      <h3>The full consent form</h3>
      <p class="muted small">
        The official form has not been attached to this deployment yet. The
        summary above describes the study, and a consent record will still be
        generated. <strong>Do not collect real participant data until the
        approved form is in place.</strong>
      </p>
    </div>`;

  const contact = cfg.sponsorContact
    ? `<p class="small muted">Questions? Contact ${esc(cfg.sponsor || "the adult sponsor")}
        at <a href="mailto:${esc(cfg.sponsorContact)}">${esc(cfg.sponsorContact)}</a>.</p>`
    : "";

  return `
  <div class="wrap" style="padding-top:28px;padding-bottom:48px">
    <h1>${esc(cfg.title)}</h1>
    <p class="muted">${esc(cfg.studyTitle)}</p>
    <p class="small muted">Student researcher: ${esc(cfg.researcher || "")}</p>

    <div class="card">
      <h3>What this involves</h3>
      <ul>${summary}</ul>
      ${contact}
    </div>

    ${pdfBlock}

    <div class="card">
      <h3>Your details</h3>
      <label for="pid">Participant code</label>
      <input id="pid" type="text" inputmode="latin" autocomplete="off"
             placeholder="P01" maxlength="7" aria-describedby="pidhelp">
      <p class="small muted" id="pidhelp" style="margin:6px 0 0">
        The anonymous code you were given, like <span class="mono">P01</span>.
        Never your name.
      </p>

      <div style="margin-top:16px">
        <label for="age">Are you under ${esc(cfg.minorAgeThreshold)}?</label>
        <select id="age">
          <option value="">Choose&hellip;</option>
          <option value="no">No, I am ${esc(cfg.minorAgeThreshold)} or older</option>
          <option value="yes">Yes, I am under ${esc(cfg.minorAgeThreshold)}</option>
        </select>
        <p class="small muted" style="margin:6px 0 0">
          If you are under ${esc(cfg.minorAgeThreshold)}, a parent or guardian must also sign.
        </p>
      </div>
    </div>

    <div class="card">
      <h3>Please confirm</h3>
      ${statements}
    </div>

    <div class="card">
      <h3>Research participant signature</h3>
      <label for="sname">Printed name</label>
      <input id="sname" type="text" autocomplete="name" placeholder="Type your full name">
      <p class="small muted" style="margin:6px 0 10px">
        This name stays on this device, inside the PDF that is saved. It is not uploaded.
      </p>
      <label for="sigpad">Sign below</label>
      <canvas id="sigpad" tabindex="0" aria-label="Signature area"
              style="width:100%;height:150px;background:#fff;border:1px dashed var(--line);border-radius:10px;display:block"></canvas>
      <div class="row" style="margin-top:10px">
        <button type="button" id="sigclear">Clear signature</button>
      </div>
    </div>

    <div class="card hidden" id="parentblock">
      <h3>Parent / guardian permission</h3>
      <label for="pname">Parent / guardian printed name</label>
      <input id="pname" type="text" autocomplete="off" placeholder="Type their full name">
      <label for="psigpad" style="margin-top:12px">Parent / guardian signature</label>
      <canvas id="psigpad" tabindex="0" aria-label="Parent or guardian signature area"
              style="width:100%;height:150px;background:#fff;border:1px dashed var(--line);border-radius:10px;display:block"></canvas>
      <div class="row" style="margin-top:10px">
        <button type="button" id="psigclear">Clear signature</button>
      </div>
    </div>

    <div class="card">
      <p id="gatewhy" class="small muted" aria-live="polite"></p>
      <div class="row">
        <button type="button" id="agree" class="primary" disabled>
          I agree &mdash; save my consent and begin
        </button>
        <button type="button" id="decline">No thanks</button>
      </div>
      <p class="small muted" style="margin:14px 0 0">
        A copy of the signed consent downloads to this device for the
        researcher's records. Only your code, the form version, the time, and a
        one-way hash of your signature are sent &mdash; never your name, your
        signature image, or any video.
      </p>
    </div>
  </div>`;
}

/* Resolves with the consent receipt once the participant has signed.
 * Never resolves if they decline; it replaces the page instead. */
export function gate(container, { onDecline } = {}) {
  return new Promise((resolve) => {
    (async () => {
      let cfg;
      try {
        cfg = await loadConfig();
      } catch (err) {
        container.innerHTML = `<div class="wrap" style="padding-top:40px"><div class="card">
          <h2>Could not start</h2><p>${esc(err.message)}</p></div></div>`;
        return;
      }

      flushConsentQueue();

      // An existing consent for THIS form version skips the gate.
      const prior = storedConsent();
      if (prior && prior.version === cfg.version && prior.participant) {
        resolve(prior);
        return;
      }

      container.innerHTML = template(cfg);
      const $ = (sel) => container.querySelector(sel);

      const pid = $("#pid");
      const age = $("#age");
      const sname = $("#sname");
      const pname = $("#pname");
      const agree = $("#agree");
      const why = $("#gatewhy");
      const parentBlock = $("#parentblock");

      const pad = new SignaturePad($("#sigpad"));
      const ppad = new SignaturePad($("#psigpad"));
      $("#sigclear").addEventListener("click", () => pad.clear());
      $("#psigclear").addEventListener("click", () => ppad.clear());

      const statements = Array.from(container.querySelectorAll("[data-statement]"));
      const isMinor = () => age.value === "yes";

      function problems() {
        const out = [];
        const id = pid.value.trim().toUpperCase();
        if (!id) out.push("enter your participant code");
        else if (!ID_PATTERN.test(id)) out.push("the code should look like P01, letters then numbers, not a name");
        if (!age.value) out.push(`say whether you are under ${cfg.minorAgeThreshold}`);
        if (statements.some((c) => !c.checked)) out.push("tick every confirmation box");
        if (!sname.value.trim()) out.push("type your printed name");
        else if (!sname.value.trim().includes(" ")) out.push("type your full name, first and last");
        if (!pad.isSigned) out.push("sign in the signature box");
        if (isMinor()) {
          if (!pname.value.trim()) out.push("add a parent or guardian's printed name");
          if (!ppad.isSigned) out.push("have a parent or guardian sign");
        }
        return out;
      }

      function refresh() {
        parentBlock.classList.toggle("hidden", !isMinor());
        if (isMinor()) ppad.resize();
        const probs = problems();
        agree.disabled = probs.length > 0;
        why.textContent = probs.length
          ? `Still needed: ${probs.join("; ")}.`
          : "Ready. Your signed consent will download when you continue.";
      }

      [pid, age, sname, pname].forEach((el) => {
        el.addEventListener("input", refresh);
        el.addEventListener("change", refresh);
      });
      statements.forEach((c) => c.addEventListener("change", refresh));
      $("#sigpad").addEventListener("signature:change", refresh);
      $("#psigpad").addEventListener("signature:change", refresh);
      refresh();

      $("#decline").addEventListener("click", () => {
        container.innerHTML = `<div class="wrap" style="padding-top:40px"><div class="card">
          <h2>No problem</h2>
          <p>Nothing has been recorded and your camera was never switched on.
             You can close this tab.</p></div></div>`;
        if (onDecline) onDecline();
      });

      agree.addEventListener("click", async () => {
        if (problems().length) {
          refresh();
          return;
        }
        agree.disabled = true;
        agree.textContent = "Saving your consent…";

        // Must happen inside this click, or autoplay policy leaves the audio
        // context suspended and the first alert of the session is silent.
        unlockAudio();

        const participant = pid.value.trim().toUpperCase();
        const signedAt = new Date().toISOString().replace("T", " ").slice(0, 16);
        const signerPNG = pad.toPNG();
        const parentPNG = isMinor() ? ppad.toPNG() : null;

        const record = {
          version: cfg.version,
          studyTitle: cfg.studyTitle,
          researcher: cfg.researcher,
          participant,
          signerName: sname.value.trim(),
          signerPNG,
          parentName: isMinor() ? pname.value.trim() : "",
          parentPNG,
          isMinor: isMinor(),
          signedAt,
          statements: cfg.statements,
        };

        const signatureHash = await sha256Hex(
          [
            record.version,
            participant,
            record.signerName,
            signerPNG || "",
            record.parentName,
            parentPNG || "",
            signedAt,
          ].join("|"),
        );

        let pdfSaved = false;
        try {
          const blob = await buildSignedPDF(record, cfg);
          downloadBlob(blob, `consent_${participant}_${cfg.version}.pdf`);
          pdfSaved = true;
        } catch (err) {
          console.warn("consent PDF could not be built", err);
        }

        const receipt = {
          participant,
          version: cfg.version,
          signedAt,
          isMinor: record.isMinor,
          signatureHash,
          statementsAgreed: cfg.statements.length,
          pdfSaved,
        };

        try {
          localStorage.setItem(STORE_KEY, JSON.stringify(receipt));
        } catch {
          /* private mode */
        }
        await postReceipt(receipt);

        pad.destroy();
        ppad.destroy();

        if (!pdfSaved) {
          // Never silently proceed without the researcher's copy: ISEF requires
          // the original signed form be retained.
          container.innerHTML = `<div class="wrap" style="padding-top:40px"><div class="card">
            <h2>Consent recorded, but the PDF did not save</h2>
            <p>Your consent was recorded, but the signed copy could not be
               generated on this device. Please tell the researcher before
               continuing so they can record it another way.</p>
            <button type="button" class="primary" id="anyway">Continue anyway</button>
          </div></div>`;
          container.querySelector("#anyway").addEventListener("click", () => resolve(receipt));
          return;
        }

        resolve(receipt);
      });
    })();
  });
}
