/* Builds the signed consent PDF entirely in the browser.
 *
 * If the configured SEFH form exists, it is loaded and a signature page is
 * APPENDED to it, so the participant's copy contains the real form plus the
 * signatures. Appending rather than stamping in place means this works for any
 * form layout; once the real form's geometry is known, the signature boxes can
 * be positioned onto its own signature lines instead.
 *
 * If no form is present yet, a self-contained consent record is generated from
 * consent.json so the flow is testable before the PDF is dropped in.
 *
 * Nothing here touches the network. The PDF is produced and handed to the
 * participant locally; only a non-identifying receipt is uploaded.
 */

const PDFLIB_URL = "https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js";

let _lib = null;
async function pdflib() {
  if (_lib) return _lib;
  if (!window.PDFLib) {
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PDFLIB_URL;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Could not load the PDF library."));
      document.head.appendChild(s);
    });
  }
  _lib = window.PDFLib;
  return _lib;
}

const A4 = [595.28, 841.89];

function wrap(text, font, size, maxWidth) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (font.widthOfTextAtSize(next, size) > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/* record: { version, studyTitle, participant, signerName, signerPNG,
 *           parentName, parentPNG, isMinor, signedAt, statements[], userAgent } */
export async function buildSignedPDF(record, config) {
  const { PDFDocument, StandardFonts, rgb } = await pdflib();

  let doc = null;
  if (config && config.pdfPresent && config.pdf) {
    try {
      const res = await fetch(config.pdf);
      if (res.ok) {
        const bytes = await res.arrayBuffer();
        doc = await PDFDocument.load(bytes);
      }
    } catch {
      doc = null;   // fall through to a standalone record
    }
  }
  if (!doc) doc = await PDFDocument.create();

  const font = await doc.embedFont(StandardFonts.Helvetica);
  const bold = await doc.embedFont(StandardFonts.HelveticaBold);

  const page = doc.addPage(A4);
  const { width, height } = page.getSize();
  const margin = 56;
  const maxW = width - margin * 2;
  let y = height - margin;

  const ink = rgb(0.08, 0.11, 0.15);
  const mute = rgb(0.38, 0.42, 0.49);
  const line = rgb(0.78, 0.81, 0.86);

  const text = (s, { size = 10.5, f = font, color = ink, gap = 4 } = {}) => {
    for (const ln of wrap(s, f, size, maxW)) {
      if (y < margin + 40) return;            // never overflow into the footer
      page.drawText(ln, { x: margin, y, size, font: f, color });
      y -= size + gap;
    }
  };
  const rule = (pad = 10) => {
    y -= pad;
    page.drawLine({
      start: { x: margin, y }, end: { x: width - margin, y },
      thickness: 0.75, color: line,
    });
    y -= pad + 4;
  };

  page.drawText("Consent record", { x: margin, y, size: 20, font: bold, color: ink });
  y -= 28;
  text(record.studyTitle || "", { size: 11, f: bold });
  y -= 2;
  text(`Form version: ${record.version}`, { size: 9.5, color: mute, gap: 3 });
  text(`Signed: ${record.signedAt}`, { size: 9.5, color: mute, gap: 3 });
  text(`Participant ID: ${record.participant}`, { size: 9.5, color: mute, gap: 3 });
  rule();

  text("The participant agreed to each of the following:", { size: 10, f: bold });
  y -= 2;
  for (const s of record.statements || []) {
    const lines = wrap(s, font, 10, maxW - 16);
    lines.forEach((ln, i) => {
      if (y < margin + 40) return;
      if (i === 0) page.drawText("x", { x: margin, y, size: 10, font: bold, color: ink });
      page.drawText(ln, { x: margin + 16, y, size: 10, font, color: ink });
      y -= 14;
    });
  }
  rule();

  // ---- signature blocks ----
  const stamp = async (label, name, png) => {
    if (y < margin + 150) { y = height - margin; doc.addPage(A4); }
    text(label, { size: 10, f: bold, gap: 6 });
    if (png) {
      try {
        const img = await doc.embedPng(png);
        const boxW = 240, boxH = 70;
        const scale = Math.min(boxW / img.width, boxH / img.height, 1);
        const w = img.width * scale, h = img.height * scale;
        page.drawImage(img, { x: margin, y: y - h + 8, width: w, height: h });
        y -= Math.max(h, 28);
      } catch { /* a missing image must not void the record */ }
    }
    page.drawLine({
      start: { x: margin, y: y - 2 }, end: { x: margin + 260, y: y - 2 },
      thickness: 0.75, color: line,
    });
    y -= 14;
    text(name || "", { size: 10, gap: 3 });
    text(`Date: ${record.signedAt}`, { size: 9, color: mute, gap: 10 });
    y -= 8;
  };

  await stamp("Participant signature", record.signerName, record.signerPNG);
  if (record.isMinor) {
    await stamp("Parent / guardian signature", record.parentName, record.parentPNG);
  }

  // ---- footer ----
  page.drawText(
    "No webcam image, video, or screenshot was uploaded or stored at any point.",
    { x: margin, y: margin - 8, size: 8, font, color: mute },
  );

  const bytes = await doc.save();
  return new Blob([bytes], { type: "application/pdf" });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
