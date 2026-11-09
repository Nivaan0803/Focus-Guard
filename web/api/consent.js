/* POST /api/consent - a consent RECEIPT, never the consent itself.
 *
 * What arrives here: participant code, form version, timestamp, minor flag,
 * and a SHA-256 hash of the signature. What never arrives: the participant's
 * name, the signature image, or the signed PDF. Those stay on the researcher's
 * device by design (see lib/consent.js), so this table can prove that P01
 * consented to form version X at a time without holding identifying data about
 * a minor in a cloud database.
 */

import { isConfigured, sql, notConfigured, readJsonBody, validParticipant } from "./_db.js";

const HEX64 = /^[0-9a-f]{64}$/;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "method_not_allowed" });
  }
  if (!isConfigured()) return notConfigured(res);

  const body = readJsonBody(req);
  if (!body || !body.version || !body.signedAt) {
    return res.status(400).json({ ok: false, error: "bad_payload" });
  }
  if (!validParticipant(body.participant)) {
    return res.status(400).json({ ok: false, error: "bad_participant_code" });
  }
  // Reject anything that is not a bare hash, so a name can never be smuggled
  // into this column by a modified client.
  const hash = typeof body.signatureHash === "string" && HEX64.test(body.signatureHash)
    ? body.signatureHash
    : null;

  try {
    const q = await sql();
    await q`
      INSERT INTO consent_receipts (
        participant, form_version, signed_at, is_minor,
        signature_hash, statements, pdf_saved
      ) VALUES (
        ${body.participant}, ${body.version}, ${body.signedAt},
        ${Boolean(body.isMinor)}, ${hash},
        ${Number.isFinite(Number(body.statementsAgreed)) ? Number(body.statementsAgreed) : null},
        ${Boolean(body.pdfSaved)}
      )
      ON CONFLICT (participant, form_version, signed_at) DO NOTHING`;

    return res.status(200).json({ ok: true });
  } catch (err) {
    console.error("consent receipt failed", err);
    return res.status(500).json({ ok: false, error: "server_error" });
  }
}
