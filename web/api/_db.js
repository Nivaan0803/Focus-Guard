/* Shared database helpers.
 *
 * The database is OPTIONAL. If no Postgres connection string is configured the
 * endpoints return a clear "not configured" response instead of a 500, and the
 * client falls back to its CSV download. That means the site is deployable and
 * usable before any storage is provisioned.
 */

export function isConfigured() {
  return Boolean(
    process.env.POSTGRES_URL ||
    process.env.POSTGRES_URL_NON_POOLING ||
    process.env.DATABASE_URL,
  );
}

let _sql = null;
export async function sql() {
  if (_sql) return _sql;
  const mod = await import("@vercel/postgres");
  _sql = mod.sql;
  return _sql;
}

export function notConfigured(res) {
  return res.status(503).json({
    ok: false,
    error: "storage_not_configured",
    message:
      "No database is attached to this deployment yet. Add Vercel Postgres in " +
      "the Storage tab and run api/schema.sql, then redeploy.",
  });
}

export function readJsonBody(req) {
  // Vercel parses JSON bodies automatically, but a keepalive/beacon POST can
  // arrive as a raw string.
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string" && req.body) {
    try {
      return JSON.parse(req.body);
    } catch {
      return null;
    }
  }
  return null;
}

/* Export is protected by a bearer token: it returns human-participants data
 * and must never be publicly fetchable. Uses a length-independent comparison
 * to avoid leaking the token length through timing. */
export function authorized(req) {
  const expected = process.env.EXPORT_TOKEN;
  if (!expected) return false;
  const header = req.headers.authorization || "";
  const bearer = header.startsWith("Bearer ") ? header.slice(7) : "";
  const supplied = bearer || (req.query && req.query.token) || "";
  if (!supplied) return false;

  const a = Buffer.from(String(supplied));
  const b = Buffer.from(String(expected));
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

const CODE = /^[A-Z]{1,3}\d{1,4}$/;
export function validParticipant(value) {
  return typeof value === "string" && CODE.test(value);
}

export function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
