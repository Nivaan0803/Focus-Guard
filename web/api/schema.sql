-- Run once against the Vercel Postgres database (Storage tab -> Query),
-- THEN run schema_sart.sql, which adds the SART aggregate columns and the
-- sart_trials / probe_responses tables.
--
-- The two are kept as separate files rather than folded into one on purpose:
-- schema_sart.sql is additive and idempotent, so running base-then-migration
-- gives a fresh database exactly the same result as an existing one, and
-- there is no second copy of those statements to drift out of step.
--
-- Two deliberately SEPARATE tables.
--
-- `consent_receipts` is the compliance record: it proves a participant code
-- consented to a form version at a time. It holds no name and no signature
-- image; those live only in the PDF saved on the researcher's device.
--
-- `sessions` and `feature_rows` are the de-identified dataset. They are keyed
-- by the same participant code, which is what makes the data analysable, but
-- nothing in either table identifies a person on its own.

CREATE TABLE IF NOT EXISTS consent_receipts (
  id             BIGSERIAL PRIMARY KEY,
  participant    TEXT        NOT NULL,
  form_version   TEXT        NOT NULL,
  signed_at      TEXT        NOT NULL,
  is_minor       BOOLEAN     NOT NULL DEFAULT FALSE,
  signature_hash TEXT,
  statements     INTEGER,
  pdf_saved      BOOLEAN     NOT NULL DEFAULT FALSE,
  received_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (participant, form_version, signed_at)
);

CREATE TABLE IF NOT EXISTS sessions (
  id              TEXT        PRIMARY KEY,   -- client-generated, makes POSTs idempotent
  participant     TEXT        NOT NULL,
  condition       TEXT        NOT NULL,
  finished        TEXT        NOT NULL,
  minutes         NUMERIC,
  attempted       INTEGER,
  correct         INTEGER,
  accuracy        NUMERIC,
  correct_per_min NUMERIC,
  seconds_away    NUMERIC,
  times_left      INTEGER,
  self_rating     INTEGER,
  alerts          INTEGER,
  model           TEXT,
  user_agent      TEXT,
  received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per detector tick. Column names match session_logger.py:18 exactly
-- so /api/export can emit a byte-compatible CSV.
CREATE TABLE IF NOT EXISTS feature_rows (
  id                   BIGSERIAL PRIMARY KEY,
  session_id           TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  elapsed              NUMERIC,
  task_tab_active      NUMERIC,
  tab_switches_per_min NUMERIC,
  time_on_current_tab  NUMERIC,
  keyboard_inactivity  NUMERIC,
  mouse_inactivity     NUMERIC,
  calibrated           NUMERIC,
  face_visible         NUMERIC,
  head_pitch           NUMERIC,
  head_yaw             NUMERIC,
  iris_delta           NUMERIC,
  eye_openness         NUMERIC,
  phone_present        NUMERIC,
  label                TEXT
);

CREATE INDEX IF NOT EXISTS feature_rows_session ON feature_rows (session_id);
CREATE INDEX IF NOT EXISTS sessions_participant ON sessions (participant);
