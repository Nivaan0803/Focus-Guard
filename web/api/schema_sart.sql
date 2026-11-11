-- SART + thought probes migration.  See docs/sart-redesign.md s6.
--
-- Additive and idempotent: every statement is IF NOT EXISTS, so running it
-- twice is harmless and running it against a database that already holds
-- arithmetic sessions changes none of their rows. The same statements are
-- folded into schema.sql so a fresh database gets everything in one pass.
--
-- The design rule behind all of this: a SART measure NEVER reuses an
-- arithmetic column. analyze_study.py:68 does a bare float() on whatever is
-- in a column, so a commission rate parked in `accuracy` would be averaged in
-- with real arithmetic accuracies and nothing would raise. Separate columns
-- plus the task_type discriminator are what keep the two populations apart.

-- ---------------------------------------------------------------- sessions

-- The anti-pooling guard. Existing rows are arithmetic by definition, which
-- is why the default is not NULL: analyze_study.py can then key on
-- (participant, task_type) without special-casing older data.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS task_type     TEXT NOT NULL DEFAULT 'arithmetic';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS task_version  TEXT;

-- Probes stop the clock, so wall time exceeds minutes*60 and `seconds_away`
-- was measured over a longer interval than the nominal session length. Both
-- are stored so that denominator stays recoverable in analysis.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS wall_seconds  NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probe_seconds NUMERIC;

-- SART aggregates. All nullable: an arithmetic session has none of them, and
-- NULL is the honest value rather than 0.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_trials         INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_targets        INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_go_trials      INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_dropped_trials INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_accuracy       NUMERIC;
-- Pressing on a target: the standard operational marker of an attention lapse.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS commission_errors   INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS commission_rate     NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS omission_errors     INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS omission_rate       NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_mean_ms          NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_sd_ms            NUMERIC;
-- Coefficient of variation: how erratic the responses are, independent of how
-- fast. A validated marker of mind-wandering, and continuous rather than the
-- binary a correct/incorrect task gives you.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_cv               NUMERIC;

-- Probe-reported off-task proportion. This is the headline measure of the
-- redesign: a DIRECT report of mind-wandering rather than a look-away proxy.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probes_shown        INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probes_answered     INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS mw_rate             NUMERIC;

CREATE INDEX IF NOT EXISTS sessions_task_type ON sessions (task_type);

-- ------------------------------------------------------------- sart_trials
--
-- One row per presented digit. feature_rows cannot hold this: it is one row
-- per detector tick with fixed numeric columns, and a trial is a different
-- grain entirely.

CREATE TABLE IF NOT EXISTS sart_trials (
  id            BIGSERIAL PRIMARY KEY,
  session_id    TEXT     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  trial         INTEGER  NOT NULL,     -- 0-based presentation index
  onset         NUMERIC  NOT NULL,     -- seconds since task t0, 3dp
  digit         SMALLINT NOT NULL,
  is_target     BOOLEAN  NOT NULL,
  responded     BOOLEAN  NOT NULL,
  rt_ms         NUMERIC,               -- NULL when no response, or when dropped
  -- hit | correct_omission | commission_error | omission_error | dropped.
  -- `dropped` means the tab was throttled so the trial's window was never
  -- watched; those are excluded from every aggregate, because scoring them
  -- as omissions would turn omission_rate into a second copy of seconds_away.
  outcome       TEXT     NOT NULL,
  extra_presses SMALLINT NOT NULL DEFAULT 0,
  -- Makes a retried POST idempotent at the row level, independently of the
  -- ON CONFLICT guard on sessions.
  UNIQUE (session_id, trial)
);

CREATE INDEX IF NOT EXISTS sart_trials_session ON sart_trials (session_id, onset);

-- --------------------------------------------------------- probe_responses
--
-- The only ground truth the web study has. The detector cannot label its own
-- training data, and a remote participant has no operator to flip the
-- focused/distracted toggle main.py relies on.
--
-- Raw answers are stored and never pre-interpreted. The mapping to
-- focused/distracted and the window length are research parameters applied at
-- export time by lib/labels.js, so retuning them is a re-download rather than
-- a re-collection.

CREATE TABLE IF NOT EXISTS probe_responses (
  id            BIGSERIAL PRIMARY KEY,
  session_id    TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  probe_index   INTEGER NOT NULL,
  -- Seconds since task t0 when the card OPENED: the right-hand edge of the
  -- window the participant is reporting on.
  elapsed       NUMERIC NOT NULL,
  -- on_task | task_related | mind_wandering | external; NULL if unanswered.
  response      TEXT,
  latency_ms    NUMERIC,
  trials_before INTEGER,
  UNIQUE (session_id, probe_index)
);

CREATE INDEX IF NOT EXISTS probe_responses_session ON probe_responses (session_id, elapsed);
