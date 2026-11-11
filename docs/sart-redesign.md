# SART + thought probes: study redesign

Status: **design approved, not yet implemented.**
Last updated: 2026-11-10

---

## 1. Why

Two problems with the study as it stands.

**The web study collects no focus labels.** `web/study/flow.js:369` passes
`session.condition` into `Detector.tick()`, and that value lands in the `label`
column of every feature row (`web/lib/upload.js:124-130`). So every task row is
labelled `"alerts"` and every calibration row `"calibration"`. Feeding
`/api/export?kind=features` to `train_model.py` would train a classifier that
predicts *which condition a session was*, not whether anyone was distracted.

The desktop path does have real ground truth — an operator flipping a
`["focused","distracted"]` combobox with F8 (`main.py:380-381`,
`session_logger.py`) — but the browser port dropped it, and a remote
participant has no operator.

**Arithmetic is the wrong task for this question.** It keeps people heads-down
and engaged, which is the opposite of the state we are trying to detect, and it
yields ~40 scored items per session with no per-item timing.

The redesign fixes both: a task that reliably induces mind-wandering and emits
~10x the data, plus periodic self-reports that supply genuine labels.

---

## 2. Decisions taken

| Decision | Choice |
|---|---|
| Task | SART (Sustained Attention to Response Task) |
| Thought probes | Yes, ~60 s interval |
| Getting labelled rows | Third `label_only` session, outside the A/B |
| `task_related` probe answer | Dropped from training by default, tunable |
| Clock during a probe | Paused; record `wall_seconds` + `probe_seconds` |
| Desktop SART port | **No** — web only |
| Arithmetic | Kept, selectable via `?task=arithmetic` |

### 2.1 The three conditions

Probes run in **all three**, so the A/B arms stay comparable to each other and
`mw_rate` becomes a direct outcome measure.

| condition | camera | alerts fire | probes | role |
|---|---|---|---|---|
| `alerts` | yes | yes | yes | A/B arm |
| `no_alerts` | **no** | no | yes | A/B arm |
| `label_only` | yes | no (silent) | yes | training data only |

This preserves the trade documented at `web/study/flow.js:11-15` — `no_alerts`
remains "don't run Focus Guard at all", so the counterbalanced comparison is
unchanged. Training labels come from `label_only`, where the camera runs but
nothing interrupts, so labels are **not** collected under the intervention being
measured.

---

## 3. Architecture

### 3.1 Task seam

`task()` at `flow.js:286-397` currently fuses five jobs: stimulus DOM, input and
grading, the 250 ms away-poll, the 90 ms detector tick, and result assembly.
Split it.

**Harness** (stays in `flow.js`): owns `started`/`ends`, both timers, banner and
meter, the probe scheduler, and `finish()`. Knows nothing about digits.

**Task module** — one factory, no base class:

```js
createTask({ root, participant, condition, minutes, rng }) -> {
  mount(), start(t0), suspend(), resume(), stop(),
  summary(),   // { task_type, task_version, ...aggregates }
  trials,      // per-trial records
}
```

Selected by `?task=sart|arithmetic`, default `sart`. `arithmetic.js` is adapted
to the same shape so a regression is a URL parameter, not a git revert — and so
the `task_type` DB column can ever hold more than one value.

### 3.2 RNG seam

Extract `xmur3` + `mulberry32` from `web/study/arithmetic.js:17-39` into
`web/study/rng.js`.

**Do this safely:** add a parity check pinning arithmetic's first 10 problems
for seed `P01-alerts` *before* extracting, so the refactor is provably
behaviour-preserving.

SART seeds with `` `${participant}-${condition}-sart` `` — deliberately
different from arithmetic's `` `${participant}-${condition}` `` so a participant
who sees both tasks doesn't get correlated streams. The probe schedule draws
from the same seeded stream, so probe timing is reproducible too.

### 3.3 Label derivation — at export time, in shared JS

New `web/lib/labels.js`, pure functions, no DOM:

```js
deriveLabels(rows, probes, { windowSeconds = 15, taskRelated = "drop" })
PROBE_TO_LABEL   // mapping table, exported as data
```

Imported by **both** `web/api/export.js` and `web/selftest.html`, so the parity
test exercises the real production path rather than a twin. `export.js` is
already a JS serverless function and `MAX_FEATURE_ROWS = 20000` bounds a
session, so in-memory labelling is cheap.

Raw probe rows and raw feature rows stay untouched in the DB. Tuning knobs live
in the URL:

```
/api/export?kind=training&window=15&task_related=drop&condition=label_only
```

Retuning the window is a re-download, not a re-collection. **Do not** add a SQL
view or a client-side copy of this rule — three implementations will drift.

> Verify `vercel build` traces `../lib/labels.js` into
> `.vercel/output/functions/api/export.func/`. If not, duplicate at
> `web/api/_labels.js` and re-export from one canonical copy.

### 3.4 `label` becomes provenance, not verdict

| value | written at | meaning |
|---|---|---|
| `calibration` | `flow.js:258` (unchanged) | calibration window |
| `task` | task tick | participant was doing the task |
| `probe` | while a probe card is up | participant was answering a probe |

The model's verdict is still never stored. `kind=features` keeps its 14-column
shape; only the values change, and only for new sessions.

---

## 4. SART specification

### 4.1 Sequence

Digits 1–9. Target is **3** (no-go). Press SPACE for every digit *except* 3.

Do **not** sample i.i.d. at p=0.11 — that clumps targets and varies the target
count per session, making commission rate noisy across participants. Use a
**constrained block shuffle**: blocks of 9 containing exactly one `3` and eight
drawn from `{1,2,4,5,6,7,8,9}`, seeded Fisher-Yates within block, rotate any
block whose first element is a target when the previous block ended with one.
Exactly 1/9 = 11.1%, matching Robertson's 25/225, fully reproducible.

### 4.2 Timing

- 250 ms digit, 900 ms mask → **1150 ms per trial**
- 10 min = 600 s → ~521 trials, minus ~10 probes × ~6 s → **~460 trials, ~51 targets**
- SPACE accepted from stimulus onset to end of mask. RT from onset via
  `performance.now()`. A second press is ignored but counted as `extra_presses`.

**Scheduler:** do not chain two `setTimeout`s per trial — drift accumulates.
Compute nominal onset as `t0 + i * TRIAL_MS` and drive transitions from a single
~16–30 ms interval (or rAF) that advances when `now >= nextBoundary`. Record
both nominal and actual onset.

**Data-integrity rule (easy to forget, matters a lot):** background tabs throttle
timers to ≥1 s. A backgrounded tab means the participant is away — already
recorded by the away-poll. If the scheduler just resumes, every skipped trial
scores as an omission error and `omission_rate` degenerates into a second copy
of `seconds_away`. Detect any boundary jump > ~1 trial and mark those trials
`outcome: "dropped"`, excluded from all aggregates.

### 4.3 Outcomes

| is_target | responded | outcome |
|---|---|---|
| false | true | `hit` |
| false | false | `omission_error` |
| true | true | `commission_error` ← the mind-wandering marker |
| true | false | `correct_omission` |
| — | — | `dropped` (throttled / hidden) |

### 4.4 Records and aggregates

Per trial: `{ trial, onset, digit, is_target, responded, rt_ms, outcome, extra_presses }`

Session: `sart_trials`, `sart_targets`, `sart_go_trials`, `sart_dropped_trials`,
`commission_errors`, `commission_rate` (commissions/targets), `omission_errors`,
`omission_rate` (omissions/go_trials), `rt_mean_ms`, `rt_sd_ms` (**sample** SD,
n−1), `rt_cv` (sd/mean), `sart_accuracy`.

RT statistics over **hits only** (standard practice).

> `performance.now()` is coarsened (100 µs–5 ms). Fine at 1150 ms trials, but do
> not claim sub-5 ms RT precision — `rt_sd_ms` sits on a quantised grid.

### 4.5 Input cleanup

`flow.js:394` installs `addEventListener("focus", () => { if (!answerEl.disabled)
answerEl.focus(); })`. **Remove it with the text input** — a window-level SPACE
handler plus a stolen caret is a bug waiting to happen. The SPACE handler needs
`ev.code === "Space"`, `ev.preventDefault()` (stops page scroll), and
`if (ev.repeat) return`.

---

## 5. Thought probes

### 5.1 Schedule

Interval uniform in **[45, 75] s** (mean 60), from the seeded RNG. No probe
before ~45 s; none within ~30 s of session end. Yields **~9–10 per session**.

Rationale for 60 s over 75 s: `train_model.py:97` hard-exits when only one class
is present, and literature off-task rates run ~20–40%, so a participant can
easily produce zero `distracted` probes. More probes materially reduces that
risk.

### 5.2 Lifecycle

1. Scheduler fires at a **trial boundary**, never mid-stimulus — interrupting
   mid-trial corrupts that trial's RT.
2. `task.suspend()` — blank stimulus, stop accepting SPACE.
3. Render the probe card.
4. **Clock:** stop `pollTimer`, record `probeOpened`. On answer:
   `ends += (now() - probeOpened)`, `lastPoll = now()`, restart `pollTimer`.
   Without this, an alt-tab during a probe increments `times_left` and probe
   deliberation is charged against the task budget.
   Consequence: wall time exceeds `minutes * 60`, so store `wall_seconds` and
   `probe_seconds` to keep the denominator recoverable.
5. **Detector keeps ticking** with `detector.tick("probe")`. Stopping it would
   punch gaps into a continuous series that look like dropouts; `probe` rows are
   excluded from training anyway.
6. `task.resume()`.

### 5.3 Question

> **Where was your attention just now?**

| value | label | → derived |
|---|---|---|
| `on_task` | On the task | `focused` |
| `task_related` | Thinking about the task, but not doing it | **dropped** (tunable) |
| `mind_wandering` | Mind-wandering — thinking about something unrelated | `distracted` |
| `external` | Distracted by something around me | `distracted` |

Keys `1`–`4` plus clicks. SPACE must be inert while the card is up, or a
reflexive press gets swallowed as an answer.

`task_related` is the ambiguous middle; forcing it either way injects label
noise into a 2-class tree. Dropped by default, exposed as `?task_related=` so a
sensitivity analysis is cheap.

### 5.4 Labelling rule

For each row with `label = 'task'`, find the first *answered* probe at or after
the row's `elapsed`; if `probe.elapsed - row.elapsed <= windowSeconds` (default
15), inherit that probe's mapped verdict. Rows matching no probe, rows labelled
`probe`/`calibration`, and rows whose probe mapped to null are **omitted** from
`kind=training`.

At 90 ms ticks: ~167 rows per probe, **~1,600 labelled rows per session**.

Per probe: `{ probe_index, elapsed, response, latency_ms, trials_before }` —
`elapsed` is when the card *opened* (right-hand edge of the window).

---

## 6. DDL

New `web/api/schema_sart.sql` (additive, idempotent, safe against the live DB);
fold the same statements into `web/api/schema.sql` so a fresh DB gets everything.

```sql
-- sessions ------------------------------------------------------------
-- task_type is the anti-pooling guard: arithmetic and SART sessions must
-- never land in the same analyze_study.py pairing.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS task_type     TEXT NOT NULL DEFAULT 'arithmetic';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS task_version  TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS wall_seconds  NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probe_seconds NUMERIC;

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_trials         INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_targets        INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_go_trials      INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_dropped_trials INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sart_accuracy       NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS commission_errors   INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS commission_rate     NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS omission_errors     INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS omission_rate       NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_mean_ms          NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_sd_ms            NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rt_cv               NUMERIC;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probes_shown        INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS probes_answered     INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS mw_rate             NUMERIC;

CREATE INDEX IF NOT EXISTS sessions_task_type ON sessions (task_type);

-- sart_trials ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS sart_trials (
  id            BIGSERIAL PRIMARY KEY,
  session_id    TEXT     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  trial         INTEGER  NOT NULL,
  onset         NUMERIC  NOT NULL,
  digit         SMALLINT NOT NULL,
  is_target     BOOLEAN  NOT NULL,
  responded     BOOLEAN  NOT NULL,
  rt_ms         NUMERIC,
  outcome       TEXT     NOT NULL,
  extra_presses SMALLINT NOT NULL DEFAULT 0,
  UNIQUE (session_id, trial)
);
CREATE INDEX IF NOT EXISTS sart_trials_session ON sart_trials (session_id, onset);

-- probe_responses -----------------------------------------------------
CREATE TABLE IF NOT EXISTS probe_responses (
  id            BIGSERIAL PRIMARY KEY,
  session_id    TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  probe_index   INTEGER NOT NULL,
  elapsed       NUMERIC NOT NULL,
  response      TEXT,
  latency_ms    NUMERIC,
  trials_before INTEGER,
  UNIQUE (session_id, probe_index)
);
CREATE INDEX IF NOT EXISTS probe_responses_session ON probe_responses (session_id, elapsed);
```

The `UNIQUE` constraints make a retried POST idempotent.

---

## 7. API changes

### `web/api/submit.js`

- Add `MAX_TRIALS = 2000`, `MAX_PROBES = 50` beside `MAX_FEATURE_ROWS`.
- **Add `label_only`** to the condition whitelist at `submit.js:27`.
- Whitelist `task_type` ∈ {`arithmetic`, `sart`}; 400 otherwise.
- Extend the `sessions` INSERT list (`submit.js:38-48`).
- Insert trials and probes **after** the `inserted.rowCount === 0` early return
  at `submit.js:55-57`, chunked at 500, so a retry re-inserts nothing.

### `web/api/export.js`

Existing kinds keep their exact column lists and formatting.

| kind | columns | status |
|---|---|---|
| `study` | existing 11, unchanged formatting | **byte-compat preserved**; now filtered `WHERE task_type = ?`, default `arithmetic` |
| `study_sart` | behavioural columns + SART aggregates + `mw_rate` | new |
| `features` | existing 14, unchanged | **preserved**; `label` now carries provenance |
| `training` | **identical 14 columns**, `label` ∈ {`focused`,`distracted`}, unlabelled omitted | new — `train_model.py` consumes unmodified |
| `trials` | per-trial records | new |
| `probes` | per-probe records | new |

Two rules that matter:

1. `kind=training` must emit the **identical 14 columns** as
   `session_logger.py:18`. Do not add a `probe_response` column — it kills the
   byte-compat claim for no benefit. Probe detail belongs in `kind=probes`.
2. **Do not reuse** `attempted`/`correct`/`correct_per_min` for SART analogues.
   `analyze_study.py:68` does a bare `float(on)`, so a reused column silently
   pools two different constructs. Leave them NULL for SART and let `task_type`
   do the separating. This is the single most important anti-pooling guard here.

---

## 8. Python side

### `analyze_study.py`

The measure loop (`:63-84`) is already generic over the `MEASURES` tuple and a
`"higher"`/`"lower"` direction, so this is a second tuple plus task-type
selection.

| current | survives? | action |
|---|---|---|
| `correct_per_min` | no | **replace** with `commission_rate` (lower) |
| `accuracy` | near ceiling for SART | keep as `sart_accuracy` (higher), or drop |
| `seconds_away` | **yes** | keep (lower) |
| `times_left` | **yes** | keep (lower) |
| `self_rating` | **yes** | keep (higher) |
| — | new | **`rt_cv`** (lower) — validated mind-wandering marker |
| — | new | **`mw_rate`** (lower) — probe-reported off-task proportion |
| — | new | `omission_rate` (lower) |

`mw_rate` is the headline: a **direct** measure of mind-wandering rather than a
look-away proxy, which speaks straight to the "Limits" paragraph at
`web/index.html:114`.

**Bug to fix in the same commit.** `analyze_study.py:38`:

```python
sessions.setdefault(row["participant"], {})[row["condition"]] = row
```

Keyed only on participant and condition. Feed it a mixed CSV and an arithmetic
`alerts` session pairs against a SART `no_alerts` session, silently. Key on
`(participant, task_type)`, defaulting `task_type` to `arithmetic` when the
column is absent so existing desktop CSVs still work, and refuse to pair across
task types.

`_save_chart` is `figsize=(3.2 * len(summary), 4.2)` (`:100`) — 7 measures is
unreadably wide. Cap at 5, or use a 2-row grid.

### New `analyze_sart.py`

Keeps trial-level and probe-level analysis out of `analyze_study.py`. The
highest-value figure for the poster lives here: **the detector's distraction
score in the 15 s before `mind_wandering` probes versus before `on_task`
probes.** That is independent validation that probes and detector agree — much
stronger evidence than tree accuracy on self-labelled data.

### `focus_study.py` — do not port SART

1. Probes, per-trial logging and a second timing implementation would all need
   duplicating, with a different RNG and tkinter's `after()` (worse timing
   fidelity). The two paths wouldn't be comparable, so the parity claim this
   repo is otherwise careful about couldn't be honoured.
2. The desktop's unique value was being the only source of real
   focused/distracted ground truth. Probes now supply that.
3. Freezing it preserves the arithmetic CSVs and the byte-compat diff target.

Add a docstring note. **Cost:** SART needs the deployed site or `vercel dev`. If
collection may be offline, the CSV fallback in `done()` already works from a
locally served static site — extend those fallback downloads to cover trials and
probes.

---

## 9. Parity / selftest

All 28 existing checks are untouched. Add ~20 in new sections, with fixtures
under new `sart` and `labels` keys in `web/data/parity.json`.

**7. RNG lock** (regression, not Python parity — there is no Python SART):
first 8 outputs for `P01-alerts-sart`; arithmetic's first 10 problems for
`P01-alerts` (**add this one before extracting `rng.js`**).

**8. Trial sequence:** 450 trials for a fixed seed — exact target count, rate in
[0.10, 0.125], every digit in 1–9, no two targets adjacent, first 20 digits match.

**9. Outcome and aggregate math:** the `(is_target, responded) → outcome` truth
table as data (4 checks), plus a ~10-trial fixture with expected
`commission_rate`, `omission_rate`, `rt_mean_ms`, `rt_sd_ms` (n−1), `rt_cv`.
**Generate those five with NumPy once and paste them** — genuine Python parity
for the only formulas that matter, and it pins the n−1 convention.

**10. Label derivation** against fixture rows + probes: inside window → labelled;
outside → omitted; `probe`/`calibration` rows → omitted; `task_related` under all
three mappings; unanswered probe → omitted; equidistant row takes the *later*
probe.

Because `api/export.js` imports the same `labels.js`, these exercise production code.

---

## 10. Work sequence

| # | Work | Files |
|---|---|---|
| 1 | Pin arithmetic's sequence in parity, **then** extract `rng.js` | `parity.json`, `selftest.html`, `study/rng.js`, `study/arithmetic.js` |
| 2 | `sart.js` **pure logic only** (sequence, outcomes, aggregates), no DOM | `study/sart.js` |
| 3 | `labels.js` pure derivation + fixtures | `lib/labels.js`, `selftest.html`, `parity.json` |
| 4 | **flow.js harness refactor** — introduce the seam, keep arithmetic working *through it*, run one session end-to-end, **then** default to SART | `study/flow.js` |
| 5 | SART DOM, keyboard, throttle→`dropped`; remove the `focus` listener | `study/sart.js`, `study/flow.js` |
| 6 | Probe scheduler, card, suspend/resume, clock pause, `tick("probe")` | `study/probes.js`, `study/flow.js` |
| 7 | CSS: large digit, mask, probe card | `study.html` |
| 8 | Payload: `task_type`, `trials[]`, `probes[]`, aggregates; CSV fallbacks | `lib/upload.js`, `study/flow.js` |
| 9 | DDL migration against the live DB | `api/schema_sart.sql`, `api/schema.sql` |
| 10 | `submit.js`: columns, inserts, caps, `label_only` + `task_type` whitelists | `api/submit.js` |
| 11 | `export.js`: new kinds, `task_type` filter, `labels.js` import; **verify `vercel build` traces it** | `api/export.js` |
| 12 | Python: `analyze_study.py` task-awareness + measures + pairing fix; `analyze_sart.py` | `analyze_study.py`, `analyze_sart.py` |
| 13 | Docs: README routes + study section, `study.html` header, `consent.json` **version bump**, `index.html` Limits | several |
| 14 | **Pilot on yourself**: 2 sessions, then `?kind=training` → `train_model.py`. Confirm `kind=study` still diffs clean against a desktop CSV | — |

Steps 1–3 are pure and independently testable. Step 4 is the risky one — its own
commit, with arithmetic still working at the end of it.

---

## 11. Risks

1. **Label yield.** ~9–10 probes/session, and a participant can still yield zero
   `distracted` probes; `train_model.py:97` exits on a single class. Expect to
   need several participants before `kind=training` is usable.
2. **Tab throttling** silently converts away-time into omission errors unless the
   `dropped` detection lands. Easy to forget; it is the difference between a
   clean measure and a duplicate of `seconds_away`.
3. **The flow.js refactor** touches the one function owning both timers.
   Mitigate by keeping arithmetic runnable through the new seam.
4. **Vercel import tracing** of `../lib/labels.js` — verify, fallback in §3.3.
5. **Rule drift** — do not reintroduce a SQL or client-side copy of the labelling rule.
6. **`performance.now()` quantisation** — do not over-claim RT precision.
7. `MAX_FEATURE_ROWS = 20000` stays comfortable: ~6,700 rows per 10 min, ~7,300
   with probe pauses.

---

## 12. Protocol changes — needs sponsor / SRC attention

These are **experimental-protocol changes, not code changes**:

1. **Task substitution: mental arithmetic → SART.** Different cognitive demand
   and fatigue profile. `consent.json`'s "timed computer tasks" wording covers
   it, but the ISEF research plan almost certainly names mental arithmetic.
2. **New category of participant-reported data: thought probes about
   attentional state.** A mind-wandering self-report is arguably more personal
   than an arithmetic score. Update the `consent.json` summary and **bump
   `version`** — the DB is empty, so no existing receipts become unattributable.
3. **New `label_only` session** — a third ~10 min session per participant,
   changing the per-participant time commitment.
4. **Interruption every ~60 s** — largely covered by the existing risk line about
   "frustration from alerts or automatic pauses", but worth naming.
5. **Session length** — pausing the clock for probes pushes total time slightly
   past the advertised 40–45 min, plus the third session.

---

## 13. Expected performance — calibrate before collecting

Published webcam mind-wandering detection reaches **F1 ≈ 0.40**. One MOOC study
found webcam features (0.40) essentially matched a dedicated eye tracker (0.41),
only 11–16% above chance. Positive-class F1 across this literature runs
~0.25–0.41, with low base rates (~17% in one study).

So a detector reporting 90% accuracy almost certainly indicates class imbalance
or label leakage, not success. **Beating F1 0.40 on probe-labelled data would be
a genuinely strong result.** Report F1 on the positive class, not accuracy.

### Secondary data

Optional external validation: run MediaPipe over public engagement datasets to
extract the same 12 features.

| Dataset | Scale | Catch |
|---|---|---|
| DAiSEE | ~9,068 clips, 112 subjects | Crowdsourced labels, quality questioned; verify access terms |
| EngageNet | ~31 h, 127 participants | Lecture-watching, not vigilance |
| EngageWild | 263 videos, 91 subjects | Small |

### Non-circular auto-labelling (optional, separate from probes)

Labels and features must come from **different sensors**, or the model just
re-learns the rule that made the labels:

- **Labels** (behavioural): `task_tab_active`, `tab_switches_per_min`,
  `time_on_current_tab`, `keyboard_inactivity`, `mouse_inactivity`
- **Features** (camera only): `head_pitch`, `head_yaw`, `iris_delta`,
  `eye_openness`, `face_visible`, `phone_present`

This fills the empty `camera_only` baseline slot in `web/data/validation.json`
and scales to every tick. The heuristic tree in `distraction_model.py:114-133`
stays a **baseline to beat**, never a label source.
