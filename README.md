# AI Webcam Focus Tracker

This is a Windows desktop focus-session program. It watches the active window title, your head pose and gaze (webcam Face Mesh), keyboard/mouse activity, and whether a phone is on camera, fuses them through a decision tree, and warns you when you have been distracted longer than a delay you choose. Frames and window titles are processed in RAM and are never saved.

**Architecture:** `main.py` runs a light Tk loop. `vision_worker.py` runs the camera + MediaPipe + phone detector on one background thread and the screen grab on another. `distraction_model.py` holds the decision-tree classifier (built-in heuristic, or a trained scikit-learn tree). `session_logger.py` / `train_model.py` handle optional data logging and training. `settings.py` persists preferences.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The main window shows only what you need to run a session: the task phrase, the alert delay, the status banner, a distraction meter, the previews, and Start / Calibrate / Pause. Everything else — webcam on/off, camera number, screen preview, sensitivity, re-alert interval, calibration length, alarm tone, and data logging — lives behind the **⚙ Settings** button. A collapsible **Live signals** panel at the bottom shows the raw feature values.

The webcam and screen previews run on their own background threads, so the UI stays responsive. If the camera shows the wrong device, change the **Camera number** in Settings — the worker reconnects automatically, no restart needed.

1. Open the task tab you want to protect and notice a distinctive phrase in its title—for example, `Research Essay - Google Docs`.
2. Run the program and enter enough of that phrase to uniquely identify the tab, such as `Research Essay`.
3. Set the delay, then click **Start focus session** (or press **F5**).
4. Work in that tab. If the fused distraction score stays high past the delay, the status turns red and the program plays a beep.

Your settings (task phrase, delay, camera, volume, sensitivity, window size) are saved to `settings.json` next to the app and restored next launch. That file holds preferences only — never frames, screenshots, or titles.

**Keyboard shortcuts:** F5 start/stop · F6 pause alerts (5 min) · F7 calibrate gaze · F8 flip the data-logging label.

**Calibrate gaze (F7):** with the webcam on, press F7 and keep working normally while looking at your screen—not at the camera. It collects a baseline (default 10 s, adjustable) and, from the *spread* of that baseline, sets look-away thresholds tuned to how still you sit. The **Sensitivity** slider scales those thresholds. Recalibrate whenever you move the laptop or camera. This is an estimate of screen-facing posture, not a measurement of thoughts or productivity.

**Smoothing, pause, re-alert:** per-frame predictions feed a leaky "distraction score" (the amber bar) so a single noisy frame never alerts. **Pause alerts** (F6) suppresses alerts for 5 minutes for a legit break. **Re-alert every N s** nudges again if you stay distracted (0 = alert once per episode). When a session ends you get a summary (focused %, longest streak, what caused the alerts); tick **Save session summaries** to also append them to `session_history.csv`.

The app also includes a local EfficientDet object detector for **cell phones**. A red bounding box and `PHONE DETECTED` label appear when it sees one; it only beeps after the phone remains visible for the selected delay. The model file is included at `models/efficientdet_lite0.tflite` and runs on-device. Very dark scenes, glare, an obscured phone, or a phone facing away from the camera can reduce accuracy.

## Decision-tree distraction model

Every tick the app now builds one feature vector from all of its signals — task-tab
active, tab switches/min, time on current tab, keyboard/mouse inactivity, whether gaze
is calibrated, face visible, head pitch/yaw vs. your neutral pose, iris offset, eye
openness, and phone present — and passes it through a single decision tree that outputs
`focused` or `distracted` plus a short reason. That one decision drives the alert
(after the grace period), instead of three separate threshold checks. The current
model and reason are shown in the metrics panel.

Out of the box it uses a small built-in heuristic tree that reproduces the original
thresholds, so no setup is required. To train your own tree on real data:

1. `pip install -r requirements.txt` (installs `scikit-learn` + `joblib`).
2. Run the app, start a focus session, click **Start logging**, choose a CSV file, and
   set the label (`focused` / `distracted`) to match what you are actually doing.
   Change the label whenever your state changes. Rows are only written while a session
   runs. Logging is off by default and never happens without this explicit action.
3. Collect a few sessions across different lighting/desk setups, then:
   `python train_model.py focus_session*.csv`
   This prints a cross-validated accuracy and the learned rules, and writes
   `models/distraction_tree.joblib`.
4. Restart the app. It loads that file automatically; the metrics panel shows
   "Model: trained decision tree". Delete the file to return to the heuristic.

## Engineering-project write-up

**Engineering problem / design question:** How can an on-device webcam program give a useful focus reminder after a user looks away for a configurable time, while avoiding alerts for brief natural breaks?

**Background:** Machine learning identifies patterns in data rather than requiring a separate rule for every image. This design uses MediaPipe Face Mesh, a pre-trained computer-vision model, to locate facial landmarks. OpenCV estimates head orientation from 3-D facial reference points and their 2-D image positions. Unlike website blockers, this system measures an observable proxy for attention—whether the person's face points toward the screen. It does not determine thoughts, productivity, or actual attention. Its difference is privacy: images remain local and are discarded after each frame.

**Design goal / prediction:** With a calibrated neutral pose, a 25-degree look-away threshold, and a 3-second grace period, the prototype will detect sustained intentional turns away while producing few alerts during a 2-minute focused session.

**Engineering rationale:** Calibration accounts for each user's posture. The grace period prevents a quick glance or posture change from producing an alert. Face Mesh is lightweight and avoids uploading video.

| Test category | Change (independent/design parameter) | Measure (dependent/performance metric) | Keep controlled |
| --- | --- | --- | --- |
| Sensitivity | Angle: 15, 25, 35 degrees | Recall: planned look-aways correctly alerted / planned look-aways | Camera, lighting, person, 3 s grace |
| Alert timing | Grace period: 2, 3, 5 seconds | Mean alert delay and false-alert count | Camera, 25-degree angle, task |
| Usability | Settings above | User rating and unwanted alerts | Same 10-minute task/environment |

**Procedure:** (1) Position camera and calibrate. (2) Run a 2-minute focused baseline and count alerts. (3) At each angle, perform 10 trials: screen for 5 seconds, look away for 5 seconds, return. Record alert/no alert and delay. (4) Repeat the full test three times. (5) Compare recall, false alerts, and average delay, then choose the best trade-off.

**Limits and ethical use:** Looking away is only a proxy; it can misclassify note-taking, legitimate breaks, accessibility-related movement, or off-screen work. It should be voluntary self-management, never surveillance or discipline. Phone detection runs a separate, independently trained object-detection model (EfficientDet-Lite0) rather than being inferred from face landmarks, and its own accuracy limits (dark scenes, glare, an obscured or face-down phone) are noted above — its errors do not carry over into the gaze/pose accuracy figures below.

## Validating the model (feature importance, confusion matrix, cross-condition accuracy)

`train_model.py` now reports out-of-fold accuracy — each row is scored by a
cross-validation fold that never trained on it, so the number reflects
generalisation rather than the tree simply memorising its own training data.
It also saves two figures to `models/`:

* `feature_importance.png` — which signal the tree actually leans on.
* `confusion_matrix.png` — the out-of-fold confusion matrix.

To characterise the model across conditions (lighting, person, desk setup)
rather than a single mixed dataset, log one CSV per condition, then run:

```powershell
python evaluate_model.py bright=logs/bright*.csv dim=logs/dim*.csv person2=logs/person2*.csv
```

This prints per-condition accuracy/precision/recall for both the built-in
heuristic tree and the trained tree, reports how often the two models agree
(a quick ablation of whether training on real data actually changed the
decision boundary), and saves `models/condition_comparison.png`.

Every alert a trained tree raises now also names the exact split that fired
it — e.g. `iris_delta 0.24 > 0.18` — instead of a generic "trained model"
label, shown live in the metrics panel's "Model state" row. Ticking **Save
session summaries** also saves a `session_charts/session_<timestamp>.png`
line chart of the distraction score over the session with alert markers, and
the end-of-session summary reports how often the trained and heuristic
models agreed and how many times a mid-session posture-drift prompt fired
(the app compares resting head pose during "focused" ticks against the
calibrated neutral pose and suggests recalibrating (F7) if it has drifted).
`matplotlib` is optional — install it to get the PNGs; everything else works
without it.

### Does it work on people it has never seen? (leave-one-person-out)

Ordinary k-fold cross-validation is optimistic here: neighbouring 1/8-second
ticks from the same session are almost identical, so near-copies of the test
rows end up in training. Name each CSV group after the person it came from and
`train_model.py` also runs a leave-one-person-out test. It trains on everyone
except one person, scores that person, and repeats for each person:

```powershell
python train_model.py p01="logs/p01_*.csv" p02="logs/p02_*.csv" p03="logs/p03_*.csv"
```

It prints per-person accuracy/precision/recall and saves
`models/leave_one_person_out.png`. Both `main.py` logs (`label` column) and
`run_trial.py` logs (`ground_truth` column) are accepted.

### Does each signal earn its place? (baselines)

`evaluate_model.py` also pools all the data and compares the full model with
single-signal versions of the heuristic tree: **phone only**, **tab only**, and
**camera only**. It saves `models/baseline_comparison.png`. If the full model
beats every baseline, combining signals is justified. Note that `run_trial.py`
fixes the tab signals to "on task", so a fair *tab only* number needs
`main.py` logs that include real tab switching.

## Does real-time feedback actually help? (user study)

`focus_study.py` is a timed mental-arithmetic task that measures focus
independently of the detector: correct answers per minute, accuracy, seconds
spent outside the task window, how many times the participant left it, and a
1-10 self-rating. Each participant does it twice, once with Focus Guard
alerting (task phrase `Focus Study`) and once without. Odd-numbered
participants do *alerts* first and even-numbered ones do *no alerts* first, so
practice and fatigue cancel out. The full protocol is in the script's docstring.

```powershell
python focus_study.py --participant P01 --condition alerts
python focus_study.py --participant P01 --condition no_alerts
python analyze_study.py
```

`analyze_study.py` compares each person with themselves. It reports the mean
change and a paired t-test p-value for every measure, and saves
`study/study_results.png`, where each grey line is one person. Use anonymous
IDs and get each participant's consent (and a parent's, if under 18), as your
fair's human-subjects rules require.

## Booth demo mode (F9)

Press **F9** (or **▶ Demo mode**) for a full-screen view you can read from
across a room. It shows a giant FOCUSED / CHECKING / DISTRACTED banner, the
exact reason the model gave, live head pose, gaze, phone and window signals,
and the last 60 seconds of the distraction score with the alert threshold. It
starts a session automatically if one isn't running. **F7** calibrates, **F5**
starts/stops, **Esc** exits.

By default it records nothing. Tick **Record booth demos (F9) to `demo_logs/`**
in Settings → Data & model and each demo also leaves a paper trail, so a run
with a visitor at the booth becomes evidence instead of a memory:

| File | One row per | Holds |
| --- | --- | --- |
| `demo_logs/demo_<timestamp>.csv` | half-second of the demo | the twelve features the tree saw, its verdict, the smoothed score, the reason it named |
| `demo_logs/demo_index.csv` | demo | start time, length, tick count, share of distracted ticks, alerts fired, most common reason |

The footer of the demo view names the file it is writing while recording is on.
Still no webcam frames, screenshots or window titles — the same numbers the
decision tree sees, nothing more. The per-row `label` is always `unlabelled`,
because a visitor's demo has no ground truth: this data describes how the
detector behaved across many faces and lighting conditions, it is not training
data. (Explicitly labelling it also stops `train_model.py` from reading these
rows as "focused" if they ever get globbed in by mistake.) Pass the index to a
spreadsheet for booth statistics — "n demos, median length, alert rate" — or a
single demo CSV to the same plotting as a session log.

## Web app: participant data collection (`web/`)

The desktop app cannot be deployed to a server — it needs a local webcam, the
Windows active-window title, global input hooks, and a Tk window. But
participants should not have to install Python to take part in the study, so
`web/` is a **static site** that runs the detector in the participant's own
browser via MediaPipe Tasks for Web, pinned to `0.10.21` — the same MediaPipe
version as `requirements.txt`.

| Route | What it is |
| --- | --- |
| `/` | Showcase: how the fusion works, what the browser can and cannot see, results |
| `/study` | The data gatherer: consent + signature → calibrate → SART + thought probes → upload |
| `/demo` | Booth demo: the live detector, recording nothing |
| `/selftest` | Asserts the JavaScript ports match the Python, against values in `data/parity.json` |

No Python runs on the server, and no video, screenshot, or window title ever
leaves the participant's device — only the derived numbers.

### The task: SART, not arithmetic

`/study` runs a **Sustained Attention to Response Task**. Digits 1–9 appear one
at a time (250 ms, then a 900 ms mask); the participant presses space for every
digit *except* 3. Pressing on a 3 is a **commission error**, the standard
operational marker of an attention lapse.

It replaced mental arithmetic for two reasons. Arithmetic keeps people engaged,
which is the opposite of the state the detector is meant to catch; and it
produced ~40 scored items per session with no per-item timing, against the
SART's ~460 trials each carrying a reaction time. `?task=arithmetic` still
selects the old task.

### Thought probes, and why they matter more than the task change

Every ~60 seconds (jittered 45–75 s, so it cannot be anticipated) the task
pauses and asks where the participant's attention actually was. This is the
**only ground truth the web study has.**

Before probes, every feature row's `label` held the experimental condition, so
`train_model.py` would have learned to predict which arm a session was, not
whether anyone was distracted. The desktop app gets real labels from an
operator flipping a toggle (`main.py:380-381`); a remote participant has no
operator. A probe makes the participant their own annotator.

Answers back-label the preceding 15 s of feature rows, derived at **export
time** by `web/lib/labels.js` — so the window length and the handling of the
ambiguous "thinking about the task" answer stay tunable by re-downloading
rather than re-collecting.

| Condition | Camera | Alerts | Probes | Role |
| --- | --- | --- | --- | --- |
| `alerts` | yes | yes | yes | A/B arm |
| `no_alerts` | no | no | yes | A/B arm |
| `label_only` | yes | silent | yes | training data only |

`label_only` sits outside the counterbalanced comparison so training labels are
not collected while the system is actively beeping at the participant — the
very intervention the study measures.

Design notes and the full rationale: `docs/sart-redesign.md`.

### Consent gate

`/study` opens on the ISEF informed-consent form
(`web/consent/isef-informed-consent-v1.pdf`), captures a drawn signature
(plus a parent's, for under-18s), and blocks the camera and all recording until
it is signed. The signed PDF is generated **in the browser** and downloads to
that device; only a non-identifying *receipt* (participant code, form version,
timestamp, signature hash) is uploaded. The dataset therefore stays
de-identified while a reviewer can still confirm that a given code consented to
a given form version.

ISEF requires the researcher to keep the original signed form, so run sessions
on the researcher's own laptop where the download lands, or collect the PDFs
from remote participants by hand.

Edit `web/consent/consent.json` to change the wording, and **bump its `version`
whenever the form changes** — that string is stored with every receipt.

### Parity with the desktop app

These are direct ports, and `/selftest` checks them:

| Ported from | To |
| --- | --- |
| `distraction_model.py` `HeuristicTree`, `_decision_reason` | `web/lib/tree.js` |
| `vision_worker.py` `iris_position`, `eye_openness` | `web/lib/features.js` |
| `vision_worker.py` `_advance_calibration` | `web/lib/calibrate.js` |
| `main.py:605-613`, `main.py:670-683` | `web/lib/score.js` |
| `focus_study.py` `make_problem` and the task loop | `web/study/` |

Two deliberate differences, both documented in the source: head pose comes from
MediaPipe's facial transformation matrix rather than `cv2.solvePnP` (valid
because the feature is a delta from the calibrated neutral pose, but **verify
the scale side by side before collecting data**), and the problem sequence uses
a different PRNG, so browser and desktop study rows should not be pooled as
though participants saw identical problems.

### Exported data drops into the existing scripts

`/api/export` emits CSVs byte-compatible with the desktop writers:

```
GET /api/export?kind=study       -> focus_study.py:45-46  -> analyze_study.py
GET /api/export?kind=study_sart  -> the SART session measures -> analyze_study.py
GET /api/export?kind=training    -> session_logger.py:18  -> train_model.py
GET /api/export?kind=features    -> session_logger.py:18  (raw, provenance labels)
GET /api/export?kind=trials      -> one row per digit     -> analyze_sart.py
GET /api/export?kind=probes      -> one row per probe     -> analyze_sart.py
GET /api/export?kind=consent     -> the consent registry
GET /api/export?kind=participants-> what has been collected so far
```

Authenticate with `Authorization: Bearer $EXPORT_TOKEN`.

**`kind=training` is the one to train on.** It emits the same 14 columns as
`session_logger.py:18`, but `label` holds a real `focused`/`distracted` verdict
derived from the probes. `kind=features` emits the raw rows, where `label` is
provenance (`task`, `probe`, `calibration`) and not a verdict — useful for
inspection, useless for training.

Two defaults will look like bugs if you have not read them. `kind=study`
returns **arithmetic** sessions only, because a SART session has no `attempted`
or `correct` and `analyze_study.py:68` would `float()` its way into nonsense;
pass `?task_type=sart|all`. `kind=training` returns **`label_only`** sessions
only, for the confound described above; pass `?condition=all` to accept it. An
empty training file usually means no `label_only` session has been run yet —
`kind=participants` reports the counts that tell you so.

Tuning knobs, applied at export time rather than at collection time:
`?window=15` (seconds a probe describes) and `?task_related=drop|focused|distracted`.

### Checking the probes are measuring anything

```powershell
python analyze_sart.py study/sart_trials.csv study/probe_responses.csv
```

The probes are self-reports, and nothing forces them to be true. If they are
meaningful, lapses and response variability should already be elevated in the
seconds *before* an off-task report. That prediction does not involve the
detector, so it validates the labels without assuming what the project is
trying to show — and if it fails, no detector accuracy computed from those
labels means anything.

**Expect modest numbers.** Published webcam mind-wandering detection reaches
F1 ≈ 0.40 on the positive class, barely above chance. A detector reporting 90%
accuracy has almost certainly found the class imbalance, not the signal. Report
F1 on the distracted class, not accuracy.

### Deploying

Import the repo at vercel.com → Framework Preset **Other** → **Root Directory
`web`** → Build Command empty. Root Directory `web` is what keeps the `.py`
source from being served as static text.

The database is optional: without one, `/api/submit` returns a clear
"not configured" and the participant can download their session as CSV instead.
To enable it, add Vercel Postgres in the Storage tab, run `web/api/schema.sql`,
set `EXPORT_TOKEN`, and redeploy.

Test locally with `python -m http.server 8000 --directory web`
(`localhost` counts as a secure context, so the camera works; the `/api` routes
will not run without the Vercel CLI, and the CSV fallback covers that).

### Optional: use a trained tree in the browser

```powershell
python train_model.py focus_session*.csv   # writes models/distraction_tree.joblib
python export_tree_json.py                 # writes web/data/distraction_tree.json
```

Without that file the web app uses the built-in heuristic tree, exactly as the
desktop app does.
