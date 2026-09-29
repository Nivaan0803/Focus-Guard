# AI Webcam Focus Tracker

This is a Windows desktop focus-session program. It watches the active window title, your head pose and gaze (webcam Face Mesh), keyboard/mouse activity, and whether a phone is on camera, fuses them through a decision tree, and warns you when you have been distracted longer than a delay you choose. Frames and window titles are processed in RAM and are never saved.

**Architecture:** `main.py` runs a light Tk loop. `vision_worker.py` runs the camera + MediaPipe + phone detector on one background thread and the screen grab on another. `distraction_model.py` holds the decision-tree classifier (built-in heuristic, or a trained scikit-learn tree). `session_logger.py` / `train_model.py` handle optional data logging and training. `settings.py` persists preferences.

## Run

```powershell
python -m venv .venv
```
To run
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The main window shows only what you need to run a session: the task phrase, the alert delay, the status banner, a distraction meter, the previews, and Start / Calibrate / Pause. Everything else — webcam on/off, camera number, screen preview, sensitivity, re-alert interval, calibration length, alarm tone, and data logging — lives behind the **⚙ Settings** button. A collapsible **Live signals** panel at the bottom shows the raw feature values. Only on windows.

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
