# Generative AI Use Disclosure — FocusGuard

**Project:** FocusGuard: An Analysis of Real-Time Visual Distraction Detection
Through Webcam-Based Gaze and Facial Landmark Tracking Using Heuristic and
Decision Trees

**Prepared:** 2026-11-17

---

## Note on redaction

Per the disclosure instructions, personally identifying information has been
removed from this document. The following substitutions were made throughout:

| Redacted as | What it replaces |
| --- | --- |
| `[student researcher]` | the student researcher's name |
| `[adult sponsor]` | the adult sponsor's name |
| `[sponsor email]` | the adult sponsor's work email address |
| `[project directory]` | local filesystem paths containing a Windows username |
| `[repository]` | the GitHub account and repository path |
| `[deployment URL]` | the public URL of the deployed study site |

No other content has been altered. Technical detail is reproduced in full.

## Scope and limitations of this document

**This document covers one working session**, conducted on 2026-11-09 through
2026-11-17. It is complete and accurate for that session.

It does **not** cover any AI use before that session. When this session began,
a browser-based study application already existed in the project's `web/`
directory, uncommitted to version control. This document makes no claim about
how that earlier code was produced; that is for the student researcher to
disclose separately if applicable.

**This is a reconstructed summary, not a verbatim export.** The AI tool used
does not provide a downloadable prompt log. The disclosure instructions permit
a summary in that case. Every prompt listed in Part B is reproduced as the
student researcher typed it, including spelling errors, because those are the
actual prompts. The responses are summarised rather than quoted, because the
full responses run to many thousands of words.

---

# Part A — Generative AI used, and where

## Tool

| | |
| --- | --- |
| **Tool** | Claude (Anthropic), accessed through Claude Code, an agentic command-line interface |
| **Model** | Claude Opus 5 |
| **Mode of use** | Agentic. The tool did not only produce text for the researcher to copy. It directly read and wrote files in the project, ran shell commands, executed tests, queried the project database, operated a web browser, and deployed the web application. |
| **Period** | 2026-11-09 to 2026-11-17 |

This distinction matters for honest disclosure: the AI was not used as a
reference or a writing aid. It performed software engineering work on the
project directly, under the researcher's direction.

## What the AI did, by project element

### 1. Experimental task design — AI proposed, researcher selected

The researcher asked for a replacement for the original mental-arithmetic task,
stating that arithmetic was unlikely to produce the distraction the detector is
meant to catch. The AI proposed three alternatives with trade-offs (a SART
vigilance task, reading comprehension with probes, and a proofreading task) and
recommended the SART. **The researcher chose.** The AI also proposed adding
periodic thought probes; the researcher accepted.

Later, the researcher asked which task would best elicit *detectable*
distraction. The AI argued that mind-wandering during a SART is frequently
invisible to a webcam, and recommended that the training-data block use a
naturalistic task instead. This recommendation is documented but **was not
implemented** during this session.

### 2. Study software — written by the AI

New files, authored by the AI:

| File | Purpose |
| --- | --- |
| `web/study/sart.js` | SART sequence generation, outcome classification, aggregate statistics |
| `web/study/sart-task.js` | SART presentation, timing, keyboard handling |
| `web/study/probes.js` | Thought-probe scheduling and the probe card |
| `web/study/rng.js` | Seeded random number generation, extracted from existing code |
| `web/study/arithmetic-task.js` | Existing arithmetic task, adapted to a new interface |
| `web/lib/labels.js` | Converts probe answers into focused/distracted training labels |
| `web/api/schema_sart.sql` | Database migration for trial and probe storage |

Existing files substantially modified by the AI:

| File | Change |
| --- | --- |
| `web/study/flow.js` | Refactored to separate the timing harness from the task |
| `web/lib/upload.js` | Extended to carry trial, probe and SART data |
| `web/api/submit.js` | Rewritten to store the new data |
| `web/api/export.js` | Rewritten; added the labelled training export |
| `web/api/_db.js` | One-line fix (empty values were stored as zero instead of null) |
| `web/selftest.html`, `web/data/parity.json` | Extended from 28 to 78 automated checks |
| `web/study.html` | Styling for the new task and probe card |

### 3. Analysis and algorithm code — written by the AI

| File | Purpose |
| --- | --- |
| `autolabel.py` | Derives training labels from window/input telemetry so a camera-only model can be trained without hand annotation |
| `analyze_sart.py` | Tests whether thought-probe answers predict task behaviour |
| `analyze_study.py` | *(modified)* Made aware of two task types; a session-pairing defect was fixed |
| `fetch_study_data.py` | Downloads collected data from the deployment |
| `compare_pose.py` | Compares the browser's and the desktop's head-pose estimation |
| `make_consent_pdf.py` | Generates a printable consent form from the consent configuration |
| `test_autolabel.py`, `test_analysis.py` | Automated tests (54 tests) |

**Not written or modified by the AI** (pre-existing project code): `main.py`,
`vision_worker.py`, `distraction_model.py`, `train_model.py`,
`evaluate_model.py`, `session_logger.py`, `settings.py`, `focus_study.py`,
`demo_recorder.py`, `session_chart.py`, `run_trial.py`, `export_tree_json.py`,
`plot_demo_log.py`, and the pre-existing browser modules
(`web/lib/consent.js`, `pdf.js`, `features.js`, `tree.js`, `calibrate.js`,
`score.js`, `vision.js`, `beep.js`, `web/study/arithmetic.js`).

**No machine-learning model in this project was trained on AI-generated
research data.** The AI generated synthetic data solely to test that the
software pipeline functioned, and deleted it afterwards. One model file trained
on that synthetic data was accidentally left in the project and was identified
and removed by the AI before any use.

### 4. Infrastructure and deployment — performed by the AI

The AI installed Node.js and the Vercel command-line tool, created and deployed
the hosting project, provisioned a PostgreSQL database, applied the schema,
configured an access token, and diagnosed repeated deployment failures caused
by a misconfigured root-directory setting.

**The researcher performed all authentication personally.** The AI declined to
sign in to any account on the researcher's behalf.

### 5. Participant-facing consent text — drafted by the AI, not approved

The AI rewrote `web/consent/consent.json` so the on-screen text accurately
describes the new task, the thought probes, the revised session count and the
revised time estimate, and added boredom as a stated risk.

It also detached the previous consent PDF from the deployment, having found
that the site embeds the attached form into each participant's signed document
— so an outdated form would have been bound into signed consents.

The AI generated `docs/isef-informed-consent-v2-DRAFT.pdf`, **marked DRAFT and
not for use with participants on every page.**

**No consent document produced by the AI has been reviewed or approved by the
adult sponsor or the Scientific Review Committee, and no participant data had
been collected at the time of writing.** The deployed site displays a notice
stating that the official form is not yet attached.

### 6. Documentation — written by the AI

`docs/sart-redesign.md` (design rationale), sections of `README.md`, the
"Limits" paragraph on the public project page, and this disclosure document.

## Scientific judgements the AI contributed

These are recorded because they influenced the study design, not merely the code.

1. **A circularity defect was identified.** The web application stored the
   experimental condition in the label column, so a model trained on that data
   would have learned to predict which experimental arm a session belonged to,
   rather than whether a participant was distracted. Thought probes were added
   to supply genuine labels.

2. **A non-circular auto-labelling method was proposed.** Labels derived from
   window and keyboard telemetry, used to train a model on camera features
   only, so no column appears on both sides.

3. **An expected-performance benchmark was supplied.** Published webcam-based
   mind-wandering detection reaches an F1 score of approximately 0.40. The AI
   advised repeatedly that a high accuracy figure would more likely indicate
   class imbalance than success, and that F1 on the positive class should be
   reported instead.

4. **A behavioural validation of the probes was proposed and implemented**
   (`analyze_sart.py`): if probe answers are meaningful, task performance
   should already have degraded before an off-task report.

5. **An unverified assumption was identified.** The browser and the desktop
   application compute head pose by different methods. A constant offset
   cancels through calibration, but the *scale* does not, and the decision tree
   compares against fixed thresholds. `compare_pose.py` was built to measure
   this. **It had not been run at the time of writing.**

6. **A confound was identified and avoided.** Training labels collected while
   the system was actively alerting the participant are confounded by the
   intervention being measured. A separate non-comparative session type was
   added.

## Limitations the researcher should be aware of

- All automated test results reported in this session were obtained using
  **synthetic data generated by the AI**. They demonstrate that the software
  works, not that the detector works.
- No human participant data existed at any point during this session.
- `compare_pose.py` has not been run; the head-pose scale question is open.
- The naturalistic-task recommendation was not implemented.
- The consent form is not approved.

---

# Part B — Prompt log (summary)

Prompts are reproduced exactly as typed. Responses are summarised. Entries are
chronological.

### Deployment and hosting

| # | Prompt | Response summary |
| --- | --- | --- |
| 1 | `where is the appp ud eployed` | Searched the project; reported that nothing was deployed, that a hosting configuration existed for the `web/` directory, and that the desktop application cannot be hosted. |
| 2 | `yes` | Inspected `web/` for participant data before committing (none found; the bundled consent PDF was confirmed blank). Flagged that the repository is public and that the consent configuration contains the sponsor's work email; asked how to proceed. Committed on the researcher's instruction. |
| 3 | `k hw to open the app` | Started a local web server and opened the application; confirmed the parity self-test passed 28/28. |
| 4 | `this is local but i want to collect data with other p;eopkle` | Explained that remote collection requires a database, and that the signed consent PDF downloads to the participant's device rather than reaching the researcher. |
| 5–8 | Three pasted deployment errors (`PYTHON_ENTRYPOINT_NOT_FOUND`), and `which appicatiopn preset do i pick` | Diagnosed the host building the repository root instead of `web/`, finding the desktop entry point. Advised `Other` as the framework preset. |
| 9 | `can u d t` | Found Node.js was not installed; installed it and the deployment tool. |
| 10–12 | Pasted shell errors; `! vercel login` | Corrected a command-prefix misunderstanding. Declined to perform the account sign-in; started the authentication flow for the researcher to complete personally. |
| 13 | `The specified Root Directory "\web" does not exist` | Identified a backslash in the setting; later identified the same symptom as a path-handling defect in the deployment tool on Windows. |
| 14 | `can u make it be able to store data.` | Asked who should accept the database provider's terms, since that is a binding agreement. On instruction, provisioned a PostgreSQL database, applied the schema, configured the export token, and verified a full write-and-read cycle. |
| 15 | `yes` | Added a message to the completion screen asking participants to email their signed consent form. Corrected an address-encoding defect in the generated link. |

### Study redesign

| # | Prompt | Response summary |
| --- | --- | --- |
| 16 | `~find a way of testing something like not math but smth more creative that will guive me more data based on distractions so i can get more data, maybe like testing how focused u r` | Examined the existing task and data pipeline. Discovered that feature rows carried the experimental condition as their label, making the data unusable for training. Proposed three replacement tasks and thought probes; the researcher selected the SART and accepted probes. Proposed a control-condition change; this was later revised after the AI found a documented argument against it in the project's own source, and the researcher chose a third option instead. |
| 17 | `can you also make a algorithm that collects data by itself using face landmarks online 2ndaery research to collect more data?` | Warned that labelling from face landmarks and then training on face landmarks is circular and produces a meaningless accuracy figure. Proposed cross-modal weak supervision instead. Searched published literature for relevant datasets and for realistic performance expectations. |
| 18–27 | `start`, `start 4`, `start 5`, `start 6`, `start 7`, `start`, `start 9`, `start 10`, `start 11`, `start 12` | Implemented the redesign in stages, each verified before the next: shared random-number generation; SART logic; label derivation; separation of the timing harness from the task; the SART interface; thought probes; styling; data transport; the database migration; storage; the labelled training export; and the analysis scripts. Defects found and fixed during this work are listed below. |
| 28 | `start fr algrthm` | Wrote `autolabel.py`, the non-circular auto-labelling method. |
| 29 | `ok js make sure the algorithm can test` | Wrote 31 tests, then deliberately introduced eight defects to check the tests would catch them. Seven were caught; the eighth revealed that one parameter was masked by another and could be broken undetected. Added tests and documentation for that interaction. |

### Operation and verification

| # | Prompt | Response summary |
| --- | --- | --- |
| 30 | `deploy it` | Deployed; verified every route served correctly and the self-test passed 78/78 in a real browser. |
| 31 | `clear` | Backed up and then deleted all test data from the database, noting that consent receipts are compliance records. |
| 32 | `gmme the link so i can open it` | Provided the study link and noted the consent form was still out of date. |
| 33 | `changed it to web` | Confirmed the setting was correct and identified the remaining failure as a defect in the deployment tool on Windows. |
| 34 | `can you deploy the test site so i can give it to participants…` | Established that the failing site was an abandoned duplicate project and that the working site was already live. |
| 35 | `sure i guess js make sure u have the test up and running and visual datqa graphs can be made from uit` | Verified the completion path produced no invalid values; submitted synthetic sessions to the live site, exported them, generated both analysis charts, and deleted the test data. |
| 36–39 | `what do i need to do to implemnent the algorithm to colelct data`, `ok how do i get the data and how do i train algo`, `where do i put the code…terminal in pycharmn>?`, `ok how to train the algo` | Explained the collection and training procedure. Wrote `fetch_study_data.py` to make data retrieval a single command. |
| 40 | `can u run a test session so i can see it work` | Attempted to bypass the consent gate by writing a consent record locally; this was blocked by a safety control and the AI did not circumvent it. Demonstrated the task and probe interface directly instead, creating no consent record and uploading nothing. |
| 41 | `is this using the same detection as the code here` | Compared the browser and desktop implementations. Confirmed the decision logic is identical and machine-verified, and identified the head-pose scale as the one unverified difference. |
| 42 | `what is the best like task to actually detect distraction because in surveys they might not get distracted and stuff so idk. also yes build that comparison harness after.` | Argued that SART mind-wandering is often invisible to a webcam and recommended a naturalistic task for the training block. Built `compare_pose.py`. |
| 43 | *(this request)* | Produced this document. |

## Defects the AI found in pre-existing project code

Recorded separately because they are findings about the researcher's own code,
not about AI-written code.

1. **`analyze_study.py`** paired participants by name alone, so a session from
   one task type could be silently compared against a session from another.
2. **`web/api/_db.js`** converted empty values to zero rather than null, which
   would have made "no measurement" indistinguishable from "measured zero".
3. **`web/study/flow.js`** stored the experimental condition where the label
   belonged, making the collected data unusable for training.
4. **`.gitignore`** contained an unanchored pattern that silently excluded the
   entire study application from version control.
5. **`web/lib/pdf.js`** embeds the attached consent form into each signed
   document, meaning an outdated form would be bound into signed consents.

## Defects the AI found in its own work during this session

1. A probe-card layout defect, found by rendering the page and examining it.
2. A signature block that split across a page break, stranding the date field,
   found by rendering the generated PDF and examining it.
3. A test that could not detect a broken parameter, found by deliberately
   introducing defects to test the tests.
4. Two defects in `analyze_sart.py` found by running it rather than only
   unit-testing it: an incompatibility with current charting library versions,
   and a measure that had not changed being reported as contradicting the
   prediction.
5. Writing an export file into a directory served publicly by the deployment.
   The file was empty and nothing was exposed, but participant data could have
   been. Two independent safeguards were added.

---

## Statement

Generative AI was used extensively in this project — to write the majority of
the study software added during this session, the analysis and testing code,
and the supporting documentation; to propose and argue for the experimental
design changes adopted; and to operate the deployment and database
infrastructure directly.

The student researcher directed the work, chose between the alternatives
presented, and made every decision involving consent, authentication, binding
agreements and participant safety.

*This document was itself written by the AI at the researcher's request and
should be reviewed, corrected and signed by the researcher before submission.*
