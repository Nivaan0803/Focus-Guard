# Generative AI Use — Summary (Application Implementation)

**Project:** FocusGuard: An Analysis of Real-Time Visual Distraction Detection
Through Webcam-Based Gaze and Facial Landmark Tracking Using Heuristic and
Decision Trees

**Prepared:** 2026-11-17

---

## What this document covers

This is a summary of the prompts given to a generative AI tool, and the work it
performed, **for the implementation of the study application only.**

A fuller disclosure covering every element of the project — hosting, the
database, the consent documents, the analysis and training scripts, and the
head-pose verification tool — is provided separately as
*Generative AI Use Disclosure — FocusGuard*. Nothing has been omitted here to
conceal it; this document is narrower on purpose.

**In scope:** the browser application participants use, and the server code
that stores and exports what it collects.

**Out of scope:** deployment and hosting, database provisioning, consent
form wording, the Python analysis and model-training scripts, and the
head-pose comparison tool.

## Redaction

Identifying information has been removed per the disclosure instructions:
`[student researcher]`, `[adult sponsor]`, `[sponsor email]`,
`[project directory]`, `[repository]` and `[deployment URL]` replace the
student researcher's name, the sponsor's name and work email, local file paths
containing a Windows username, the repository path, and the public study URL.
No other content has been altered.

## Scope and honesty notes

- This covers **one working session**, 2026-11-09 to 2026-11-17.
- A browser study application **already existed** in the project when that
  session began, uncommitted to version control. This document describes what
  was built on top of it. How that earlier code was produced is not something
  this document can speak to.
- This is a **reconstructed summary, not a verbatim export** — the tool
  provides no downloadable prompt log. Prompts are reproduced exactly as
  typed, including spelling errors. Responses are summarised.

---

## Tool

| | |
| --- | --- |
| Tool | Claude (Anthropic), via Claude Code, an agentic command-line interface |
| Model | Claude Opus 5 |
| Mode | Agentic — the tool wrote files directly, ran the test suite, operated a browser to check the result, and queried the project database. It was not used as a reference the researcher typed code from. |

---

## Summary of prompts and responses

### Deciding what the app should do

| # | Prompt | What the AI did |
| --- | --- | --- |
| 1 | `~find a way of testing something like not math but smth more creative that will guive me more data based on distractions so i can get more data, maybe like testing how focused u r` | Read the existing task and data pipeline. Found that every feature row stored the experimental condition in its label column, meaning a model trained on the collected data would have learned to predict which experimental arm a session belonged to rather than whether anyone was distracted. Proposed three replacement tasks with trade-offs and recommended a SART vigilance task; also proposed adding periodic thought probes to supply genuine labels. **The researcher chose the SART and accepted the probes.** |
| 2 | `can you also make a algorithm that collects data by itself using face landmarks online 2ndaery research to collect more data?` | Advised against the approach as stated: labelling data from face landmarks and then training on face landmarks is circular and yields a meaningless accuracy figure. Proposed deriving labels from a different sensor instead. Searched published literature for realistic performance expectations. |

### Building it

Implemented in ten stages, each verified before the next began. The researcher
advanced each stage with a one-word prompt.

| # | Prompt | What the AI did |
| --- | --- | --- |
| 3 | `start` | Extracted the seeded random-number generator into a shared module, after first pinning the existing task's output so the extraction could be proven not to change it. Wrote the SART logic — sequence generation, outcome classification, aggregate statistics — as pure code with no interface, so it could be tested automatically. Wrote the module that converts probe answers into training labels. |
| 4 | `start 4` | Separated the timing harness from the task in the main study flow. Previously one function owned the clock, both timers, the on-screen elements and the arithmetic input handling together, so changing the task meant rewriting the timers. Kept the old task working through the new structure and verified it behaved identically before going further. |
| 5 | `start 5` | Built the SART interface: digit presentation, the 250 ms / 1150 ms timing, and space-bar handling. Added detection for browser tab throttling, so trials the browser skipped while the participant was away are marked unusable rather than counted as errors. |
| 6 | `start 6` | Built the thought probes: an unpredictable schedule, the question card, and the handling that stops the clock while a probe is open. Hid the detector's on-screen verdict during a probe, so the system's own opinion could not influence the participant's self-report. |
| 7 | `start 7` | Styling for the task and the probe card. Rendered the page and inspected it, finding a layout defect in the probe options. |
| 8 | `start` | Extended the data the app sends so trial-level and probe-level records reach the server. Ensured a SART session leaves the arithmetic-specific measures empty rather than filling them with zeros, which would have been indistinguishable from a real score. |
| 9 | `start 9` | Wrote and applied the database migration adding two tables and nineteen columns, then verified the constraints behaved correctly. |
| 10 | `start 10` | Rewrote the upload endpoint to store the new data, with validation that rejects unrecognised values rather than quietly coercing them, and safeguards so a retried upload cannot duplicate records. |
| 11 | `start 11` | Rewrote the export endpoint, adding the labelled training export that converts probe answers into usable labels at download time, so the labelling rules can be adjusted later without re-collecting data. |
| 12 | `start 12` | Updated the analysis scripts for the new task. *(Analysis code is covered in the separate full disclosure.)* |

### Checking it worked

| # | Prompt | What the AI did |
| --- | --- | --- |
| 13 | `sure i guess js make sure u have the test up and running and visual datqa graphs can be made from uit` | Verified the session-completion path produced no invalid values, submitted synthetic sessions to the running site, exported them, produced the analysis charts, then deleted the test data. |
| 14 | `can u run a test session so i can see it work` | Attempted to skip the consent screen by writing a consent record locally. **A safety control blocked this and the AI did not work around it.** Demonstrated the task and probe interface directly instead, creating no consent record and uploading nothing. |
| 15 | `is this using the same detection as the code here` | Compared the browser's detection code against the project's Python implementation. Confirmed the decision rules, thresholds and confidence values are identical and automatically verified, and identified head-pose estimation as the one unverified difference between them. |
| 16 | `what is the best like task to actually detect distraction because in surveys they might not get distracted and stuff so idk` | Argued that mind-wandering during a SART is frequently invisible to a webcam — a still head and forward gaze look identical to being on task — and recommended a naturalistic task for the training block instead. **This recommendation was not implemented.** |

---

## Application files written by the AI

New:

| File | Purpose |
| --- | --- |
| `web/study/sart.js` | Sequence generation, outcome classification, aggregates |
| `web/study/sart-task.js` | Presentation, timing, keyboard handling |
| `web/study/probes.js` | Probe scheduling and the question card |
| `web/study/rng.js` | Seeded random numbers, extracted from existing code |
| `web/study/arithmetic-task.js` | Existing task, adapted to the new structure |
| `web/lib/labels.js` | Probe answers to training labels |
| `web/api/schema_sart.sql` | Database migration |

Modified:

| File | Change |
| --- | --- |
| `web/study/flow.js` | Timing harness separated from the task |
| `web/lib/upload.js` | Carries trial, probe and SART data |
| `web/api/submit.js` | Stores the new data |
| `web/api/export.js` | Adds the labelled training export |
| `web/api/_db.js` | One-line fix: empty values were stored as zero, not null |
| `web/selftest.html`, `web/data/parity.json` | Automated checks extended from 28 to 78 |
| `web/study.html` | Styling |

Pre-existing application code **not** written or modified by the AI:
`web/lib/consent.js`, `pdf.js`, `features.js`, `tree.js`, `calibrate.js`,
`score.js`, `vision.js`, `beep.js`, and `web/study/arithmetic.js`.

## Verification performed

The AI wrote its own tests and ran them. At completion: **78 automated checks**
in the application's self-test page, passing in a real browser; **53 further
checks** covering the task interface, the probe scheduling and the timing
behaviour; and **70 checks** against the live database covering storage and
export.

Defects the AI found in its own work and corrected include a probe-card layout
error found by rendering the page and looking at it, and a test suite that
could not detect a broken parameter — found by deliberately introducing eight
defects to confirm the tests would catch them, of which one was missed.

## Statement

The application implementation described here was substantially written by
generative AI. The researcher directed the work, chose between the design
alternatives presented, and made the decisions involving participant consent
and safety.

No participant data existed at any point during this work. All test data was
synthetic, generated by the AI to confirm the software functioned, and deleted
afterwards.

*This document was written by the AI at the researcher's request and should be
reviewed, corrected and signed by the researcher before submission.*
