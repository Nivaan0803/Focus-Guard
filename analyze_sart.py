"""Trial- and probe-level analysis of the SART sessions.

Usage:
    python analyze_sart.py study/sart_trials.csv study/probe_responses.csv

Both files come from the deployed study:

    curl -H "Authorization: Bearer $EXPORT_TOKEN" ".../api/export?kind=trials" > study/sart_trials.csv
    curl -H "Authorization: Bearer $EXPORT_TOKEN" ".../api/export?kind=probes" > study/probe_responses.csv

WHAT THIS IS FOR
----------------
analyze_study.py answers "did alerts help?". This answers a prior and more
important question: **are the thought probes measuring anything real?**

The probes are the only ground truth in the study, and they are self-reports.
If a participant says "mind-wandering", nothing in the data forces that to be
true. But if the probes are meaningful, behaviour should already have degraded
in the seconds before an off-task report - more lapses, and more erratic
response times - because that is what attention lapses do to a vigilance task.
That prediction is independent of the detector, so testing it validates the
probes without assuming the thing the project is trying to show.

If it holds, the labels are worth training on and the poster can say so. If it
does not, the labels are noise and no detector accuracy computed from them
means anything - which is worth finding out BEFORE collecting a full sample.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# How far back a probe is taken to describe. Matches the default labelling
# window in web/lib/labels.js so the behavioural check and the training labels
# refer to the same stretch of time.
DEFAULT_WINDOW_S = 15.0

OFF_TASK = {"mind_wandering", "external"}
ON_TASK = {"on_task"}
# Deliberately in neither group: engaged with the task but not performing it is
# the ambiguous middle, and forcing it either way would blur the contrast this
# whole script exists to measure.
AMBIGUOUS = {"task_related"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value):
    """Floats out of CSV cells, where an absent measure is an empty string."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def window_stats(trials):
    """Performance inside one probe's window.

    `dropped` trials are excluded: the tab was throttled, so those windows were
    never observed and scoring them would import away-time into the measure.
    Returns None for a figure there is not enough data to compute, rather than
    a zero that would be averaged in as though it were one.
    """
    usable = [t for t in trials if t["outcome"] != "dropped"]
    if not usable:
        return None

    targets = [t for t in usable if t["is_target"]]
    go = [t for t in usable if not t["is_target"]]
    commissions = sum(1 for t in targets if t["outcome"] == "commission_error")
    omissions = sum(1 for t in go if t["outcome"] == "omission_error")
    rts = [t["rt_ms"] for t in usable if t["outcome"] == "hit" and t["rt_ms"] is not None]

    rt_mean = float(np.mean(rts)) if rts else None
    # Sample SD (n-1), matching web/study/sart.js and the parity fixture.
    rt_sd = float(np.std(rts, ddof=1)) if len(rts) >= 3 else None

    return {
        "n_trials": len(usable),
        "n_targets": len(targets),
        "commission_rate": commissions / len(targets) if targets else None,
        "omission_rate": omissions / len(go) if go else None,
        "rt_mean_ms": rt_mean,
        "rt_cv": (rt_sd / rt_mean) if (rt_sd is not None and rt_mean) else None,
    }


def build_windows(trials_rows, probe_rows, window_s):
    """One record per answered probe: its response plus the behaviour before it."""
    by_session = defaultdict(list)
    for r in trials_rows:
        by_session[r["session_id"]].append({
            "onset": fnum(r["onset"]),
            "is_target": r["is_target"] in ("1", "true", "True"),
            "rt_ms": fnum(r["rt_ms"]),
            "outcome": r["outcome"],
        })
    for rows in by_session.values():
        rows.sort(key=lambda x: (x["onset"] if x["onset"] is not None else 0.0))

    windows = []
    for p in probe_rows:
        response = (p.get("response") or "").strip()
        if not response:
            continue                        # unanswered: nothing to describe
        at = fnum(p["elapsed"])
        if at is None:
            continue
        trials = [t for t in by_session.get(p["session_id"], [])
                  if t["onset"] is not None and at - window_s <= t["onset"] <= at]
        stats = window_stats(trials)
        if stats is None:
            continue
        windows.append({
            "participant": p.get("participant", "?"),
            "condition": p.get("condition", "?"),
            "response": response,
            **stats,
        })
    return windows


def compare(windows, key, label, higher_means_worse=True):
    off = [w[key] for w in windows if w["response"] in OFF_TASK and w[key] is not None]
    on = [w[key] for w in windows if w["response"] in ON_TASK and w[key] is not None]
    if len(off) < 2 or len(on) < 2:
        print(f"== {label} ==")
        print(f"   not enough windows yet (on-task {len(on)}, off-task {len(off)})\n")
        return None

    p_value = None
    try:
        from scipy.stats import ttest_ind
        p_value = float(ttest_ind(off, on, equal_var=False).pvalue)
    except ImportError:
        pass

    mean_on, mean_off = float(np.mean(on)), float(np.mean(off))
    delta = mean_off - mean_on

    # A measure that did not move is not evidence against the prediction, and
    # calling it "OPPOSITE" would overstate a difference of zero. This happens
    # for real: omission errors are rare, so a short window often has none
    # either side.
    if abs(delta) < 1e-9:
        direction, verdict = "no difference", "nothing to see"
    else:
        direction = "higher" if delta > 0 else "lower"
        expected = "higher" if higher_means_worse else "lower"
        verdict = "as predicted" if direction == expected else "OPPOSITE to prediction"

    print(f"== {label} ==")
    print(f"   before 'on task' probes  : {mean_on:.4f}  (n={len(on)})")
    print(f"   before off-task probes   : {mean_off:.4f}  (n={len(off)})")
    print(f"   difference               : {delta:+.4f}  -> {direction}, {verdict}")
    print(f"   Welch t-test             : {'p = %.4f' % p_value if p_value is not None else 'p = n/a (pip install scipy)'}\n")
    return {"label": label, "on": on, "off": off, "p": p_value}


def main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials", type=Path, help="CSV from /api/export?kind=trials")
    ap.add_argument("probes", type=Path, help="CSV from /api/export?kind=probes")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                    help=f"seconds before each probe to summarise (default {DEFAULT_WINDOW_S})")
    ap.add_argument("--chart", type=Path, default=None,
                    help="write a figure here (default study/sart_probe_validation.png)")
    args = ap.parse_args(argv)

    for path in (args.trials, args.probes):
        if not path.is_file():
            sys.exit(f"No file at {path}. Export it from /api/export first.")

    trials_rows = read_csv(args.trials)
    probe_rows = read_csv(args.probes)
    if not trials_rows:
        sys.exit("The trials file is empty - no SART session has been uploaded yet.")
    if not probe_rows:
        sys.exit("The probes file is empty - no thought probe has been answered yet.")

    # ---- what was collected -------------------------------------------
    answered = [p for p in probe_rows if (p.get("response") or "").strip()]
    counts = defaultdict(int)
    for p in answered:
        counts[p["response"]] += 1
    by_condition = defaultdict(lambda: [0, 0])
    for p in answered:
        slot = by_condition[p.get("condition", "?")]
        slot[0] += 1
        if p["response"] in OFF_TASK:
            slot[1] += 1

    print(f"{len(trials_rows)} trials, {len(probe_rows)} probes "
          f"({len(answered)} answered), window {args.window:g} s\n")
    print("Probe answers:")
    for response in ("on_task", "task_related", "mind_wandering", "external"):
        print(f"  {response:<16}{counts.get(response, 0)}")
    print("\nMind-wandering rate by condition (off-task / answered):")
    for condition, (total, off) in sorted(by_condition.items()):
        print(f"  {condition:<14}{off}/{total}"
              + (f"  = {off / total:.1%}" if total else ""))
    print()

    windows = build_windows(trials_rows, probe_rows, args.window)
    if not windows:
        sys.exit("No probe had any usable trials in its window. Check that the "
                 "trials and probes files are from the same sessions.")

    print(f"{'=' * 62}")
    print("DO THE PROBES PREDICT BEHAVIOUR?")
    print("If off-task reports are real, lapses and response variability")
    print(f"should already be elevated in the {args.window:g} s before one.")
    print(f"{'=' * 62}\n")

    results = [
        compare(windows, "commission_rate", "Lapses (pressed on a 3)"),
        compare(windows, "rt_cv", "Response variability (RT CV)"),
        compare(windows, "omission_rate", "Missed non-targets"),
        compare(windows, "rt_mean_ms", "Mean reaction time"),
    ]
    results = [r for r in results if r]

    ambiguous = sum(1 for w in windows if w["response"] in AMBIGUOUS)
    if ambiguous:
        print(f"({ambiguous} 'task_related' window(s) excluded from the comparison - "
              "engaged but not performing is\n the ambiguous middle, and forcing it "
              "either way would blur the contrast.)\n")

    _save_chart(results, args.chart or args.trials.parent / "sart_probe_validation.png")


def _save_chart(results, out_path: Path) -> None:
    if not results:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the figure")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 4.4), squeeze=False)
    for ax, r in zip(axes.flatten(), results):
        data = [r["on"], r["off"]]
        # Not boxplot(labels=...): that keyword was deprecated in matplotlib
        # 3.9 and removed, and renaming it to tick_labels would break older
        # installs instead. Setting the ticks afterwards works on both.
        ax.boxplot(data, widths=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["on task", "off task"])
        for i, values in enumerate(data, start=1):
            jitter = (np.random.default_rng(0).random(len(values)) - 0.5) * 0.12
            ax.plot(np.full(len(values), i) + jitter, values, "o",
                    color="#8b98ab", markersize=5, alpha=0.7)
        title = r["label"] if r["p"] is None else f"{r['label']}\np = {r['p']:.4f}"
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Behaviour in the window before each probe (one point = one probe)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
