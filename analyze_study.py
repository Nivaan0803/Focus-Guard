"""Analyse the focus_study.py results: does Focus Guard actually help people focus?

Usage:
    python analyze_study.py                      # reads study/results.csv
    python analyze_study.py path/to/results.csv

Each participant is compared with themselves (alerts vs. no_alerts), which
cancels out people simply being faster or slower at arithmetic. For every
measure it prints the per-person numbers, the average difference, and a
paired t-test p-value (p < 0.05 is the usual bar for "probably not luck").
With fewer than ~8 participants, treat p-values as a rough hint, and say so
on the poster. Saves study/study_results.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from focus_study import DEFAULT_OUT

# Measures shared by every task: they come from the window and the
# participant, not from what the participant was doing.
SHARED_MEASURES = (
    ("seconds_away", "Seconds outside task window", "lower"),
    ("times_left", "Times left task window", "lower"),
    ("self_rating", "Self-rated focus (1-10)", "higher"),
)

ARITHMETIC_MEASURES = (
    ("correct_per_min", "Correct answers / min", "higher"),
    ("accuracy", "Accuracy", "higher"),
) + SHARED_MEASURES

# No column name is shared with the arithmetic measures. A SART session has no
# "accuracy" in the arithmetic sense, and reusing the column would let the two
# be averaged together by the float() below without anything complaining.
SART_MEASURES = (
    # The headline: a DIRECT report of mind-wandering rather than a look-away
    # proxy. This is what the probes were added for.
    ("mw_rate", "Mind-wandering reported (0-1)", "lower"),
    # Pressing on a target: the standard operational marker of a lapse.
    ("commission_rate", "Lapses (pressed on a 3)", "lower"),
    # How erratic responses are, independent of how fast - a validated marker
    # of mind-wandering, and continuous rather than binary.
    ("rt_cv", "Response variability (RT CV)", "lower"),
) + SHARED_MEASURES


def detect_task_type(fieldnames) -> str:
    """Work out which export produced this file.

    Neither export carries a task_type column - kind=study cannot, because its
    11 columns are fixed by focus_study.py:45-46 - so the measure columns are
    the discriminator. They are disjoint by design, which is what makes this
    unambiguous.
    """
    names = set(fieldnames or ())
    if names & {"commission_rate", "mw_rate", "rt_cv"}:
        return "sart"
    return "arithmetic"


def load(path: Path):
    """Pair each participant's alerts session with their own no_alerts session.

    Keyed on (participant, task_type), NOT on participant alone. Keying on the
    participant was a real bug: feed this a file holding both task types and an
    arithmetic `alerts` session would pair against a SART `no_alerts` session,
    producing a silent comparison of two different constructs. Sessions are
    never paired across task types now.

    `label_only` sessions are loaded but never paired - they sit outside the
    counterbalanced comparison on purpose.
    """
    sessions = {}
    file_task = "arithmetic"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        file_task = detect_task_type(reader.fieldnames)
        for row in reader:
            # A row may carry its own task_type (a future export, or a merged
            # file); otherwise the whole file is one task type.
            task = (row.get("task_type") or file_task).strip() or file_task
            key = (row["participant"], task)
            # A participant's latest run of each condition wins (re-runs overwrite).
            sessions.setdefault(key, {})[row["condition"]] = row
    paired = {k: v for k, v in sessions.items() if {"alerts", "no_alerts"} <= set(v)}
    return paired, file_task


def paired_p_value(a, b):
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return None
    if len(a) < 2 or np.allclose(np.asarray(a) - np.asarray(b), 0):
        return None
    return float(ttest_rel(a, b).pvalue)


def main(argv) -> None:
    path = Path(argv[0]) if argv else DEFAULT_OUT
    if not path.is_file():
        sys.exit(f"No results file at {path}. Run focus_study.py first.")
    paired, _file_task = load(path)
    if not paired:
        sys.exit("No participant has both an 'alerts' and a 'no_alerts' session yet.")

    # One file can legitimately hold both task types (a merged export). They
    # are reported separately and never pooled.
    tasks = sorted({task for _, task in paired})
    for task in tasks:
        measures = SART_MEASURES if task == "sart" else ARITHMETIC_MEASURES
        people = sorted(person for person, t in paired if t == task)
        if len(tasks) > 1:
            print(f"{'=' * 60}\n{task.upper()} SESSIONS\n{'=' * 60}\n")
        print(f"{len(people)} participant(s) with both conditions: {', '.join(people)}\n")

        summary = []
        for key, label, better in measures:
            pairs = []
            for person in people:
                cells = paired[(person, task)]
                on = cells["alerts"].get(key, "")
                off = cells["no_alerts"].get(key, "")
                if on not in ("", None) and off not in ("", None):
                    pairs.append((person, float(on), float(off)))
            if not pairs:
                continue
            on_vals = [p[1] for p in pairs]
            off_vals = [p[2] for p in pairs]
            diff = np.mean(on_vals) - np.mean(off_vals)
            improved = sum((a > b) if better == "higher" else (a < b) for _, a, b in pairs)
            p_value = paired_p_value(on_vals, off_vals)

            print(f"== {label} ({better} is better) ==")
            print(f"{'person':<10}{'no alerts':>11}{'alerts':>9}")
            for person, a, b in pairs:
                print(f"{person:<10}{b:>11.2f}{a:>9.2f}")
            p_text = f"p = {p_value:.3f}" if p_value is not None else "p = n/a"
            print(f"mean: {np.mean(off_vals):.2f} -> {np.mean(on_vals):.2f}  (change {diff:+.2f}), "
                  f"improved for {improved}/{len(pairs)} people, paired t-test {p_text}\n")
            summary.append((label, pairs, p_value))

        if summary and not any(p is not None for *_, p in summary):
            print("(Install scipy for p-values: pip install scipy. It comes with scikit-learn.)")
        suffix = f"_{task}" if len(tasks) > 1 else ""
        _save_chart(summary, path.parent / f"study_results{suffix}.png")


def _save_chart(summary, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping study_results.png")
        return

    if not summary:
        return

    # SART reports six measures. In one row that is a 19-inch figure with
    # unreadable axes, so wrap at three.
    per_row = min(len(summary), 3)
    n_rows = (len(summary) + per_row - 1) // per_row
    fig, axes = plt.subplots(n_rows, per_row,
                             figsize=(3.2 * per_row, 4.2 * n_rows), squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(summary):]:
        ax.set_visible(False)
    for ax, (label, pairs, p_value) in zip(flat, summary):
        # One line per person: no_alerts on the left, alerts on the right.
        for _, on, off in pairs:
            ax.plot([0, 1], [off, on], color="#8b98ab", linewidth=1.5, marker="o", markersize=6, alpha=0.8)
        ax.plot([0, 1], [np.mean([p[2] for p in pairs]), np.mean([p[1] for p in pairs])],
                color="#2a78d6", linewidth=3, marker="o", markersize=9, label="average")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no alerts", "alerts"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(label if p_value is None else f"{label}\np = {p_value:.3f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    flat[0].legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("Same people, with vs. without Focus Guard (grey = one person)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
