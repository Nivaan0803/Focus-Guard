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

MEASURES = (
    ("correct_per_min", "Correct answers / min", "higher"),
    ("accuracy", "Accuracy", "higher"),
    ("seconds_away", "Seconds outside task window", "lower"),
    ("times_left", "Times left task window", "lower"),
    ("self_rating", "Self-rated focus (1-10)", "higher"),
)


def load(path: Path):
    sessions = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # A participant's latest run of each condition wins (re-runs overwrite).
            sessions.setdefault(row["participant"], {})[row["condition"]] = row
    return {p: c for p, c in sessions.items() if {"alerts", "no_alerts"} <= set(c)}


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
    paired = load(path)
    if not paired:
        sys.exit("No participant has both an 'alerts' and a 'no_alerts' session yet.")
    people = sorted(paired)
    print(f"{len(people)} participant(s) with both conditions: {', '.join(people)}\n")

    summary = []
    for key, label, better in MEASURES:
        pairs = []
        for person in people:
            on, off = paired[person]["alerts"].get(key, ""), paired[person]["no_alerts"].get(key, "")
            if on != "" and off != "":
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

    if not any(p is not None for *_, p in summary):
        print("(Install scipy for p-values: pip install scipy. It comes with scikit-learn.)")
    _save_chart(summary, path.parent / "study_results.png")


def _save_chart(summary, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping study_results.png")
        return

    fig, axes = plt.subplots(1, len(summary), figsize=(3.2 * len(summary), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (label, pairs, p_value) in zip(axes, summary):
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
    axes[0].legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("Same people, with vs. without Focus Guard (grey = one person)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
