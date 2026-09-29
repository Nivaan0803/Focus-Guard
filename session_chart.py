"""Render a Focus Guard session's distraction score over time to a PNG.

Pure function, no Tkinter or camera dependency, so it is easy to unit-test
and reuse from a script. Matplotlib is optional; if it is not installed,
``save_chart`` returns False and the caller should skip silently — nothing
else in the app depends on this file existing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


def save_chart(
    path: Path,
    score_history: Sequence[Tuple[float, float]],
    alert_events: Sequence[Tuple[float, str]],
    threshold: float = 0.55,
) -> bool:
    """Save a line chart of the distraction score with alert markers.

    ``score_history`` is a list of (elapsed_seconds, score) samples.
    ``alert_events`` is a list of (elapsed_seconds, reason) for each alert
    actually fired. Returns True if a file was written.
    """
    if len(score_history) < 2:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    times = [t for t, _ in score_history]
    scores = [s for _, s in score_history]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times, scores, color="#2c7fb8", linewidth=1.5, label="distraction score")
    ax.axhline(threshold, color="#e0574a", linestyle="--", linewidth=1,
               label=f"alert threshold ({threshold:.2f})")
    ax.fill_between(times, scores, threshold,
                     where=[s >= threshold for s in scores],
                     color="#e0574a", alpha=0.15, interpolate=True)

    seen = set()
    for elapsed, reason in alert_events:
        ax.axvline(elapsed, color="#e0a13a", linewidth=1, alpha=0.7)
        if reason not in seen:
            ax.annotate(reason, (elapsed, 1.03), rotation=90, fontsize=7, ha="center",
                        va="bottom", color="#8b98ab", xycoords=("data", "axes fraction"))
            seen.add(reason)

    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("Session time (s)")
    ax.set_ylabel("Distraction score")
    ax.set_title("Distraction score over the session")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
