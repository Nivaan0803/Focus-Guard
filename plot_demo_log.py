"""Chart a booth-demo log (``demo_logs/demo_<timestamp>.csv``).

``session_chart.py`` plots a live session as it ends. This does the same for a
recorded demo log after the fact, so a run captured at the booth can go on a
poster. It reads only the columns ``demo_recorder.py`` writes.

Usage:
    python plot_demo_log.py demo_logs/demo_20261020_175133.csv
    python plot_demo_log.py demo_logs/demo_*.csv --out charts/
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
from pathlib import Path

ALERT_AT = 0.55          # main.py:613
METER_AMBER = 0.4        # main.py:610
METER_RED = 0.7

# Status palette, taken from main.py:43-58 so the figures match the app.
GREEN, AMBER, RED = "#35b17e", "#e0a13a", "#e0574a"
INK, MUTE, GRID = "#141b26", "#5a6678", "#dde3ec"


def read_log(path: Path):
    with io.open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no rows.")
    missing = {"elapsed", "score", "verdict"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    return rows


def spans(times, flags):
    """Contiguous [start, end] runs where flag is True — for shading."""
    out, start = [], None
    for t, flag in zip(times, flags):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, times[-1]))
    return out


def plot(path: Path, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_log(path)
    t = [float(r["elapsed"]) for r in rows]
    score = [float(r["score"]) for r in rows]
    distracted = [r["verdict"].strip() == "distracted" for r in rows]
    pitch = [float(r.get("head_pitch", 0) or 0) for r in rows]
    yaw = [float(r.get("head_yaw", 0) or 0) for r in rows]
    phone = [float(r.get("phone_present", 0) or 0) > 0.5 for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")

    # --- panel 1: the smoothed distraction score -------------------------
    for a, b in spans(t, distracted):
        ax1.axvspan(a, b, color=RED, alpha=0.09, linewidth=0)
    for a, b in spans(t, phone):
        ax1.axvspan(a, b, color=AMBER, alpha=0.16, linewidth=0)

    ax1.axhline(ALERT_AT, color=MUTE, linestyle=(0, (5, 4)), linewidth=1.2)
    ax1.text(t[-1], ALERT_AT + 0.02, f"alert threshold {ALERT_AT}",
             ha="right", va="bottom", fontsize=8.5, color=MUTE)

    ax1.plot(t, score, color=INK, linewidth=2, solid_capstyle="round")

    # Mark where an alert would fire: score crosses up through the threshold.
    crossings = [t[i] for i in range(1, len(score))
                 if score[i - 1] < ALERT_AT <= score[i]]
    for x in crossings:
        ax1.plot([x], [ALERT_AT], marker="o", markersize=8, color=RED,
                 markeredgecolor="white", markeredgewidth=1.6, zorder=5)

    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("distraction score")
    ax1.set_title(f"Distraction score over time  ·  {path.name}",
                  loc="left", fontsize=12.5, color=INK, pad=10)

    pct = round(100 * sum(distracted) / len(distracted))
    ax1.text(0.0, 1.14, f"{len(rows)} ticks · {t[-1]:.0f}s · {pct}% distracted · "
                        f"{len(crossings)} alert crossing(s)",
             transform=ax1.transAxes, fontsize=9, color=MUTE)

    # --- panel 2: head pose, both in degrees so one axis is honest -------
    ax2.axhline(0, color=GRID, linewidth=1)
    ax2.plot(t, yaw, color=RED, linewidth=1.8, label="head yaw")
    ax2.plot(t, pitch, color=GREEN, linewidth=1.8, label="head pitch")
    ax2.set_ylabel("degrees from neutral")
    ax2.set_xlabel("seconds into the demo")
    ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTE, labelsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="demo CSV file(s); globs allowed")
    parser.add_argument("--out", type=Path, default=Path("charts"))
    args = parser.parse_args()

    paths = [Path(p) for pattern in args.logs for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit("No matching log files.")
    for p in paths:
        print("wrote", plot(p, args.out))


if __name__ == "__main__":
    main()
