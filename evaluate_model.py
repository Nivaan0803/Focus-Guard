"""Evaluate Focus Guard's distraction models across labelled conditions.

Turns the engineering-write-up test matrix (different lighting, people,
desk setups) into hard numbers: per-condition accuracy/precision/recall for
both the built-in heuristic tree and a trained model (if one exists at
models/distraction_tree.joblib), plus how often the two agree.

Usage:
    python evaluate_model.py bright=logs/bright*.csv dim=logs/dim*.csv person2=logs/person2*.csv

Each argument is `condition_name=glob_pattern`. Collect the CSVs first with
the app's "Start logging" feature (see README) — one or more files per
condition, labelled to match what you were actually doing.
"""
from __future__ import annotations

import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from distraction_model import DISTRACTED, FEATURE_NAMES, HeuristicTree, load_model
from train_model import load_rows


def _predict_all(model, rows):
    return [model.predict(dict(zip(FEATURE_NAMES, row)))[0] for row in rows]


def evaluate(argv) -> None:
    if not argv:
        sys.exit(__doc__)

    conditions = []
    for arg in argv:
        if "=" not in arg:
            sys.exit(f"Expected condition_name=glob_pattern, got: {arg!r}")
        name, pattern = arg.split("=", 1)
        conditions.append((name, pattern))

    heuristic = HeuristicTree()
    trained = load_model()
    has_trained = not isinstance(trained, HeuristicTree)
    if not has_trained:
        print("No trained model found at models/distraction_tree.joblib — "
              "reporting heuristic-tree results only.\n")

    rows_table = []
    for name, pattern in conditions:
        X, y = load_rows([pattern])
        if len(X) == 0:
            print(f"[{name}] no rows found for pattern {pattern!r} — skipping.")
            continue

        heur_preds = _predict_all(heuristic, X)
        heur_acc = accuracy_score(y, heur_preds)
        heur_p, heur_r, _, _ = precision_recall_fscore_support(
            y, heur_preds, labels=[DISTRACTED], zero_division=0)

        row = {"condition": name, "n": len(X), "heuristic_acc": heur_acc,
               "heuristic_precision": heur_p[0], "heuristic_recall": heur_r[0]}

        if has_trained:
            trained_preds = _predict_all(trained, X)
            trained_acc = accuracy_score(y, trained_preds)
            trained_p, trained_r, _, _ = precision_recall_fscore_support(
                y, trained_preds, labels=[DISTRACTED], zero_division=0)
            agreement = sum(a == b for a, b in zip(heur_preds, trained_preds)) / len(X)
            row.update({"trained_acc": trained_acc, "trained_precision": trained_p[0],
                        "trained_recall": trained_r[0], "agreement": agreement})

        rows_table.append(row)

    if not rows_table:
        sys.exit("No usable data across any condition.")

    header = f"{'condition':<14}{'n':>6}{'heur acc':>10}{'heur prec':>11}{'heur rec':>10}"
    if has_trained:
        header += f"{'trn acc':>10}{'trn prec':>10}{'trn rec':>9}{'agree':>8}"
    print(header)
    for row in rows_table:
        line = (f"{row['condition']:<14}{row['n']:>6}{row['heuristic_acc']:>10.2f}"
                f"{row['heuristic_precision']:>11.2f}{row['heuristic_recall']:>10.2f}")
        if has_trained:
            line += (f"{row['trained_acc']:>10.2f}{row['trained_precision']:>10.2f}"
                     f"{row['trained_recall']:>9.2f}{row['agreement']:>8.2f}")
        print(line)

    _save_chart(rows_table, has_trained)


def _save_chart(rows_table, has_trained) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping condition_comparison.png "
              "(pip install matplotlib for a poster-ready figure).")
        return

    from distraction_model import MODEL_PATH

    names = [row["condition"] for row in rows_table]
    positions = list(range(len(names)))
    width = 0.35 if has_trained else 0.6

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.4), 4))
    heuristic_x = [i - width / 2 for i in positions] if has_trained else positions
    ax.bar(heuristic_x, [row["heuristic_acc"] for row in rows_table], width,
           label="heuristic tree", color="#8b98ab")
    if has_trained:
        ax.bar([i + width / 2 for i in positions], [row["trained_acc"] for row in rows_table],
               width, label="trained tree", color="#2c7fb8")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model accuracy by test condition")
    ax.legend()
    fig.tight_layout()

    path = MODEL_PATH.parent / "condition_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    evaluate(sys.argv[1:])
