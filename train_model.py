"""Train the Focus Guard distraction decision tree from logged sessions.

Usage:
    python train_model.py session1.csv session2.csv ...
    python train_model.py "logs/*.csv"

Collect data first: run main.py, start a focus session, click "Start logging",
and set the label ("focused" / "distracted") to match what you are actually
doing. Repeat across several sessions and lighting conditions.

This writes models/distraction_tree.joblib. With no trained model present the
app falls back to its built-in heuristic tree, so training is optional.
"""
from __future__ import annotations

import csv
import glob
import sys

import numpy as np

from distraction_model import FEATURE_NAMES, MODEL_PATH

try:
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text
except ImportError:
    sys.exit("scikit-learn and joblib are required: pip install scikit-learn joblib")


def load_rows(patterns):
    features, labels = [], []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            with open(path, newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    try:
                        features.append([float(record[name]) for name in FEATURE_NAMES])
                    except (KeyError, ValueError, TypeError):
                        continue
                    labels.append((record.get("label") or "focused").strip() or "focused")
    return np.asarray(features, dtype=float), np.asarray(labels)


def main(argv) -> None:
    if not argv:
        sys.exit(__doc__)

    X, y = load_rows(argv)
    if len(X) < 20:
        sys.exit(f"Need at least ~20 labelled rows, found {len(X)}.")

    classes = sorted(set(y))
    if len(classes) < 2:
        sys.exit(f"Need both 'focused' and 'distracted' rows; found only {classes}.")

    counts = {label: int((y == label).sum()) for label in classes}
    print("Label counts:", counts)

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=max(10, len(X) // 50),
        class_weight="balanced",
        random_state=0,
    )

    folds = min(5, min(counts.values()))
    labels_sorted = classes
    matrix = None
    if folds >= 2:
        scores = cross_val_score(tree, X, y, cv=folds)
        print(f"{folds}-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Out-of-fold predictions: each row is scored by a fold that never saw
        # it during training, so this report reflects generalisation rather
        # than the resubstitution accuracy of a tree fit on all the data.
        oof_predictions = cross_val_predict(tree, X, y, cv=folds)
        print("\nOut-of-fold classification report (unseen-fold predictions):\n")
        print(classification_report(y, oof_predictions, zero_division=0))
        matrix = confusion_matrix(y, oof_predictions, labels=labels_sorted)
        print(f"Confusion matrix (rows=true, cols=predicted), labels={labels_sorted}:")
        print(matrix)
    else:
        print("Fewer than 2 rows in the smallest class — skipping cross-validation report.")

    tree.fit(X, y)
    print("\nLearned rules:\n")
    print(export_text(tree, feature_names=list(FEATURE_NAMES)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": tree, "feature_names": list(FEATURE_NAMES)}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    _save_charts(tree, list(FEATURE_NAMES), matrix, labels_sorted)


def _save_charts(tree, feature_names, matrix, labels_sorted) -> None:
    """Save poster-ready PNGs: feature importance and the out-of-fold confusion matrix.

    Optional — skipped with a note if matplotlib isn't installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping chart PNGs "
              "(pip install matplotlib for poster-ready figures).")
        return

    importances = tree.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([feature_names[i] for i in order], importances[order], color="#2c7fb8")
    ax.set_xlabel("Importance (mean Gini reduction)")
    ax.set_title("Which signals drive the distraction decision?")
    fig.tight_layout()
    importance_path = MODEL_PATH.parent / "feature_importance.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)
    print(f"Saved {importance_path}")

    if matrix is None:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (out-of-fold)")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    cm_path = MODEL_PATH.parent / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved {cm_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
