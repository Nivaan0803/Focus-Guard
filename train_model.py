"""Train the Focus Guard distraction decision tree from logged sessions.

Usage:
    python train_model.py session1.csv session2.csv ...
    python train_model.py "logs/*.csv"
    python train_model.py alex="logs/alex_*.csv" sam="logs/sam_*.csv" ...

Naming each argument after a person (person=glob) turns on a
leave-one-person-out test: the tree is trained on everyone except one
person, then scored on that person, once per person. That answers "does it
work on someone it has never seen?", which ordinary k-fold can't, since
neighbouring 1/8-second ticks from the same session are nearly identical and
leak between folds.

Both main.py "Start logging" CSVs (``label`` column) and run_trial.py CSVs
(``ground_truth`` column) are accepted.

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

from distraction_model import DISTRACTED, FEATURE_NAMES, MODEL_PATH

try:
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
                    label = record.get("label") or record.get("ground_truth") or "focused"
                    labels.append(label.strip() or "focused")
    return np.asarray(features, dtype=float), np.asarray(labels)


def load_grouped(args):
    """Load ``name=glob`` (or bare glob) arguments; return X, y and a per-row group name.

    Groups are None when no argument is named.
    """
    named = [arg for arg in args if "=" in arg]
    if named and len(named) != len(args):
        sys.exit("Either name every argument (person=glob) or none of them.")
    if not named:
        X, y = load_rows(args)
        return X, y, None
    Xs, ys, groups = [], [], []
    for arg in args:
        name, pattern = arg.split("=", 1)
        X, y = load_rows([pattern])
        if len(X) == 0:
            print(f"[{name}] no rows found for {pattern!r} - skipping.")
            continue
        Xs.append(X)
        ys.append(y)
        groups.extend([name] * len(X))
    if not Xs:
        return np.empty((0, len(FEATURE_NAMES))), np.asarray([]), None
    return np.vstack(Xs), np.concatenate(ys), np.asarray(groups)


def main(argv) -> None:
    if not argv:
        sys.exit(__doc__)

    X, y, groups = load_grouped(argv)
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

    person_results = None
    if groups is not None and len(set(groups)) >= 2:
        person_results = leave_one_person_out(tree, X, y, groups)

    tree.fit(X, y)
    print("\nLearned rules:\n")
    print(export_text(tree, feature_names=list(FEATURE_NAMES)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": tree, "feature_names": list(FEATURE_NAMES)}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    _save_charts(tree, list(FEATURE_NAMES), matrix, labels_sorted, person_results)


def leave_one_person_out(tree, X, y, groups):
    """Train on all-but-one person, test on the held-out person, for each person."""
    from sklearn.base import clone

    results = []
    print("\nLeave-one-person-out (trained without this person, tested on them):\n")
    print(f"{'person':<14}{'n':>6}{'accuracy':>10}{'precision':>11}{'recall':>8}")
    for person in sorted(set(groups)):
        test = groups == person
        train_y = y[~test]
        if len(set(train_y)) < 2:
            print(f"{person:<14}  skipped - the other people's data has only one label")
            continue
        model = clone(tree).fit(X[~test], train_y)
        predicted = model.predict(X[test])
        acc = accuracy_score(y[test], predicted)
        prec, rec, _, _ = precision_recall_fscore_support(
            y[test], predicted, labels=[DISTRACTED], zero_division=0)
        results.append((person, int(test.sum()), acc, prec[0], rec[0]))
        print(f"{person:<14}{int(test.sum()):>6}{acc:>10.2f}{prec[0]:>11.2f}{rec[0]:>8.2f}")
    if results:
        accs = np.array([r[2] for r in results])
        print(f"\nMean accuracy on unseen people: {accs.mean():.3f} "
              f"(range {accs.min():.2f}-{accs.max():.2f}, {len(results)} people)")
        print("Precision/recall are for the 'distracted' class.")
    return results


def _save_charts(tree, feature_names, matrix, labels_sorted, person_results=None) -> None:
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

    if person_results:
        _save_person_chart(plt, person_results)

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


def _save_person_chart(plt, person_results) -> None:
    names = [r[0] for r in person_results]
    accs = [r[2] for r in person_results]
    mean = float(np.mean(accs))
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 4))
    ax.bar(names, accs, color="#2a78d6", width=0.6)
    ax.axhline(mean, color="#52514e", linestyle="--", linewidth=1.5)
    ax.text(len(names) - 0.5, mean, f" mean {mean:.2f}", va="bottom", ha="right", color="#52514e")
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy on held-out person")
    ax.set_title("Does it work on people it was never trained on?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = MODEL_PATH.parent / "leave_one_person_out.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
