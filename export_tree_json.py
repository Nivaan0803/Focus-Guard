"""Export a trained decision tree to JSON for the web app.

``train_model.py`` writes ``models/distraction_tree.joblib``. The browser
cannot read a joblib pickle, so this dumps the fitted tree's arrays to
``web/data/distraction_tree.json``, which ``web/lib/tree.js`` loads and walks
with the same logic as ``distraction_model.py``.

If the JSON is absent the web app falls back to the built-in heuristic tree,
exactly as the desktop app does (``distraction_model.py:156-168``), so running
this is optional.

Usage:
    python export_tree_json.py
    python export_tree_json.py --model models/distraction_tree.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distraction_model import FEATURE_NAMES, MODEL_PATH

DEFAULT_OUT = Path(__file__).resolve().parent / "web" / "data" / "distraction_tree.json"


def export(model_path: Path, out_path: Path) -> int:
    try:
        import joblib
    except ImportError:
        print("joblib is not installed. Run: python -m pip install -r requirements.txt")
        return 1

    if not model_path.is_file():
        print(f"No trained model at {model_path}.")
        print("Train one first:  python train_model.py focus_session*.csv")
        print("Until then the web app uses the built-in heuristic tree, which is")
        print("the same behaviour as the desktop app with no model file.")
        return 1

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
    tree = model.tree_

    # value is (n_nodes, n_outputs, n_classes); the app is single-output, so
    # take output 0 and keep the per-class counts for the confidence.
    payload = {
        "feature_names": feature_names,
        "classes": [str(c) for c in model.classes_],
        "tree": {
            "children_left": [int(v) for v in tree.children_left],
            "children_right": [int(v) for v in tree.children_right],
            "feature": [int(v) for v in tree.feature],
            "threshold": [float(v) for v in tree.threshold],
            "value": [[float(x) for x in node[0]] for node in tree.value],
        },
        "n_nodes": int(tree.node_count),
        "max_depth": int(tree.max_depth),
        "source": str(model_path.name),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    leaves = sum(1 for left in tree.children_left if left == -1)
    print(f"Wrote {out_path}")
    print(f"  {tree.node_count} nodes ({leaves} leaves), depth {tree.max_depth}")
    print(f"  classes: {', '.join(payload['classes'])}")
    print(f"  features: {len(feature_names)}")
    if feature_names != list(FEATURE_NAMES):
        print("  NOTE: this tree's feature order differs from FEATURE_NAMES;")
        print("        the exported order is stored with it, so the web app will")
        print("        build its rows to match.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="trained joblib bundle (default: models/distraction_tree.joblib)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="where to write the JSON (default: web/data/distraction_tree.json)")
    args = parser.parse_args()
    raise SystemExit(export(args.model, args.out))


if __name__ == "__main__":
    main()
