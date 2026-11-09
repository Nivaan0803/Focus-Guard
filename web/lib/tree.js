/* Distraction classifier ported from distraction_model.py.
 *
 * Two interchangeable back ends, exactly as on the desktop:
 *   TrainedTree  - reads data/distraction_tree.json (export_tree_json.py)
 *   HeuristicTree - the built-in transparent tree, used when no export exists
 * loadModel() mirrors distraction_model.py:156-168: trained if available,
 * heuristic otherwise, and any failure falls back rather than throwing.
 */

export const FOCUSED = "focused";
export const DISTRACTED = "distracted";

// distraction_model.py:23-36 — the fixed vector order. The exported tree
// indexes into this, so it must not be reordered.
export const FEATURE_NAMES = Object.freeze([
  "task_tab_active",
  "tab_switches_per_min",
  "time_on_current_tab",
  "keyboard_inactivity",
  "mouse_inactivity",
  "calibrated",
  "face_visible",
  "head_pitch",
  "head_yaw",
  "iris_delta",
  "eye_openness",
  "phone_present",
]);

// distraction_model.py:45-58
export const FEATURE_LABELS = Object.freeze({
  task_tab_active: "task tab active",
  tab_switches_per_min: "tab switches/min",
  time_on_current_tab: "time on current tab",
  keyboard_inactivity: "keyboard idle (s)",
  mouse_inactivity: "mouse idle (s)",
  calibrated: "gaze calibrated",
  face_visible: "face visible",
  head_pitch: "head pitch (deg)",
  head_yaw: "head yaw (deg)",
  iris_delta: "iris offset",
  eye_openness: "eye openness",
  phone_present: "phone present",
});

export const SIGNAL_GROUPS = Object.freeze(["phone", "tab", "camera"]);

// distraction_model.py:91-93
export function featureRow(features, names = FEATURE_NAMES) {
  return names.map((n) => Number(features[n] ?? 0));
}

/* distraction_model.py:99-133. Rule order is load-bearing: phone, then
 * non-task window, then the calibrated camera checks, then rapid switching.
 * Confidences and reason strings are kept verbatim so a browser-collected row
 * is indistinguishable from a desktop one. */
export class HeuristicTree {
  constructor(signals = SIGNAL_GROUPS) {
    this.signals = new Set(signals);
    const all = this.signals.size === SIGNAL_GROUPS.length &&
      SIGNAL_GROUPS.every((s) => this.signals.has(s));
    this.name = all
      ? "built-in heuristic tree"
      : SIGNAL_GROUPS.filter((s) => this.signals.has(s)).join(" + ") + " only";
  }

  predict(f) {
    if (this.signals.has("phone") && f.phone_present) {
      return [DISTRACTED, 0.90, "phone in view"];
    }
    // Python: `not features.get("task_tab_active", 1.0)` — absent means 1.0,
    // i.e. assume on-task when the signal is missing.
    if (this.signals.has("tab") && !(f.task_tab_active ?? 1.0)) {
      return [DISTRACTED, 0.80, "non-task window active"];
    }
    if (this.signals.has("camera") && f.calibrated) {
      if (!f.face_visible) return [DISTRACTED, 0.70, "face not visible"];
      const pitch = Math.abs(f.head_pitch ?? 0);
      const yaw = Math.abs(f.head_yaw ?? 0);
      const iris = Math.abs(f.iris_delta ?? 0);
      if (pitch > 18 || yaw > 22 || iris > 0.18) {
        return [DISTRACTED, 0.75, "looking away from screen"];
      }
    }
    if (this.signals.has("tab") && (f.tab_switches_per_min ?? 0) >= 15) {
      return [DISTRACTED, 0.55, "rapid tab switching"];
    }
    return [FOCUSED, 0.65, "on task"];
  }
}

/* distraction_model.py:61-88 _decision_reason — walk the path this row took and
 * name the last internal node before the leaf. That is the split that actually
 * separated it, which is what makes an alert legible. */
function decisionReason(tree, row, names) {
  try {
    const path = [];
    let node = 0;
    while (tree.children_left[node] !== -1) {
      path.push(node);
      const i = tree.feature[node];
      node = row[i] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
    }
    for (let k = path.length - 1; k >= 0; k--) {
      const id = path[k];
      const i = tree.feature[id];
      if (i < 0) continue;
      const name = names[i];
      const threshold = tree.threshold[id];
      const value = row[i];
      const direction = value <= threshold ? "<=" : ">";
      const label = FEATURE_LABELS[name] ?? name;
      return `${label} ${value.toFixed(2)} ${direction} ${threshold.toFixed(2)}`;
    }
  } catch {
    /* fall through */
  }
  return "trained model";
}

/* distraction_model.py:136-153 SklearnTree over the JSON export. */
export class TrainedTree {
  constructor(bundle) {
    this.name = "trained decision tree";
    this.tree = bundle.tree;
    this.classes = bundle.classes;
    this.featureNames = bundle.feature_names || FEATURE_NAMES;
  }

  predict(f) {
    const row = featureRow(f, this.featureNames);
    let node = 0;
    const t = this.tree;
    while (t.children_left[node] !== -1) {
      node = row[t.feature[node]] <= t.threshold[node]
        ? t.children_left[node]
        : t.children_right[node];
    }
    const counts = t.value[node];
    let best = 0;
    for (let i = 1; i < counts.length; i++) if (counts[i] > counts[best]) best = i;
    const total = counts.reduce((a, b) => a + b, 0);
    const label = String(this.classes[best]);
    const confidence = total > 0 ? counts[best] / total : 1.0;
    const reason = label === DISTRACTED
      ? decisionReason(t, row, this.featureNames)
      : "on task";
    return [label, confidence, reason];
  }
}

/* distraction_model.py:156-168 */
export async function loadModel(url = "data/distraction_tree.json") {
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) return new HeuristicTree();
    const bundle = await res.json();
    if (!bundle || !bundle.tree || !Array.isArray(bundle.tree.children_left)) {
      return new HeuristicTree();
    }
    return new TrainedTree(bundle);
  } catch {
    return new HeuristicTree();
  }
}

export function describeModel(model) {
  return model && model.name ? model.name : "unknown model";
}
