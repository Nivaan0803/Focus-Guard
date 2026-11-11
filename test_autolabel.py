"""Tests for autolabel.py.

Run:  python test_autolabel.py        (or: python -m unittest test_autolabel)

These matter more than most tests in this repo. autolabel.py produces the
labels a model is then trained and scored on, so a bug here does not crash
anything - it quietly produces a number that looks like a result. The two
properties worth protecting are:

  1. No telemetry column ever reaches the features. That is the only thing
     standing between this method and the circular version that scores 99%
     and means nothing.
  2. Run lengths and guard bands are applied exactly. Labelling a transition
     teaches the model noise, and labelling a one-tick misclick as
     disengagement inflates the positive class.

Tick spacing is 0.1 s and the thresholds used for the exact-count tests are
deliberately off-grid (0.95, 1.95, 4.95). With round numbers a boundary row
sits exactly on the comparison and binary floating point decides whether it
is kept - the test would then be asserting an artefact of 9.9 - 8.9 rather
than the labelling rule.
"""

import csv
import os
import tempfile
import unittest

import autolabel
from autolabel import (
    CAMERA_FEATURES, TELEMETRY_FEATURES, DISTRACTED, FOCUSED,
    balance, blank_telemetry, label_session, load_session, validate_split,
    write_rows,
)
from distraction_model import FEATURE_NAMES

TICK = 0.1
GUARD = 0.95      # off-grid on purpose; see the module docstring
MIN_AWAY = 1.95
MIN_ON = 4.95
GAP = 1.0

# (start, end, task_tab_active) - end exclusive
SCHEDULE = [
    (0.0, 10.0, 1),    # long on-task run      -> focused
    (10.0, 13.0, 0),   # 3 s excursion         -> distracted
    (13.0, 20.0, 1),   # long on-task run      -> focused
    (20.0, 20.5, 0),   # 0.5 s blip            -> too short, unlabelled
    (20.5, 30.0, 1),   # long on-task run      -> focused
]


def _state(t):
    for start, end, on in SCHEDULE:
        if start - 1e-9 <= t < end - 1e-9:
            return on
    return 1


def build_rows(count=300, tick=TICK, schedule=True, on_task=1):
    """Feature rows with the same keys load_session would produce."""
    rows = []
    for i in range(count):
        t = round(i * tick, 2)
        on = _state(t) if schedule else on_task
        row = {name: 0.0 for name in FEATURE_NAMES}
        row["elapsed"] = t
        row["task_tab_active"] = float(on)
        # Camera values vary so a zeroing bug cannot hide behind constants.
        row["head_yaw"] = 2.0 if on else 20.0
        row["head_pitch"] = 1.0 if on else 9.0
        row["face_visible"] = 1.0
        row["calibrated"] = 1.0
        # Telemetry values are non-zero so blanking is observable.
        row["keyboard_inactivity"] = 0.5
        row["mouse_inactivity"] = 0.7
        row["time_on_current_tab"] = 1.0
        rows.append(row)
    return rows


def labelled(rows, **kw):
    opts = dict(min_away=MIN_AWAY, min_on=MIN_ON, guard=GUARD, gap=GAP, switch_rate=0.0)
    opts.update(kw)
    return label_session(rows, **opts)


class TestSplit(unittest.TestCase):
    """The assumption the whole method rests on."""

    def test_validate_split_passes(self):
        validate_split()   # exits the process if the split is broken

    def test_camera_and_telemetry_are_disjoint(self):
        self.assertEqual(set(CAMERA_FEATURES) & set(TELEMETRY_FEATURES), set())

    def test_split_covers_every_feature(self):
        self.assertEqual(set(CAMERA_FEATURES) | set(TELEMETRY_FEATURES), set(FEATURE_NAMES))

    def test_phone_is_treated_as_camera(self):
        # phone_present comes from an object detector running on the webcam
        # frame. Putting it on the label side would make the method circular.
        self.assertIn("phone_present", CAMERA_FEATURES)
        self.assertNotIn("phone_present", TELEMETRY_FEATURES)


class TestLabelling(unittest.TestCase):

    def setUp(self):
        self.rows = build_rows()
        self.labels = labelled(self.rows)

    def test_exact_counts(self):
        n_focused = sum(1 for x in self.labels if x == FOCUSED)
        n_distracted = sum(1 for x in self.labels if x == DISTRACTED)
        # Hand-derived: runs of 9.9 / 2.9 / 6.9 / 0.4 / 9.4 s, each trimmed by
        # a 0.95 s guard at both ends, with the 0.4 s run below min_away.
        self.assertEqual(n_focused, 205)
        self.assertEqual(n_distracted, 10)
        self.assertEqual(len(self.labels), len(self.rows))

    def test_short_excursion_is_never_labelled(self):
        for row, label in zip(self.rows, self.labels):
            if 20.0 <= row["elapsed"] < 20.5:
                self.assertIsNone(label, f"0.5 s blip labelled at t={row['elapsed']}")

    def test_guard_drops_rows_either_side_of_a_transition(self):
        # The switch to off-task is at t=10.0; nothing within 0.95 s of it,
        # on either side, may carry a label.
        for row, label in zip(self.rows, self.labels):
            if 9.05 < row["elapsed"] < 10.95:
                self.assertIsNone(label, f"transition row labelled at t={row['elapsed']}")

    def test_distracted_rows_are_inside_the_excursion(self):
        for row, label in zip(self.rows, self.labels):
            if label == DISTRACTED:
                self.assertTrue(10.0 <= row["elapsed"] < 13.0)
                self.assertEqual(row["task_tab_active"], 0.0)

    def test_min_away_binds_independently_of_the_guard(self):
        """min_away has to do its own work, and the obvious test cannot see it.

        A run only survives the guard if it is longer than 2*guard, so with the
        shipped numbers (min_away 2.0, guard 1.0) the guard already discards
        everything min_away would have rejected - and a broken min_away passes
        unnoticed. Mutation testing found exactly that hole. These parameters
        separate the two: the 3 s excursion clears the guard easily, so only
        min_away can reject it.
        """
        rows = build_rows()                       # 3.0 s excursion at t=10
        self.assertNotIn(DISTRACTED, labelled(rows, min_away=4.0),
                         "min_away is not rejecting a run shorter than it")
        self.assertIn(DISTRACTED, labelled(rows, min_away=2.5),
                      "min_away is rejecting a run longer than it")

    def test_min_on_binds_independently_of_the_guard(self):
        rows = build_rows(count=70, schedule=False, on_task=1)   # 6.9 s
        self.assertNotIn(FOCUSED, labelled(rows, min_on=8.0))
        self.assertIn(FOCUSED, labelled(rows, min_on=6.0))

    def test_a_run_just_under_the_minimum_is_dropped(self):
        rows = build_rows(count=40, schedule=False, on_task=1)
        # 4.0 s of on-task, below min_on of 4.95
        self.assertTrue(all(x is None for x in labelled(rows)))

    def test_a_run_just_over_the_minimum_is_kept(self):
        rows = build_rows(count=70, schedule=False, on_task=1)   # 6.9 s
        self.assertIn(FOCUSED, labelled(rows))

    def test_time_gap_breaks_a_run(self):
        """Rows either side of a gap are not contiguous evidence.

        Without this, two short visits separated by a dead period would be
        credited as one long run and labelled on a duration never observed.
        """
        rows = build_rows(count=30, schedule=False, on_task=1)      # 2.9 s
        later = build_rows(count=30, schedule=False, on_task=1)
        for r in later:
            r["elapsed"] = round(r["elapsed"] + 60.0, 2)            # 60 s gap
        joined = rows + later
        self.assertTrue(all(x is None for x in labelled(joined)),
                        "a 60 s gap was treated as one continuous run")

    def test_switch_rate_rule_is_off_by_default(self):
        rows = build_rows(count=70, schedule=False, on_task=1)
        for r in rows:
            r["tab_switches_per_min"] = 30.0
        self.assertNotIn(DISTRACTED, labelled(rows))

    def test_switch_rate_rule_marks_distracted_when_enabled(self):
        rows = build_rows(count=70, schedule=False, on_task=1)
        for r in rows:
            r["tab_switches_per_min"] = 30.0
        self.assertIn(DISTRACTED, labelled(rows, switch_rate=15.0))

    def test_all_on_task_yields_one_class(self):
        # train_model.py exits when only one class is present, so this is a
        # real situation the report has to warn about rather than a hypothetical.
        rows = build_rows(count=200, schedule=False, on_task=1)
        got = {x for x in labelled(rows) if x is not None}
        self.assertEqual(got, {FOCUSED})

    def test_empty_input(self):
        self.assertEqual(labelled([]), [])


class TestOutput(unittest.TestCase):

    def setUp(self):
        self.rows = build_rows()
        labels = labelled(self.rows)
        self.kept = [(r, l) for r, l in zip(self.rows, labels) if l is not None]

    def test_blank_telemetry_zeroes_only_telemetry(self):
        row = self.rows[0]
        out = blank_telemetry(row)
        for name in TELEMETRY_FEATURES:
            self.assertEqual(out[name], 0.0)
        for name in CAMERA_FEATURES:
            self.assertEqual(out[name], row[name])

    def test_blank_telemetry_does_not_mutate_input(self):
        row = self.rows[0]
        before = row["keyboard_inactivity"]
        blank_telemetry(row)
        self.assertEqual(row["keyboard_inactivity"], before)

    def _written(self, keep_telemetry):
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        try:
            write_rows(path, [r for r, _ in self.kept], [l for _, l in self.kept],
                       keep_telemetry)
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        finally:
            os.unlink(path)

    def test_written_columns_match_session_logger(self):
        out = self._written(False)
        self.assertEqual(list(out[0].keys()), ["elapsed", *FEATURE_NAMES, "label"])

    def test_telemetry_is_constant_in_the_training_file(self):
        """The guarantee that makes camera-only training structural.

        sklearn cannot split on a constant feature, so zeroing is what stops a
        telemetry column leaking into the model - no flag to forget.
        """
        out = self._written(False)
        for name in TELEMETRY_FEATURES:
            self.assertEqual({r[name] for r in out}, {"0.0000"}, f"{name} not blanked")

    def test_camera_columns_still_vary(self):
        out = self._written(False)
        self.assertGreater(len({r["head_yaw"] for r in out}), 1)

    def test_keep_telemetry_preserves_real_values(self):
        out = self._written(True)
        self.assertNotEqual({r["keyboard_inactivity"] for r in out}, {"0.0000"})

    def test_only_two_labels_are_ever_written(self):
        out = self._written(False)
        self.assertTrue({r["label"] for r in out} <= {FOCUSED, DISTRACTED})

    def test_round_trip_through_load_session(self):
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        try:
            write_rows(path, [r for r, _ in self.kept], [l for _, l in self.kept], False)
            reloaded = load_session(path)
            self.assertEqual(len(reloaded), len(self.kept))
            self.assertTrue(all(name in reloaded[0] for name in FEATURE_NAMES))
        finally:
            os.unlink(path)


class TestBalance(unittest.TestCase):

    def test_balance_equalises_classes(self):
        rows = build_rows(count=100, schedule=False)
        labels = [FOCUSED] * 90 + [DISTRACTED] * 10
        _, out = balance(rows, labels)
        self.assertEqual(out.count(FOCUSED), out.count(DISTRACTED))
        self.assertEqual(out.count(DISTRACTED), 10)

    def test_balance_keeps_rows_and_labels_aligned(self):
        rows = build_rows(count=100, schedule=False)
        for i, r in enumerate(rows):
            r["head_yaw"] = float(i)
        labels = [FOCUSED if i % 10 else DISTRACTED for i in range(100)]
        kept_rows, kept_labels = balance(rows, labels)
        for row, label in zip(kept_rows, kept_labels):
            self.assertEqual(label, labels[int(row["head_yaw"])])

    def test_balance_is_a_noop_for_one_class(self):
        rows = build_rows(count=10, schedule=False)
        labels = [FOCUSED] * 10
        out_rows, out_labels = balance(rows, labels)
        self.assertEqual(len(out_rows), 10)
        self.assertEqual(out_labels, labels)

    def test_balance_is_deterministic(self):
        rows = build_rows(count=100, schedule=False)
        labels = [FOCUSED] * 90 + [DISTRACTED] * 10
        first = balance(rows, labels)[1]
        second = balance(rows, labels)[1]
        self.assertEqual(first, second)


class TestDefaults(unittest.TestCase):
    """The shipped defaults, asserted loosely - exact counts belong above."""

    def test_defaults_label_the_obvious_cases(self):
        rows = build_rows()
        labels = label_session(rows, min_away=2.0, min_on=5.0, guard=1.0,
                               gap=1.0, switch_rate=0.0)
        self.assertIn(FOCUSED, labels)
        self.assertIn(DISTRACTED, labels)
        self.assertIn(None, labels)

    def test_module_constants_are_the_documented_ones(self):
        self.assertEqual(autolabel.FOCUSED, "focused")
        self.assertEqual(autolabel.DISTRACTED, "distracted")
        self.assertEqual(autolabel.COLUMNS, ("elapsed", *FEATURE_NAMES, "label"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
