"""Tests for analyze_study.py and analyze_sart.py.

Run:  python test_analysis.py

The load() tests are the important ones. Pairing used to key on the
participant alone, so a file holding both task types would pair an arithmetic
`alerts` session against a SART `no_alerts` session and print a confident
comparison of two different constructs. Nothing crashed - that is exactly why
it needs a test.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import analyze_study
import analyze_sart
from analyze_study import (
    ARITHMETIC_MEASURES, SART_MEASURES, SHARED_MEASURES, detect_task_type, load,
)

ARITH_COLS = ["finished", "participant", "condition", "minutes", "attempted",
              "correct", "accuracy", "correct_per_min", "seconds_away",
              "times_left", "self_rating"]
SART_COLS = ["finished", "participant", "condition", "minutes", "commission_rate",
             "rt_cv", "mw_rate", "seconds_away", "times_left", "self_rating"]


def write_csv(cols, rows):
    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return Path(path)


def arith(participant, condition, **kw):
    return {"finished": "x", "participant": participant, "condition": condition,
            "minutes": "10", "attempted": "40", "correct": "34", "accuracy": "0.850",
            "correct_per_min": "3.40", "seconds_away": "12.5", "times_left": "2",
            "self_rating": "4", **kw}


def sart(participant, condition, **kw):
    return {"finished": "x", "participant": participant, "condition": condition,
            "minutes": "10", "commission_rate": "0.2353", "rt_cv": "0.2153",
            "mw_rate": "0.2500", "seconds_away": "9.4", "times_left": "3",
            "self_rating": "6", **kw}


class TestTaskDetection(unittest.TestCase):

    def test_arithmetic_file(self):
        self.assertEqual(detect_task_type(ARITH_COLS), "arithmetic")

    def test_sart_file(self):
        self.assertEqual(detect_task_type(SART_COLS), "sart")

    def test_unknown_defaults_to_arithmetic(self):
        self.assertEqual(detect_task_type(["participant", "condition"]), "arithmetic")

    def test_none_is_safe(self):
        self.assertEqual(detect_task_type(None), "arithmetic")

    def test_measure_columns_are_disjoint(self):
        """The detection above relies on this, and so does the database design."""
        a = {k for k, _, _ in ARITHMETIC_MEASURES}
        s = {k for k, _, _ in SART_MEASURES}
        shared = {k for k, _, _ in SHARED_MEASURES}
        self.assertEqual(a & s, shared,
                         "arithmetic and SART share a measure that is not a shared one")
        self.assertNotIn("accuracy", s)
        self.assertNotIn("correct_per_min", s)


class TestPairing(unittest.TestCase):

    def test_pairs_within_one_task(self):
        p = write_csv(ARITH_COLS, [arith("P01", "alerts"), arith("P01", "no_alerts")])
        try:
            paired, task = load(p)
            self.assertEqual(task, "arithmetic")
            self.assertEqual(set(paired), {("P01", "arithmetic")})
        finally:
            os.unlink(p)

    def test_unpaired_participant_excluded(self):
        p = write_csv(ARITH_COLS, [arith("P01", "alerts")])
        try:
            self.assertEqual(load(p)[0], {})
        finally:
            os.unlink(p)

    def test_label_only_never_pairs(self):
        """label_only sits outside the counterbalanced comparison by design."""
        p = write_csv(SART_COLS, [sart("P01", "alerts"), sart("P01", "label_only")])
        try:
            self.assertEqual(load(p)[0], {}, "label_only was treated as a comparison arm")
        finally:
            os.unlink(p)

    def test_does_not_pair_across_task_types(self):
        """THE BUG. Keyed on participant alone these two would have paired."""
        cols = ARITH_COLS + ["task_type", "commission_rate", "rt_cv", "mw_rate"]
        p = write_csv(cols, [
            arith("P01", "alerts", task_type="arithmetic"),
            sart("P01", "no_alerts", task_type="sart"),
        ])
        try:
            paired, _ = load(p)
            self.assertEqual(paired, {},
                             "an arithmetic session was paired against a SART session")
        finally:
            os.unlink(p)

    def test_both_task_types_pair_independently(self):
        cols = ARITH_COLS + ["task_type", "commission_rate", "rt_cv", "mw_rate"]
        p = write_csv(cols, [
            arith("P01", "alerts", task_type="arithmetic"),
            arith("P01", "no_alerts", task_type="arithmetic"),
            sart("P01", "alerts", task_type="sart"),
            sart("P01", "no_alerts", task_type="sart"),
        ])
        try:
            paired, _ = load(p)
            self.assertEqual(set(paired), {("P01", "arithmetic"), ("P01", "sart")})
        finally:
            os.unlink(p)

    def test_rerun_overwrites_same_condition(self):
        p = write_csv(ARITH_COLS, [
            arith("P01", "alerts", correct_per_min="1.00"),
            arith("P01", "alerts", correct_per_min="9.99"),
            arith("P01", "no_alerts"),
        ])
        try:
            paired, _ = load(p)
            self.assertEqual(paired[("P01", "arithmetic")]["alerts"]["correct_per_min"], "9.99")
        finally:
            os.unlink(p)


class TestWindowStats(unittest.TestCase):

    def trial(self, onset, is_target, outcome, rt=None):
        return {"onset": onset, "is_target": is_target, "outcome": outcome, "rt_ms": rt}

    def test_dropped_trials_excluded(self):
        s = analyze_sart.window_stats([
            self.trial(0, False, "hit", 400.0),
            self.trial(1, False, "dropped", 999.0),
        ])
        self.assertEqual(s["n_trials"], 1)

    def test_all_dropped_returns_none(self):
        self.assertIsNone(analyze_sart.window_stats([self.trial(0, False, "dropped")]))

    def test_empty_returns_none(self):
        self.assertIsNone(analyze_sart.window_stats([]))

    def test_commission_rate(self):
        s = analyze_sart.window_stats([
            self.trial(0, True, "commission_error", 300.0),
            self.trial(1, True, "correct_omission"),
            self.trial(2, False, "hit", 400.0),
        ])
        self.assertAlmostEqual(s["commission_rate"], 0.5)

    def test_no_targets_gives_none_not_zero(self):
        """A window with no target cannot have a commission rate of 0."""
        s = analyze_sart.window_stats([self.trial(0, False, "hit", 400.0)])
        self.assertIsNone(s["commission_rate"])

    def test_rt_cv_needs_three_hits(self):
        two = analyze_sart.window_stats([
            self.trial(0, False, "hit", 400.0), self.trial(1, False, "hit", 500.0)])
        self.assertIsNone(two["rt_cv"])
        three = analyze_sart.window_stats([
            self.trial(0, False, "hit", 400.0), self.trial(1, False, "hit", 500.0),
            self.trial(2, False, "hit", 600.0)])
        self.assertIsNotNone(three["rt_cv"])

    def test_rt_uses_hits_only(self):
        s = analyze_sart.window_stats([
            self.trial(0, False, "hit", 400.0),
            self.trial(1, False, "hit", 400.0),
            self.trial(2, False, "hit", 400.0),
            self.trial(3, True, "commission_error", 9999.0),
        ])
        self.assertAlmostEqual(s["rt_mean_ms"], 400.0)


class TestBuildWindows(unittest.TestCase):

    def setUp(self):
        self.trials = [
            {"session_id": "S1", "onset": "50.0", "is_target": "1", "rt_ms": "", "outcome": "correct_omission"},
            {"session_id": "S1", "onset": "55.0", "is_target": "1", "rt_ms": "300", "outcome": "commission_error"},
            {"session_id": "S1", "onset": "58.0", "is_target": "0", "rt_ms": "400", "outcome": "hit"},
            # outside a 15 s window ending at 60
            {"session_id": "S1", "onset": "10.0", "is_target": "1", "rt_ms": "300", "outcome": "commission_error"},
        ]
        self.probes = [
            {"session_id": "S1", "participant": "P01", "condition": "label_only",
             "elapsed": "60.0", "response": "mind_wandering"},
        ]

    def test_window_bounds(self):
        w = analyze_sart.build_windows(self.trials, self.probes, 15.0)
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]["n_trials"], 3, "a trial outside the window was included")
        self.assertAlmostEqual(w[0]["commission_rate"], 0.5)

    def test_unanswered_probe_skipped(self):
        probes = [{**self.probes[0], "response": ""}]
        self.assertEqual(analyze_sart.build_windows(self.trials, probes, 15.0), [])

    def test_probe_from_another_session_gets_nothing(self):
        probes = [{**self.probes[0], "session_id": "S2"}]
        self.assertEqual(analyze_sart.build_windows(self.trials, probes, 15.0), [])

    def test_widening_the_window_pulls_in_more(self):
        w = analyze_sart.build_windows(self.trials, self.probes, 60.0)
        self.assertEqual(w[0]["n_trials"], 4)

    def test_off_task_and_on_task_are_disjoint(self):
        self.assertEqual(analyze_sart.OFF_TASK & analyze_sart.ON_TASK, set())
        self.assertNotIn("task_related", analyze_sart.OFF_TASK)
        self.assertNotIn("task_related", analyze_sart.ON_TASK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
