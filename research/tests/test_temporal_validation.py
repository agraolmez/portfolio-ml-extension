"""Synthetic chronology/selection checks; no historical grid or fit is selected."""
from dataclasses import replace
import unittest

import pandas as pd

from temporal_extension.validation import (
    Candidate, FoldScore, ParentArtifact, ParentRegistry, SelectionPlan,
    ValidationFold, select_candidate,
)


def parent(**changes):
    values = dict(artifact_id="synthetic-parent-1999", seed=1, cutoff="1999-12-31",
                  training_last="1999-11-30", outcomes_available_at="1999-12-31",
                  settings_available_at="1999-12-31", model_sha256="1" * 64,
                  receipt_sha256="2" * 64, settings_sha256="3" * 64)
    values.update(changes)
    return ParentArtifact(**values)


def fold(**changes):
    values = dict(name="synthetic-2000", seed=1, training_start="1998-01-31",
                  training_end="1999-11-30", training_cutoff="1999-12-31",
                  validation_start="2000-01-31", validation_end="2000-03-31",
                  parent_cutoff="1999-12-31")
    values.update(changes)
    return ValidationFold(**values)


def score(candidate, item, source):
    return FoldScore(candidate.name, item.name, "synthetic_score", "validation",
                     candidate.parameters["synthetic_parameter"], item.validation_dates,
                     str((pd.Timestamp(item.validation_end) + pd.offsets.MonthEnd(1)).date()),
                     item.training_end, item.training_cutoff,
                     source.fingerprint)


class TemporalValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = SelectionPlan((fold(),), "2000-04-30", "synthetic_score")
        self.registry = ParentRegistry([parent()])
        self.candidates = (Candidate("first", {"synthetic_parameter": .2}),
                           Candidate("second", {"synthetic_parameter": .3}))

    def run_selection(self, **changes):
        arguments = dict(plan=self.plan, registry=self.registry,
                         candidates=self.candidates, evaluator=score,
                         aggregate=lambda scores: sum(item.score for item in scores),
                         maximize=True)
        arguments.update(changes)
        return select_candidate(**arguments)

    def test_explicit_grid_and_score_direction_with_labeled_receipts(self):
        actual = self.run_selection()
        self.assertEqual(actual.selected.name, "second")
        self.assertEqual(actual.candidate_scores, (("first", .2), ("second", .3)))
        self.assertEqual(actual.criterion_role, "validation")
        self.assertEqual(actual.max_selection_date, "2000-04-30")
        self.assertEqual(self.run_selection(maximize=False).selected.name, "first")
        self.assertEqual(len(actual.fold_scores), 2)

    def test_exact_tie_retains_declared_order(self):
        tied = lambda c, f, p: replace(score(c, f, p), score=1.)
        self.assertEqual(self.run_selection(evaluator=tied).selected.name, "first")
        self.assertEqual(self.run_selection(evaluator=tied, candidates=self.candidates[::-1]).selected.name,
                         "second")

    def test_missing_cutoff_specific_parent_fails_before_any_callback(self):
        calls = []
        later = parent(artifact_id="later", cutoff="2000-12-31", training_last="2000-11-30",
                       outcomes_available_at="2000-12-31", settings_available_at="2000-12-31")
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_selection(registry=ParentRegistry([later]),
                               evaluator=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_all_parent_dependencies_resolve_before_first_candidate_starts(self):
        next_fold = fold(name="later", training_end="2000-11-30", training_cutoff="2000-12-31",
                         parent_cutoff="2000-12-31", validation_start="2001-01-31",
                         validation_end="2001-03-31")
        plan = SelectionPlan((fold(), next_fold), "2001-04-30", "synthetic_score")
        calls = []
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_selection(plan=plan, evaluator=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_future_parent_and_post_cutoff_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Parent cutoff"):
            fold(parent_cutoff="2000-12-31")
        with self.assertRaisesRegex(ValueError, "settings"):
            parent(settings_available_at="2000-01-31")
        with self.assertRaisesRegex(ValueError, "training outcomes"):
            parent(outcomes_available_at="2000-01-31")
        earlier_training_fold = fold(training_end="1999-10-31")
        with self.assertRaisesRegex(ValueError, "beyond the fold"):
            self.run_selection(plan=SelectionPlan((earlier_training_fold,), "2000-04-30", "synthetic_score"))

    def test_duplicate_registry_records_and_bad_hashes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ParentRegistry([parent(), parent()])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            parent(model_sha256="unverified")
        with self.assertRaisesRegex(ValueError, "inventory"):
            ParentRegistry([])

    def test_bad_and_nonmonthly_fold_dates_are_rejected(self):
        for changes in (dict(training_start="2000-01-31"),
                        dict(training_cutoff="2000-01-31"),
                        dict(validation_end="1999-12-31"),
                        dict(validation_start="2000-01-30"),
                        dict(training_cutoff="1999-12-31T01:00:00"),
                        dict(training_cutoff="1999-12-31T00:00:00Z")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fold(**changes)

    def test_overlapping_out_of_order_or_duplicate_folds_are_rejected(self):
        overlap = fold(name="overlap", validation_start="2000-03-31", validation_end="2000-05-31")
        later = fold(name="later", validation_start="2000-04-30", validation_end="2000-05-31")
        for folds in ((fold(), overlap), (later, fold()), (fold(), fold())):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                SelectionPlan(folds, "2000-06-30", "synthetic_score")
        with self.assertRaisesRegex(ValueError, "separate for each seed"):
            SelectionPlan((fold(), replace(later, seed=2)), "2000-06-30", "synthetic_score")

    def test_explicit_selection_boundary_cannot_be_crossed(self):
        with self.assertRaisesRegex(ValueError, "selection boundary"):
            SelectionPlan((fold(),), "2000-02-29", "synthetic_score")
        with self.assertRaises(TypeError):
            SelectionPlan(folds=(fold(),), criterion_name="no-default-boundary")

    def test_grids_are_explicit_finite_consistent_and_immutable(self):
        for parameters in ({}, {"p": float("nan")}, {"p": float("inf")}, {"p": True}, {"p": "0.1"}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                Candidate("invalid", parameters)
        with self.assertRaises(TypeError):
            self.candidates[0].parameters["synthetic_parameter"] = 100.
        for candidates in ((), (self.candidates[0], self.candidates[0]),
                           (self.candidates[0], Candidate("duplicate", {"synthetic_parameter": .2})),
                           (self.candidates[0], Candidate("mismatch", {"other": .1}))):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.run_selection(candidates=candidates)

    def test_callback_cannot_label_training_objective_as_validation(self):
        for changes in (dict(criterion_role="training"), dict(criterion_name="training_loss"),
                        dict(candidate_name="other"), dict(fold_name="other")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "validation criterion"):
                self.run_selection(evaluator=lambda c, f, p: replace(score(c, f, p), **changes))

    def test_callback_parent_and_settings_identity_must_match(self):
        other = parent(settings_sha256="4" * 64)
        with self.assertRaisesRegex(ValueError, "different parent"):
            self.run_selection(evaluator=lambda c, f, p: replace(
                score(c, f, p), parent_fingerprint=other.fingerprint))

    def test_validation_months_must_be_complete_ordered_unique_and_within_bounds(self):
        dates = fold().validation_dates
        for invalid in (dates[:-1], dates[::-1], (dates[0], dates[0], dates[2]),
                        ("1999-12-31", *dates[1:]), (*dates[:-1], "2000-04-30")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "every declared month"):
                self.run_selection(evaluator=lambda c, f, p: replace(score(c, f, p), validation_dates=invalid))

    def test_callback_outcome_availability_respects_training_and_selection_cutoffs(self):
        for changes in (dict(training_last="1999-10-31"),
                        dict(training_outcomes_available_at="2000-01-31"),
                        dict(training_outcomes_available_at="1999-10-31"),
                        dict(validation_outcomes_available_at="2000-05-31"),
                        dict(validation_outcomes_available_at="2000-02-29")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_selection(evaluator=lambda c, f, p: replace(score(c, f, p), **changes))

    def test_decision_month_returns_cannot_be_claimed_available_in_same_month(self):
        with self.assertRaisesRegex(ValueError, "training outcomes"):
            parent(outcomes_available_at="1999-11-30")
        with self.assertRaisesRegex(ValueError, "not mature"):
            fold(training_cutoff="1999-11-30", parent_cutoff="1999-11-30")
        with self.assertRaisesRegex(ValueError, "selection boundary"):
            SelectionPlan((fold(),), "2000-03-31", "synthetic_score")
        for changes in (dict(training_outcomes_available_at="1999-11-30"),
                        dict(validation_outcomes_available_at="2000-03-31")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_selection(evaluator=lambda c, f, p: replace(score(c, f, p), **changes))

    def test_nonfinite_scores_aggregation_and_absent_policy_are_rejected(self):
        for invalid in (float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "score must be finite"):
                self.run_selection(evaluator=lambda c, f, p: replace(score(c, f, p), score=invalid))
            with self.subTest(aggregate=invalid), self.assertRaisesRegex(ValueError, "score must be finite"):
                self.run_selection(aggregate=lambda scores: invalid)
        for changes in (dict(maximize=None), dict(evaluator=None), dict(aggregate=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_selection(**changes)


if __name__ == "__main__":
    unittest.main()
