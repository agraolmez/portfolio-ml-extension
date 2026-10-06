"""Analytic software fixtures; all settings here are synthetic, not a protocol."""
import unittest
from unittest.mock import patch

import numpy as np

from temporal_extension.optimizer import OptimizerSettings, optimize


def fixture_settings(**changes):
    values = dict(lower=0., upper=2., tolerance=1e-10, max_steps=80,
                  max_evaluations=160, max_backtracks=12, initial_step=1.,
                  armijo=1e-4, wall_seconds=60.)
    values.update(changes)
    return OptimizerSettings(**values)


def quadratic(target, curvature=1.):
    target = np.asarray(target, dtype=float)
    calls = []

    def evaluate(gains, *, with_gradient):
        calls.append((gains.copy(), with_gradient))
        difference = gains - target
        # Explicitly regenerate conditional coefficients for each proposed point.
        beta = np.concatenate(([1.], gains**2))
        return dict(value=float(-.5 * curvature * (difference @ difference)),
                    gradient=-curvature * difference, beta=beta, solve_residual=0.)

    return evaluate, calls


class OptimizerTests(unittest.TestCase):
    def test_all_settings_are_required_and_invalid_settings_fail_before_callback(self):
        with self.assertRaises(TypeError):
            OptimizerSettings()
        for name, value in (("tolerance", 0.), ("tolerance", np.nan),
                            ("initial_step", -1.), ("wall_seconds", None),
                            ("armijo", 1.), ("max_steps", .5),
                            ("max_evaluations", True), ("max_backtracks", -1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                fixture_settings(**{name: value})
        evaluate, calls = quadratic([1.2, .7])
        for lower, upper in ((None, 2.), (True, 2.), ([0.], [2.]),
                             (2., 3.), (-1., 0.), (np.nan, 2.),
                             ([0., 1.5], [2., 1.]), ("0", "2")):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                optimize(evaluate, 2, fixture_settings(lower=lower, upper=upper))
        for count in (0, 2., True):
            with self.assertRaises(ValueError):
                optimize(evaluate, count, fixture_settings())
        self.assertEqual(calls, [])

    def test_interior_optimum_and_conditional_coefficients_on_last_permitted_step(self):
        evaluate, calls = quadratic([1.3, .6])
        result = optimize(evaluate, 2, fixture_settings(max_steps=1, max_evaluations=2))
        self.assertTrue(result["converged"])
        self.assertEqual(result["status"], "converged")
        self.assertEqual(result["accepted_steps"], 1)
        self.assertEqual(result["evaluation_count"], 2)
        np.testing.assert_array_equal(calls[0][0], [1., 1.])
        np.testing.assert_allclose(result["gains"], [1.3, .6], atol=1e-15)
        np.testing.assert_allclose(result["beta"], [1., 1.3**2, .6**2])
        self.assertEqual(result["projected_gradient_norm"], 0.)
        self.assertTrue(all(with_gradient for _, with_gradient in calls))

    def test_boundary_optimum_certifies_constrained_not_unconstrained_stationarity(self):
        evaluate, _ = quadratic([-1., 3., 1.])
        result = optimize(evaluate, 3, fixture_settings())
        np.testing.assert_array_equal(result["gains"], [0., 2., 1.])
        self.assertTrue(result["converged"])
        self.assertEqual(result["projected_gradient_norm"], 0.)
        self.assertGreater(np.linalg.norm(result["gradient"]), 1.)
        self.assertEqual(result["boundary_flags"],
                         dict(at_lower=[True, False, False], at_upper=[False, True, False]))

    def test_fixed_coordinate_and_initial_stationarity(self):
        evaluate, _ = quadratic([3., 1.])
        result = optimize(evaluate, 2, fixture_settings(lower=[1., 0.], upper=[1., 2.], max_steps=0))
        self.assertTrue(result["converged"])
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(result["evaluation_count"], 1)

    def test_deterministic_backtracking_records_every_accepted_and_rejected_trial(self):
        settings = fixture_settings(initial_step=8.)
        evaluate, calls = quadratic([1.3, .6], curvature=4.)
        actual = optimize(evaluate, 2, settings)
        again, _ = quadratic([1.3, .6], curvature=4.)
        expected = optimize(again, 2, settings)
        self.assertTrue(actual["converged"])
        np.testing.assert_array_equal(actual["gains"], expected["gains"])
        self.assertEqual(actual["trials"], expected["trials"])
        self.assertEqual(actual["evaluations"], expected["evaluations"])
        self.assertEqual(len(calls), actual["evaluation_count"])
        self.assertEqual(len(actual["trials"]) + 1, len(calls))
        self.assertTrue(any(t["reason"] == "armijo_rejected" for t in actual["trials"]))
        self.assertTrue(any(t["accepted"] for t in actual["trials"]))
        self.assertTrue(np.all(np.diff(actual["objective_trace"]) >= 0))
        self.assertEqual(len(actual["projected_gradient_trace"]), actual["accepted_steps"] + 1)
        for trial in actual["trials"]:
            if trial["accepted"]:
                self.assertGreaterEqual(trial["value"], trial["armijo_required"])

    def test_zero_evaluation_budget_does_not_invoke_profile(self):
        evaluate, calls = quadratic([1.3])
        result = optimize(evaluate, 1, fixture_settings(max_evaluations=0))
        self.assertEqual(calls, [])
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "evaluation_limit")
        self.assertIsNone(result["value"])
        self.assertIsNone(result["beta"])
        self.assertIsNone(result["projected_gradient_norm"])

    def test_zero_step_and_one_evaluation_budgets_remain_nonconvergence(self):
        for settings, status in ((fixture_settings(max_steps=0), "step_limit"),
                                 (fixture_settings(max_evaluations=1), "evaluation_limit")):
            evaluate, calls = quadratic([1.3])
            result = optimize(evaluate, 1, settings)
            self.assertFalse(result["converged"])
            self.assertEqual(result["status"], status)
            self.assertEqual(len(calls), 1)
            self.assertGreater(result["projected_gradient_norm"], settings.tolerance)

    def test_nonstationary_final_accepted_step_reports_cap_exhaustion(self):
        evaluate, _ = quadratic([1.3])
        result = optimize(evaluate, 1, fixture_settings(max_steps=1, initial_step=.1))
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["accepted_steps"], 1)
        self.assertAlmostEqual(result["gains"][0], 1.03)
        self.assertAlmostEqual(result["projected_gradient_norm"], .27)

    def test_rejected_trial_counts_against_evaluation_budget(self):
        evaluate, calls = quadratic([1.3], curvature=4.)
        result = optimize(evaluate, 1, fixture_settings(initial_step=8., max_evaluations=2))
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "evaluation_limit")
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["trials"][0]["reason"], "armijo_rejected")
        self.assertEqual(result["trials"][-1]["reason"], "evaluation_limit")
        np.testing.assert_array_equal(result["gains"], [1.])

    def test_exhausted_backtracking_is_not_convergence(self):
        evaluate, _ = quadratic([1.3], curvature=4.)
        result = optimize(evaluate, 1, fixture_settings(initial_step=8., max_backtracks=0))
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "line_search_stalled")
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(len(result["trials"]), 1)

    def test_invalid_profile_results_fail_closed_with_receipts(self):
        valid = dict(value=1., gradient=np.ones(2), beta=np.ones(3), solve_residual=0.)
        invalid = [None, {}, {**valid, "value": np.nan}, {**valid, "value": True},
                   {**valid, "gradient": None}, {**valid, "gradient": [1.]},
                   {**valid, "gradient": [1., np.inf]}, {**valid, "beta": []},
                   {**valid, "beta": [None]}, {**valid, "solve_residual": -.01}]
        for item in invalid:
            with self.subTest(item=item):
                result = optimize(lambda gains, with_gradient: item, 2, fixture_settings())
                self.assertEqual(result["status"], "evaluation_failed")
                self.assertFalse(result["converged"])
                self.assertEqual(result["evaluation_count"], 1)
                self.assertEqual(result["evaluations"][0]["status"], "failed")
                self.assertIsNone(result["value"])

    def test_failed_trial_retains_previous_valid_point_and_reports_failure(self):
        evaluate, calls = quadratic([1.3])
        def failing(gains, *, with_gradient):
            if calls:
                raise RuntimeError("fixture solve failed")
            return evaluate(gains, with_gradient=with_gradient)
        result = optimize(failing, 1, fixture_settings())
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertFalse(result["converged"])
        np.testing.assert_array_equal(result["gains"], [1.])
        self.assertIsNotNone(result["beta"])
        self.assertEqual(result["trials"][0]["reason"], "evaluation_failed")
        self.assertEqual(result["evaluations"][-1]["error"], "fixture solve failed")

    def test_wall_limit_stops_between_evaluations_and_discloses_uninterruptible_work(self):
        clock = [0.]
        evaluate, calls = quadratic([1.3])
        def slow(gains, *, with_gradient):
            result = evaluate(gains, with_gradient=with_gradient)
            clock[0] = 61.
            return result
        with patch("temporal_extension.optimizer.monotonic", side_effect=lambda: clock[0]):
            result = optimize(slow, 1, fixture_settings())
        self.assertEqual(result["status"], "wall_time_limit")
        self.assertFalse(result["converged"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["evaluations"][0]["status"], "completed_after_time_limit")
        self.assertEqual(result["wall_time_enforcement"], "checked_between_evaluations_only")

    def test_callback_cannot_mutate_optimizer_coordinates(self):
        def mutation(gains, *, with_gradient):
            gains[0] = 0.
        result = optimize(mutation, 1, fixture_settings())
        self.assertEqual(result["status"], "evaluation_failed")
        np.testing.assert_array_equal(result["gains"], [1.])

    def test_profile_coefficient_dimension_cannot_change_during_fit(self):
        evaluate, calls = quadratic([1.3])
        def changing(gains, *, with_gradient):
            result = evaluate(gains, with_gradient=with_gradient)
            if len(calls) > 1:
                result["beta"] = np.ones(3)
            return result
        result = optimize(changing, 1, fixture_settings())
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertFalse(result["converged"])
        self.assertIn("dimension changed", result["evaluations"][-1]["error"])


if __name__ == "__main__":
    unittest.main()
