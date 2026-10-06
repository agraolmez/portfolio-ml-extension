"""Synthetic engineering checks only; these settings authorize no historical fit.

Identity tolerances are declared here before execution: exact input/features,
1e-11 absolute / 1e-10 relative moments and coefficients, and 2e-7 absolute /
2e-5 relative full-objective finite differences. No performance test is run.
"""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from test_feature_geometry import fixture
from baseline.core import transition
from feature_geometry.features import FeatureMap, fingerprint
from feature_geometry.training import profile as parent_profile
from geometry_study.kernel import LeanBasis, evaluate as parent_evaluate
from temporal_extension.reference import CalendarReference, Innovation
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.profile import evaluate


ATOL = 1e-11
RTOL = 1e-10


def temporal_fixture(dimensions=115, frequencies=256, months=2):
    rng, parent, dense, settings, theta = fixture(dimensions, frequencies, months)
    groups = np.arange(dimensions) % min(13, dimensions)
    fmap = TemporalFeatureMap(parent, theta, groups)
    bases = [LeanBasis(b.sample) for b in dense]
    innovations = []
    for basis in bases:
        history = []
        for z in basis.sample.market.characteristics:
            valid = rng.uniform(size=z.shape) > .1
            values = rng.normal(size=z.shape)
            values[~valid] = np.nan
            history.append(Innovation(values, valid, {"synthetic": True}))
        innovations.append(history)
    return rng, parent, dense, bases, settings, theta, fmap, innovations


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_neutral_115_inputs_512_features_moments_coefficients_and_objective(self):
        _, parent, dense, bases, settings, theta, fmap, innovations = temporal_fixture()
        gains = np.ones(13)
        self.assertEqual(fmap.gain_count, 13)
        self.assertEqual(parent.dimensions, 115)
        self.assertEqual(parent.columns, 513)  # 512 nonlinear features plus constant.
        for basis, history in zip(bases, innovations):
            market = basis.sample.market
            new = []
            old = []
            for z, vol, innov in zip(market.characteristics, market.volatility, history):
                np.testing.assert_array_equal(fmap.inputs(z, innov, gains), z)
                new.append(fmap.signals(z, vol, innov, gains))
                old.append(parent.signals(z, vol, theta))
                np.testing.assert_array_equal(new[-1], old[-1])
            for actual, expected in zip(basis.operators(new), basis.operators(old)):
                np.testing.assert_array_equal(actual, expected)
        actual = evaluate(bases, innovations, fmap, gains, settings,
                          rho=.03, objective_scale=.7)
        expected = parent_evaluate(bases, parent, theta, settings)
        independent = parent_profile(dense, parent, theta, settings, gradient=False)
        for key in ("score", "risk", "cost", "beta"):
            np.testing.assert_allclose(actual[key], expected[key], atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(actual[key], independent[key], atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(actual["parent_value"], expected["value"], atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(actual["value"], expected["value"] / .7, atol=ATOL, rtol=RTOL)
        system = actual["risk"] + actual["cost"] + settings.ridge * np.eye(513)
        np.testing.assert_allclose(system @ actual["beta"], actual["score"], atol=ATOL, rtol=RTOL)
        self.assertLess(actual["solve_residual"], 1e-10)

    def test_missing_current_and_history_are_neutral_without_nan_leakage(self):
        parent = FeatureMap(np.array([[1., .3], [-.4, 2.]]))
        fmap = TemporalFeatureMap(parent, np.array([.2, -.2]), np.array([0, 1]))
        z = np.array([[.5, .8], [.2, .5], [.9, .3]])
        ref = CalendarReference(["2000-01-31"] * 3, [1, 2, 3], z,
                                np.array([[False, True], [True, False], [True, True]]), [1.])
        innov = ref.at("2000-01-31", [1, 2, 3], z)
        self.assertEqual(innov.reasons["bypassed"], 6)
        np.testing.assert_array_equal(innov.values, np.zeros_like(z))
        masked_nan = Innovation(np.full_like(z, np.nan), np.zeros_like(z, dtype=bool), {})
        for invalid in (innov, masked_nan):
            for gains in (np.ones(2), np.array([0., 2.])):
                np.testing.assert_array_equal(fmap.inputs(z, invalid, gains), z)
                np.testing.assert_array_equal(fmap.signals(z, np.ones(3), invalid, gains),
                                              parent.signals(z, np.ones(3), fmap.theta))

    def test_gain_math_frozen_scale_applied_exactly_once_and_shared_by_group(self):
        parent = FeatureMap(np.array([[1., .3], [-.4, 2.], [.2, -.1]]))
        theta = np.array([.3, -.1, -.2])
        fmap = TemporalFeatureMap(parent, theta, np.array([0, 1, 0]))
        z = np.array([[.5, .8, .1], [.2, .4, .7], [.9, .3, .2]])
        level = np.array([[.3, .1, .5], [.1, .7, .2], [.6, .4, .8]])
        innov = Innovation(z - level, np.ones_like(z, dtype=bool), {})
        for gain in (0., 1., 1.7):
            gains = np.full(2, gain)
            expected = level + gain * (z - level)
            np.testing.assert_allclose(fmap.inputs(z, innov, gains), expected, atol=1e-15)
            np.testing.assert_allclose(fmap.signals(z, np.ones(3), innov, gains),
                                       parent.signals(expected, np.ones(3), theta), atol=1e-13)
        np.testing.assert_allclose(fmap.inputs(z, innov, [0., 2.]),
                                   np.column_stack((level[:, 0], 2*z[:, 1]-level[:, 1], level[:, 2])))

    def test_calendar_gaps_are_not_compressed_and_security_ids_are_isolated(self):
        ref = CalendarReference(
            ["2000-01-31", "2000-03-31", "2000-01-31", "2000-02-29", "2000-03-31"],
            [1, 1, 2, 2, 2], np.array([[1.], [5.], [10.], [20.], [30.]]),
            np.ones((5, 1), dtype=bool), [.5, .5])
        actual = ref.at("2000-03-31", [2, 1], [[30.], [5.]])
        np.testing.assert_array_equal(actual.valid, [[True], [False]])
        np.testing.assert_array_equal(actual.values, [[15.], [0.]])
        self.assertEqual(actual.reasons["prior_security_month_absent"], 1)

    def test_future_changes_do_not_change_any_earlier_reference_or_feature(self):
        dates = pd.date_range("1999-10-31", periods=6, freq="ME")
        ids = np.tile([1, 2, 3], len(dates))
        days = np.repeat(dates, 3)
        values = np.arange(36, dtype=float).reshape(18, 2) / 37.
        observed = np.ones_like(values, dtype=bool)
        before = CalendarReference(days, ids, values, observed, [.5, .5])
        changed = values.copy()
        changed[days > pd.Timestamp("2000-01-31")] += 1000.
        after = CalendarReference(days, ids, changed, observed, [.5, .5])
        fmap = TemporalFeatureMap(FeatureMap(np.array([[1., .7], [-.4, .1]])),
                                  np.array([.1, -.1]), np.array([0, 1]))
        for day in dates[dates <= pd.Timestamp("2000-01-31")]:
            z = values[days == day]
            a, b = before.at(day, [1, 2, 3], z), after.at(day, [1, 2, 3], z)
            np.testing.assert_array_equal(a.values, b.values)
            np.testing.assert_array_equal(a.valid, b.valid)
            np.testing.assert_array_equal(fmap.signals(z, np.ones(3), a, [.6, 1.4]),
                                          fmap.signals(z, np.ones(3), b, [.6, 1.4]))

    def test_current_value_is_excluded_and_history_survives_year_boundary(self):
        dates = ["1999-11-30", "1999-12-31", "2000-01-31"]
        raw = np.array([[2.], [4.], [10.]])
        observed = np.ones_like(raw, dtype=bool)
        first = CalendarReference(dates, [7]*3, raw, observed, [.5, .5])
        actual = first.at(dates[-1], [7], [[10.]])
        np.testing.assert_array_equal(actual.values, [[7.]])  # reference=(2+4)/2.
        updated = raw.copy(); updated[-1] = 20.
        second = CalendarReference(dates, [7]*3, updated, observed, [.5, .5])
        np.testing.assert_array_equal(second.at(dates[-1], [7], [[20.]]).values, [[17.]])
        self.assertTrue(actual.valid.all())

    def test_observed_half_is_valid_but_unobserved_half_is_bypassed(self):
        dates = ["1999-12-31", "2000-01-31"]
        values = np.array([[.5, .5], [.7, .7]])
        ref = CalendarReference(dates, [1, 1], values,
                                np.array([[True, False], [True, True]]), [1.])
        actual = ref.at(dates[-1], [1], values[-1:])
        np.testing.assert_array_equal(actual.valid, [[True, False]])
        np.testing.assert_allclose(actual.values, [[.2, 0.]])

    def test_zero_weight_lag_does_not_require_an_observation(self):
        ref = CalendarReference(["2000-01-31", "2000-03-31"], [1, 1], [[.2], [.6]],
                                np.ones((2, 1), dtype=bool), [0., 1.])
        actual = ref.at("2000-03-31", [1], [[.6]])
        np.testing.assert_allclose(actual.values, [[.4]])
        self.assertTrue(actual.valid.all())

    def test_gain_gradient_through_all_13_slices_normalization_and_conditional_solve(self):
        _, _, _, bases, settings, _, fmap, innovations = temporal_fixture()
        gains = np.linspace(.45, 1.55, 13)
        result = evaluate(bases, innovations, fmap, gains, settings,
                          rho=.03, objective_scale=.7)
        epsilon = 1e-6
        fd = np.empty(13)
        for j in range(13):
            delta = np.eye(13)[j] * epsilon
            plus = evaluate(bases, innovations, fmap, gains + delta, settings,
                            rho=.03, objective_scale=.7, with_gradient=False)
            minus = evaluate(bases, innovations, fmap, gains - delta, settings,
                             rho=.03, objective_scale=.7, with_gradient=False)
            fd[j] = (plus["value"] - minus["value"]) / (2 * epsilon)
        np.testing.assert_allclose(result["gradient"], fd, atol=2e-7, rtol=2e-5)
        neutral = evaluate(bases, innovations, fmap, np.ones(13), settings,
                           rho=.03, objective_scale=.7, with_gradient=False)
        self.assertGreater(np.linalg.norm(result["beta"] - neutral["beta"]), 1e-4)
        self.assertGreater(np.linalg.norm(result["cost"] - neutral["cost"]), 1e-4)

    def test_signal_gradient_constant_column_and_frozen_economic_inputs(self):
        rng, parent, _, bases, settings, theta, fmap, innovations = temporal_fixture(13, 7)
        market = bases[0].sample.market
        before = fingerprint(parent.frequencies, theta, fmap.theta, *market.characteristics,
                             *market.volatility, *market.growth, market.m, market.covariance,
                             market.impact, market.wealth, market.gamma, settings.ridge)
        z, vol, innov = market.characteristics[0], market.volatility[0], innovations[0][0]
        upstream = np.zeros((len(z), parent.columns)); upstream[:, 0] = rng.normal(size=len(z))
        np.testing.assert_array_equal(fmap.vjp(z, vol, innov, np.ones(13), upstream), np.zeros(13))
        evaluate(bases, innovations, fmap, np.linspace(.3, 1.7, 13), settings,
                 rho=.04, objective_scale=1.)
        after = fingerprint(parent.frequencies, theta, fmap.theta, *market.characteristics,
                            *market.volatility, *market.growth, market.m, market.covariance,
                            market.impact, market.wealth, market.gamma, settings.ridge)
        self.assertEqual(before, after)

    def test_continuous_neutral_path_across_annual_parents_entries_exits_and_actual_costs(self):
        rng, parent, _, bases, settings, theta, _, innovations = temporal_fixture(13, 7, 3)
        ids = [np.arange(16), np.arange(1, 18), np.arange(2, 20)]
        dates = ["1999-11-30", "1999-12-31", "2000-01-31"]
        old_h = rng.normal(size=16) * .01
        new_h = old_h.copy()
        for t, (basis, history) in enumerate(zip(bases, innovations)):
            market = basis.sample.market
            # The next return year's parent changes in December, not January.
            annual_theta = theta if t == 0 else -theta
            fmap = TemporalFeatureMap(parent, annual_theta, np.arange(13))
            parent_fit = parent_evaluate([basis], parent, annual_theta, settings)
            new_fit = evaluate([basis], [history], fmap, np.ones(13), settings,
                               rho=.03, objective_scale=1., with_gradient=False)
            z, vol = market.characteristics[0], market.volatility[0]
            old_aim = parent.signals(z, vol, annual_theta) @ parent_fit["beta"]
            new_aim = fmap.signals(z, vol, history[0], np.ones(13)) @ new_fit["beta"]
            total = rng.uniform(-.15, .2, len(z)); excess = total - .001
            next_ids = ids[t + 1] if t < 2 else ids[t]
            args = (market.m, total, .004, market.impact, market.wealth, excess, next_ids)
            old = transition(ids[t], old_h, old_aim, *args)
            new = transition(ids[t], new_h, new_aim, *args)
            for key in ("weights", "trades", "next_pretrade", "gross_excess", "impact_cost"):
                np.testing.assert_allclose(new[key], old[key], atol=ATOL, rtol=RTOL, err_msg=dates[t])
            direct_orders = market.wealth * (new["weights"] - new_h)
            direct_fractional_cost = .5 * np.sum(market.impact * direct_orders**2) / market.wealth
            np.testing.assert_allclose(new["impact_cost"], direct_fractional_cost, atol=1e-13)
            self.assertEqual(new["source_exit_trade_cost"], 0.)  # Inherited source convention.
            if t < 2:
                self.assertEqual(new["dropped_ids"].tolist(), [t])
                self.assertEqual(new["next_pretrade"][-1], 0.)
                self.assertGreater(np.linalg.norm(new["next_pretrade"][:-2]), 0.)
            old_h, new_h = old["next_pretrade"], new["next_pretrade"]

    def test_invalid_masks_dates_duplicate_ids_and_current_mismatches_rejected(self):
        with self.assertRaises(ValueError):
            CalendarReference(["2000-01-31"], [1], [[.5]], [[1]], [1.])
        with self.assertRaises(ValueError):
            CalendarReference(["2000-01-30"], [1], [[.5]], np.ones((1, 1), bool), [1.])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            CalendarReference(["2000-01-31"]*2, [1, 1], [[.5], [.6]], np.ones((2, 1), bool), [1.])
        with self.assertRaises(ValueError):
            CalendarReference(["2000-01-31"], [None], [[.5]], np.ones((1, 1), bool), [1.])
        for weights in ([.2, .2], [-1., 2.], [], [np.nan]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                CalendarReference(["2000-01-31"], [1], [[.5]], np.ones((1, 1), bool), weights)
        ref = CalendarReference(["2000-01-31"], [1], [[.5]], np.ones((1, 1), bool), [1.])
        with self.assertRaisesRegex(ValueError, "differs"):
            ref.at("2000-01-31", [1], [[.4]])
        with self.assertRaises(ValueError):
            ref.at("2000-01-31", [1, 1], [[.5], [.5]])
        with self.assertRaises(ValueError):
            Innovation([[np.nan]], np.ones((1, 1), bool), {})

    def test_invalid_pipeline_shapes_nonfinite_inputs_and_training_maturity_rejected(self):
        _, parent, _, bases, settings, theta, fmap, innovations = temporal_fixture(13, 7)
        with self.assertRaises(ValueError):
            TemporalFeatureMap(parent, theta, np.arange(13) * 2)
        with self.assertRaises(ValueError):
            TemporalFeatureMap(parent, theta[:-1], np.arange(13))
        market = bases[0].sample.market
        for gains in (np.ones(12), np.full(13, np.nan)):
            with self.assertRaises(ValueError):
                fmap.inputs(market.characteristics[0], innovations[0][0], gains)
        with self.assertRaises(ValueError):
            fmap.inputs(market.characteristics[0][:, :-1], innovations[0][0], np.ones(13))
        with self.assertRaisesRegex(ValueError, "Unmatured"):
            evaluate(bases, innovations, fmap, np.ones(13), replace(settings, cutoff="1999-01-31"),
                     rho=.03, objective_scale=1.)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate([bases[0], bases[0]], [innovations[0], innovations[0]], fmap, np.ones(13), settings,
                     rho=.03, objective_scale=1.)
        for rho, scale in ((-.1, 1.), (0., 0.), (0., np.nan)):
            with self.assertRaises(ValueError):
                evaluate(bases, innovations, fmap, np.ones(13), settings, rho=rho, objective_scale=scale)
        with self.assertRaises(ValueError):
            evaluate(bases, [row[:12] for row in innovations], fmap, np.ones(13), settings,
                     rho=.03, objective_scale=1.)


if __name__ == "__main__":
    unittest.main()
