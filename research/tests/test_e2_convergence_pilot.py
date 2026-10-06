"""Independent toy checks; never open real research data or run historical fits."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from test_feature_geometry import fixture
from feature_geometry.features import project_scales
from feature_geometry.policy import GeometryModel
from geometry_study import kernel

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e2_pilot_review_target", ROOT / "scripts/e2_convergence_pilot_20260928.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        _, cls.fmap, old, settings, _ = fixture(dimensions=6, frequencies=8, months=3)
        cls.bases = [kernel.LeanBasis(b.sample) for b in old]
        cls.settings = replace(settings, max_steps=8, gradient_tolerance=1e-20)
        cls.normalizer = kernel.evaluate(
            cls.bases, cls.fmap, np.zeros(cls.fmap.dimensions), cls.settings)["value"]
        cls.first = kernel.matrix_fit(cls.bases, cls.fmap, cls.settings,
                                      objective_scale=cls.normalizer)
        cls.current = kernel.evaluate(cls.bases, cls.fmap, cls.first["theta"], cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def resume(self, **changes):
        args = dict(bases=self.bases, fmap=self.fmap, settings=self.settings,
                    normalizer=self.normalizer, theta=self.first["theta"],
                    current=self.current, start_update=8, final_update=32)
        args.update(changes)
        return pilot.continue_updates(**args)

    def test_resume_recovers_uninterrupted_solver_and_every_accepted_update(self):
        actual = self.resume()
        expected = kernel.matrix_fit(self.bases, self.fmap,
                                     replace(self.settings, max_steps=32),
                                     objective_scale=self.normalizer)
        self.assertEqual(self.first["trace"][-1]["iteration"], 8)
        self.assertEqual(actual["update"], 32)
        self.assertEqual(actual["status"], expected["status"])
        np.testing.assert_allclose(actual["theta"], expected["theta"], atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(actual["current"]["beta"], expected["beta"], atol=1e-12, rtol=1e-11)
        self.assertEqual(len(actual["trace"]), 24)
        for old, new in zip(expected["trace"][9:], actual["trace"]):
            for key in ("iteration", "backtracks"):
                self.assertEqual(old[key], new[key])
            for key in ("value", "step", "projected_gradient_before"):
                np.testing.assert_allclose(old[key], new[key], atol=1e-12, rtol=1e-11)
        final = actual["theta"]
        gradient = kernel.gradient(self.bases, self.fmap, final,
                                   actual["current"]["beta"], self.settings, "full") / self.normalizer
        expected_norm = np.max(abs(project_scales(final + gradient, self.settings.bound) - final))
        self.assertAlmostEqual(actual["projected_gradient_final"], expected_norm, places=13)
        self.assertTrue(np.all(np.diff([self.current["value"]] +
                                     [row["value"] for row in actual["trace"]]) >= 0))

    def test_stopping_uses_current_gradient_and_neutral_continuation_is_identity(self):
        callback = []
        result = self.resume(settings=replace(self.settings, gradient_tolerance=10.),
                             on_accept=lambda *args: callback.append(args))
        self.assertEqual(result["status"], "PROJECTED_GRADIENT_TOLERANCE")
        self.assertEqual(result["update"], 8)
        self.assertEqual(callback, [])
        np.testing.assert_array_equal(result["theta"], self.first["theta"])
        np.testing.assert_array_equal(result["current"]["beta"], self.current["beta"])
        self.assertEqual(result["projected_gradient_start"], result["projected_gradient_final"])

    def test_unchanged_budget_and_inputs_reject_invalid_states(self):
        for normalizer in (0., -1., np.nan, np.inf):
            with self.subTest(normalizer=normalizer), self.assertRaises(ValueError):
                self.resume(normalizer=normalizer)
        for start, final in ((9, 8), (-1, 8), (8, 33), (8.5, 32)):
            with self.subTest(budget=(start, final)), self.assertRaises(ValueError):
                self.resume(start_update=start, final_update=final)
        for theta in (np.full(6, np.nan), np.ones(6),
                      np.array([1., -1., 0., 0., 0., 0.]), np.zeros(5)):
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                self.resume(theta=theta)
        for key, bad in (("beta", np.full(self.fmap.columns, np.nan)),
                         ("beta", np.zeros(self.fmap.columns - 1)),
                         ("solve_residual", .01)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.resume(current=dict(self.current, **{key: bad}))

    def test_causality_duplicate_and_unordered_dates_fail_before_updates(self):
        cases = [dict(settings=replace(self.settings, cutoff="1999-02-01")),
                 dict(bases=[self.bases[0], self.bases[0]]),
                 dict(bases=list(reversed(self.bases))), dict(bases=[])]
        for changes in cases:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.resume(**changes)

    def test_checkpoint_interruption_recovers_same_final_fit_and_tamper_fails(self):
        with tempfile.TemporaryDirectory(prefix="e2-independent-checkpoint-") as temporary:
            folder = Path(temporary)
            previous = pilot.save_checkpoint(folder, 8, self.first["theta"], self.current,
                                             {}, "locked-test", None)

            def save(update, theta, current, detail):
                nonlocal previous
                previous = pilot.save_checkpoint(folder, update, theta, current, detail,
                                                 "locked-test", previous)

            self.resume(final_update=17, on_accept=save)
            arrays, records, tail = pilot.recover_checkpoints(folder, "locked-test")
            self.assertEqual(records[-1]["update"], 17)
            self.assertEqual(tail, previous)
            resumed = self.resume(theta=arrays["theta"], current=pilot.current_from_arrays(arrays),
                                  start_update=17)
            expected = self.resume()
            np.testing.assert_array_equal(resumed["theta"], expected["theta"])
            np.testing.assert_array_equal(resumed["current"]["beta"], expected["current"]["beta"])
            with self.assertRaises(ValueError):
                pilot.recover_checkpoints(folder, "different-lock")
            commit = folder / "checkpoints/update_010.json"
            record = json.loads(commit.read_text())
            record["previous_commit_sha256"] = "bad-parent"
            commit.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                pilot.recover_checkpoints(folder, "locked-test")


class ValidationAccountingTests(unittest.TestCase):
    def test_neutral_twelve_month_path_charges_cost_and_carries_own_holdings(self):
        """Exercise actual validate_year against a separately calculated toy archive."""
        rng, fmap, bases, settings, theta = fixture()
        model = GeometryModel(fmap, theta, np.linspace(-.15, .1, fmap.columns),
                              "1999-12-31", settings.bound, "toy-causal-parent")
        dates = pd.date_range("1999-12-31", "2000-11-30", freq="ME")
        markets, months, outcome_records, saved_monthly = {}, {}, [], []
        ids = np.array([10, 20, 30, 40])
        holdings = np.array([.02, -.03, .01, .04])
        synthetic_archives = []
        for index, day in enumerate(dates):
            n = len(ids)
            z = rng.normal(size=(n, fmap.dimensions))
            vol = rng.uniform(.8, 1.2, n)
            inertia = np.diag(np.linspace(.15, .6, n)) + np.full((n, n), .015)
            market = replace(bases[0].sample.market, decision_date=str(day.date()),
                             ids=ids.copy(), characteristics=(z,) * 13,
                             volatility=(vol,) * 13, growth=(np.ones(n),) * 13,
                             m=inertia, covariance=np.eye(n) * .02,
                             impact=np.linspace(.001, .004, n), wealth=2.)
            markets[day] = market
            excess = np.array([.015, -.025, .035, -.005]) + .001 * index
            total = excess + .002
            market_return = .01
            months[day] = pd.DataFrame({"id": ids, "ret_ld1": excess}).set_index("id", drop=False)
            for sid, ret in zip(ids, total):
                outcome_records.append(dict(id=sid, eom=day, tr_ld1=ret, mu_ld1=market_return))
            target = model.aim(z, vol, str(day.date()))
            weight = holdings + (np.eye(n) - inertia) @ (target - holdings)
            trade = weight - holdings
            drift = weight * (1 + total) / (1 + market_return)
            next_ids = np.array([ids[2], ids[0], ids[3], 100 + index])
            lookup = dict(zip(ids, drift))
            next_h = np.array([lookup.get(sid, 0.) for sid in next_ids])
            gross = math.fsum(float(w) * float(r) for w, r in zip(weight, excess))
            cost = market.wealth / 2 * math.fsum(float(l) * float(q)**2
                                                for l, q in zip(market.impact, trade))
            saved_monthly.append(dict(seed=1, method="full", decision_eom=day,
                                      gross_excess=gross, impact_cost=cost, net_excess=gross-cost,
                                      turnover=abs(trade).sum(), gross_exposure=abs(weight).sum()))
            synthetic_archives.append(dict(ids=ids.copy(), next_ids=next_ids,
                                           pretrade=holdings[:, None], target=target[:, None],
                                           weight=weight[:, None], seeds=np.array([1]),
                                           methods=np.array(["full"])))
            holdings, ids = next_h, next_ids

        class FakeHistory:
            def __init__(self, *args, **kwargs):
                self.data = None
                self.months = months
                self.consumed = {}

            def market(self, day):
                return markets[day]

        with tempfile.TemporaryDirectory(prefix="e2-independent-path-") as temporary:
            root = Path(temporary)
            parent = root / "parent"
            (parent / "paths").mkdir(parents=True)
            folder = root / "diagnostic"
            folder.mkdir()
            inventory = {}
            for day, arrays in zip(dates, synthetic_archives):
                path = parent / "paths" / f"{day.date()}.npz"
                np.savez_compressed(path, **arrays)
                inventory[str(path.relative_to(root))] = pilot.sha(path)

            def parquet(path, *args, **kwargs):
                return (pd.DataFrame(saved_monthly) if Path(path).name == "monthly.parquet"
                        else pd.DataFrame(outcome_records))

            with patch.object(pilot, "ROOT", root), patch.object(pilot, "PARENT", parent), \
                    patch.object(pilot, "CanonicalHistory", FakeHistory), \
                    patch.object(pilot.pd, "read_parquet", side_effect=parquet), \
                    patch.object(pilot, "event"):
                result, attempt = pilot.validate_year(2000, model, model.theta, model.beta,
                                                       {}, folder, inventory, {})
            self.assertTrue(result["passed"])
            self.assertEqual(result["months"], 12)
            self.assertFalse(result["statistical_significance_established"])
            self.assertFalse(result["intermediate_iteration_selected_using_returns"])
            self.assertEqual(result["mean_monthly_target_l2_change"], 0.)
            for value in result["continued_minus_archived"].values():
                self.assertAlmostEqual(value, 0., places=14)
            values = pd.read_csv(attempt / "monthly.csv")
            old = values[values.method == "archived8"]
            gross, net = old.gross_excess.to_numpy(), old.net_excess.to_numpy()
            self.assertGreater(old.iloc[0].impact_cost, 0.)
            metrics = result["metrics"]["archived8"]
            self.assertAlmostEqual(metrics["source_gross_sample_variance_utility"],
                                   12 * (net.mean() - 5 * gross.var(ddof=1)), places=13)
            self.assertAlmostEqual(metrics["net_population_utility"],
                                   12 * (net.mean() - 5 * net.var(ddof=0)), places=13)
            with np.load(attempt / "2000-01-31.npz") as snapshot:
                np.testing.assert_allclose(snapshot["pretrade"][:, 0],
                                           synthetic_archives[1]["pretrade"][:, 0], atol=1e-14)
                self.assertEqual(snapshot["pretrade"][-1, 0], 0.)


if __name__ == "__main__":
    unittest.main()
