"""Independent calculus, original-kernel, chronology and decision-contract checks."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from baseline.core import inertia, moment, normalize_signal, transition
from feature_geometry.features import FeatureMap, project_scales
from feature_geometry.moments import MarketMoment, MomentBasis, TrainingSample
from feature_geometry.policy import GeometryModel, act
from feature_geometry.training import FitSettings, fit_geometry, profile, joint_value_gradient


def fixture(dimensions=4, frequencies=3, months=2):
    rng = np.random.default_rng(20260914)
    fmap = FeatureMap(rng.normal(size=(dimensions, frequencies)), "synthetic seed 20260914")
    bases = []
    for t in range(months):
        n = 16 + t
        x = rng.normal(size=(n, 4))
        sigma = x @ x.T * .002 + np.eye(n) * .015
        impact = rng.uniform(.01, .05, n)
        m, _ = inertia(sigma, impact, 2., .001)
        market = MarketMoment(f"1999-{t+1:02d}-28", np.arange(n),
                              tuple(rng.normal(size=(n, dimensions)) for _ in range(13)),
                              tuple(rng.uniform(.5, 1.5, n) for _ in range(13)),
                              tuple(rng.uniform(.9, 1.1, n) for _ in range(13)),
                              m, sigma, impact, 2.)
        bases.append(MomentBasis(TrainingSample(market, rng.normal(size=n), f"1999-{t+2:02d}-28")))
    settings = FitSettings("1999-12-31", ridge=.3, scale_penalty=.01, max_steps=4)
    theta = project_scales(np.linspace(-.2, .2, dimensions))
    return rng, fmap, bases, settings, theta


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_identity_features_match_original_exactly(self):
        _, fmap, bases, _, _ = fixture()
        for basis in bases:
            for z, vol in zip(basis.sample.market.characteristics, basis.sample.market.volatility):
                arg = z @ fmap.frequencies
                expected = normalize_signal(np.column_stack((np.cos(arg), np.sin(arg))), vol)
                np.testing.assert_array_equal(fmap.signals(z, vol, np.zeros(fmap.dimensions)), expected)

    def test_identity_all_moments_and_constant_ridge_match_original(self):
        _, fmap, bases, settings, _ = fixture()
        original = []
        for basis in bases:
            sample, market = basis.sample, basis.sample.market
            signals = basis.signals(fmap, np.zeros(fmap.dimensions))
            expected = moment(signals, market.growth, market.m, market.covariance, market.impact,
                              market.wealth, sample.returns, return_operators=True)
            actual = basis.matrices(fmap, np.zeros(fmap.dimensions))
            for a, b in (("score", "r_tilde"), ("risk", "risk"), ("cost", "tc"), ("omega", "omega"), ("change", "omega_change")):
                np.testing.assert_allclose(actual[a], expected[b], atol=1e-12, rtol=1e-11)
            original.append(expected)
        score = np.mean([r["r_tilde"] for r in original], axis=0)
        matrix = np.mean([r["denom"] for r in original], axis=0)
        beta = np.linalg.solve(matrix + settings.ridge*np.eye(fmap.columns), score)
        actual = profile(bases, fmap, np.zeros(fmap.dimensions), settings)
        np.testing.assert_allclose(actual["beta"], beta, atol=1e-12, rtol=1e-11)

    def test_signal_adjoint_includes_center_norm_and_volatility(self):
        rng, fmap, bases, _, theta = fixture()
        market = bases[0].sample.market
        z, vol = market.characteristics[0], market.volatility[0]
        direction = project_scales(rng.normal(size=fmap.dimensions))
        upstream = rng.normal(size=(len(z), fmap.columns))
        epsilon = 1e-6
        fd = np.sum(upstream*(fmap.signals(z, vol, theta+epsilon*direction)-fmap.signals(z, vol, theta-epsilon*direction)))/(2*epsilon)
        analytic = fmap.vjp(z, vol, theta, upstream)@direction
        np.testing.assert_allclose(analytic, fd, atol=2e-8, rtol=2e-7)
        constant_only = np.zeros_like(upstream); constant_only[:, 0] = 1
        np.testing.assert_array_equal(fmap.vjp(z, vol, theta, constant_only), np.zeros(fmap.dimensions))

    def test_profile_gradients_for_each_objective(self):
        _, fmap, bases, settings, theta = fixture()
        direction = project_scales(np.array([.3, -.2, .4, -.5]))
        for objective in ("full", "risk_only", "return_prediction"):
            with self.subTest(objective=objective):
                analytic = profile(bases, fmap, theta, settings, objective)["gradient"]@direction
                epsilon = 1e-6
                fd = (profile(bases, fmap, theta+epsilon*direction, settings, objective, False)["value"]
                      - profile(bases, fmap, theta-epsilon*direction, settings, objective, False)["value"])/(2*epsilon)
                np.testing.assert_allclose(analytic, fd, atol=2e-8, rtol=2e-6)

    def test_joint_gradient_matches_finite_difference_and_profile_envelope(self):
        rng, fmap, bases, settings, theta = fixture()
        optimal = profile(bases, fmap, theta, settings)
        joint = joint_value_gradient(bases, fmap, theta, optimal["beta"], settings)
        np.testing.assert_allclose(joint["value"], optimal["value"], atol=1e-12)
        np.testing.assert_allclose(joint["theta_gradient"], optimal["gradient"], atol=1e-12)
        np.testing.assert_allclose(joint["beta_gradient"], 0, atol=1e-12)
        beta = rng.normal(size=fmap.columns)
        dt, db = rng.normal(size=fmap.dimensions), rng.normal(size=fmap.columns)
        joint = joint_value_gradient(bases, fmap, theta, beta, settings)
        epsilon = 1e-6
        fd = (joint_value_gradient(bases, fmap, theta+epsilon*dt, beta+epsilon*db, settings)["value"]
              - joint_value_gradient(bases, fmap, theta-epsilon*dt, beta-epsilon*db, settings)["value"])/(2*epsilon)
        np.testing.assert_allclose(joint["theta_gradient"]@dt+joint["beta_gradient"]@db, fd, atol=1e-7, rtol=1e-6)

    def test_projection_satisfies_box_zero_sum_and_kkt(self):
        value = np.array([10., -5., 2., .1, .2])
        bound = np.log(2.)
        projected = project_scales(value, bound)
        self.assertLess(abs(projected.sum()), 1e-12)
        self.assertLessEqual(np.max(abs(projected)), bound)
        interior = abs(projected) < bound-1e-10
        shifts = value[interior]-projected[interior]
        self.assertGreater(len(shifts), 0)
        np.testing.assert_allclose(shifts, shifts[0], atol=1e-12)
        np.testing.assert_allclose(project_scales(projected, bound), projected, atol=1e-12)

    def test_feature_scaling_cancels_but_input_scaling_changes_span(self):
        rng, fmap, bases, _, theta = fixture()
        z = rng.normal(size=(120, fmap.dimensions)); vol = rng.uniform(.5, 1.5, len(z))
        arg = z@fmap.frequencies
        raw = np.column_stack((np.cos(arg), np.sin(arg)))
        np.testing.assert_allclose(normalize_signal(raw, vol), normalize_signal(raw*np.arange(1, fmap.columns), vol), atol=1e-12)
        original = fmap.signals(z, vol, np.zeros(fmap.dimensions))
        changed = fmap.signals(z, vol, theta)@rng.normal(size=fmap.columns)
        residual = changed-original@np.linalg.lstsq(original, changed, rcond=None)[0]
        self.assertGreater(np.linalg.norm(residual)/np.linalg.norm(changed), .01)

    def test_post_normalization_rotation_invariance(self):
        rng, fmap, bases, settings, theta = fixture()
        q, _ = np.linalg.qr(rng.normal(size=(fmap.columns, fmap.columns)))
        result = profile(bases, fmap, theta, settings)
        k, r = result["risk"]+result["cost"], result["score"]
        rotated_beta = np.linalg.solve(q.T@k@q + settings.ridge*np.eye(len(r)), q.T@r)
        np.testing.assert_allclose(q@rotated_beta, result["beta"], atol=1e-12)

    def test_fit_is_feasible_monotone_and_refits_full_coefficients_for_controls(self):
        _, fmap, bases, settings, _ = fixture()
        for objective in ("full", "risk_only", "return_prediction"):
            result = fit_geometry(bases, fmap, settings, objective)
            values = [r["value"] for r in result["trace"]]
            self.assertTrue(np.all(np.diff(values) >= 0))
            self.assertGreater(values[-1], values[0])
            self.assertLess(abs(result["theta"].sum()), 1e-12)
            self.assertLessEqual(np.max(abs(result["theta"])), settings.bound+1e-12)
            beta = np.linalg.solve(result["risk"]+result["cost"]+settings.ridge*np.eye(fmap.columns), result["score"])
            np.testing.assert_allclose(result["beta"], beta, atol=1e-12)
            if objective != "full":
                self.assertGreater(np.linalg.norm(result["geometry_beta"]-result["beta"]), 1e-6)

    def test_115_characteristics_and_source_sized_output(self):
        rng = np.random.default_rng(5)
        fmap = FeatureMap(rng.normal(size=(115, 256)))
        z = rng.normal(size=(20, 115)); vol = np.ones(20)
        theta = project_scales(np.linspace(-1., 1., 115))
        signals = fmap.signals(z, vol, theta)
        self.assertEqual(signals.shape, (20, 513))
        self.assertEqual(fmap.vjp(z, vol, theta, np.ones_like(signals)).shape, (115,))

    def test_unmatured_or_duplicate_dates_are_rejected(self):
        _, fmap, bases, settings, theta = fixture()
        with self.assertRaisesRegex(ValueError, "Unmatured"):
            profile(bases, fmap, theta, replace(settings, cutoff="1999-01-31"))
        with self.assertRaisesRegex(ValueError, "unique"):
            profile([bases[0], bases[0]], fmap, theta, settings)

    def test_alternate_iso_formats_cannot_bypass_maturity(self):
        _, fmap, bases, settings, theta = fixture()
        early = replace(settings, cutoff="19990131")
        self.assertEqual(early.cutoff, "1999-01-31")
        with self.assertRaisesRegex(ValueError, "Unmatured"):
            profile(bases, fmap, theta, early)
        market = replace(bases[0].sample.market, decision_date="19990128")
        sample = replace(bases[0].sample, market=market, available_at="19990228")
        self.assertEqual(sample.market.decision_date, "1999-01-28")
        self.assertEqual(sample.available_at, "1999-02-28")

    def test_decision_uses_own_holdings_original_m_and_no_future_arguments(self):
        rng, fmap, bases, settings, theta = fixture()
        result = fit_geometry(bases, fmap, replace(settings, max_steps=1))
        model = GeometryModel.from_fit(fmap, result)
        market = bases[0].sample.market
        m_before = market.m.copy()
        h = rng.normal(size=len(market.ids))
        decision = act(model, settings.cutoff, market.ids, market.characteristics[0], market.volatility[0], market.m, market.ids, h)
        original = transition(market.ids, h, decision["aim"], market.m, np.zeros(len(h)), .01,
                              market.impact, market.wealth, np.zeros(len(h)), market.ids)
        np.testing.assert_array_equal(decision["weights"], original["weights"])
        np.testing.assert_array_equal(market.m, m_before)
        changed = act(model, settings.cutoff, market.ids, market.characteristics[0], market.volatility[0], market.m, market.ids, h+1)
        np.testing.assert_allclose(changed["weights"]-decision["weights"], market.m@np.ones(len(h)), atol=1e-12)
        self.assertNotIn("returns", inspect.signature(act).parameters)
        with self.assertRaisesRegex(ValueError, "future"):
            model.aim(market.characteristics[0], market.volatility[0], "1999-01-01")
        with self.assertRaisesRegex(ValueError, "IDs"):
            act(model, settings.cutoff, market.ids, market.characteristics[0], market.volatility[0], market.m, market.ids[::-1], h)

    def test_serialization_roundtrip_refuses_overwrite_and_detects_tampering(self):
        _, fmap, bases, settings, theta = fixture()
        model = GeometryModel.from_fit(fmap, fit_geometry(bases, fmap, replace(settings, max_steps=0)))
        with tempfile.TemporaryDirectory(prefix="geometry_model_") as folder:
            path = Path(folder)/"model.npz"
            model.save(path)
            loaded = GeometryModel.load(path)
            np.testing.assert_array_equal(model.beta, loaded.beta)
            np.testing.assert_array_equal(model.feature_map.frequencies, loaded.feature_map.frequencies)
            with self.assertRaises(FileExistsError): model.save(path)
            with np.load(path, allow_pickle=False) as z:
                altered = {k:z[k].copy() for k in z.files}
            altered["beta"][0] += .1
            np.savez_compressed(Path(folder)/"changed.npz", **altered)
            with self.assertRaisesRegex(ValueError, "Changed"):
                GeometryModel.load(Path(folder)/"changed.npz")

    def test_geometry_cache_keys_include_inputs_cutoff_and_frequency_source(self):
        _, fmap, bases, settings, theta = fixture()
        settings = replace(settings, max_steps=0)
        original = fit_geometry(bases, fmap, settings)
        changed_cutoff = fit_geometry(bases, fmap, replace(settings, cutoff="2000-01-31"))
        changed_geometry = fit_geometry(bases, fmap, settings, initial_theta=theta)
        changed_seed = fit_geometry(bases, FeatureMap(fmap.frequencies, "different provenance"), settings)
        self.assertEqual(len({r["geometry_cache_key"] for r in (original, changed_cutoff, changed_geometry, changed_seed)}), 4)

    def test_invalid_inputs_fail_without_silent_repair(self):
        _, fmap, bases, settings, theta = fixture()
        market = bases[0].sample.market
        with self.assertRaises(ValueError): FeatureMap(np.full((4, 3), np.nan))
        with self.assertRaises(ValueError): fmap.signals(np.ones((10, 4)), np.ones(10), theta)
        with self.assertRaises(ValueError): replace(settings, ridge=-1)
        with self.assertRaises(ValueError): replace(market, characteristics=market.characteristics[:12])
        with self.assertRaises(ValueError): replace(market, impact=-market.impact)
        with self.assertRaises(ValueError): profile(bases, fmap, theta, settings, objective="typo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
