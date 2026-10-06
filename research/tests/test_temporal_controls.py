"""Synthetic current-only control checks; no historical settings are chosen."""
import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from feature_geometry.features import fingerprint
from geometry_study.kernel import evaluate as parent_evaluate
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.profile import evaluate
from temporal_extension.reference import Innovation
from temporal_extension.streaming import StreamingObjective
from temporal_extension.model import TemporalModel
from temporal_extension.optimizer import OptimizerSettings
from temporal_study.controls import (CURRENT_ONLY_KIND, CurrentOnlyFeatureMap, CurrentOnlyModel,
                                     current_only_batch_factory, current_only_reference_fingerprint,
                                     train_current_only_profile, save_current_only_result,
                                     fit_fixed_profile, save_fixed_result)
from test_temporal_extension import temporal_fixture


class CurrentOnlyControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def fixture(self, frequencies=9):
        rng, parent, _, bases, settings, theta, temporal, histories = temporal_fixture(
            115, frequencies)
        control = CurrentOnlyFeatureMap(parent, theta, temporal.groups)
        return rng, parent, bases, settings, theta, temporal, control, histories

    def test_neutral_full_shape_features_moments_and_coefficients_match_parent(self):
        _, parent, bases, settings, theta, _, control, histories = self.fixture(256)
        self.assertEqual(control.gain_count, 13)
        self.assertEqual(parent.dimensions, 115)
        self.assertEqual(parent.columns, 513)
        for basis, history in zip(bases, histories):
            market = basis.sample.market
            new, old = [], []
            for z, vol, innovation in zip(market.characteristics, market.volatility, history):
                np.testing.assert_array_equal(control.inputs(z, innovation, np.ones(13)), z)
                new.append(control.signals(z, vol, innovation, np.ones(13)))
                old.append(parent.signals(z, vol, theta))
                np.testing.assert_array_equal(new[-1], old[-1])
            for actual, expected in zip(basis.operators(new), basis.operators(old)):
                np.testing.assert_array_equal(actual, expected)
        actual = evaluate(bases, histories, control, np.ones(13), settings,
                          rho=.03, objective_scale=.7)
        expected = parent_evaluate(bases, parent, theta, settings)
        for key in ('score', 'risk', 'cost', 'beta'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(actual['parent_value'], expected['value'], atol=1e-11)
        self.assertLess(actual['solve_residual'], 1e-9)

    def test_current_gain_arithmetic_and_once_only_frozen_scales(self):
        _, parent, bases, _, theta, _, control, histories = self.fixture()
        market = bases[0].sample.market
        z, vol = market.characteristics[0], market.volatility[0]
        gains = np.linspace(.25, 1.75, 13)
        expected = z * gains[control.groups]
        np.testing.assert_array_equal(control.inputs(z, histories[0][0], gains), expected)
        np.testing.assert_array_equal(control.signals(z, vol, histories[0][0], gains),
                                      parent.signals(expected, vol, theta))
        np.testing.assert_array_equal(control.theta, theta)

    def test_current_only_has_no_history_or_source_mask_dependency(self):
        _, _, bases, _, _, _, control, histories = self.fixture()
        market = bases[0].sample.market
        z, vol = market.characteristics[0], market.volatility[0]
        gains = np.linspace(.4, 1.6, 13)
        absent = Innovation(np.full_like(z, np.nan), np.zeros_like(z, dtype=bool), {})
        arbitrary = Innovation(np.full_like(z, 10000.), np.ones_like(z, dtype=bool), {})
        expected = control.signals(z, vol, None, gains)
        for history in (absent, arbitrary, histories[0][0], object()):
            np.testing.assert_array_equal(control.signals(z, vol, history, gains), expected)
        self.assertGreater(np.linalg.norm(control.inputs(z, absent, gains) - z), 0.)

    def test_signal_derivative_includes_centering_normalization_and_volatility(self):
        rng, _, bases, _, _, _, control, _ = self.fixture()
        market = bases[0].sample.market
        z, vol = market.characteristics[0], market.volatility[0]
        gains = np.linspace(.45, 1.55, 13)
        upstream = rng.normal(size=(len(z), control.parent.columns))
        actual = control.vjp(z, vol, None, gains, upstream)
        finite_difference = np.empty(13)
        epsilon = 1e-6
        for j in range(13):
            delta = np.eye(13)[j] * epsilon
            plus = np.sum(upstream * control.signals(z, vol, None, gains + delta))
            minus = np.sum(upstream * control.signals(z, vol, None, gains - delta))
            finite_difference[j] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(actual, finite_difference, atol=2e-7, rtol=2e-5)
        upstream[:, 1:] = 0.
        np.testing.assert_array_equal(control.vjp(z, vol, None, gains, upstream), np.zeros(13))

    def test_profile_gradient_rebuilds_all_slices_and_conditional_beta(self):
        _, _, bases, settings, _, _, control, histories = self.fixture(16)
        gains = np.linspace(.45, 1.55, 13)
        actual = evaluate(bases, histories, control, gains, settings,
                          rho=.03, objective_scale=.7)
        finite_difference = np.empty(13)
        epsilon = 1e-6
        for j in range(13):
            delta = np.eye(13)[j] * epsilon
            plus = evaluate(bases, histories, control, gains + delta, settings,
                            rho=.03, objective_scale=.7, with_gradient=False)
            minus = evaluate(bases, histories, control, gains - delta, settings,
                             rho=.03, objective_scale=.7, with_gradient=False)
            finite_difference[j] = (plus['value'] - minus['value']) / (2 * epsilon)
        np.testing.assert_allclose(actual['gradient'], finite_difference, atol=2e-7, rtol=2e-5)
        neutral = evaluate(bases, histories, control, np.ones(13), settings,
                           rho=.03, objective_scale=.7, with_gradient=False)
        self.assertGreater(np.linalg.norm(actual['beta'] - neutral['beta']), 1e-4)
        self.assertGreater(np.linalg.norm(actual['cost'] - neutral['cost']), 1e-4)
        system = actual['risk'] + actual['cost'] + settings.ridge * np.eye(control.parent.columns)
        np.testing.assert_allclose(system @ actual['beta'], actual['score'], atol=1e-11)

    def test_streaming_control_matches_reference_profile_without_history(self):
        _, _, bases, settings, _, _, control, histories = self.fixture()
        calls = []
        def basis_factory():
            calls.append(True)
            return iter(bases)
        factory = current_only_batch_factory(basis_factory)
        gains = np.linspace(.4, 1.6, 13)
        actual = StreamingObjective(factory, control, settings,
                                    rho=.03, objective_scale=.7)(gains)
        expected = evaluate(bases, histories, control, gains, settings,
                            rho=.03, objective_scale=.7)
        for key in ('value', 'parent_value', 'beta', 'score', 'risk', 'cost', 'gradient'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=1e-10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(actual['training_months'], 2)
        self.assertEqual(actual['coverage']['all_13_slices']['bypassed'], 0)
        self.assertTrue(actual['coverage']['reason_counts_complete'])

    def test_fingerprints_distinguish_transform_type_and_frozen_parent_content(self):
        _, parent, _, _, theta, temporal, control, _ = self.fixture()
        self.assertEqual(control.kind, CURRENT_ONLY_KIND)
        self.assertNotEqual(control.fingerprint, temporal.fingerprint)
        self.assertEqual(control.fingerprint, CurrentOnlyFeatureMap(parent, theta, control.groups).fingerprint)
        self.assertNotEqual(control.fingerprint,
                            CurrentOnlyFeatureMap(parent, -theta, control.groups).fingerprint)

    def test_economics_and_parent_data_are_not_mutated(self):
        _, parent, bases, settings, theta, _, control, histories = self.fixture()
        def identity():
            return fingerprint(parent.frequencies, theta, control.theta,
                               *(basis.sample.market.fingerprint for basis in bases),
                               *(basis.sample.market.m for basis in bases),
                               *(basis.sample.market.covariance for basis in bases),
                               *(basis.sample.market.impact for basis in bases), settings.ridge)
        before = identity()
        evaluate(bases, histories, control, np.linspace(.2, 1.8, 13), settings,
                 rho=.04, objective_scale=1.)
        self.assertEqual(before, identity())

    def test_invalid_map_inputs_and_parent_batch_contract_are_rejected(self):
        _, parent, bases, _, theta, _, control, _ = self.fixture()
        z = bases[0].sample.market.characteristics[0]
        for bad_z, gains in ((z[:, :-1], np.ones(13)), (z, np.ones(12)),
                             (z, np.full(13, np.nan))):
            with self.assertRaises(ValueError):
                control.inputs(bad_z, None, gains)
        with self.assertRaises(ValueError):
            CurrentOnlyFeatureMap(parent, theta, np.zeros(115, dtype=float))
        with self.assertRaises(ValueError):
            current_only_batch_factory([])
        with self.assertRaises(ValueError):
            list(current_only_batch_factory(lambda: [object()])())

    def training_problem(self, neutral=False):
        _, parent, bases, settings, theta, temporal, control, histories = self.fixture()
        beta = parent_evaluate(bases, parent, theta, settings)['beta']
        optimizer = OptimizerSettings(lower=1. if neutral else 0., upper=1. if neutral else 2.,
                                      tolerance=1e-8, max_steps=3, max_evaluations=40,
                                      max_backtracks=12, initial_step=.1, armijo=1e-4,
                                      wall_seconds=30.)
        kwargs = dict(expected_neutral_beta=beta,
                      expected_parent_feature_fingerprint=temporal.fingerprint,
                      parent_fingerprint='synthetic-parent', source_fingerprint='synthetic-source',
                      rho=.03, objective_scale=.7, optimizer_settings=optimizer)
        return bases, settings, temporal, control, histories, kwargs

    def test_training_profiles_current_gains_and_persists_explicit_model_kind(self):
        bases, settings, temporal, control, _, kwargs = self.training_problem()
        model, receipt = train_current_only_profile(lambda: iter(bases), control, settings, **kwargs)
        self.assertIs(type(model), CurrentOnlyModel)
        self.assertTrue(receipt['neutral_gate']['passed'])
        self.assertGreater(receipt['accepted_steps'], 0)
        self.assertGreater(np.linalg.norm(model.gains - 1.), 0.)
        self.assertGreater(np.linalg.norm(model.beta - kwargs['expected_neutral_beta']), 0.)
        self.assertEqual(receipt['model_kind'], 'learned_current_only')
        self.assertEqual(receipt['parent_feature_fingerprint'], temporal.fingerprint)
        self.assertEqual(receipt['control_feature_fingerprint'], control.fingerprint)
        self.assertEqual(receipt['status'], 'step_limit')
        self.assertFalse(receipt['eligible_for_portfolio_evaluation'])
        with tempfile.TemporaryDirectory(prefix='current-only-persistence-') as temporary:
            folder = Path(temporary) / 'fit'
            saved = save_current_only_result(folder, model, receipt)
            self.assertEqual(saved['model_file'], 'diagnostic_model.npz')
            path = folder / saved['model_file']
            loaded = CurrentOnlyModel.load(path, expected_parent_fingerprint='synthetic-parent')
            for name in ('gains', 'beta', 'lower', 'upper'):
                np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
            self.assertEqual(loaded.feature_map.fingerprint, model.feature_map.fingerprint)
            self.assertEqual(saved['parent_feature_fingerprint'], temporal.fingerprint)
            with self.assertRaises(ValueError):
                TemporalModel.load(path, expected_parent_fingerprint='synthetic-parent')
            with self.assertRaises(ValueError):
                CurrentOnlyModel.load(path, expected_parent_fingerprint='different-parent')
            with self.assertRaises(FileExistsError):
                save_current_only_result(folder, model, receipt)

    def test_failed_neutral_gate_cannot_produce_current_only_model(self):
        bases, settings, _, control, _, kwargs = self.training_problem()
        model, receipt = train_current_only_profile(lambda: iter(bases), control, settings,
             **dict(kwargs, expected_neutral_beta=kwargs['expected_neutral_beta'] + 1.))
        self.assertIsNone(model)
        self.assertEqual(receipt['status'], 'evaluation_failed')
        self.assertEqual(receipt['accepted_steps'], 0)
        self.assertFalse(receipt['eligible_for_portfolio_evaluation'])
        with self.assertRaises(ValueError):
            train_current_only_profile(lambda: iter(bases), control, settings,
                 **dict(kwargs, expected_parent_feature_fingerprint='wrong-parent-map'))

    def test_current_model_aim_is_parent_neutral_without_history_and_checks_cutoff(self):
        bases, settings, _, control, _, kwargs = self.training_problem(neutral=True)
        model, receipt = train_current_only_profile(lambda: iter(bases), control, settings, **kwargs)
        self.assertTrue(receipt['eligible_for_portfolio_evaluation'])
        market = replace(bases[0].sample.market, decision_date='2000-01-31')
        z, vol = market.characteristics[0], market.volatility[0]
        expected = control.parent.signals(z, vol, control.theta) @ kwargs['expected_neutral_beta']
        np.testing.assert_allclose(model.aim(market.decision_date, market.ids, z, vol),
                                   expected, atol=1e-11, rtol=1e-10)
        with self.assertRaisesRegex(ValueError, 'future training'):
            model.aim('1999-11-30', market.ids, z, vol)
        with self.assertRaisesRegex(ValueError, 'IDs'):
            model.aim(market.decision_date, np.zeros(len(z)), z, vol)
        with self.assertRaisesRegex(ValueError, 'no historical reference'):
            replace(model, reference_fingerprint='temporal-reference')

    def test_wrong_transform_type_and_persisted_parameter_changes_are_rejected(self):
        bases, settings, temporal, control, _, kwargs = self.training_problem(neutral=True)
        model, _ = train_current_only_profile(lambda: iter(bases), control, settings, **kwargs)
        with self.assertRaisesRegex(ValueError, 'current-only feature map'):
            replace(model, feature_map=temporal)
        with tempfile.TemporaryDirectory(prefix='current-only-integrity-') as temporary:
            path = Path(temporary) / 'model.npz'
            model.save(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
                metadata = json.loads(str(data['metadata'].item()))
                self.assertEqual(metadata['transform_kind'], CURRENT_ONLY_KIND)
            arrays['gains'][0] += .01
            changed = Path(temporary) / 'changed.npz'
            np.savez_compressed(changed, **arrays)
            with self.assertRaisesRegex(ValueError, 'Changed current-only'):
                CurrentOnlyModel.load(changed, expected_parent_fingerprint='synthetic-parent')
            model.gains.setflags(write=True)
            model.gains[0] += .01
            with self.assertRaisesRegex(ValueError, 'content changed'):
                model.save(Path(temporary) / 'mutated.npz')

    def test_current_only_receipt_parameter_and_eligibility_changes_fail_before_write(self):
        bases, settings, _, control, _, kwargs = self.training_problem(neutral=True)
        model, receipt = train_current_only_profile(lambda: iter(bases), control, settings, **kwargs)
        changes = [dict(parent_feature_fingerprint=control.fingerprint),
                   dict(control_feature_fingerprint='wrong'),
                   dict(optimizer_settings=dict(receipt['optimizer_settings'], upper=2.)),
                   dict(parent_settings=dict(receipt['parent_settings'], cutoff='2001-01-31')),
                   dict(status='step_limit'), dict(neutral_gate=dict(passed=False)),
                   dict(projected_gradient_norm=.1)]
        with tempfile.TemporaryDirectory(prefix='current-only-receipt-') as temporary:
            for i, change in enumerate(changes):
                destination = Path(temporary) / str(i)
                with self.assertRaises(ValueError):
                    save_current_only_result(destination, model, dict(receipt, **change))
                self.assertFalse(destination.exists())

    def test_fixed_temporal_and_current_only_controls_reprofile_and_persist(self):
        bases, settings, temporal, control, histories, kwargs = self.training_problem()
        gains = np.linspace(.7, 1.3, 13)
        common = {k: v for k, v in kwargs.items() if k != 'optimizer_settings'}
        for fmap, factory, reference in (
                (temporal, lambda: zip(bases, histories), 'synthetic-reference'),
                (control, current_only_batch_factory(lambda: iter(bases)),
                 current_only_reference_fingerprint('synthetic-source'))):
            model, receipt = fit_fixed_profile(factory, fmap, settings, gains=gains,
                lower=0., upper=2., reference_fingerprint=reference, **common)
            expected = evaluate(bases, histories, fmap, gains, settings, rho=.03, objective_scale=.7)
            np.testing.assert_allclose(model.beta, expected['beta'], atol=1e-11, rtol=1e-10)
            self.assertEqual(receipt['status'], 'fixed_configuration_solved')
            self.assertTrue(receipt['eligible_for_portfolio_evaluation'])
            self.assertNotIn('converged', receipt)
            self.assertTrue(receipt['neutral_gate']['passed'])
            np.testing.assert_array_equal(receipt['prescribed_gains'], gains)
            with tempfile.TemporaryDirectory(prefix='fixed-control-receipt-') as temporary:
                destination = Path(temporary) / 'fit'
                saved = save_fixed_result(destination, model, receipt)
                loaded = type(model).load(destination / saved['model_file'],
                                          expected_parent_fingerprint='synthetic-parent')
                np.testing.assert_array_equal(loaded.beta, model.beta)
                with self.assertRaises(FileExistsError):
                    save_fixed_result(destination, model, receipt)

    def test_fixed_control_neutral_gate_bounds_and_receipt_changes_fail_closed(self):
        bases, settings, temporal, _, histories, kwargs = self.training_problem()
        common = {k: v for k, v in kwargs.items() if k != 'optimizer_settings'}
        supplied = dict(gains=np.full(13, .8), lower=0., upper=2.,
                        reference_fingerprint='synthetic-reference', **common)
        factory = lambda: zip(bases, histories)
        with self.assertRaises(AssertionError):
            fit_fixed_profile(factory, temporal, settings,
                **dict(supplied, expected_neutral_beta=kwargs['expected_neutral_beta'] + 1.))
        with self.assertRaises(ValueError):
            fit_fixed_profile(factory, temporal, settings, **dict(supplied, gains=np.full(13, 3.)))
        model, receipt = fit_fixed_profile(factory, temporal, settings, **supplied)
        changes = [dict(gains=np.ones(13)), dict(prescribed_gains=np.ones(13)),
                   dict(upper=np.full(13, 3.)), dict(solve_residual=1e-3),
                   dict(neutral_gate=dict(passed=False)), dict(status='converged')]
        with tempfile.TemporaryDirectory(prefix='fixed-control-failed-') as temporary:
            for i, change in enumerate(changes):
                destination = Path(temporary) / str(i)
                with self.assertRaises(ValueError):
                    save_fixed_result(destination, model, dict(receipt, **change))
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
