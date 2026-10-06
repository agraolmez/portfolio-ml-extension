"""Synthetic training/persistence/decision integration; no historical fitting."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_temporal_extension import temporal_fixture
from feature_geometry.features import fingerprint
from geometry_study.kernel import evaluate as parent_evaluate
from temporal_extension.history import BoundInnovation
from temporal_extension.model import TemporalModel, decide
from temporal_extension.optimizer import OptimizerSettings
from temporal_extension.training import train_profile, save_result


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def problem(self, fixed=False):
        _, parent, _, bases, settings, theta, fmap, histories = temporal_fixture(13, 7)
        beta = parent_evaluate(bases, parent, theta, settings)['beta']
        optimizer = OptimizerSettings(lower=1. if fixed else 0., upper=1. if fixed else 2.,
                                      tolerance=1e-8, max_steps=3, max_evaluations=40,
                                      max_backtracks=12, initial_step=.1, armijo=1e-4, wall_seconds=30.)
        kwargs = dict(expected_neutral_beta=beta, expected_parent_feature_fingerprint=fmap.fingerprint,
                      parent_fingerprint='synthetic-parent', source_fingerprint='synthetic-source',
                      reference_fingerprint='synthetic-reference', rho=.03, objective_scale=.7,
                      optimizer_settings=optimizer)
        return bases, settings, fmap, histories, kwargs

    def test_gain_training_reprofiles_beta_and_preserves_parent_then_roundtrips(self):
        bases, settings, fmap, histories, kwargs = self.problem()
        before = fmap.fingerprint
        model, receipt = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        self.assertTrue(receipt['neutral_gate']['passed'])
        self.assertGreater(receipt['accepted_steps'], 0)
        self.assertTrue(np.all(np.diff(receipt['objective_trace']) >= 0))
        self.assertGreater(np.linalg.norm(model.gains - 1.), 0.)
        self.assertGreater(np.linalg.norm(model.beta - kwargs['expected_neutral_beta']), 0.)
        self.assertEqual(fmap.fingerprint, before)
        self.assertEqual(receipt['status'], 'step_limit')
        self.assertFalse(receipt['converged'])
        self.assertFalse(receipt['eligible_for_portfolio_evaluation'])
        with tempfile.TemporaryDirectory(prefix='temporal-training-test-') as temporary:
            folder = Path(temporary)/'new-fit'
            saved = save_result(folder, model, receipt)
            self.assertEqual(saved['model_file'], 'diagnostic_model.npz')
            loaded = TemporalModel.load(folder/saved['model_file'], expected_parent_fingerprint='synthetic-parent')
            np.testing.assert_array_equal(loaded.gains, model.gains)
            np.testing.assert_array_equal(loaded.beta, model.beta)
            np.testing.assert_array_equal(loaded.feature_map.theta, fmap.theta)
            with self.assertRaises(FileExistsError):
                save_result(folder, model, receipt)
            with self.assertRaisesRegex(ValueError, 'parent artifact'):
                TemporalModel.load(folder/saved['model_file'], expected_parent_fingerprint='different-parent')

    def test_neutral_coefficients_and_original_rule_from_supplied_holdings(self):
        bases, settings, fmap, histories, kwargs = self.problem(fixed=True)
        model, receipt = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        self.assertTrue(receipt['converged'])
        self.assertTrue(receipt['eligible_for_portfolio_evaluation'])
        np.testing.assert_allclose(model.beta, kwargs['expected_neutral_beta'], atol=1e-11, rtol=1e-10)
        market = replace(bases[0].sample.market, decision_date='2000-01-31')
        z = market.characteristics[0]
        bound = BoundInnovation(histories[0][0], market.decision_date, market.ids,
                                fingerprint(z), 'synthetic-source', 'synthetic-reference')
        expected = fmap.parent.signals(z, market.volatility[0], fmap.theta) @ kwargs['expected_neutral_beta']
        actual = model.aim(market.decision_date, market.ids, z, market.volatility[0], bound)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
        h = np.linspace(-.01, .02, len(z))
        result = decide(model, market, bound, market.ids, h)
        np.testing.assert_allclose(result['weights'], market.m@h + (np.eye(len(h))-market.m)@expected,
                                   atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(result['trades'], result['weights']-h, atol=1e-15)
        with self.assertRaises(ValueError):
            decide(model, market, bound, market.ids[::-1], h)
        with self.assertRaisesRegex(ValueError, 'future training'):
            model.aim('1999-11-30', market.ids, z, market.volatility[0], bound)
        with self.assertRaisesRegex(ValueError, 'binding'):
            model.aim('2000-02-29', market.ids, z, market.volatility[0], bound)

    def test_wrong_archived_coefficients_fail_before_any_gain_update(self):
        bases, settings, fmap, histories, kwargs = self.problem()
        kwargs['expected_neutral_beta'] = kwargs['expected_neutral_beta'] + 1.
        model, receipt = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        self.assertIsNone(model)
        self.assertEqual(receipt['status'], 'evaluation_failed')
        self.assertEqual(receipt['accepted_steps'], 0)
        self.assertEqual(receipt['evaluation_count'], 1)
        self.assertFalse(receipt['eligible_for_portfolio_evaluation'])
        with tempfile.TemporaryDirectory(prefix='temporal-failed-test-') as temporary:
            saved = save_result(Path(temporary)/'failed-fit', model, receipt)
            self.assertNotIn('model_file', saved)

    def test_parent_mismatch_mutation_and_receipt_mismatch_fail_closed(self):
        bases, settings, fmap, histories, kwargs = self.problem()
        with self.assertRaisesRegex(ValueError, 'verified frozen parent'):
            train_profile(lambda: zip(bases, histories), fmap, settings,
                          **dict(kwargs, expected_parent_feature_fingerprint='wrong'))
        model, receipt = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        with tempfile.TemporaryDirectory(prefix='temporal-integrity-test-') as temporary:
            with self.assertRaisesRegex(ValueError, 'fitting receipt'):
                save_result(Path(temporary)/'bad', model, dict(receipt, parent_fingerprint='wrong'))
            model.gains.setflags(write=True)
            model.gains[0] += .01
            model.gains.setflags(write=False)
            with self.assertRaisesRegex(ValueError, 'content changed'):
                model.save(Path(temporary)/'mutated.npz')

    def test_serialized_parameter_change_is_detected(self):
        bases, settings, fmap, histories, kwargs = self.problem(fixed=True)
        model, _ = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        with tempfile.TemporaryDirectory(prefix='temporal-persistence-test-') as temporary:
            path = Path(temporary)/'model.npz'; model.save(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k:data[k].copy() for k in data.files}
            arrays['beta'][0] += .1
            changed = Path(temporary)/'changed.npz'; np.savez_compressed(changed, **arrays)
            with self.assertRaisesRegex(ValueError, 'Changed temporal model'):
                TemporalModel.load(changed, expected_parent_fingerprint='synthetic-parent')

    def test_saved_cutoff_bounds_and_convergence_receipt_must_match(self):
        bases, settings, fmap, histories, kwargs = self.problem(fixed=True)
        model, receipt = train_profile(lambda: zip(bases, histories), fmap, settings, **kwargs)
        changes = [dict(parent_settings=dict(receipt['parent_settings'], cutoff='2000-12-31')),
                   dict(optimizer_settings=dict(receipt['optimizer_settings'], upper=2.)),
                   dict(status='step_limit'), dict(projected_gradient_norm=.1)]
        with tempfile.TemporaryDirectory(prefix='temporal-receipt-test-') as temporary:
            for i, change in enumerate(changes):
                destination = Path(temporary)/str(i)
                with self.assertRaises(ValueError):
                    save_result(destination, model, dict(receipt, **change))
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
