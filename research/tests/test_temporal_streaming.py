"""Streaming mathematical equivalence and changed-source rejection; synthetic only."""
from dataclasses import replace
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_temporal_extension import temporal_fixture
from temporal_extension.profile import evaluate
from temporal_extension.streaming import StreamingObjective
from temporal_extension.reference import Innovation


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def make(self):
        _, _, _, bases, settings, _, fmap, histories = temporal_fixture()
        return bases, settings, fmap, histories

    def test_matches_reference_value_moments_conditional_beta_and_gradient(self):
        bases, settings, fmap, histories = self.make()
        objective = StreamingObjective(lambda: zip(bases, histories), fmap, settings,
                                       rho=.03, objective_scale=.7)
        for gains in (np.ones(13), np.linspace(.4, 1.6, 13)):
            expected = evaluate(bases, histories, fmap, gains, settings, rho=.03, objective_scale=.7)
            actual = objective(gains)
            for key in ('value', 'parent_value', 'beta', 'score', 'risk', 'cost', 'gradient'):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=1e-10)
            self.assertEqual(actual['training_months'], 2)
            self.assertLess(actual['solve_residual'], 1e-9)
            no_gradient = objective(gains, with_gradient=False)
            self.assertIsNone(no_gradient['gradient'])
            np.testing.assert_array_equal(no_gradient['beta'], actual['beta'])

    def test_factory_is_reopened_and_nonrepeatable_generator_is_rejected(self):
        bases, settings, fmap, histories = self.make()
        calls = []
        def factory():
            calls.append(True)
            return zip(bases, histories)
        objective = StreamingObjective(factory, fmap, settings, rho=0., objective_scale=1.)
        objective(np.ones(13))
        self.assertEqual(len(calls), 2)
        exhausted = iter(zip(bases, histories))
        bad = StreamingObjective(lambda: exhausted, fmap, settings, rho=0., objective_scale=1.)
        with self.assertRaisesRegex(ValueError, 'observations'):
            bad(np.ones(13))

    def test_changed_innovations_rejected_between_passes_and_evaluations(self):
        bases, settings, fmap, histories = self.make()
        changed = [list(x) for x in histories]
        first = histories[0][0]
        changed[0][0] = Innovation(first.values + .1, first.valid, {})
        calls = [0]
        def factory():
            calls[0] += 1
            return zip(bases, histories if calls[0] == 1 else changed)
        objective = StreamingObjective(factory, fmap, settings, rho=.01, objective_scale=1.)
        with self.assertRaisesRegex(ValueError, 'between objective passes'):
            objective(np.ones(13))
        calls[0] = 0
        objective = StreamingObjective(factory, fmap, settings, rho=.01, objective_scale=1.)
        objective(np.ones(13), with_gradient=False)
        with self.assertRaisesRegex(ValueError, 'across gain evaluations'):
            objective(np.ones(13), with_gradient=False)

    def test_maturity_order_shape_empty_and_configuration_changes_rejected(self):
        bases, settings, fmap, histories = self.make()
        for factory, setting in ((lambda: [], settings),
                                 (lambda: zip(bases[::-1], histories[::-1]), settings),
                                 (lambda: zip(bases, histories), replace(settings, cutoff='1999-01-31')),
                                 (lambda: zip(bases, [h[:12] for h in histories]), settings)):
            with self.assertRaises(ValueError):
                StreamingObjective(factory, fmap, setting, rho=.01, objective_scale=1.)(np.ones(13))
        objective = StreamingObjective(lambda: zip(bases, histories), fmap, settings,
                                       rho=.01, objective_scale=1.)
        objective.rho = .02
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            objective(np.ones(13))

    def test_live_sample_and_operator_mutation_is_rejected(self):
        for target in ('returns', 'operator'):
            bases, settings, fmap, histories = self.make()
            objective = StreamingObjective(lambda: zip(bases, histories), fmap, settings,
                                           rho=.01, objective_scale=1.)
            objective(np.ones(13), with_gradient=False)
            data = bases[0].sample.returns if target == 'returns' else bases[0].current_lu[0]
            data.setflags(write=True)
            data.flat[0] += .1
            data.setflags(write=False)
            with self.assertRaisesRegex(ValueError, 'changed'):
                objective(np.ones(13), with_gradient=False)


if __name__ == '__main__':
    unittest.main()
