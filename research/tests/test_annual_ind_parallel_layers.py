from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_fast_v4.deferred import OrderedObjective
from temporal_fast_v4.settings import VARIANTS
from annual_ind.parallel_layers import ParallelTemporalObjective, conditional_sums
from test_temporal_extension import temporal_fixture


class ParallelLayerTests(unittest.TestCase):
    def problem(self, workers):
        _, _, _, bases, settings, _, fmap, history = temporal_fixture(
            dimensions=13, frequencies=7, months=3)
        bases = [LeanBasis(replace(b.sample, market=replace(b.sample.market,
            decision_date=str(pd.Timestamp(b.sample.market.decision_date).to_period('M').to_timestamp('M').date()))))
            for b in bases]
        args = (lambda: zip(bases, history), fmap, settings)
        kwargs = dict(rho=.1, objective_scale=.7, trig_backend='accelerate', **dict(VARIANTS))
        return args, kwargs

    def test_neutral_and_nonneutral_profiles_preserve_every_value(self):
        args, kwargs = self.problem(8)
        original = OrderedObjective(*args, workers=4, **kwargs)
        parallel = ParallelTemporalObjective(*args, workers=8, **kwargs)
        for gains in (np.ones(13), np.linspace(.7, 1.3, 13)):
            expected = original(gains)
            actual = parallel(gains)
            for key in ('score', 'risk', 'cost', 'beta', 'gradient', 'value'):
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual['training_source_fingerprint'], expected['training_source_fingerprint'])
            self.assertEqual(actual['coverage'], expected['coverage'])

    def test_execution_and_source_mutation_checks_retained(self):
        args, kwargs = self.problem(8)
        objective = ParallelTemporalObjective(*args, workers=8, **kwargs)
        objective.workers = 4
        with self.assertRaisesRegex(ValueError, 'Execution settings changed'):
            objective(np.ones(13))
        for workers in (0, True, 3, 12):
            with self.assertRaises(ValueError):
                ParallelTemporalObjective(*args, workers=workers, **kwargs)

    def test_conditional_reduction_preserves_chronological_rounding(self):
        def moments(v):
            return dict(score=np.array([v]), risk=np.array([[v]]), cost=np.zeros((1, 1)))
        inputs = [1e16, -1e16, 1.] * 5
        serial = conditional_sums(inputs, moments, columns=1, workers=1, guard=lambda: None)
        parallel = conditional_sums(inputs, moments, columns=1, workers=8, guard=lambda: None)
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)

    def test_empty_invalid_and_resource_error_fail(self):
        with self.assertRaises(ValueError):
            conditional_sums([], lambda _: {}, columns=2, workers=8, guard=lambda: None)
        def stop():
            raise MemoryError('budget')
        with self.assertRaises(MemoryError):
            conditional_sums([1], lambda _: {}, columns=2, workers=8, guard=stop)
        with self.assertRaises(ValueError):
            conditional_sums([1], lambda _: dict(score=np.ones(1),risk=np.ones((1,1)),cost=np.ones((1,1))),
                             columns=2, workers=8, guard=lambda: None)


if __name__ == '__main__':
    unittest.main()
