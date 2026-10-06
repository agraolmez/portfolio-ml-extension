import unittest
import numpy as np
from temporal_numerics.optimizer import optimize, LBFGSBSettings
from annual_ind.parallel_layers import ParallelTemporalObjective
from annual_ind.temporal_reuse import SharedTemporalProfiles
import test_annual_ind_parallel_layers as fixtures


class TemporalReuseTests(unittest.TestCase):
    def problem(self, limit=16*2**20):
        args, kwargs = fixtures.ParallelLayerTests().problem(8)
        kwargs.pop('rho')
        kwargs['workers'] = 8
        shared = SharedTemporalProfiles(*args, **kwargs, maximum_bytes=limit, guard=lambda: None)
        return args, kwargs, shared

    def test_each_restraint_and_cache_hit_are_identical(self):
        args, kwargs, shared = self.problem()
        for rho in (.1, .01, .001):
            original = ParallelTemporalObjective(*args, rho=rho, **kwargs)
            cached = shared.for_restraint(rho)
            for gains in (np.ones(13), np.linspace(.7, 1.3, 13)):
                for gradient in (True, False):
                    expected = original(gains, with_gradient=gradient)
                    actual = cached(gains, with_gradient=gradient)
                    for key in ('value', 'parent_value', 'beta', 'score', 'risk', 'cost', 'gradient'):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    for key in ('training_source_fingerprint', 'frozen_model_fingerprint', 'coverage'):
                        self.assertEqual(actual[key], expected[key])
        self.assertEqual(shared.hits, 8)

    def test_hit_rechecks_source_arrays(self):
        args, _, shared = self.problem()
        evaluate = shared.for_restraint(.1)
        evaluate(np.ones(13))
        basis, _ = next(iter(args[0]()))
        value = basis.sample.market.characteristics[0]
        value.setflags(write=True)
        value[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'arrays changed'):
            evaluate(np.ones(13))

    def test_complete_sequential_restraint_fits_match(self):
        args, kwargs, shared = self.problem(limit=128*2**20)
        settings = LBFGSBSettings(lower=0., upper=2., tolerance=1e-5,
            max_steps=100, max_evaluations=250, max_line_searches=16,
            history_size=10, relative_ftol=0., wall_seconds=60.)
        for rho in (.1, .01, .001):
            expected = optimize(ParallelTemporalObjective(*args, rho=rho, **kwargs), 13, settings)
            actual = optimize(shared.for_restraint(rho), 13, settings)
            self.assertEqual(actual['status'], expected['status'], actual['evaluations'][-1])
            for key in ('gains', 'beta', 'gradient', 'objective_trace', 'projected_gradient_trace'):
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_configuration_guard_copy_isolation_and_memory_bound(self):
        _, _, shared = self.problem()
        evaluate = shared.for_restraint(.01)
        first = evaluate(np.ones(13))
        expected = first['beta'].copy()
        first['beta'][:] = 99
        np.testing.assert_array_equal(evaluate(np.ones(13))['beta'], expected)
        shared.engine.workers = 4
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            evaluate(np.ones(13))
        _, _, empty = self.problem(limit=0)
        f = empty.for_restraint(.1)
        f(np.ones(13)); f(np.ones(13))
        self.assertEqual(empty.retained_bytes, 0)
        self.assertEqual(empty.hits, 0)


if __name__ == '__main__':
    unittest.main()
