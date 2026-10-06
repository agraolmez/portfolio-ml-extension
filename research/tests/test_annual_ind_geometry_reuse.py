from dataclasses import replace
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from geometry_study import kernel
from geometry_study.features import CachedFeatureMap
from annual_ind.geometry_reuse import NeutralMoments, fit_full
from test_feature_geometry import fixture


class NeutralReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def problem(self):
        _, fmap, bases, settings, _ = fixture()
        bases = [kernel.LeanBasis(b.sample) for b in bases]
        return bases, CachedFeatureMap(fmap.frequencies, fmap.source), settings

    def test_ridges_and_complete_optimization_traces_identical(self):
        bases, fmap, settings = self.problem()
        neutral = NeutralMoments.build(bases, fmap, settings.cutoff)
        for ridge in (0., .01, .3, 10000.):
            setting = replace(settings, ridge=ridge)
            expected = kernel.evaluate(bases, fmap, np.zeros(fmap.dimensions), setting)
            reused = neutral.evaluate(bases, fmap, setting)
            for key in expected:
                np.testing.assert_array_equal(expected[key], reused[key])
            original = kernel.fit(bases, fmap, setting, objective_scale=expected['value'])
            actual = fit_full(bases, fmap, setting, neutral=neutral, objective_scale=expected['value'])
            for key in set(original)-{'seconds'}:
                if key == 'trace':
                    self.assertEqual(original[key], actual[key])
                else:
                    np.testing.assert_array_equal(original[key], actual[key])

    def test_history_frequency_and_cutoff_changes_rejected(self):
        bases, fmap, settings = self.problem()
        neutral = NeutralMoments.build(bases, fmap, settings.cutoff)
        variants = [(bases[:-1], fmap, settings),
                    (bases, CachedFeatureMap(fmap.frequencies * 2, fmap.source), settings),
                    (bases, fmap, replace(settings, cutoff='2000-12-31'))]
        changed = list(bases)
        changed[0] = kernel.LeanBasis(replace(bases[0].sample, returns=bases[0].sample.returns+1))
        variants.append((changed, fmap, settings))
        for bb, ff, ss in variants:
            with self.assertRaisesRegex(ValueError, 'another training context'):
                neutral.evaluate(bb, ff, ss)

    def test_no_future_training_and_no_invalid_normalizer(self):
        bases, fmap, settings = self.problem()
        with self.assertRaisesRegex(ValueError, 'Unmatured'):
            NeutralMoments.build(bases, fmap, '1999-01-01')
        neutral = NeutralMoments.build(bases, fmap, settings.cutoff)
        for value in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                fit_full(bases, fmap, settings, neutral=neutral, objective_scale=value)
        with self.assertRaises(ValueError):
            neutral.score[0] = 0.


if __name__ == '__main__':
    unittest.main()
