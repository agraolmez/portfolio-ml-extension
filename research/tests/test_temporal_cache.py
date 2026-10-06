"""Cache keeps economics immutable and enforces annual availability."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from test_temporal_extension import temporal_fixture
from temporal_study.cache import AnnualBatchCache, batch_bytes


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def fixture(self):
        _, _, _, bases, settings, _, _, histories = temporal_fixture()
        from geometry_study.kernel import LeanBasis
        dated = []
        for basis, day in zip(bases, pd.date_range('1999-01-31', periods=len(bases), freq='ME')):
            market = replace(basis.sample.market, decision_date=day.date().isoformat())
            sample = replace(basis.sample, market=market,
                             available_at=(day + pd.offsets.MonthEnd(1)).date().isoformat())
            dated.append(LeanBasis(sample))
        bases = dated
        by_date = {b.sample.market.decision_date: (b, h) for b, h in zip(bases, histories)}
        class History:
            source_fingerprint = 'source'
            reference_fingerprint = 'reference'
            source_id = bases[0].sample.market.source_id
            files = SimpleNamespace(unchanged=lambda: None)
            def training_sample(self, day, cutoff):
                return by_date[day][0].sample
            def innovations(self, market):
                return by_date[market.decision_date][1]
            def validate_market(self, market, bound):
                return bound
        parent = SimpleNamespace(settings=settings,
            metadata={'training_first': min(by_date), 'training_last': max(by_date)},
            assert_cutoff=lambda cutoff: None)
        return History(), parent, bases, histories

    def test_reuses_bases_with_identical_operators(self):
        history, parent, bases, histories = self.fixture()
        cache = AnnualBatchCache(history, maximum_bytes=10**8)
        first = list(cache.factory(parent)())
        second = list(cache.factory(parent)())
        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.misses, 2)
        for (a, _), (b, _), original in zip(first, second, bases):
            self.assertIs(a, b)
            for actual, expected in zip(a.current_lu, original.current_lu):
                np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(cache.peak_retained_bytes, cache.maximum_bytes)

    def test_budget_eviction_and_single_month_rejection(self):
        history, parent, bases, histories = self.fixture()
        from geometry_study.kernel import LeanBasis
        size = max(batch_bytes(LeanBasis(b.sample), h) for b, h in zip(bases, histories))
        cache = AnnualBatchCache(history, maximum_bytes=size)
        list(cache.factory(parent)())
        self.assertEqual(cache.diagnostics()['retained_months'], 1)
        list(cache.factory(parent)())
        self.assertEqual(cache.misses, 4)
        with self.assertRaises(MemoryError):
            list(AnnualBatchCache(history, maximum_bytes=1).factory(parent)())

    def test_cutoff_and_changed_reference_rejected(self):
        history, parent, _, _ = self.fixture()
        cache = AnnualBatchCache(history, maximum_bytes=10**8)
        factory = cache.factory(parent)
        history.reference_fingerprint = 'different'
        with self.assertRaisesRegex(ValueError, 'source or reference'):
            list(factory())
        parent.settings = SimpleNamespace(cutoff=parent.metadata['training_last'])
        with self.assertRaisesRegex(ValueError, 'Unmatured'):
            cache.factory(parent)


if __name__ == '__main__':
    unittest.main()
