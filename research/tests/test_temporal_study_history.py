"""Extra temporal warm-up reads only ranks/masks, preserving parent markets."""
from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_geometry.features import fingerprint
from temporal_extension.history import AcceptedTemporalHistory, RANKS, RAW, SCALES
from temporal_study.history import StudyHistory, _join_reference_month


class StudyHistoryTests(unittest.TestCase):
    def fixture(self):
        history = StudyHistory.__new__(StudyHistory)
        history.features = ('x', 'y')
        history.weights = np.full(6, 1 / 6)
        history.cache_limit = 19
        history._months = OrderedDict()
        history._reference_months = OrderedDict()
        history.source_id = 'economic-source'
        history.source_fingerprint = 'accepted-ranks-and-masks'
        history.reference_fingerprint = fingerprint('strict_calendar_lags_source_mask_bypass_v1',
            history.weights, 2, history.source_fingerprint)
        history.files = SimpleNamespace(unchanged=lambda: None, path=lambda rel: rel)
        reads = []
        def read(path, *, columns, filters):
            day = filters[0][2]
            reads.append((path, day))
            k = day.to_period('M').ordinal
            data = dict(id=[10, 20], eom=[day] * 2)
            if path == SCALES:
                if day < pd.Timestamp('1964-11-30'):
                    raise AssertionError('No volatility data for reference-only warm-up months')
                data['vol_scale'] = [1., 2.]
            else:
                data.update(x=[k / 1000., k / 1000. + .1], y=[.5, .5])
                if path == RAW:
                    data['y'] = [np.nan, 3.]
                if path == RANKS:
                    data.update(valid=[True, True], ret_ld1=[.01, .02], tr_ld0=[.01, .02],
                        mu_ld0=[0., 0.], tr_ld1=[.01, .02], mu_ld1=[0., 0.],
                        lambda_diagonal=[.1, .1], wealth_usd=[100., 100.])
            return pd.DataFrame(data)[columns]
        return history, read, reads

    def test_earliest_training_reference_needs_no_warmup_volatility(self):
        history, read, reads = self.fixture()
        day = pd.Timestamp('1965-11-30')
        before = (history.source_id, history.source_fingerprint, history.reference_fingerprint)
        with patch('temporal_study.history.pd.read_parquet', side_effect=read):
            slices = tuple(history._month(day-pd.offsets.MonthEnd(lag))[0][['x', 'y']].to_numpy()
                           for lag in range(13))
            market = SimpleNamespace(decision_date=str(day.date()), ids=np.array([10, 20]),
                                     characteristics=slices, source_id=history.source_id)
            bound = history.innovations(market)
            innovations = history.validate_market(market, bound)
            self.assertEqual(len(innovations), 13)
            self.assertEqual(min(history.cached_reference_months), pd.Timestamp('1964-05-31'))
            self.assertEqual(len(history.cached_reference_months), 19)
            for innovation in innovations:
                np.testing.assert_allclose(innovation.values[:, 0], .0035, atol=1e-15)
                np.testing.assert_array_equal(innovation.valid[:, 1], [False, True])
                self.assertEqual(innovation.values[0, 1], 0.)
                np.testing.assert_allclose(innovation.values[:, 1], [0., 0.], atol=1e-15)
            self.assertFalse(any(path == SCALES and date < pd.Timestamp('1964-11-30')
                                 for path, date in reads))
            old_count = len(reads)
            history.innovations(market)
            self.assertEqual(len(reads), old_count)
        self.assertEqual(before, (history.source_id, history.source_fingerprint, history.reference_fingerprint))
        self.assertIs(StudyHistory.market, AcceptedTemporalHistory.market)

    def test_reference_join_preserves_ranks_and_genuine_missingness(self):
        day = pd.Timestamp('1964-05-31')
        ranks = pd.DataFrame(dict(id=[20, 10], eom=[day] * 2, x=[.5, .5], y=[.7, .2]))
        raw = pd.DataFrame(dict(id=[10, 20], eom=[day] * 2, x=[np.nan, 2.], y=[1., 4.]))
        frame, mask = _join_reference_month(ranks, raw, ('x', 'y'))
        np.testing.assert_array_equal(frame[['x', 'y']], [[.5, .2], [.5, .7]])
        np.testing.assert_array_equal(mask, [[False, True], [True, True]])
        self.assertNotIn('vol_scale', frame)
        with self.assertRaisesRegex(ValueError, 'matching source-mask'):
            _join_reference_month(ranks, raw.iloc[:1], ('x', 'y'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            _join_reference_month(ranks, pd.concat([raw, raw.iloc[:1]]), ('x', 'y'))

    def test_separate_reference_cache_is_bounded_and_source_mismatch_fails(self):
        history, read, _ = self.fixture()
        with patch('temporal_study.history.pd.read_parquet', side_effect=read):
            for day in pd.date_range('1964-01-31', periods=30, freq='ME'):
                history._reference_month(day)
                self.assertLessEqual(len(history.cached_reference_months), 19)
        self.assertEqual(len(history.cached_months), 0)
        with self.assertRaisesRegex(ValueError, 'source'):
            history.innovations(SimpleNamespace(source_id='wrong'))


if __name__ == '__main__':
    unittest.main()
