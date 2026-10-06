"""Small deterministic provenance/history tests; no historical fitting."""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace, MappingProxyType
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_geometry.features import FeatureMap, fingerprint
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.reference import CalendarReference
from temporal_extension.history import (
    AcceptedTemporalHistory, BoundReference, VerifiedParent, _AcceptedFiles,
    _join_month, load_parent, RANKS, RAW, SCALES,
)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("1999-10-31", periods=18, freq="ME")
        self.ids = np.tile([10, 20], 18)
        self.z = np.arange(72, dtype=float).reshape(36, 2) / 100.
        self.mask = np.ones_like(self.z, dtype=bool)
        self.ref = BoundReference(CalendarReference(np.repeat(self.dates, 2), self.ids,
            self.z, self.mask, [.25, .75]), source_fingerprint="synthetic-source-v1")

    def test_innovation_is_bound_to_exact_date_ordered_ids_all_inputs_and_reference(self):
        date = self.dates[-1]; z = self.z[-2:]
        bound = self.ref.at(date, [10, 20], z)
        self.assertIs(bound.validate(date, [10, 20], z, self.ref.source_fingerprint,
                                     self.ref.reference_fingerprint), bound.innovation)
        cases = [(self.dates[-2], [10, 20], z, self.ref.source_fingerprint, self.ref.reference_fingerprint),
                 (date, [20, 10], z, self.ref.source_fingerprint, self.ref.reference_fingerprint),
                 (date, [10, 20], z + .01, self.ref.source_fingerprint, self.ref.reference_fingerprint),
                 (date, [10, 20], z, "another-source", self.ref.reference_fingerprint),
                 (date, [10, 20], z, self.ref.source_fingerprint, "another-reference")]
        for args in cases:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "binding"):
                bound.validate(*args)
        with self.assertRaises(ValueError):
            bound.ids[0] = 99

    def test_all_thirteen_slices_have_real_calendar_dates_and_no_annual_reset(self):
        day = self.dates[-1]
        slices = tuple(self.z[2*(17-lag):2*(18-lag)] for lag in range(13))
        market = SimpleNamespace(decision_date=str(day.date()), ids=np.array([10, 20]),
                                 characteristics=slices)
        bound = self.ref.for_market(market)
        expected = self.ref.validate_market(market, bound)
        self.assertEqual(len(expected), 13)
        for lag, item in enumerate(bound):
            self.assertEqual(item.date, str((day-pd.offsets.MonthEnd(lag)).date()))
            self.assertTrue(item.innovation.valid.all())
        # Swapping equal-shaped adjacent slices must be rejected.
        with self.assertRaisesRegex(ValueError, "binding"):
            self.ref.validate_market(market, (bound[1], bound[0], *bound[2:]))
        with self.assertRaises(ValueError):
            self.ref.validate_market(market, bound[:-1])

    def test_imputed_current_cells_must_still_equal_parent_inputs(self):
        ref = BoundReference(CalendarReference(["2000-01-31"], [10], [[.5, .5]],
            np.array([[False, True]]), [1.]), source_fingerprint="explicit-mask")
        correct = ref.at("2000-01-31", [10], [[.5, .5]])
        np.testing.assert_array_equal(correct.innovation.values, [[0., 0.]])
        with self.assertRaisesRegex(ValueError, "including imputed"):
            ref.at("2000-01-31", [10], [[.7, .5]])

    def test_reference_mutation_rejected_and_weights_identify_reference(self):
        other = BoundReference(CalendarReference(np.repeat(self.dates, 2), self.ids,
            self.z, self.mask, [.75, .25]), source_fingerprint="synthetic-source-v1")
        self.assertNotEqual(other.reference_fingerprint, self.ref.reference_fingerprint)
        key = next(iter(self.ref.reference.rows))
        self.ref.reference.rows[key][0][0] += 1
        with self.assertRaisesRegex(ValueError, "mutated"):
            self.ref.at(self.dates[-1], [10, 20], self.z[-2:])

    def test_source_mask_join_uses_keys_not_row_order_or_half_rank(self):
        ranks = pd.DataFrame(dict(id=[20, 10], eom=[pd.Timestamp("2000-01-31")]*2,
                                  x=[.5, .5], y=[.7, .5]))
        raw = pd.DataFrame(dict(id=[10, 20], eom=[pd.Timestamp("2000-01-31")]*2,
                                x=[np.nan, 123.], y=[4., 5.]))
        scales = pd.DataFrame(dict(id=[20, 10], eom=[pd.Timestamp("2000-01-31")]*2,
                                   vol_scale=[2., np.nan]))
        frame, observed = _join_month(ranks, raw, scales, ("x", "y"))
        self.assertEqual(frame.id.tolist(), [10, 20])
        np.testing.assert_array_equal(observed.to_numpy(), [[False, True], [True, True]])
        np.testing.assert_array_equal(frame.vol_scale, [2., 2.])
        np.testing.assert_array_equal(frame[["x", "y"]].to_numpy(), [[.5, .5], [.5, .7]])
        with self.assertRaisesRegex(ValueError, "matching source-mask"):
            _join_month(ranks, raw.iloc[:1], scales, ("x", "y"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _join_month(ranks, pd.concat([raw, raw.iloc[:1]]), scales, ("x", "y"))

    def test_accepted_files_reject_changed_unknown_and_midrun_sources(self):
        with tempfile.TemporaryDirectory(prefix="temporal_history_fixture_") as tmp:
            root = Path(tmp); path = root / "accepted.txt"; path.write_text("accepted")
            files = _AcceptedFiles(root, {"accepted.txt": sha(path)})
            self.assertEqual(files.path("accepted.txt"), path.resolve())
            with self.assertRaisesRegex(ValueError, "Unaccepted"):
                files.path("unknown.txt")
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "during use"):
                files.unchanged()
            with self.assertRaisesRegex(ValueError, "Changed accepted"):
                _AcceptedFiles(root, {"accepted.txt": "0"*64}).path("accepted.txt")

    def test_month_cache_is_bounded_and_histories_cross_annual_boundaries(self):
        source = AcceptedTemporalHistory.__new__(AcceptedTemporalHistory)
        source.root = Path("/synthetic"); source.features = ("x", "y")
        source.weights = np.array([.5, .5]); source.cache_limit = 15
        source._months = OrderedDict(); source.source_id = "economic-v1"
        source.source_fingerprint = "ranks-and-raw-mask-v1"
        source.reference_fingerprint = fingerprint("strict_calendar_lags_source_mask_bypass_v1",
            source.weights, 2, source.source_fingerprint)
        source.files = SimpleNamespace(unchanged=lambda: None, path=lambda rel: rel)
        reads = []
        def read(path, *, columns, filters):
            day = filters[0][2]; reads.append((path, day))
            k = day.to_period("M").ordinal
            data = dict(id=[10, 20], eom=[day]*2)
            if path == SCALES:
                data["vol_scale"] = [1., 2.]
            else:
                data.update(x=[k / 1000., k / 1000. + .1], y=[.5, .5])
                if path == RANKS:
                    data.update(valid=[True, True], ret_ld1=[.01, .02], tr_ld0=[.01, .02],
                                mu_ld0=[0., 0.], tr_ld1=[.01, .02], mu_ld1=[0., 0.],
                                lambda_diagonal=[.1, .1], wealth_usd=[100., 100.])
            return pd.DataFrame(data)[columns]
        with patch("temporal_extension.history.pd.read_parquet", side_effect=read):
            for day in pd.date_range("1999-01-31", periods=30, freq="ME"):
                source._month(day)
                self.assertLessEqual(len(source.cached_months), 15)
            day = pd.Timestamp("2000-01-31")
            slices = tuple(source._month(day-pd.offsets.MonthEnd(lag))[0][["x", "y"]].to_numpy()
                           for lag in range(13))
            market = SimpleNamespace(decision_date=str(day.date()), ids=np.array([10, 20]),
                                     characteristics=slices, source_id=source.source_id)
            bound = source.innovations(market)
            checked = source.validate_market(market, bound)
            self.assertTrue(all(item.valid.all() for item in checked))
            np.testing.assert_allclose(checked[0].values[:, 0], .0015, atol=1e-15)
            self.assertEqual(bound[1].date, "1999-12-31")
            self.assertLessEqual(len(source.cached_months), 15)
            old_reads = len(reads); source._month(source.cached_months[-1])
            self.assertEqual(len(reads), old_reads)
            with self.assertRaisesRegex(ValueError, "source"):
                source.innovations(SimpleNamespace(**{**market.__dict__, "source_id": "wrong"}))

    def test_parent_cutoff_and_annual_decision_cannot_use_later_model(self):
        fmap = FeatureMap(np.array([[1.], [.3]]))
        model = GeometryModel(fmap, np.zeros(2), np.zeros(3), "1999-12-31", .7, "synthetic")
        parent = VerifiedParent(2000, 1, model, TemporalFeatureMap(fmap, model.theta, np.arange(2)),
            FitSettings("1999-12-31"), 1., MappingProxyType(dict(training_first="1965-11-30",
            training_last="1999-11-30")), MappingProxyType({"fixture": "test"}))
        parent.assert_cutoff("1999-12-31")
        parent.assert_decision("1999-12-31"); parent.assert_decision("2000-11-30")
        for date in ("1999-11-30", "2000-12-31"):
            with self.assertRaisesRegex(ValueError, "annual model"):
                parent.assert_decision(date)
        with self.assertRaisesRegex(ValueError, "earlier fold"):
            parent.assert_cutoff("1998-12-31")
        source = AcceptedTemporalHistory.__new__(AcceptedTemporalHistory)
        for first, last in (("1965-10-31", "1999-11-30"), ("1965-11-30", "1999-12-31")):
            with self.assertRaisesRegex(ValueError, "cutoff"):
                source.batch_factory(parent, first, last)
        with self.assertRaisesRegex(ValueError, "earlier fold"):
            load_parent("/unused", 2000, 1, expected_cutoff="1998-12-31")


if __name__ == "__main__":
    unittest.main()
