"""Study history separates temporal references from parent economic inputs.

The extra lagged-reference months require only archived unscaled ranks and raw
source observation masks. They need no volatility scale, market matrix or return.
Parent market construction is inherited unchanged, including its volatility
requirements for the original thirteen dynamic-feature slices.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from temporal_extension.history import (AcceptedTemporalHistory, BoundReference,
                                         RANKS, RAW, _date)
from temporal_extension.reference import CalendarReference


def _join_reference_month(ranks, raw, features):
    """Exact accepted rank/source-mask join, with no economic-data dependency."""
    keys = ['id', 'eom']
    for frame in (ranks, raw):
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError('Duplicate or missing security/calendar-month keys')
    if not ranks.empty and ranks.eom.nunique() != 1:
        raise ValueError('Reference month must contain exactly one calendar month')
    ranks = ranks.sort_values('id').copy()
    source = raw.set_index(keys)
    positions = source.index.get_indexer(pd.MultiIndex.from_frame(ranks[keys]))
    if np.any(positions < 0):
        raise ValueError('Prepared rank has no matching source-mask key')
    values = source.iloc[positions][list(features)].to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(ranks[list(features)].to_numpy()).all():
        raise ValueError('Nonfinite accepted rank/source value')
    observed = pd.DataFrame(~np.isnan(values), index=ranks.id.to_numpy(), columns=features)
    return ranks.set_index('id', drop=False), observed


class StudyHistory(AcceptedTemporalHistory):
    """Same accepted source/reference identities; separate rank-and-mask cache."""

    def __init__(self, root, preflight_receipt, *, weights):
        super().__init__(root, preflight_receipt, weights=weights)
        self._reference_months = OrderedDict()

    @property
    def cached_reference_months(self):
        return tuple(self._reference_months)

    def _reference_month(self, date):
        day = pd.Timestamp(_date(date))
        self.files.unchanged()
        if day in self._reference_months:
            self._reference_months.move_to_end(day)
            return self._reference_months[day]
        columns = ['id', 'eom', *self.features]
        filters = [('eom', '==', day)]
        ranks = pd.read_parquet(self.files.path(RANKS), columns=columns, filters=filters)
        raw = pd.read_parquet(self.files.path(RAW), columns=columns, filters=filters)
        self._reference_months[day] = _join_reference_month(ranks, raw, self.features)
        while len(self._reference_months) > self.cache_limit:
            self._reference_months.popitem(last=False)
        return self._reference_months[day]

    def innovations(self, market):
        if market.source_id != self.source_id:
            raise ValueError('Market source differs from accepted temporal history')
        dates, ids, values, masks = [], [], [], []
        day = pd.Timestamp(market.decision_date)
        for lag in reversed(range(self.cache_limit)):
            date = day - pd.offsets.MonthEnd(lag)
            frame, observed = self._reference_month(date)
            block = frame[frame.id.isin(market.ids)]
            dates.extend([date] * len(block))
            ids.extend(block.id.tolist())
            values.extend(block[list(self.features)].to_numpy())
            masks.extend(observed.loc[block.id].to_numpy())
        reference = BoundReference(CalendarReference(
            dates, ids, np.asarray(values).reshape(-1, len(self.features)),
            np.asarray(masks, dtype=bool).reshape(-1, len(self.features)), self.weights),
            source_fingerprint=self.source_fingerprint)
        if reference.reference_fingerprint != self.reference_fingerprint:
            raise ValueError('Reference specification changed during study history construction')
        return reference.for_market(market)
