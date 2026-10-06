"""Bounded reuse of immutable economic bases; no new scientific settings."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import month_key


def batch_bytes(basis, innovations):
    """Conservative array inventory, including every stored parent operator."""
    sample = basis.sample
    market = sample.market
    arrays = [market.ids, *market.characteristics, *market.volatility,
              *market.growth, market.m, market.covariance, market.impact,
              sample.returns, *basis.current_lu, *basis.prior_lu,
              *getattr(basis, 'current', ()), *getattr(basis, 'prior', ())]
    arrays += [a for innovation in innovations for a in (innovation.values, innovation.valid)]
    unique = {id(a): a for a in arrays if isinstance(a, np.ndarray)}
    return int(sum(a.nbytes for a in unique.values()) * 1.1) + 65536


class AnnualBatchCache:
    """LRU cache shared across candidates/seeds, with an explicit memory bound.

    Only geometry-independent bases and source-bound innovations are reused.
    Features and profiled coefficients are recomputed by StreamingObjective.
    Each yielded sample is checked against the requested parent's cutoff.
    A too-small cache remains correct but may need to rebuild earlier months.
    """
    def __init__(self, history, *, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError('Explicit positive cache memory budget required')
        self.history = history
        self.maximum_bytes = maximum_bytes
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.hits = self.misses = 0
        self._items = OrderedDict()

    def factory(self, parent):
        parent.assert_cutoff(parent.settings.cutoff)
        dates = [parent.metadata['training_first'], parent.metadata['training_last'], parent.settings.cutoff]
        for day in dates:
            month_key(day)
        first, last, cutoff = map(pd.Timestamp, dates)
        source = self.history.source_fingerprint
        reference = self.history.reference_fingerprint
        if first > last or last + pd.offsets.MonthEnd(1) > cutoff:
            raise ValueError('Unmatured or inverted annual training interval')

        def batches():
            if (self.history.source_fingerprint != source
                    or self.history.reference_fingerprint != reference):
                raise ValueError('Cached history source or reference changed')
            self.history.files.unchanged()
            for day in pd.date_range(first, last, freq='ME'):
                key = day.date().isoformat()
                if key in self._items:
                    self.hits += 1
                    basis, innovations, size = self._items.pop(key)
                    self._items[key] = (basis, innovations, size)
                else:
                    self.misses += 1
                    sample = self.history.training_sample(key, parent.settings.cutoff)
                    bound = self.history.innovations(sample.market)
                    innovations = self.history.validate_market(sample.market, bound)
                    basis = LeanBasis(sample)
                    size = batch_bytes(basis, innovations)
                    if size > self.maximum_bytes:
                        raise MemoryError('One training month exceeds the declared cache budget')
                    while self.retained_bytes + size > self.maximum_bytes:
                        _, (_, _, old_size) = self._items.popitem(last=False)
                        self.retained_bytes -= old_size
                    self._items[key] = (basis, innovations, size)
                    self.retained_bytes += size
                    self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
                if (basis.sample.market.decision_date != key
                        or pd.Timestamp(basis.sample.available_at) > cutoff
                        or basis.sample.market.source_id != self.history.source_id):
                    raise ValueError('Cached sample source, decision or cutoff mismatch')
                yield basis, innovations
            self.history.files.unchanged()
        return batches

    def diagnostics(self):
        return dict(retained_months=len(self._items), retained_bytes=self.retained_bytes,
                    peak_retained_bytes=self.peak_retained_bytes,
                    maximum_bytes=self.maximum_bytes, hits=self.hits, misses=self.misses,
                    cached_dates=list(self._items))
