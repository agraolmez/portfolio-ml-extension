"""Keep a bounded chronological prefix; stream overflow without cyclic eviction."""
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import month_key
from temporal_study.cache import AnnualBatchCache, batch_bytes


class RetainedBatchCache(AnnualBatchCache):
    def __init__(self, history, *, maximum_bytes):
        super().__init__(history, maximum_bytes=maximum_bytes)
        self.admission_closed = False
        self.streamed_months = 0

    def factory(self, parent):
        parent.assert_cutoff(parent.settings.cutoff)
        dates = [parent.metadata['training_first'], parent.metadata['training_last'], parent.settings.cutoff]
        for day in dates:
            month_key(day)
        first, last, cutoff = map(pd.Timestamp, dates)
        source, reference = self.history.source_fingerprint, self.history.reference_fingerprint
        if first > last or last + pd.offsets.MonthEnd(1) > cutoff:
            raise ValueError('Unmatured or inverted annual training interval')

        def batches():
            if (self.history.source_fingerprint != source or self.history.reference_fingerprint != reference):
                raise ValueError('Cached history source or reference changed')
            self.history.files.unchanged()
            for day in pd.date_range(first, last, freq='ME'):
                key = day.date().isoformat()
                if key in self._items:
                    self.hits += 1
                    basis, innovations, _ = self._items[key]
                else:
                    self.misses += 1
                    sample = self.history.training_sample(key, parent.settings.cutoff)
                    bound = self.history.innovations(sample.market)
                    innovations = self.history.validate_market(sample.market, bound)
                    basis = LeanBasis(sample)
                    size = batch_bytes(basis, innovations)
                    if size > self.maximum_bytes:
                        raise MemoryError('One training month exceeds the declared cache budget')
                    if not self.admission_closed and self.retained_bytes + size <= self.maximum_bytes:
                        self._items[key] = (basis, innovations, size)
                        self.retained_bytes += size
                        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
                    else:
                        self.admission_closed = True
                        self.streamed_months += 1
                if (basis.sample.market.decision_date != key or pd.Timestamp(basis.sample.available_at) > cutoff
                        or basis.sample.market.source_id != self.history.source_id):
                    raise ValueError('Cached sample source, decision or cutoff mismatch')
                yield basis, innovations
            self.history.files.unchanged()
        return batches

    def diagnostics(self):
        return dict(super().diagnostics(), policy='retained_prefix_stream_overflow',
                    admission_closed=self.admission_closed, streamed_months=self.streamed_months)
