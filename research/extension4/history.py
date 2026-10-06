"""Bounded reuse of unchanged market inputs, never of fitted conditional features."""
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_study.cache import batch_bytes


class TrainingBatches:
    def __init__(self, history, maximum_bytes, guard):
        if type(maximum_bytes) is not int or maximum_bytes <= 0: raise ValueError('Explicit cache budget required')
        self.history, self.maximum_bytes, self.guard = history, maximum_bytes, guard
        self.items = {}; self.used = 0; self.hits = self.misses = 0
        self.identity = (history.source_fingerprint, history.reference_fingerprint, history.source_id)

    def batches(self, dates, cutoff):
        if self.identity != (self.history.source_fingerprint,self.history.reference_fingerprint,self.history.source_id):
            raise ValueError('Cached market source changed')
        self.history.files.unchanged()
        for date in dates:
            self.guard()
            key=str(pd.Timestamp(date).date())
            if key in self.items:
                self.hits+=1;basis,bound=self.items[key]
            else:
                self.misses+=1
                sample=self.history.training_sample(key,cutoff)
                bound=self.history.innovations(sample.market)
                self.history.validate_market(sample.market,bound)
                basis=LeanBasis(sample)
                size=batch_bytes(basis,[b.innovation for b in bound])
                if self.used+size<=self.maximum_bytes:
                    self.items[key]=(basis,bound);self.used+=size
            if (basis.sample.market.decision_date!=key or basis.sample.available_at>cutoff
                    or basis.sample.market.source_id!=self.history.source_id):
                raise ValueError('Cached training source/date/maturity differs')
            self.guard()
            yield basis,bound
        self.history.files.unchanged()
        if hasattr(self.history,'panels'):self.history.panels.release()

    def diagnostics(self):
        return dict(retained_months=len(self.items),retained_bytes=self.used,maximum_bytes=self.maximum_bytes,
                    hits=self.hits,misses=self.misses,overflow_policy='rebuild_from_verified_inputs_without_new_disk_cache')
