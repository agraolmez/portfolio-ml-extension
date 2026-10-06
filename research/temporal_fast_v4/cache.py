"""Memory-bounded prefix plus immutable local reuse of overflow operators."""
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import month_key
from temporal_numerics.cache import RetainedBatchCache
from .storage import OverflowStore
from .source_checks import AcceptedBatchVerifier
from .deferred import DeferredMonth
import threading
from temporal_study.cache import batch_bytes


class StorageBatchCache(RetainedBatchCache):
    def __init__(self, history, *, maximum_bytes, directory, maximum_disk_bytes, study_lock_sha256):
        super().__init__(history, maximum_bytes=maximum_bytes)
        self.disk_hits = self.disk_writes = 0
        self._counter_lock=threading.Lock()

        def verify_input(relative, expected):
            history.files.path(relative)
            if history.files.consumed.get(relative) != expected:
                raise ValueError('Restored cache input is not in the accepted source inventory')
        self.store = OverflowStore(directory, source_fingerprint=history.source_fingerprint,
            reference_fingerprint=history.reference_fingerprint, study_lock_sha256=study_lock_sha256,
            maximum_bytes=maximum_disk_bytes, verify_input=verify_input,verify_inputs=AcceptedBatchVerifier(history.files))

    def factory(self, parent):
        parent.assert_cutoff(parent.settings.cutoff)
        dates = [parent.metadata['training_first'], parent.metadata['training_last'], parent.settings.cutoff]
        for date in dates: month_key(date)
        first,last,cutoff = map(pd.Timestamp,dates)
        source,reference = self.history.source_fingerprint,self.history.reference_fingerprint
        if first > last or last+pd.offsets.MonthEnd(1) > cutoff:
            raise ValueError('Unmatured or inverted annual training interval')

        def batches():
            if (self.history.source_fingerprint != source or self.history.reference_fingerprint != reference):
                raise ValueError('Cached history source or reference changed')
            self.history.files.unchanged()
            for date in pd.date_range(first,last,freq='ME'):
                day=date.date().isoformat()
                if day in self._items:
                    self.hits+=1
                    basis,innovations,_=self._items[day]
                else:
                    path,commit=self.store._paths(day)
                    if self.admission_closed and path.is_file() and commit.is_file():
                        def load(day=day):
                            value=self.store.get(day,parent.settings.cutoff)
                            if value is None:raise ValueError('Previously committed overflow disappeared')
                            with self._counter_lock:self.disk_hits+=1;self.streamed_months+=1
                            return value
                        yield DeferredMonth(day,parent.settings.cutoff,self.history.source_id,load)
                        continue
                    restored=self.store.get(day,parent.settings.cutoff)
                    if restored is None:
                        self.misses+=1
                        sample=self.history.training_sample(day,parent.settings.cutoff)
                        innovations=self.history.validate_market(sample.market,self.history.innovations(sample.market))
                        basis=LeanBasis(sample)
                    else:
                        self.disk_hits+=1
                        basis,innovations=restored
                    size=batch_bytes(basis,innovations)
                    if size>self.maximum_bytes:
                        raise MemoryError('One training month exceeds the declared memory cache budget')
                    if not self.admission_closed and self.retained_bytes+size<=self.maximum_bytes:
                        self._items[day]=(basis,innovations,size)
                        self.retained_bytes+=size
                        self.peak_retained_bytes=max(self.peak_retained_bytes,self.retained_bytes)
                    else:
                        self.admission_closed=True
                        self.streamed_months+=1
                        if restored is None:
                            self.store.put(day,basis,innovations,dict(self.history.files.consumed))
                            self.disk_writes+=1
                if (basis.sample.market.decision_date!=day or pd.Timestamp(basis.sample.available_at)>cutoff
                        or basis.sample.market.source_id!=self.history.source_id):
                    raise ValueError('Cached sample source, decision or cutoff mismatch')
                yield basis,innovations
            self.history.files.unchanged()
            if hasattr(self.history,'panels'):self.history.panels.release()
        return batches

    def diagnostics(self):
        return dict(super().diagnostics(), overflow_reuse='verified_local_arrays_and_operators',deferred_workers_share_numerical_pool=True,
                    disk_hits=self.disk_hits,disk_writes=self.disk_writes,
                    disk_budget_bytes=self.store.maximum_bytes)
