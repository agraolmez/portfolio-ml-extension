"""Read-only reuse of verified, model-independent Extension-3 input operators.

Only months within the requested expanding window are read. No parent model,
learned scale, temporal gain or portfolio decision is imported from this cache.
Uncached months use the existing bounded source and operator preparation.
"""
import hashlib
import pandas as pd
from feature_geometry.history import sha
from annual_ind.inputs import YearInputs, check_sample
from annual_ind.retained_inputs import InputCache
from temporal_fast_v4.storage import OverflowStore
from temporal_fast_v4.source_checks import AcceptedBatchVerifier


def open_parent_store(root, history, expected_lock_hash):
    folder=root/'output/extension3_full_study_20260919'
    if sha(folder/'study_lock.json')!=expected_lock_hash:
        raise ValueError('Existing input-cache study lock changed')
    def verify_input(relative, expected):
        history.files.path(relative)
        if history.files.consumed.get(relative)!=expected:
            raise ValueError('Input-cache source is not accepted')
    return OverflowStore(folder/'operator_cache',
        source_fingerprint=history.source_fingerprint,
        reference_fingerprint=history.reference_fingerprint,
        study_lock_sha256=expected_lock_hash,maximum_bytes=6*2**30,
        verify_input=verify_input,verify_inputs=AcceptedBatchVerifier(history.files))


class ReusedInputCache(InputCache):
    def __init__(self, maximum_bytes, guard, store):
        super().__init__(maximum_bytes,guard)
        self.store=store;self.disk_hits=0;self.disk_misses=0

    def get_batch(self,history,day,cutoff,operators):
        self.guard()
        if pd.Timestamp(day)+pd.offsets.MonthEnd(1)>pd.Timestamp(cutoff):
            raise ValueError('Requested cached month has an outcome after cutoff')
        source=(history.source_fingerprint,history.reference_fingerprint)
        expected=(self.store.context['source_fingerprint'],self.store.context['reference_fingerprint'])
        if source!=expected or (self.source is not None and source!=self.source):
            raise ValueError('Input cache source context changed')
        self.source=source
        restored=self.store.get(day,cutoff)
        if restored is None:
            self.disk_misses+=1
            sample,innovation=super().get(history,day,cutoff)
            return operators.basis(sample),innovation
        basis,innovation=restored
        check_sample(basis.sample)
        if (basis.sample.market.decision_date!=day or basis.sample.available_at>cutoff
                or basis.sample.market.source_id!=history.source_id):
            raise ValueError('Restored sample source, day or cutoff differs')
        self.disk_hits+=1;self.guard()
        return basis,innovation

    def diagnostics(self):
        return dict(super().diagnostics(),disk_hits=self.disk_hits,
                    disk_misses=self.disk_misses,disk_writes=0,
                    reused_content='sample, lag innovations, model-independent LU only')


class ReusedYearInputs(YearInputs):
    def __init__(self,*args,input_cache,**kwargs):
        super().__init__(*args,**kwargs);self.input_cache=input_cache

    def items(self):
        digest=hashlib.sha256();self.passes+=1
        for i,day in enumerate(self.dates,1):
            self.guard()
            basis,innovation=self.input_cache.get_batch(self.history,day,self.cutoff,self.cache)
            digest.update(basis.sample.fingerprint.encode())
            if i==1 or i%120==0 or i==len(self):
                self.event('phase_progress',label=f'{self.year}: training inputs, pass {self.passes}',
                    completed=i,total=len(self),year=self.year,input_reuse=self.input_cache.diagnostics())
            yield basis,innovation
        identity=digest.hexdigest()
        if self.identity is not None and self.identity!=identity:
            raise ValueError('Training source changed between passes')
        self.identity=identity;self.history.files.unchanged()

    def bases(self):
        for basis,_ in self.items():yield basis

    def batches(self):yield from self.items()
