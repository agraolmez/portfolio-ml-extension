"""Bounded immutable sample/reference reuse above the checked LU cache.

Admission preserves six GiB of current system headroom; the enclosing run still
enforces its four-GiB reserve and sixteen-GiB process limit. Excess months stream
unchanged. Every hit rechecks numerical/source fingerprints. No future month is
served because it was present in a broader source panel.
"""
import hashlib
import numpy as np
import psutil
from annual_ind.inputs import YearInputs,check_sample
from feature_geometry.features import fingerprint


def content(sample,innovation):
    check_sample(sample)
    return fingerprint(sample.fingerprint,*(v for x in innovation for v in (x.values,x.valid,x.reasons)))


def retained_size(sample,innovation):
    m=sample.market
    arrays=[m.ids,*m.characteristics,*m.volatility,*m.growth,m.m,m.covariance,m.impact,
            sample.returns,*(v for x in innovation for v in (x.values,x.valid))]
    # Deliberately count every referenced view at full length: an overestimate
    # for shared arrays is preferable to admitting an uncharged backing buffer.
    return sum(a.nbytes for a in arrays)


class InputCache:
    def __init__(self,maximum_bytes,guard):
        if type(maximum_bytes) is not int or maximum_bytes<=0:raise ValueError('Explicit positive input-cache bound required')
        self.maximum_bytes,self.guard=maximum_bytes,guard;self.items={};self.bytes=0
        self.hits=self.misses=0;self.source=None

    def get(self,history,day,cutoff):
        self.guard();source=(history.source_fingerprint,history.reference_fingerprint)
        if self.source is not None and source!=self.source:raise ValueError('Input cache source context changed')
        self.source=source
        if day in self.items:
            sample,innovation,digest=self.items[day]
            if content(sample,innovation)!=digest:raise ValueError('Cached sample/reference mutated')
            self.hits+=1
        else:
            sample=history.training_sample(day,cutoff)
            bound=history.innovations(sample.market)
            innovation=history.validate_market(sample.market,bound)
            digest=content(sample,innovation);self.misses+=1
            size=retained_size(sample,innovation)
            if self.bytes+size<=self.maximum_bytes and psutil.virtual_memory().available-size>6*2**30:
                self.items[day]=(sample,innovation,digest);self.bytes+=size
        if sample.market.decision_date!=day or sample.available_at>cutoff:raise ValueError('Cached sample after requested cutoff')
        return sample,innovation

    def diagnostics(self):
        return dict(retained_months=len(self.items),retained_bytes=self.bytes,
                    maximum_bytes=self.maximum_bytes,hits=self.hits,misses=self.misses)


class RetainedYearInputs(YearInputs):
    def __init__(self,*args,input_cache,**kwargs):
        super().__init__(*args,**kwargs);self.input_cache=input_cache

    def items(self):
        digest=hashlib.sha256();self.passes+=1
        for i,day in enumerate(self.dates,1):
            self.guard()
            sample,innovation=self.input_cache.get(self.history,day,self.cutoff)
            digest.update(sample.fingerprint.encode());basis=self.cache.basis(sample)
            if i==1 or i%120==0 or i==len(self):
                self.event('phase_progress',label=f'{self.year}: training inputs, pass {self.passes}',
                           completed=i,total=len(self),year=self.year)
            yield basis,innovation
        identity=digest.hexdigest()
        if self.identity is not None and self.identity!=identity:raise ValueError('Training source changed between passes')
        self.identity=identity;self.history.files.unchanged()

    def bases(self):
        for basis,_ in self.items():yield basis

    def batches(self):yield from self.items()
