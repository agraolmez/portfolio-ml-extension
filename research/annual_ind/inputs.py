"""Expanding-window samples with bounded, model-independent LU reuse."""
import hashlib
import numpy as np
import pandas as pd
import psutil
from geometry_study.kernel import LeanBasis
from feature_geometry.features import fingerprint


def check_sample(sample):
    m=sample.market
    identity=fingerprint(m.decision_date,m.source_id,m.ids,*m.characteristics,
        *m.volatility,*m.growth,m.m,m.covariance,m.impact,m.wealth,m.gamma)
    if (identity!=m.fingerprint or
            fingerprint(identity,sample.returns,sample.available_at)!=sample.fingerprint):
        raise ValueError('Market or training sample arrays changed')


class OperatorCache:
    def __init__(self, *, maximum_bytes, guard):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError('Explicit positive operator-cache bound required')
        self.maximum_bytes,self.guard=maximum_bytes,guard
        self.items={};self.bytes=0;self.hits=self.misses=0

    def basis(self,sample):
        self.guard();check_sample(sample);key=sample.market.decision_date
        if key in self.items:
            identity,current,prior,digest=self.items[key]
            if identity!=sample.market.fingerprint or fingerprint(*current,*prior)!=digest:
                raise ValueError('Cached economic inputs/operators changed')
            basis=LeanBasis.__new__(LeanBasis);basis.sample=sample
            basis.current_lu,basis.prior_lu=current,prior;self.hits+=1
            return basis
        basis=LeanBasis(sample);self.misses+=1
        size=sum(x.nbytes for x in (*basis.current_lu,*basis.prior_lu))
        # Retention admission preserves additional room for the numerical workers.
        if self.bytes+size<=self.maximum_bytes and psutil.virtual_memory().available-size>6*2**30:
            for a in (*basis.current_lu,*basis.prior_lu):a.setflags(write=False)
            self.items[key]=(sample.market.fingerprint,basis.current_lu,basis.prior_lu,
                             fingerprint(*basis.current_lu,*basis.prior_lu))
            self.bytes+=size
        self.guard();return basis

    def release(self):
        self.items.clear();self.bytes=0

    def diagnostics(self):
        return dict(retained_months=len(self.items),retained_bytes=self.bytes,
                    maximum_bytes=self.maximum_bytes,hits=self.hits,misses=self.misses)


class YearInputs:
    def __init__(self,history,cache,year,guard,event):
        if type(year) is not int or not 1971<=year<=2020:
            raise ValueError('Annual fitting year outside frozen study')
        self.history,self.cache,self.year,self.guard,self.event=history,cache,year,guard,event
        self.cutoff=f'{year-1}-12-31'
        self.dates=tuple(str(x.date()) for x in pd.date_range('1965-11-30',f'{year-1}-11-30',freq='ME'))
        self.identity=None;self.passes=0

    def __len__(self):return len(self.dates)

    def bases(self):
        digest=hashlib.sha256();self.passes+=1
        for i,date in enumerate(self.dates,1):
            self.guard()
            sample=self.history.training_sample(date,self.cutoff)
            if sample.market.decision_date!=date or sample.available_at>self.cutoff:
                raise ValueError('Wrong or unmatured training sample')
            digest.update(sample.fingerprint.encode())
            basis=self.cache.basis(sample)
            if i==1 or i%120==0 or i==len(self):
                self.event('phase_progress',label=f'{self.year}: training inputs, pass {self.passes}',
                           completed=i,total=len(self),year=self.year)
            yield basis
        identity=digest.hexdigest()
        if self.identity is not None and identity!=self.identity:
            raise ValueError('Training source changed between passes')
        self.identity=identity;self.history.files.unchanged()

    def batches(self):
        for basis in self.bases():
            bound=self.history.innovations(basis.sample.market)
            yield basis,self.history.validate_market(basis.sample.market,bound)
