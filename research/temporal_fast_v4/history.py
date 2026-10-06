"""Preserve accepted economic inputs; batch source checks and calendar rows."""
import pandas as pd
from temporal_study.history import StudyHistory as PreviousHistory, _join_reference_month
from temporal_extension.history import _AcceptedFiles, _date, _join_month, RANKS, RAW, SCALES
from .panels import SourcePanels
from .source_checks import AcceptedBatchVerifier
from .reference import ArrayBoundReference


class AcceptedFiles(_AcceptedFiles):
    def unchanged(self):
        if self.consumed:self._batch_verify(dict(self.consumed))


class StudyHistory(PreviousHistory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        previous=self.files
        self.files=AcceptedFiles(previous.root,previous.inventory)
        self.files.consumed=previous.consumed;self.files._stats=previous._stats
        self.files._batch_verify=AcceptedBatchVerifier(self.files)
        self._economic_columns=['id','eom','valid','ret_ld1','tr_ld0','mu_ld0','tr_ld1','mu_ld1',
            'lambda_diagonal','wealth_usd',*self.features]
        self.panels=SourcePanels(self.files,{RANKS:self._economic_columns,RAW:['id','eom',*self.features],
            SCALES:['id','eom','vol_scale']},maximum_bytes=3*2**30)

    def _month(self,date):
        day=pd.Timestamp(_date(date));self.files.unchanged()
        if day in self._months:
            self._months.move_to_end(day);return self._months[day]
        ranks=self.panels.read(RANKS,self._economic_columns,day)
        raw=self.panels.read(RAW,['id','eom',*self.features],day)
        scales=self.panels.read(SCALES,['id','eom','vol_scale'],day)
        self._months[day]=_join_month(ranks,raw,scales,self.features)
        while len(self._months)>self.cache_limit:self._months.popitem(last=False)
        return self._months[day]

    def _reference_month(self,date):
        day=pd.Timestamp(_date(date));self.files.unchanged()
        if day in self._reference_months:
            self._reference_months.move_to_end(day);return self._reference_months[day]
        columns=['id','eom',*self.features]
        if day in self._months:
            frame,observed=self._months[day]
            value=(frame[columns].copy(),observed.copy())
        else:
            ranks=self.panels.read(RANKS,columns,day);raw=self.panels.read(RAW,columns,day)
            value=_join_reference_month(ranks,raw,self.features)
        self._reference_months[day]=value
        while len(self._reference_months)>self.cache_limit:self._reference_months.popitem(last=False)
        return value

    def innovations(self,market):
        if market.source_id!=self.source_id:raise ValueError('Market source differs from accepted temporal history')
        blocks=[];day=pd.Timestamp(market.decision_date)
        for lag in reversed(range(self.cache_limit)):
            date=day-pd.offsets.MonthEnd(lag)
            frame,observed=self._reference_month(date)
            block=frame[frame.id.isin(market.ids)]
            blocks.append((date,block.id.to_numpy(),block[list(self.features)].to_numpy(),observed.loc[block.id].to_numpy()))
        reference=ArrayBoundReference(blocks,self.weights,dimensions=len(self.features),source_fingerprint=self.source_fingerprint)
        if reference.reference_fingerprint!=self.reference_fingerprint:
            raise ValueError('Reference specification changed during study history construction')
        return reference.for_market(market)
