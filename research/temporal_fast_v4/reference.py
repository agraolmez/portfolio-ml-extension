"""Calendar reference with the same cell arithmetic and whole-array checks."""
import numpy as np
import pandas as pd
from feature_geometry.features import array, fingerprint
from temporal_extension.reference import CalendarReference, Innovation, month_key
from temporal_extension.history import BoundInnovation, BoundReference, _ids, _date


class ArrayBoundReference(BoundReference):
    def __init__(self, blocks, weights, *, dimensions, source_fingerprint):
        weights=array(weights,1)
        CalendarReference([],[],np.empty((0,dimensions)),np.empty((0,dimensions),bool),weights)
        if not source_fingerprint:raise ValueError('Explicit source identity required')
        self.weights=weights;self.dimensions=dimensions;self.blocks={}
        for day,ids,values,masks in blocks:
            key=month_key(day);ids=_ids(ids);values=array(values,2);masks=np.asarray(masks)
            if key in self.blocks or values.shape!=(len(ids),dimensions) or masks.shape!=values.shape or masks.dtype!=bool:
                raise ValueError('Unique months, IDs and aligned masks required')
            if not np.isfinite(values[masks]).all():raise ValueError('Nonfinite observed input')
            masks=masks.copy();masks.setflags(write=False)
            self.blocks[key]=(ids,values,masks)
        self.source_fingerprint=str(source_fingerprint)
        self.reference_fingerprint=fingerprint('strict_calendar_lags_source_mask_bypass_v1',
            self.weights,self.dimensions,self.source_fingerprint)
        self._content=self._snapshot()

    def _snapshot(self):
        return fingerprint(self.weights,*(fingerprint(key,*value) for key,value in self.blocks.items()))

    def _aligned(self,key,ids):
        rows=self.blocks.get(key)
        if rows is None:return np.zeros((len(ids),self.dimensions)),np.zeros((len(ids),self.dimensions),bool),np.zeros(len(ids),bool)
        source,values,masks=rows;positions=pd.Index(source).get_indexer(ids);present=positions>=0
        values=values[np.maximum(positions,0)] if len(source) else np.zeros((len(ids),self.dimensions))
        masks=masks[np.maximum(positions,0)].copy() if len(source) else np.zeros((len(ids),self.dimensions),bool)
        masks[~present]=False
        return values,masks,present

    def _at(self,date,ids,z):
        z=array(z,2);ids=_ids(ids);key=month_key(date)
        if z.shape!=(len(ids),self.dimensions):raise ValueError('Current IDs and parent inputs must align')
        values,mask,present=self._aligned(key,ids)
        if not np.array_equal(values[present],z[present]):
            raise ValueError('Current input differs from archived rank, including imputed cells')
        current_bad=~mask;history_bad=np.zeros_like(mask);missing_row=np.zeros_like(mask);level=np.zeros_like(z)
        # Each cell retains the exact original lag-1, lag-2, ... addition order.
        for lag,weight in enumerate(self.weights,1):
            if weight==0:continue
            values,observed,present=self._aligned(key-lag,ids)
            history_bad|=~observed;missing_row|=(~present)[:,None]
            level+=weight*np.where(observed,values,0.)
        valid=~(current_bad|history_bad)
        reasons=dict(cells=z.size,adjusted_eligible=int(valid.sum()),bypassed=int((~valid).sum()),
            current_unobserved=int(current_bad.sum()),prior_unobserved_or_absent=int(history_bad.sum()),
            prior_security_month_absent=int(missing_row.sum()))
        return BoundInnovation(Innovation(z-level,valid,reasons),_date(date),ids,fingerprint(z),
            self.source_fingerprint,self.reference_fingerprint)
