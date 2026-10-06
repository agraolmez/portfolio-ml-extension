"""Load spilled months inside the same bounded numerical worker queue."""
from dataclasses import dataclass
from datetime import date
import numpy as np
from feature_geometry.moments import MomentBasis
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import Innovation,month_key
from temporal_study.controls import current_only_batch_factory
from .streaming import OrderedObjective as PreviousObjective


@dataclass(frozen=True)
class DeferredMonth:
    day: str
    cutoff: str
    source_id: str
    loader: object
    current_only: bool=False

    def __post_init__(self):
        for value in (self.day,self.cutoff):
            month_key(value)
            if date.fromisoformat(value).isoformat()!=value:raise ValueError('Canonical month keys required')
        if not callable(self.loader) or not self.source_id or type(self.current_only) is not bool:
            raise ValueError('Bound month loader required')
        if self.day>=self.cutoff:raise ValueError('Unmatured deferred month')

    def load(self):
        basis,history=self.loader()
        m=basis.sample.market
        if m.decision_date!=self.day or m.source_id!=self.source_id or basis.sample.available_at>self.cutoff:
            raise ValueError('Deferred sample source, decision or cutoff mismatch')
        if self.current_only:return next(current_only_batch_factory(lambda:iter([basis]))())
        return basis,history


class OrderedObjective(PreviousObjective):
    def __init__(self,*args,deferred_loads=False,**kwargs):
        if type(deferred_loads) is not bool:raise ValueError('Explicit deferred-loading option required')
        super().__init__(*args,**kwargs)
        self.deferred_loads=deferred_loads;self._deferred_identity=deferred_loads

    def __call__(self,*args,**kwargs):
        if self.deferred_loads!=self._deferred_identity:raise ValueError('Deferred execution option changed')
        return super().__call__(*args,**kwargs)

    def _validate_actual(self,basis,innovations):
        if not isinstance(basis,MomentBasis):raise ValueError('An audited parent moment-basis implementation is required')
        if self.workers>1 and type(basis) is not LeanBasis:raise ValueError('Parallel work requires the audited locked-solve LeanBasis')
        m=basis.sample.market
        if basis.sample.available_at>self.settings.cutoff:raise ValueError('Unmatured outcome at fitting cutoff')
        if (m.characteristics[0].shape[1]!=self.feature_map.parent.dimensions or len(innovations)!=13
                or any(not isinstance(x,Innovation) or x.values.shape!=z.shape for x,z in zip(innovations,m.characteristics))):
            raise ValueError('Misaligned characteristic dictionary or temporal slices')
        return basis,innovations,None

    def _batches(self):
        previous=None
        for value in self.batch_factory():
            if type(value) is DeferredMonth:
                day=value.day
                if value.cutoff!=self.settings.cutoff or day>=self.settings.cutoff:
                    raise ValueError('Unmatured or mismatched deferred cutoff')
                batch=value if self.deferred_loads else self._validate_actual(*value.load())
            else:
                batch=self._validate_actual(*value);day=batch[0].sample.market.decision_date
            if previous is not None and day<=previous:raise ValueError('Training dates must be unique and ordered')
            previous=day
            yield batch

    def _audit(self,batch):
        if type(batch) is DeferredMonth:batch=self._validate_actual(*batch.load())
        return super()._audit(batch)
