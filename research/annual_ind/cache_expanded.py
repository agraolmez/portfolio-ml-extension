"""Larger bounded input retention; exactly the same source and payload checks."""
import pandas as pd
from annual_ind.inputs import check_sample,OperatorCache as OriginalOperators
from annual_ind.retained_inputs import content,retained_size
from annual_ind.preparation_checks import PreparationFiles,PreparationCheckedInputCache
from annual_ind.memory_policy import admit
from feature_geometry.features import fingerprint
from geometry_study.kernel import LeanBasis


class OperatorCache(OriginalOperators):
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
        if self.bytes+size<=self.maximum_bytes and admit(size):
            for a in (*basis.current_lu,*basis.prior_lu):a.setflags(write=False)
            self.items[key]=(sample.market.fingerprint,basis.current_lu,basis.prior_lu,
                             fingerprint(*basis.current_lu,*basis.prior_lu))
            self.bytes+=size
        self.guard();return basis


class ExpandedInputCache(PreparationCheckedInputCache):
    def _get(self,history,day,cutoff):
        if day in self.items:
            sample,innovation,digest=self.items[day]
            if content(sample,innovation)!=digest:
                raise ValueError('Cached sample/reference mutated')
            self.hits+=1
        else:
            sample=history.training_sample(day,cutoff)
            bound=history.innovations(sample.market)
            innovation=history.validate_market(sample.market,bound)
            digest=content(sample,innovation);self.misses+=1
            size=retained_size(sample,innovation)
            if self.bytes+size<=self.maximum_bytes and admit(size):
                self.items[day]=(sample,innovation,digest);self.bytes+=size
        if sample.market.decision_date!=day or sample.available_at>cutoff:
            raise ValueError('Cached sample after requested cutoff')
        return sample,innovation

    def get_batch(self,history,day,cutoff,operators):
        if not isinstance(history.files,PreparationFiles):
            raise TypeError('Explicit before/after preparation source verifier required')
        with history.files.preparation():
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
                sample,innovation=self._get(history,day,cutoff)
                result=operators.basis(sample),innovation
            else:
                basis,innovation=restored;check_sample(basis.sample)
                if (basis.sample.market.decision_date!=day or basis.sample.available_at>cutoff
                        or basis.sample.market.source_id!=history.source_id):
                    raise ValueError('Restored sample source, day or cutoff differs')
                self.disk_hits+=1;result=basis,innovation
            self.guard();return result
