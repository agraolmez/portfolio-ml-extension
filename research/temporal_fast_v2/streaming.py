"""Same complete profile with explicit, bounded computational variants."""
import hashlib
import numpy as np
from feature_geometry.features import array
from feature_geometry.training import _solve
from feature_geometry.moments import MomentBasis
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import Innovation
from temporal_fast.streaming import (OrderedObjective as PreviousObjective, ordered,
    feature_parts, signal_from_parts, gradient_from_parts)
from .kernels import fingerprint,rank_one_gradient,packed_operators,shared_horner_operators


class OrderedObjective(PreviousObjective):
    def __init__(self,*args,parallel_checks,rank_one,packed,shared=False,**kwargs):
        if any(type(x) is not bool for x in (parallel_checks,rank_one,packed,shared)) or (packed and shared):
            raise ValueError('Explicit boolean computational variants required')
        super().__init__(*args,**kwargs)
        self.parallel_checks,self.rank_one,self.packed=parallel_checks,rank_one,packed
        self.shared=shared
        self._revision_identity=(parallel_checks,rank_one,packed,shared)

    def __call__(self,gains,with_gradient=True):
        if (self.parallel_checks,self.rank_one,self.packed,self.shared)!=self._revision_identity:
            raise ValueError('Computational variants changed after construction')
        return super().__call__(gains,with_gradient)

    def _batches(self):
        if not self.parallel_checks:
            yield from super()._batches()
            return
        previous=None
        for basis,innovations in self.batch_factory():
            if not isinstance(basis,MomentBasis):
                raise ValueError('An audited parent moment-basis implementation is required')
            if self.workers>1 and type(basis) is not LeanBasis:
                raise ValueError('Parallel work requires the audited locked-solve LeanBasis')
            market=basis.sample.market
            if previous is not None and market.decision_date<=previous:
                raise ValueError('Training dates must be unique and ordered')
            if basis.sample.available_at>self.settings.cutoff:
                raise ValueError('Unmatured outcome at fitting cutoff')
            if (market.characteristics[0].shape[1]!=self.feature_map.parent.dimensions
                    or len(innovations)!=13
                    or any(not isinstance(x,Innovation) or x.values.shape!=z.shape
                           for x,z in zip(innovations,market.characteristics))):
                raise ValueError('Misaligned characteristic dictionary or temporal slices')
            previous=market.decision_date
            yield basis,innovations,None

    def _audit(self,batch):
        if not self.parallel_checks:return batch
        basis,innovations,_=batch;sample=basis.sample;market=sample.market
        market_identity=fingerprint(market.decision_date,market.source_id,market.ids,
            *market.characteristics,*market.volatility,*market.growth,market.m,
            market.covariance,market.impact,market.wealth,market.gamma)
        sample_identity=fingerprint(market_identity,sample.returns,sample.available_at)
        if market_identity!=market.fingerprint or sample_identity!=sample.fingerprint:
            raise ValueError('Market or training sample arrays changed after construction')
        operators=[*basis.current_lu,*basis.prior_lu,*getattr(basis,'current',()),*getattr(basis,'prior',())]
        identity=fingerprint(sample_identity,*operators,
            *(value for x in innovations for value in (x.values,x.valid)))
        return basis,innovations,identity

    def _evaluate(self, gains, with_gradient):
        fmap = self.feature_map
        gains = array(gains, 1)
        if gains.shape != (fmap.gain_count,): raise ValueError('Invalid temporal gain vector')
        if self._model_hash() != self._model_identity:
            raise ValueError('Frozen model or objective configuration changed')
        p = fmap.parent.columns
        score, risk, cost = np.zeros(p), np.zeros((p,p)), np.zeros((p,p))
        digest = hashlib.sha256(); count = 0
        coverage = {'all_13_slices':{}, 'current_slice':{}, 'reason_counts_complete':True}
        reasons = ('current_unobserved','prior_unobserved_or_absent','prior_security_month_absent')
        use_shared=self.shared and not np.all(gains==1.)

        def moments(batch):
            basis, history, identity = self._audit(batch)
            market = basis.sample.market
            signals = [signal_from_parts(feature_parts(fmap,z,vol,innov,gains,self.trig_backend)) for z,vol,innov in
                       zip(market.characteristics,market.volatility,history)]
            operator=shared_horner_operators if use_shared else packed_operators if self.packed else None
            omega,change = operator(basis,signals) if operator is not None else basis.operators(signals)
            rr = market.gamma*omega.T@market.covariance@omega
            cc = market.wealth*change.T@(market.impact[:,None]*change)
            counts = []
            for innovation in history:
                record = dict(cells=innovation.values.size,adjusted_eligible=int(innovation.valid.sum()),
                              bypassed=int((~innovation.valid).sum()))
                for key in reasons:
                    if key in innovation.reasons:
                        value=innovation.reasons[key]
                        if not isinstance(value,int) or not 0 <= value <= record['cells']:
                            raise ValueError('Invalid source-mask coverage reason count')
                        record[key]=value
                counts.append(record)
            return omega.T@basis.sample.returns,(rr+rr.T)/2,(cc+cc.T)/2,identity,counts

        for ss,rr,cc,identity,counts in ordered(moments,self._batches(),self.workers):
            score += ss; risk += rr; cost += cc; count += 1; digest.update(identity.encode())
            for lag,record in enumerate(counts):
                if any(key not in record for key in reasons):coverage['reason_counts_complete']=False
                for scope in ('all_13_slices','current_slice') if lag==0 else ('all_13_slices',):
                    for key,value in record.items():coverage[scope][key]=coverage[scope].get(key,0)+value
        identity = self._finish_pass(count,digest)
        score,risk,cost = score/count,risk/count,cost/count
        beta,residual = _solve(risk+cost,score,self.settings.ridge)
        parent_value = (score@beta-.5*beta@(risk+cost+self.settings.ridge*np.eye(p))@beta
                        -.5*self.settings.scale_penalty*fmap.theta@fmap.theta)
        value = parent_value/self.objective_scale-self.rho*np.mean((gains-1.)**2)
        grad = None
        if with_gradient:
            grad=np.zeros(fmap.gain_count);second_count=0;second_digest=hashlib.sha256()
            def gradient(batch):
                basis,history,identity=self._audit(batch);market=basis.sample.market
                parts=[feature_parts(fmap,z,vol,innov,gains,self.trig_backend) for z,vol,innov in
                       zip(market.characteristics,market.volatility,history)]
                signals=[(signal_from_parts(part)@beta)[:,None] for part in parts]
                omega,change=basis.operators(signals);v,u=omega[:,0],change[:,0]
                gv=(basis.sample.returns-market.gamma*(market.covariance@v))/count
                gu=-market.wealth*market.impact*u/count
                adjoints=basis.pullback(gv,gu)
                pieces=[(rank_one_gradient(fmap,z,innov,part,adj,beta) if self.rank_one else
                         gradient_from_parts(fmap,z,innov,part,np.outer(adj,beta)))
                    for z,innov,part,adj in zip(market.characteristics,history,parts,adjoints)]
                return pieces,identity
            for pieces,sample_identity in ordered(gradient,self._batches(),self.workers):
                for piece in pieces:grad += piece
                second_count += 1;second_digest.update(sample_identity.encode())
            self._finish_pass(second_count,second_digest,expected=identity)
            grad=grad/self.objective_scale-2.*self.rho*(gains-1.)/fmap.gain_count
        if not np.isfinite(value) or (grad is not None and not np.isfinite(grad).all()):
            raise ValueError('Nonfinite profiled objective or gradient')
        self._source_identity=identity
        return dict(value=float(value),parent_value=float(parent_value),beta=beta,score=score,
            risk=risk,cost=cost,solve_residual=residual,gradient=grad,training_months=count,
            training_source_fingerprint=identity[1],frozen_model_fingerprint=self._model_identity,
            coverage=coverage,coverage_unit='decision-security-characteristic cells; all_13_slices repeats overlapping history')
