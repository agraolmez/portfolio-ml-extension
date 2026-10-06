"""Bounded strict-double fused-array computation with exact neutral bypass."""
import hashlib
import numpy as np
from feature_geometry.features import array
from feature_geometry.training import _solve
from temporal_fast.streaming import ordered,feature_parts,signal_from_parts,gradient_from_parts
from temporal_fast_v2.streaming import OrderedObjective as PreviousObjective
from temporal_fast_v2.kernels import rank_one_gradient,packed_operators,shared_horner_operators
from .kernels import (feature_parts as native_parts,signal_from_parts as native_signals,
                      rank_one_gradient as native_gradient)
from . import native


class OrderedObjective(PreviousObjective):
    def __init__(self,*args,fused_parts,fused_gradient,**kwargs):
        if any(type(x) is not bool for x in (fused_parts,fused_gradient)):
            raise ValueError('Explicit boolean native variants required')
        super().__init__(*args,**kwargs)
        self.fused_parts,self.fused_gradient=fused_parts,fused_gradient
        self._native_identity=(fused_parts,fused_gradient)
        if fused_parts or fused_gradient:native.library()

    def __call__(self,gains,with_gradient=True):
        if (self.fused_parts,self.fused_gradient)!=self._native_identity:
            raise ValueError('Native variants changed after construction')
        return super().__call__(gains,with_gradient)

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
        nonneutral=not np.all(gains==1.)
        use_shared=self.shared and nonneutral
        parts_function=native_parts if self.fused_parts and nonneutral else feature_parts
        signal_function=native_signals if self.fused_parts and nonneutral else signal_from_parts
        gradient_function=native_gradient if self.fused_gradient and nonneutral else rank_one_gradient

        def moments(batch):
            basis, history, identity = self._audit(batch)
            market = basis.sample.market
            signals = [signal_function(parts_function(fmap,z,vol,innov,gains,self.trig_backend)) for z,vol,innov in
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
                parts=[parts_function(fmap,z,vol,innov,gains,self.trig_backend) for z,vol,innov in
                       zip(market.characteristics,market.volatility,history)]
                signals=[(signal_function(part)@beta)[:,None] for part in parts]
                omega,change=basis.operators(signals);v,u=omega[:,0],change[:,0]
                gv=(basis.sample.returns-market.gamma*(market.covariance@v))/count
                gu=-market.wealth*market.impact*u/count
                adjoints=basis.pullback(gv,gu)
                pieces=[(gradient_function(fmap,z,innov,part,adj,beta) if self.rank_one else
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
