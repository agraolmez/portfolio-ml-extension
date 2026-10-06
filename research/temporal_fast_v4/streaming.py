"""Bounded strict-double fused-array computation with exact neutral bypass."""
import hashlib
import numpy as np
from feature_geometry.features import array
from feature_geometry.training import _solve
from temporal_fast.streaming import ordered,feature_parts,signal_from_parts,gradient_from_parts
from temporal_fast_v3.streaming import OrderedObjective as PreviousObjective
from temporal_fast_v2.kernels import rank_one_gradient,packed_operators,shared_horner_operators
from temporal_fast_v3.kernels import (feature_parts as native_parts,signal_from_parts as native_signals,
                      rank_one_gradient as native_gradient)
from .kernels import fingerprint as accelerated_fingerprint,shared_operators,vector_signal,risk_cost,parts_with_mean
from temporal_fast_v2.kernels import fingerprint as original_fingerprint
import psutil
from . import native
from .memory import charged_bytes


class OrderedObjective(PreviousObjective):
    def __init__(self,*args,fast_hash=False,direct_solve=False,fused_horner=False,
                 vector_projection=False,symmetric_gram=False,risk_provider=None,reuse_bytes=0,reuse_mode="full",**kwargs):
        flags=(fast_hash,direct_solve,fused_horner,vector_projection,symmetric_gram)
        if any(type(x) is not bool for x in flags) or type(reuse_bytes) is not int or not 0<=reuse_bytes<=8*2**30:
            raise ValueError('Explicit bounded computational options required')
        if reuse_mode not in ('full','trig'):raise ValueError('Unknown intermediate representation')
        if risk_provider is not None and not callable(risk_provider):raise ValueError('Checked risk provider required')
        super().__init__(*args,**kwargs)
        self.fast_hash,self.direct_solve,self.fused_horner,self.vector_projection,self.symmetric_gram=flags
        self.risk_provider,self.reuse_bytes=risk_provider,reuse_bytes
        self.reuse_mode=reuse_mode
        self._options=(*flags,risk_provider,reuse_bytes,reuse_mode)
        if fast_hash or direct_solve or fused_horner or (reuse_bytes and reuse_mode=="trig"):native.library()

    def __call__(self,gains,with_gradient=True):
        if self._options!=(self.fast_hash,self.direct_solve,self.fused_horner,self.vector_projection,
                           self.symmetric_gram,self.risk_provider,self.reuse_bytes,self.reuse_mode):
            raise ValueError('Computational options changed after construction')
        return super().__call__(gains,with_gradient)

    def _audit(self,batch):
        if not self.fast_hash:return super()._audit(batch)
        basis,innovations,_=batch;sample=basis.sample;market=sample.market
        fingerprint=accelerated_fingerprint
        market_identity=fingerprint(market.decision_date,market.source_id,market.ids,
            *market.characteristics,*market.volatility,*market.growth,market.m,
            market.covariance,market.impact,market.wealth,market.gamma)
        sample_identity=fingerprint(market_identity,sample.returns,sample.available_at)
        if market_identity!=market.fingerprint or sample_identity!=sample.fingerprint:
            raise ValueError('Market or training sample arrays changed after construction')
        operators=[*basis.current_lu,*basis.prior_lu,*getattr(basis,'current',()),*getattr(basis,'prior',())]
        identity=fingerprint(sample_identity,*operators,*(v for x in innovations for v in (x.values,x.valid)))
        return basis,innovations,identity

    def _evaluate(self, gains, with_gradient):
        if np.all(np.asarray(gains)==1.):return super()._evaluate(gains,with_gradient)
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

        reuse={};retained=0;reused=0;peak_reuse=0;admission_closed=False
        process=psutil.Process()
        reuse_limit=min(self.reuse_bytes,max(0,24*2**30-charged_bytes()-4*2**30),
                        max(0,psutil.virtual_memory().available-4*2**30)) if with_gradient else 0

        def moments(batch):
            basis, history, identity = self._audit(batch)
            market = basis.sample.market
            blocks=([parts_with_mean(fmap,z,vol,innov,gains) for z,vol,innov in
                   zip(market.characteristics,market.volatility,history)] if reuse_limit and not admission_closed else None)
            parts=[block[0] for block in blocks] if blocks is not None else None
            signals=([signal_function(part) for part in parts] if parts is not None else
                     [signal_function(parts_function(fmap,z,vol,innov,gains,self.trig_backend)) for z,vol,innov in
                      zip(market.characteristics,market.volatility,history)])
            if self.direct_solve or self.fused_horner:
                omega,change=shared_operators(basis,signals,direct_solve=self.direct_solve,fused_horner=self.fused_horner)
            else:omega,change=shared_horner_operators(basis,signals)
            factors=self.risk_provider(market) if self.risk_provider is not None else None
            rr,cc=risk_cost(market,omega,change,symmetric=self.symmetric_gram,factors=factors)
            payload=None
            if parts is not None:
                # Inputs are supplied again in the checked second pass; only privately
                # generated intermediate arrays live here, within this one evaluation.
                compact=[(None,*part[1:5],part[5] if self.reuse_mode=='full' else None,part[6]) for part in parts]
                means=[block[1] for block in blocks]
                size=sum(a.nbytes for part in compact for a in part[2:] if a is not None)+sum(a.nbytes for a in means)+omega.nbytes+change.nbytes
                payload=(market.decision_date,identity,compact,means,omega,change,size)
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
            return omega.T@basis.sample.returns,rr,cc,identity,counts,payload

        for ss,rr,cc,identity,counts,payload in ordered(moments,self._batches(),self.workers):
            if payload is not None and retained+payload[-1]<=reuse_limit and charged_bytes()<20*2**30 and psutil.virtual_memory().available>4*2**30:
                reuse[payload[0]]=payload[1:];retained+=payload[-1];peak_reuse=max(peak_reuse,retained)
            elif payload is not None:admission_closed=True
            payload=None
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
                saved=reuse.pop(market.decision_date,None)
                if saved is not None:
                    saved_identity,parts,means,om,ch,size=saved
                    if saved_identity!=identity:raise ValueError('Training source changed between objective passes')
                    if self.reuse_mode=='trig':
                        parts=[(*part[:5],native.normalized_from_stats(part[3],part[4],mean,part[6]),part[6]) for part,mean in zip(parts,means)]
                    v,u=om@beta,ch@beta
                else:
                    parts=[parts_function(fmap,z,vol,innov,gains,self.trig_backend) for z,vol,innov in
                           zip(market.characteristics,market.volatility,history)]
                    signals=[(vector_signal(part,beta) if self.vector_projection else (signal_function(part)@beta)[:,None]) for part in parts]
                    omega,change=basis.operators(signals);v,u=omega[:,0],change[:,0]
                gv=(basis.sample.returns-market.gamma*(market.covariance@v))/count
                gu=-market.wealth*market.impact*u/count
                adjoints=basis.pullback(gv,gu)
                pieces=[(gradient_function(fmap,z,innov,part,adj,beta) if self.rank_one else
                         gradient_from_parts(fmap,z,innov,part,np.outer(adj,beta)))
                    for z,innov,part,adj in zip(market.characteristics,history,parts,adjoints)]
                return pieces,identity,saved is not None
            for pieces,sample_identity,hit in ordered(gradient,self._batches(),self.workers):
                reused+=int(hit)
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
            computation_diagnostics=dict(reused_months=reused,peak_reuse_bytes=peak_reuse,reuse_limit_bytes=reuse_limit),
            coverage=coverage,coverage_unit='decision-security-characteristic cells; all_13_slices repeats overlapping history')
