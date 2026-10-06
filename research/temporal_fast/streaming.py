"""Ordered monthly parallelism and reuse of feature parts within each gradient.

The original economic profile, conditional solve, provenance checks, summation
order, and gradient formula are retained. At most four months are in flight.
The outer caller owns BLAS thread limits; workers*BLAS threads must not exceed
the separately declared resource budget. No numerical or scientific defaults
are selected by this module.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np
from feature_geometry.features import array
from feature_geometry.training import _solve
from geometry_study.kernel import LeanBasis
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.streaming import StreamingObjective
from temporal_study.controls import CurrentOnlyFeatureMap
from .backend import single_thread_blas, worker_init, sincos


def ordered(function, items, workers):
    if workers == 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        pending = deque()
        source = iter(items)
        for _ in range(workers):
            try: pending.append(pool.submit(function, next(source)))
            except StopIteration: break
        while pending:
            result = pending.popleft().result()
            # Yield before submitting the replacement: live scratch + result
            # count stays bounded by the worker count, including consumers.
            yield result
            try: pending.append(pool.submit(function, next(source)))
            except StopIteration: pass


def feature_parts(fmap, z, vol, innovation, gains, trig_backend='numpy'):
    inputs=fmap.inputs(z, innovation, gains)
    if trig_backend=='numpy' or np.all(gains==1.):
        return fmap.parent._parts(inputs, vol, fmap.theta)
    if trig_backend!='accelerate':raise ValueError('Unknown trigonometric backend')
    vol=np.asarray(vol)
    if vol.shape!=(len(inputs),) or not np.isfinite(vol).all() or np.any(vol<=0):
        raise ValueError('Invalid volatility')
    with np.errstate(over='raise',invalid='raise'):
        scale=np.exp(fmap.theta);angles=(inputs*scale)@fmap.parent.frequencies
    cosine,sine=sincos(angles)
    raw=np.column_stack((cosine,sine));centered=raw-raw.mean(axis=0)
    norm=np.sqrt(np.sum(centered**2,axis=0))
    if np.any(norm==0) or not np.isfinite(norm).all():raise ValueError('Zero or invalid feature norm')
    return inputs,vol,scale,cosine,sine,centered/norm,norm


def signal_from_parts(parts):
    z, vol, _, _, _, normalized, _ = parts
    return np.column_stack((np.full(len(z), 1/np.sqrt(len(z))), normalized))/vol[:,None]


def gradient_from_parts(fmap, z, innovation, parts, upstream):
    _, vol, scale, cosine, sine, normalized, norm = parts
    upstream = array(upstream, 2)
    if upstream.shape != (len(z), fmap.parent.columns):
        raise ValueError('Signal adjoint shape mismatch')
    adjoint = upstream[:,1:]/vol[:,None]
    raw_gradient = (adjoint-adjoint.mean(axis=0)
        - normalized*np.sum(normalized*adjoint,axis=0))/norm
    k = fmap.parent.frequencies.shape[1]
    angle_gradient = -sine*raw_gradient[:,:k]+cosine*raw_gradient[:,k:]
    input_gradient = (angle_gradient@fmap.parent.frequencies.T)*scale
    derivative = z if type(fmap) is CurrentOnlyFeatureMap else innovation.values
    by_characteristic = np.sum(input_gradient*derivative,axis=0)
    return np.bincount(fmap.groups,weights=by_characteristic,minlength=fmap.gain_count)


class OrderedObjective(StreamingObjective):
    def __init__(self, *args, workers, trig_backend='numpy', **kwargs):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('An explicit one-to-four worker bound is required')
        super().__init__(*args, **kwargs)
        if type(self.feature_map) not in (TemporalFeatureMap, CurrentOnlyFeatureMap):
            raise ValueError('Only audited temporal and current-only feature maps are supported')
        self.workers = workers
        if trig_backend not in ('numpy','accelerate'):raise ValueError('Unknown trigonometric backend')
        self.trig_backend=trig_backend
        self._execution_identity=(workers,trig_backend)

    def __call__(self, gains, with_gradient=True):
        if (self.workers,self.trig_backend)!=self._execution_identity:
            raise ValueError('Execution settings changed after construction')
        with single_thread_blas():
            return super().__call__(gains,with_gradient)

    def _batches(self):
        for batch in super()._batches():
            # LeanBasis already serializes LAPACK calls. Do not dispatch the
            # prototype's unguarded dense implementation across worker threads.
            if self.workers > 1 and type(batch[0]) is not LeanBasis:
                raise ValueError('Parallel work requires the audited locked-solve LeanBasis')
            yield batch

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

        def moments(batch):
            basis, history, identity = batch
            market = basis.sample.market
            signals = [signal_from_parts(feature_parts(fmap,z,vol,innov,gains,self.trig_backend)) for z,vol,innov in
                       zip(market.characteristics,market.volatility,history)]
            omega,change = basis.operators(signals)
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
                basis,history,identity=batch;market=basis.sample.market
                parts=[feature_parts(fmap,z,vol,innov,gains,self.trig_backend) for z,vol,innov in
                       zip(market.characteristics,market.volatility,history)]
                signals=[(signal_from_parts(part)@beta)[:,None] for part in parts]
                omega,change=basis.operators(signals);v,u=omega[:,0],change[:,0]
                gv=(basis.sample.returns-market.gamma*(market.covariance@v))/count
                gu=-market.wealth*market.impact*u/count
                adjoints=basis.pullback(gv,gu)
                pieces=[gradient_from_parts(fmap,z,innov,part,np.outer(adj,beta))
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
