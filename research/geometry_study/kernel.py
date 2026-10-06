"""Memory-bounded exact geometry study kernels; original prototype remains immutable."""
import time
from dataclasses import replace
import numpy as np
from scipy.linalg import lu_solve as _lapack_lu_solve
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

# The installed LAPACK backend failed concurrent solve equivalence checks.
# Serialize only these calls; independent signal/matrix work remains concurrent.
_LU_LOCK=threading.Lock()
_POOL=ThreadPoolExecutor(max_workers=4)
def lu_solve(*args,**kwargs):
    with _LU_LOCK:return _lapack_lu_solve(*args,**kwargs)

def ordered(function,items):
    source=iter(items);pending=deque()
    for _ in range(4):
        try:pending.append(_POOL.submit(function,next(source)))
        except StopIteration:break
    while pending:
        result=pending.popleft().result()
        try:pending.append(_POOL.submit(function,next(source)))
        except StopIteration:pass
        yield result
from feature_geometry.moments import MomentBasis
from feature_geometry.features import project_scales
from feature_geometry.training import _solve, validate_training

class LeanBasis(MomentBasis):
    """Horner evaluation keeps two LU factors instead of 24 dense powers.

    The products need not commute. Reverse Horner evaluation uses A_i=m diag(g_i)
    in exactly the original order; the adjoint uses A_i' sequentially.
    """
    def __init__(self, sample):
        super().__init__(sample)
        del self.current, self.prior

    @property
    def nbytes(self):
        return self.current_lu[0].nbytes+self.prior_lu[0].nbytes

    def operators(self, signals):
        m=self.sample.market
        cur=signals[11].copy(); lag=signals[12].copy()
        for i in range(10,-1,-1):
            g=m.growth[i]; gp=m.growth[i+1]
            cur=signals[i]+m.m@(g*cur if cur.ndim==1 else g[:,None]*cur)
            lag=signals[i+1]+m.m@(gp*lag if lag.ndim==1 else gp[:,None]*lag)
        omega=lu_solve(self.current_lu,cur,check_finite=False)
        prior=lu_solve(self.prior_lu,lag,check_finite=False)
        return omega,omega-(m.growth[0]*prior if prior.ndim==1 else m.growth[0][:,None]*prior)

    def pullback(self, omega_gradient, change_gradient):
        market=self.sample.market
        cur=lu_solve(self.current_lu,omega_gradient+change_gradient,trans=1,check_finite=False)
        lag=lu_solve(self.prior_lu,-market.growth[0]*change_gradient,trans=1,check_finite=False)
        result=[np.zeros_like(cur) for _ in range(13)]
        for i in range(12):
            result[i]+=cur; result[i+1]+=lag
            if i<11:
                cur=market.growth[i]*(market.m.T@cur)
                lag=market.growth[i+1]*(market.m.T@lag)
        return result


def sums(bases,fmap,theta,objective):
    p=fmap.columns;score=np.zeros(p);risk=np.zeros((p,p));cost=np.zeros((p,p))
    def one(basis):
        sample=basis.sample;market=sample.market
        if objective=='return_prediction':
            sig=fmap.signals(market.characteristics[0],market.volatility[0],theta)
            return dict(score=sig.T@sample.returns/len(sig),risk=sig.T@sig/len(sig),cost=None)
        return basis.matrices(fmap,theta)
    for record in ordered(one,bases):
        score+=record['score'];risk+=record['risk']
        if objective=='full':cost+=record['cost']
    return score/len(bases),risk/len(bases),cost/len(bases)


def gradient(bases,fmap,theta,beta,settings,objective):
    grad=-settings.scale_penalty*theta.copy();count=len(bases)
    def one(b):
        market=b.sample.market
        if objective=='return_prediction':
            sig=fmap.signals(market.characteristics[0],market.volatility[0],theta)
            residual=(b.sample.returns-sig@beta)/(count*len(sig))
            return [fmap.vjp(market.characteristics[0],market.volatility[0],theta,np.outer(residual,beta))]
        vector=[fmap.signals(z,vol,theta)@beta for z,vol in zip(market.characteristics,market.volatility)]
        v,u=b.operators(vector)
        gv=(b.sample.returns-market.gamma*(market.covariance@v))/count
        gu=-market.wealth*market.impact*u/count if objective=='full' else np.zeros_like(u)
        return [fmap.vjp(z,vol,theta,np.outer(adj,beta)) for z,vol,adj in zip(market.characteristics,market.volatility,b.pullback(gv,gu))]
    for pieces in ordered(one,bases):
        for piece in pieces:grad+=piece
    return grad


def evaluate(bases,fmap,theta,settings,objective='full',beta=None):
    score,risk,cost=sums(bases,fmap,theta,objective)
    if beta is None:beta,residual=_solve(risk+cost,score,settings.ridge)
    else:residual=None
    value=score@beta-.5*beta@(risk+cost+settings.ridge*np.eye(len(beta)))@beta-.5*settings.scale_penalty*theta@theta
    return dict(value=float(value),beta=beta,score=score,risk=risk,cost=cost,solve_residual=residual)


def matrix_fit(bases,fmap,settings,objective='full',joint=False,progress=None,objective_scale=1.,coefficient_ridge=None):
    validate_training(bases,fmap,settings);start=time.monotonic();theta=np.zeros(fmap.dimensions)
    current=evaluate(bases,fmap,theta,settings,objective);evaluations=1
    trace=[dict(iteration=0,value=current['value'])];status='ITERATION_BUDGET_REACHED'
    for it in range(1,settings.max_steps+1):
        raw_grad=gradient(bases,fmap,theta,current['beta'],settings,objective)
        grad=raw_grad/objective_scale
        pg=float(np.max(abs(project_scales(theta+grad,settings.bound)-theta)))
        if pg<=settings.gradient_tolerance:status='PROJECTED_GRADIENT_TOLERANCE';break
        step=settings.initial_step/max(float(np.max(abs(grad))),1e-30)
        # Joint comparator starts from the same exact identity beta. Both parameter
        # blocks subsequently move by gradients, with a fixed identity-Hessian
        # preconditioner in beta. It never reprofiles beta during optimization.
        if joint:
            bg=current['score']-(current['risk']+current['cost']+settings.ridge*np.eye(len(current['beta'])))@current['beta']
            if it==1:preconditioner=current['risk']+current['cost']
            db,_=_solve(preconditioner,bg,settings.ridge)
        accepted=False
        for bt in range(settings.max_backtracks):
            th=project_scales(theta+step*grad,settings.bound);dt=th-theta
            be=current['beta']+step*db if joint else None
            slope=float(raw_grad@dt+(step*bg@db if joint else 0))
            if slope<=0:break
            trial=evaluate(bases,fmap,th,settings,objective,beta=be);evaluations+=1
            if trial['value']>=current['value']+settings.armijo*slope:accepted=True;break
            step*=.5
        if not accepted:status='LINE_SEARCH_STALLED';break
        theta,current=th,trial
        trace.append(dict(iteration=it,value=current['value'],backtracks=bt,step=step,projected_gradient_before=pg))
        if progress:progress(trace[-1])
    geometry_beta=current['beta'].copy()
    # All geometry-learning objectives receive exactly the full economic beta refit.
    # Joint raw beta is retained as a separate optimization diagnostic/path.
    final_settings=replace(settings,ridge=coefficient_ridge) if coefficient_ridge is not None else settings
    final=evaluate(bases,fmap,theta,final_settings,'full') if objective!='full' or joint else current
    return dict(theta=theta,beta=final['beta'],geometry_beta=geometry_beta,status=status,trace=trace,
                profile_evaluations=evaluations,seconds=time.monotonic()-start,
                solve_residual=final['solve_residual'],final_training_value=current['value'])


def fit(bases,fmap,settings,objective='full',joint=False,progress=None,objective_scale=1.,coefficient_ridge=None):
    if joint:
        if objective!='full':raise ValueError('Joint comparator requires the full economic objective')
        from .joint import fit_joint
        return fit_joint(bases,fmap,settings,progress,objective_scale,coefficient_ridge)
    return matrix_fit(bases,fmap,settings,objective,False,progress,objective_scale,coefficient_ridge)
