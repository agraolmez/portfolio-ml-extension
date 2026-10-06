"""Exact computational candidates; selection depends on measured equivalence."""
import numpy as np
from scipy.linalg.blas import dtrmm,dsyrk
from geometry_study.kernel import lu_solve as locked_solve
from temporal_fast_v3.kernels import signal_from_parts
from . import native
from temporal_fast.backend import sincos
from temporal_fast_v3 import native as previous_native


def fingerprint(*values):
    parts=[]
    for value in values:
        if isinstance(value,np.ndarray):
            parts.extend((str((value.dtype.str,value.shape)).encode(),np.ascontiguousarray(value)))
        else:parts.append(str(value).encode())
        parts.append(b'\0')
    return native.sha_parts(parts)


def vector_signal(parts,beta):
    normalized,vol=parts[5],parts[1]
    return ((normalized@beta[1:]+beta[0]/np.sqrt(len(vol)))/vol)[:,None]


def parts_with_mean(fmap,z,vol,innovation,gains):
    """Same v3 normalization call, retaining its already calculated mean."""
    inputs=fmap.inputs(z,innovation,gains);vol=np.asarray(vol)
    if vol.shape!=(len(inputs),) or not np.isfinite(vol).all() or np.any(vol<=0):raise ValueError('Invalid volatility')
    with np.errstate(over='raise',invalid='raise'):
        scale=np.exp(fmap.theta);angles=(inputs*scale)@fmap.parent.frequencies
    cosine,sine=sincos(angles);n,k=cosine.shape
    normalized=np.empty((n,2*k));norm=np.empty(2*k);mean=np.empty(2*k)
    if previous_native.library().normalize(n,k,cosine,sine,normalized,norm,mean):raise ValueError('Invalid feature norm')
    return (inputs,vol,scale,cosine,sine,normalized,norm),mean


def shared_operators(basis,signals,*,direct_solve=False,fused_horner=False):
    market=basis.sample.market
    lag=signals[12].copy()
    for i in range(10,-1,-1):
        scaled=market.growth[i+1][:,None]*lag
        lag=(native.gemm_add(market.m,scaled,signals[i+1]) if fused_horner else
             signals[i+1]+market.m@scaled)
    factors,pivots=basis.prior_lu
    product=dtrmm(1.,factors,signals[12],lower=0,diag=0)
    product=dtrmm(1.,factors,product,lower=1,diag=1,overwrite_b=1)
    for i in range(len(pivots)-1,-1,-1):
        if i!=pivots[i]:product[[i,pivots[i]],:]=product[[pivots[i],i],:]
    cur=signals[0]-signals[12]+market.m@(market.growth[0][:,None]*(lag-product))
    solve=native.solve_lu if direct_solve else lambda a,b:locked_solve(a,b,check_finite=False)
    omega=signals[12]+solve(basis.current_lu,cur)
    prior=solve(basis.prior_lu,lag)
    return omega,omega-market.growth[0][:,None]*prior


def gram(values):
    out=dsyrk(1.,np.asfortranarray(values),trans=1,lower=0)
    i,j=np.tril_indices(out.shape[0],-1);out[i,j]=out[j,i]
    return out


def risk_cost(market,omega,change,*,symmetric=False,factors=None):
    if factors is None:rr=market.gamma*omega.T@market.covariance@omega
    else:
        x,f,specific=factors
        projection=x.T@omega
        diagonal=gram(np.sqrt(specific)[:,None]*omega) if symmetric else omega.T@(specific[:,None]*omega)
        rr=market.gamma*(projection.T@f@projection+diagonal)
    if symmetric:cc=market.wealth*gram(np.sqrt(market.impact)[:,None]*change)
    else:cc=market.wealth*change.T@(market.impact[:,None]*change)
    return (rr+rr.T)/2,(cc+cc.T)/2
