"""Algebraically equivalent kernels, retaining double precision and LU locking."""
import hashlib
import numpy as np
from scipy.linalg.blas import dtrmm
from geometry_study.kernel import lu_solve
from temporal_study.controls import CurrentOnlyFeatureMap


def fingerprint(*values):
    """The original byte stream, without materializing a second bytes copy."""
    digest=hashlib.sha256()
    for value in values:
        if isinstance(value,np.ndarray):
            digest.update(str((value.dtype.str,value.shape)).encode())
            digest.update(np.ascontiguousarray(value))
        else:
            digest.update(str(value).encode())
        digest.update(b'\0')
    return digest.hexdigest()


def rank_one_gradient(fmap,z,innovation,parts,adjoint,beta):
    """Use the known outer-product adjoint without creating/copying that matrix."""
    _,vol,scale,cosine,sine,normalized,norm=parts
    adjoint,beta=np.asarray(adjoint),np.asarray(beta)
    if (adjoint.shape!=(len(z),) or beta.shape!=(fmap.parent.columns,)
            or not np.isfinite(adjoint).all() or not np.isfinite(beta).all()):
        raise ValueError('Finite aligned rank-one adjoint required')
    a=adjoint/vol
    raw_gradient=((a-a.mean())[:,None]-normalized*(normalized.T@a))*(beta[1:]/norm)
    k=fmap.parent.frequencies.shape[1]
    angle_gradient=-sine*raw_gradient[:,:k]+cosine*raw_gradient[:,k:]
    input_gradient=(angle_gradient@fmap.parent.frequencies.T)*scale
    derivative=z if type(fmap) is CurrentOnlyFeatureMap else innovation.values
    by_characteristic=np.sum(input_gradient*derivative,axis=0)
    return np.bincount(fmap.groups,weights=by_characteristic,minlength=fmap.gain_count)


def packed_operators(basis,signals):
    """Share one matrix multiplication dispatch for the two Horner recursions."""
    market=basis.sample.market
    if len(signals)!=13 or signals[0].ndim!=2 or any(s.shape!=signals[0].shape for s in signals):
        raise ValueError('Thirteen aligned matrix signals required')
    columns=signals[0].shape[1]
    state=np.column_stack((signals[11],signals[12]))
    work=np.empty_like(state)
    for i in range(10,-1,-1):
        state[:,:columns]*=market.growth[i][:,None]
        state[:,columns:]*=market.growth[i+1][:,None]
        np.matmul(market.m,state,out=work)
        work[:,:columns]+=signals[i]
        work[:,columns:]+=signals[i+1]
        state,work=work,state
    omega=lu_solve(basis.current_lu,state[:,:columns],check_finite=False)
    prior=lu_solve(basis.prior_lu,state[:,columns:],check_finite=False)
    return omega,omega-market.growth[0][:,None]*prior


def shared_horner_operators(basis,signals):
    """Recover the current recursion from the lag recursion and existing LU.

    Let A=sum(I,M G0,...), B=sum(I,M G1,...), and L be the lag numerator.
    The original current numerator is A S12 + S0-S12 + M G0 (L-B S12).
    B S12 is evaluated from the existing pivoted LU; no matrix is approximated,
    truncated, inverted, newly cached, or used across different gain vectors.
    """
    if len(signals)!=13 or signals[0].ndim!=2 or any(s.shape!=signals[0].shape for s in signals):
        raise ValueError('Thirteen aligned matrix signals required')
    market=basis.sample.market
    lag=signals[12].copy()
    for i in range(10,-1,-1):
        lag=signals[i+1]+market.m@(market.growth[i+1][:,None]*lag)
    factors,pivots=basis.prior_lu
    product=dtrmm(1.,factors,signals[12],lower=0,diag=0)
    product=dtrmm(1.,factors,product,lower=1,diag=1,overwrite_b=1)
    # scipy's row interchanges satisfy P B=L U; undo them in reverse order.
    for i in range(len(pivots)-1,-1,-1):
        if i!=pivots[i]:product[[i,pivots[i]],:]=product[[pivots[i],i],:]
    cur=signals[0]-signals[12]+market.m@(market.growth[0][:,None]*(lag-product))
    omega=signals[12]+lu_solve(basis.current_lu,cur,check_finite=False)
    prior=lu_solve(basis.prior_lu,lag,check_finite=False)
    return omega,omega-market.growth[0][:,None]*prior
