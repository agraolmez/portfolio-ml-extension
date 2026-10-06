"""Economic kernels retaining the released finite-iteration/finite-history rules."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh,solve
from .io import finite


def symmetric(a):
    a=np.asarray(a,dtype=float);finite(a)
    if a.ndim!=2 or a.shape[0]!=a.shape[1]:raise ValueError('Square matrix required')
    if not np.allclose(a,a.T,atol=1e-12,rtol=1e-10):raise ValueError('Asymmetric economic matrix')
    return (a+a.T)/2


def inertia(sigma,impact,wealth,rf,mu=.007,gamma=10.,iterations=10):
    sigma=symmetric(sigma);impact=np.asarray(impact,dtype=float);finite(impact,wealth,rf,mu,gamma)
    n=len(impact)
    if sigma.shape!=(n,n) or np.any(impact<=0) or wealth<=0 or gamma<=0 or 1+rf+mu<=0 or iterations!=10:
        raise ValueError('Invalid impact, wealth, growth, dimensions, or source iteration count')
    root=np.sqrt(impact);x=(gamma/wealth)*sigma/root[:,None]/root[None,:]
    values,vectors=eigh(x,driver='evd',check_finite=False)
    if values[0]<=0:raise ValueError('Positive definite transformed risk required')
    # Rationalized version of (x+2-sqrt((x+2)^2-4))/2 avoids cancellation.
    initial=2/(values+2+np.sqrt(values*(values+4)))
    mt=(vectors*initial)@vectors.T
    sgr=1+sigma/(1+rf+mu)**2
    fixed=x+np.diag(1+np.diag(sgr))
    identity=np.eye(n)
    for _ in range(iterations):
        denominator=symmetric(fixed-mt*sgr)
        mt=cho_solve(cho_factor(denominator,lower=True,check_finite=False),identity,check_finite=False)
        mt=(mt+mt.T)/2
    denominator=symmetric(fixed-mt*sgr)
    next_mt=cho_solve(cho_factor(denominator,lower=True,check_finite=False),identity,check_finite=False)
    ev=eigh(mt,eigvals_only=True,check_finite=False)
    if ev[0]<-1e-10 or ev[-1]>1+1e-10:raise ValueError('Inertia spectrum outside [0,1]')
    m=mt*root[None,:]/root[:,None]
    return m,{'transformed_risk_condition':float(values[-1]/values[0]),'inertia_eigen_min':float(ev[0]),'inertia_eigen_max':float(ev[-1]),'ten_iteration_fixed_point_residual':float(np.max(np.abs(next_mt-mt))),'inertia_asymmetry':float(np.max(np.abs(m-m.T))),'iterations':iterations}


def normalize_signal(raw,vol):
    raw=np.asarray(raw,dtype=float);vol=np.asarray(vol,dtype=float);finite(raw,vol)
    if raw.ndim!=2 or len(vol)!=len(raw) or np.any(vol<=0):raise ValueError('Invalid signal scale')
    centered=raw-raw.mean(axis=0)
    norm=np.sqrt(np.sum(centered**2,axis=0))
    if np.any(norm==0):raise ValueError('Zero cross-sectional random-feature norm')
    # Canonical order matches pfml_feat_fun: constant, cosine block, sine block.
    return np.column_stack([np.full(len(raw),1/np.sqrt(len(raw))),centered/norm])/vol[:,None]


def moment(signals,growth,m,sigma,impact,wealth,ret,gamma=10.,return_operators=False):
    # Thirteen monthly signal slices support twelve terms at t and t-1.
    if len(signals)!=13 or len(growth)!=13:raise ValueError('Full twelve-month lookback plus current signal required')
    n,p=signals[0].shape
    if any(s.shape!=(n,p) for s in signals):raise ValueError('Unaligned lookback security/feature dimensions')
    finite(*signals,*growth,m,sigma,impact,wealth,ret)
    current=np.eye(n);prior=np.eye(n)
    omega=np.zeros((n,p));omega_lag=np.zeros((n,p));const=np.zeros((n,n));const_lag=np.zeros((n,n))
    for i in range(12):
        omega+=current@signals[i];omega_lag+=prior@signals[i+1]
        const+=current;const_lag+=prior
        if i<11:
            current=current@(m*growth[i][None,:]);prior=prior@(m*growth[i+1][None,:])
    omega=solve(const,omega,assume_a='gen',check_finite=False)
    omega_lag=solve(const_lag,omega_lag,assume_a='gen',check_finite=False)
    change=omega-growth[0][:,None]*omega_lag
    risk=gamma*(omega.T@sigma@omega)
    cost=wealth*(change.T@(np.asarray(impact)[:,None]*change))
    result={'r_tilde':omega.T@ret,'risk':symmetric(risk),'tc':symmetric(cost),'denom':symmetric(risk+cost),'signal_t':signals[0]}
    if return_operators:result.update(omega=omega,omega_change=change)
    return result


def ridge_path(denom,r,lambdas):
    a=symmetric(denom);finite(r,lambdas)
    l=np.asarray(lambdas);values,vectors=eigh(a,driver='evd',check_finite=False)
    if np.any(l<0) or values[0]<=0:raise ValueError('Nonpositive moment matrix or invalid ridge penalty')
    coefs=vectors@((vectors.T@r)[:,None]/(values[:,None]+l[None,:]))
    finite(coefs)
    return coefs,{'moment_condition':float(values[-1]/values[0]),'minimum_eigenvalue':float(values[0])}


def report_returns(gross,costs,gamma=10.):
    gross=np.asarray(gross);costs=np.asarray(costs);finite(gross,costs)
    if len(gross)<2 or gross.shape!=costs.shape or np.any(costs<0):raise ValueError('At least two aligned observations and nonnegative costs required')
    net=gross-costs
    return {'months':len(net),'annual_mean_net_excess':12*net.mean(),'annual_net_volatility':np.sqrt(12*net.var(ddof=0)),
        'net_population_utility':12*(net.mean()-gamma/2*net.var(ddof=0)),
        'net_sharpe':np.sqrt(12)*net.mean()/net.std(ddof=0) if net.std(ddof=0)>0 else None,
        'source_gross_sample_variance_utility':12*(net.mean()-gamma/2*gross.var(ddof=1)),
        'source_net_over_gross_sharpe':np.sqrt(12)*net.mean()/gross.std(ddof=1) if gross.std(ddof=1)>0 else None}


def transition(ids,pretrade,aim,m,ret_total,market_growth,impact,wealth,ret_excess,next_ids):
    ids=np.asarray(ids);next_ids=np.asarray(next_ids);h=np.asarray(pretrade);aim=np.asarray(aim)
    ret_total=np.asarray(ret_total);impact=np.asarray(impact);ret_excess=np.asarray(ret_excess);m=np.asarray(m)
    n=len(ids)
    if any(x.shape!=(n,) for x in [h,aim,ret_total,impact,ret_excess]) or m.shape!=(n,n):raise ValueError('Unaligned portfolio dimensions')
    if len(np.unique(ids))!=len(ids) or len(np.unique(next_ids))!=len(next_ids):raise ValueError('Duplicate security IDs')
    finite(h,aim,m,ret_total,market_growth,impact,wealth,ret_excess)
    if 1+market_growth<=0 or wealth<=0 or np.any(np.asarray(impact)<=0):raise ValueError('Invalid growth, impact or wealth')
    w=m@h+(np.eye(len(h))-m)@aim;q=w-h
    drift=w*(1+ret_total)/(1+market_growth)
    lookup=dict(zip(ids,drift));next_h=np.array([lookup.get(i,0.) for i in next_ids])
    dropped=~np.isin(ids,next_ids)
    return {'weights':w,'trades':q,'next_pretrade':next_h,'drifted_weights':drift,'dropped_ids':ids[dropped],'dropped_drifted_weights':drift[dropped],
        'gross_excess':float(w@ret_excess),'impact_cost':float(wealth/2*np.sum(impact*q*q)),
        'turnover':float(np.abs(q).sum()),'gross_exposure':float(np.abs(w).sum()),'source_exit_trade_cost':0.}
