"""Small numerical primitives; no future outcomes are accepted by decide()."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

FLOOR = 1e-12
A0 = ['log_trade_risk','log_trade_impact','log_holdings_risk','risk_alignment']
A1 = A0 + ['log1p_aim_dispersion','alignment_times_log1p_dispersion']
VARIANTS = {'primary': (12,12,12), 'horizon36': (36,12,36), 'blocks24': (12,24,24)}


def feasible(m, impact):
    m,impact=np.asarray(m),np.asarray(impact)
    if m.shape!=(len(impact),len(impact)) or not np.isfinite(m).all() or not np.isfinite(impact).all() or np.any(impact<=0):
        raise ValueError('Invalid inertia or impact')
    root=np.sqrt(impact); transformed=m*root[:,None]/root[None,:]
    asym=float(np.max(np.abs(transformed-transformed.T)))
    if asym>1e-10: raise ValueError('Impact-transformed inertia is not symmetric')
    eigen=eigvalsh(transformed,check_finite=False)
    if eigen[0]<0 or eigen[-1]>1:raise ValueError('Inertia spectrum outside [0,1]')
    gain=1-float(eigen[0]); upper=min(1.5,1/gain) if gain>0 else 1.5
    if upper<1:raise ValueError('Baseline outside feasible interval')
    return dict(kappa_min=.5,kappa_max=upper,eigen_min=float(eigen[0]),eigen_max=float(eigen[-1]),symmetry_error=asym)


def risk_apply(x, loadings, factor_cov, specific):
    x=np.asarray(x)
    return loadings@(factor_cov@(loadings.T@x)) + (specific*x if x.ndim==1 else specific[:,None]*x)


def remap(values, ids, next_ids):
    """Map only surviving IDs; entrants zero. Return all omitted rows explicitly."""
    ids,next_ids=np.asarray(ids),np.asarray(next_ids)
    if len(np.unique(ids))!=len(ids) or len(np.unique(next_ids))!=len(next_ids):raise ValueError('Duplicate security ID')
    common,old,new=np.intersect1d(ids,next_ids,assume_unique=True,return_indices=True)
    result=np.zeros((len(next_ids),)+values.shape[1:]);result[new]=values[old]
    dropped=np.flatnonzero(~np.isin(ids,common))
    return result,dropped


def seed(protocol, variant, date, replica):
    payload=dict(protocol_id=protocol['protocol_id'],master_seed=protocol['seeds']['extension'],
        component=f'advantage_moving_blocks_{variant}',date=str(date),replica=int(replica))
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return int(digest[:32],16),digest


def blocks(dates,length,seed_value):
    dates=pd.DatetimeIndex(dates);ords=dates.to_period('M').asi8
    valid=np.array([i for i in range(len(dates)-length+1) if np.all(np.diff(ords[i:i+length])==1)],dtype=int)
    if not len(valid):raise ValueError('No eligible contiguous date block')
    rng=np.random.Generator(np.random.PCG64(seed_value))
    starts=rng.choice(valid,size=(len(dates)+length-1)//length,replace=True)
    sampled=(starts[:,None]+np.arange(length)).ravel()[:len(dates)]
    return starts,sampled,np.bincount(sampled,minlength=len(dates))


def weights(frame):
    return (1/frame.groupby('decision_eom').decision_eom.transform('size')).to_numpy(float)


def fit_ridge(x,y,w,penalty):
    x,y,w=map(lambda a:np.asarray(a,dtype=float),(x,y,w))
    if y.ndim==1:y=y[:,None]
    if not all(np.isfinite(a).all() for a in [x,y,w]) or np.any(w<0) or w.sum()<=0 or penalty<=0:raise ValueError('Invalid ridge inputs')
    w=w/w.sum();mean=w@x;sd=np.sqrt(w@((x-mean)**2));constant=sd<=FLOOR
    scale=np.where(constant,1.,sd);z=(x-mean)/scale;z[:,constant]=0
    intercept=w@y;system=z.T@(w[:,None]*z)+penalty*np.eye(x.shape[1]);rhs=z.T@(w[:,None]*(y-intercept))
    coef=np.linalg.solve(system,rhs)
    residual=np.linalg.norm(system@coef-rhs)/max(np.linalg.norm(rhs),1e-30)
    if residual>1e-10 or not np.isfinite(coef).all():raise ValueError('Ridge solve failure')
    return dict(mean=mean.tolist(),scale=scale.tolist(),constant=constant.tolist(),intercept=intercept.tolist(),
        coefficients=coef.tolist(),penalty=float(penalty),minimum=x[w>0].min(axis=0).tolist(),maximum=x[w>0].max(axis=0).tolist(),residual=float(residual))


def predict(model,x):
    x=np.asarray(x,dtype=float);z=(x-np.array(model['mean']))/np.array(model['scale']);z[...,np.array(model['constant'])]=0
    result=z@np.array(model['coefficients'])+np.array(model['intercept'])
    if not np.isfinite(result).all():raise ValueError('Nonfinite prediction')
    return result


def split(frame,cutoff):
    eligible=frame[frame.usable & frame.label_available_at.le(cutoff)].copy()
    dates=pd.DatetimeIndex(sorted(eligible.decision_eom.unique()))
    info=dict(eligible_months=len(dates),fit_cutoff=str(pd.Timestamp(cutoff).date()))
    if len(dates)<36:return eligible,None,None,dict(**info,status='BASELINE_FALLBACK',reason='fewer_than_36_matured_validation_months',training_months=0)
    validation_dates=dates[-36:]
    if not np.all(np.diff(validation_dates.to_period('M').asi8)==1):return eligible,None,None,dict(**info,status='BASELINE_FALLBACK',reason='validation_not_36_consecutive_months',training_months=0)
    first=validation_dates[0]
    train=eligible[eligible.decision_eom.lt(first)&eligible.outcome_end.lt(first)&eligible.label_available_at.lt(first)]
    validation=eligible[eligible.decision_eom.isin(validation_dates)]
    n=train.decision_eom.nunique();purged=len(dates)-36-n
    info.update(training_months=n,validation_months=36,purged_months=purged,validation_first=str(first.date()),validation_last=str(validation_dates[-1].date()))
    if n<120:return eligible,train,validation,dict(**info,status='BASELINE_FALLBACK',reason='fewer_than_120_training_months_after_purge')
    return eligible,train,validation,dict(**info,status='FITTED',reason=None)


def select_model(train,validation,all_rows,features,penalties):
    x=train[features].to_numpy();y=train[['b','c']].to_numpy();w=weights(train)
    xv=validation[features].to_numpy();yv=validation[['b','c']].to_numpy();wv=weights(validation);wv/=wv.sum()
    wn=w/w.sum();variance=np.maximum(wn@((y-wn@y)**2),FLOOR)
    candidates=[]
    for penalty in penalties:
        model=fit_ridge(x,y,w,penalty);mse=wv@((predict(model,xv)-yv)**2);loss=float(np.mean(mse/variance))
        candidates.append(dict(penalty=float(penalty),loss=loss,mse=mse.tolist(),training_target_variance=variance.tolist()))
    best=min(v['loss'] for v in candidates);tol=FLOOR*(1+abs(best))
    chosen=max(v['penalty'] for v in candidates if v['loss']<=best+tol)
    model=fit_ridge(all_rows[features].to_numpy(),all_rows[['b','c']].to_numpy(),weights(all_rows),chosen)
    model.update(features=features,candidates=candidates)
    return model


def decide(fit,method,feature_row,lower,upper):
    if not lower<=1<=upper:raise ValueError('Baseline infeasible')
    reason=None
    if fit['status']!='FITTED':reason=fit['reason']
    elif feature_row['initial_curvature']<=FLOOR:reason='initial_curvature_floor'
    elif method=='A2' and fit.get('uncertainty_status')!='FITTED':reason='uncertainty_fit_failure'
    if reason:return dict(kappa=1.,delta=0.,b_hat=None,c_raw=None,c_hat=None,u=None,out_of_range=False,out_of_range_features=[],fallback_reason=reason)
    model=fit['models']['A0' if method=='A0' else 'A1'];x=np.array([feature_row[k] for k in model['features']])
    b,c=predict(model,x);u=float(np.std([predict(rep,x)[0] for rep in fit['bootstrap']],ddof=1)) if method=='A2' else 0.
    chat=max(1.,float(c));numerator=np.sign(b)*max(abs(b)-u,0.) if method=='A2' else b
    delta=float(np.clip(numerator/(chat+1),lower-1,upper-1))
    flags=(x<np.array(model['minimum']))|(x>np.array(model['maximum']))
    return dict(kappa=1+delta,delta=delta,b_hat=float(b),c_raw=float(c),c_hat=chat,u=u,
        out_of_range=bool(flags.any()),out_of_range_features=[k for k,f in zip(model['features'],flags) if f],fallback_reason=None)
