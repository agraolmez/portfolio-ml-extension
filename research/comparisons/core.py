"""Fixed comparison policies, selection and descriptive measurements."""
import numpy as np
from learner.core import decide,remap
from learner.pipeline import features,risk

SPEEDS=[.5,.75,1.,1.25,1.5]
RHOS=[.25,.5,.75]

def family(choice=None):
    controls=[('R1','R1',None)]
    controls += [(f'B1_{v:g}','B1',v) for v in SPEEDS] if choice is None else [('B1','B1',choice['B1_kappa'])]
    controls += [('B2','B2',None)]
    controls += [(f'B3_{v:g}','B3',v) for v in RHOS] if choice is None else [('B3','B3',choice['B3_rho'])]
    return controls+[(v,v,None) for v in ['A0','A1','A2']]

def smoothing(aim,previous,previous_ids,previous_growth,ids,rho):
    if previous is None:return aim.copy(),np.empty(0,dtype=int),np.empty(0)
    drift=previous*previous_growth;mapped,dropped=remap(drift,previous_ids,ids)
    entrant=~np.isin(ids,previous_ids);mapped[entrant]=aim[entrant]
    return (1-rho)*aim+rho*mapped,previous_ids[dropped],drift[dropped]

def action(month,h,method,parameter,fit,block,cap,smoothed=None):
    # The caller supplies no outcome vector to this function.
    required=['ids','aim','m','loadings','fc','specific','impact','wealth']
    for key in required:
        if not np.isfinite(month[key]).all():raise ValueError('Corrupt baseline input '+key)
    if not np.isfinite(h).all() or h.shape!=month['aim'].shape:raise ValueError('Corrupt inherited holdings')
    aim=month['aim'];target=aim.copy();state={};reason=None
    value=dict(kappa=1.,delta=0.,b_hat=None,c_raw=None,c_hat=None,u=None,out_of_range=False,out_of_range_features=[],fallback_reason=None)
    if method=='B1':value['kappa']=float(np.clip(parameter,cap['kappa_min'],cap['kappa_max']))
    if method in ['B2','B3']:
        candidate=month.get('confidence',{}).get(block,{}).get('mean') if method=='B2' else smoothed
        if candidate is None or np.shape(candidate)!=aim.shape or not np.isfinite(candidate).all():reason='missing_or_nonfinite_extension_aim'
        else:target=np.asarray(candidate).copy()
    if method in ['A0','A1','A2']:
        try:
            q=aim-h-month['m']@(aim-h)
            state=features(month,h,q,block)
            if not all(np.isfinite(v) for v in state.values()):raise ValueError('Nonfinite extension features')
            if fit is None:raise ValueError('Absent annual fit')
            if not fit['first_decision']<=month['date']<=fit['last_decision']:raise ValueError('Annual fit date mismatch')
            value=decide(fit,method,state,cap['kappa_min'],cap['kappa_max'])
        except (KeyError,ValueError,FloatingPointError) as error:
            reason='extension_input_or_forecast_failure:'+str(error)
    if reason:
        target=aim.copy();value.update(kappa=1.,delta=0.,fallback_reason=reason)
    value['delta']=value['kappa']-1
    trade=value['kappa']*((target-h)-month['m']@(target-h))
    if not np.isfinite(trade).all():raise ValueError('Nonfinite implemented trade')
    return dict(target=target,trade=trade,weight=h+trade,decision=value,features=state)

def utility(net):
    net=np.asarray(net);return float(12*(net.mean()-5*np.var(net,ddof=0)))

def select(summary):
    scores=dict(zip(summary.method,summary.annual_net_utility));best=max(scores[f'B1_{v:g}'] for v in SPEEDS)
    speed=min([v for v in SPEEDS if scores[f'B1_{v:g}']>=best-1e-8],key=lambda v:(abs(v-1),v))
    best=max(scores[f'B3_{v:g}'] for v in RHOS);rho=min(v for v in RHOS if scores[f'B3_{v:g}']>=best-1e-8)
    candidates=['R1',f'B1_{speed:g}','B2',f'B3_{rho:g}'];best=max(scores[k] for k in candidates)
    chosen=next(k for k in candidates if scores[k]>=best-1e-8)
    return dict(B1_kappa=speed,B3_rho=rho,C_star=chosen.split('_')[0],C_star_selection_path=chosen,selection_utility={k:scores[k] for k in candidates},tie_absolute_tolerance=1e-8)

def metrics(frame):
    net=frame.net_excess.to_numpy();gross=frame.gross_excess.to_numpy();index=np.r_[1.,np.cumprod(1+net)]
    if np.any(1+net<=0):raise ValueError('Nonpositive compounded diagnostic index')
    result=dict(months=len(frame),annual_gross_excess_mean=12*gross.mean(),annual_net_excess_mean=12*net.mean(),
        annual_net_volatility=np.sqrt(12)*np.std(net,ddof=0),annual_net_utility=utility(net),
        gross_sharpe=np.sqrt(12)*gross.mean()/np.std(gross,ddof=0),net_sharpe=np.sqrt(12)*net.mean()/np.std(net,ddof=0),
        annual_impact_bps=120000*frame.impact_cost.mean(),monthly_absolute_turnover=frame.turnover.mean(),
        average_gross_exposure=frame.gross_exposure.mean(),average_net_exposure=frame.net_exposure.mean(),
        average_modeled_annual_volatility=frame.modeled_annual_volatility.mean(),
        average_maximum_absolute_weight=frame.max_absolute_weight.mean(),average_absolute_weight_HHI=frame.absolute_weight_hhi.mean(),
        net_excess_index_max_drawdown=float(np.min(index/np.maximum.accumulate(index)-1)),
        mean_kappa=frame.kappa.mean(),upper_bound_rate=frame.upper_bound.mean(),lower_bound_rate=frame.lower_bound.mean(),
        fallback_rate=frame.fallback_reason.notna().mean(),out_of_range_rate=frame.out_of_range.mean())
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite performance metric')
    return result
