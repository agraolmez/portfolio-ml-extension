"""Joint gradients from vector holdings, without rebuilding coefficient Hessians."""
from dataclasses import replace
import time
import numpy as np
from feature_geometry.features import project_scales
from feature_geometry.training import _solve,validate_training
from .kernel import ordered,evaluate


def state(bases,fmap,theta,beta,settings,gradient=True):
    count=len(bases)
    def one(b):
        market=b.sample.market
        signals=[fmap.signals(z,vol,theta) for z,vol in zip(market.characteristics,market.volatility)]
        v,u=b.operators([s@beta for s in signals]);sv=market.covariance@v
        value=(b.sample.returns@v-.5*market.gamma*v@sv-.5*market.wealth*np.sum(market.impact*u*u))/count
        if not gradient:return float(value),None,None
        adj=b.pullback((b.sample.returns-market.gamma*sv)/count,-market.wealth*market.impact*u/count)
        gb=sum(s.T@g for s,g in zip(signals,adj))
        gt=[fmap.vjp(z,vol,theta,np.outer(g,beta)) for z,vol,g in zip(market.characteristics,market.volatility,adj)]
        return float(value),gb,gt
    value=-.5*settings.ridge*(beta@beta)-.5*settings.scale_penalty*(theta@theta)
    gb=-settings.ridge*beta.copy();gt=-settings.scale_penalty*theta.copy()
    for val,bgrad,tgrad in ordered(one,bases):
        value+=val
        if gradient:
            gb+=bgrad
            for piece in tgrad:gt+=piece
    if not np.isfinite(value) or not np.isfinite(gb).all() or not np.isfinite(gt).all():raise ValueError('Nonfinite joint value/gradient')
    return dict(value=float(value),beta_gradient=gb,theta_gradient=gt)


def fit_joint(bases,fmap,settings,progress=None,objective_scale=1.,coefficient_ridge=None):
    validate_training(bases,fmap,settings);start=time.monotonic();theta=np.zeros(fmap.dimensions)
    initial=evaluate(bases,fmap,theta,settings,'full');beta=initial['beta'].copy();preconditioner=initial['risk']+initial['cost']
    value=initial['value'];trace=[dict(iteration=0,value=value)];evaluations=1;status='ITERATION_BUDGET_REACHED'
    for iteration in range(1,settings.max_steps+1):
        current=state(bases,fmap,theta,beta,settings,True);raw_grad=current['theta_gradient'];grad=raw_grad/objective_scale
        pg=float(np.max(abs(project_scales(theta+grad,settings.bound)-theta)))
        if pg<=settings.gradient_tolerance:status='PROJECTED_GRADIENT_TOLERANCE';break
        step=settings.initial_step/max(float(np.max(abs(grad))),1e-30)
        bg=current['beta_gradient'];db,_=_solve(preconditioner,bg,settings.ridge)
        accepted=False
        for bt in range(settings.max_backtracks):
            th=project_scales(theta+step*grad,settings.bound);dt=th-theta;be=beta+step*db
            slope=float(raw_grad@dt+step*bg@db)
            if slope<=0:break
            trial=state(bases,fmap,th,be,settings,False);evaluations+=1
            if trial['value']>=value+settings.armijo*slope:accepted=True;break
            step*=.5
        if not accepted:status='LINE_SEARCH_STALLED';break
        theta,beta,value=th,be,trial['value']
        trace.append(dict(iteration=iteration,value=value,backtracks=bt,step=step,projected_gradient_before=pg))
        if progress:progress(trace[-1])
    final_settings=replace(settings,ridge=coefficient_ridge) if coefficient_ridge is not None else settings
    final=evaluate(bases,fmap,theta,final_settings,'full')
    return dict(theta=theta,beta=final['beta'],geometry_beta=beta,status=status,trace=trace,profile_evaluations=evaluations,seconds=time.monotonic()-start,solve_residual=final['solve_residual'],final_training_value=value,optimizer_implementation='Exact vector portfolio objective and adjoint coefficient gradient; fixed initial Hessian preconditioner')
