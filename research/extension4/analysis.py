"""Fixed comparison family, paired monthly bootstrap and independent R handoff."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
from inference.core import stationary,utility,bootstrap_utility,basic_interval
from .storage import write_json


def analyze(frame,config,folder,*,independent_script):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    frame=frame.copy();frame['return_eom']=pd.to_datetime(frame.return_eom)
    specs=[(s,m) for s in config['history']['seeds'] for m in config['controls']]
    expected=pd.date_range(f"{config['history']['first_return_year']}-01-31",f"{config['history']['last_return_year']}-12-31",freq='ME')
    for seed,method in specs:
        rows=frame[(frame.seed==seed)&(frame.method==method)].sort_values('return_eom')
        if list(rows.return_eom)!=list(expected):raise ValueError('Missing or duplicated final method/seed/month')
    if len(frame)!=len(specs)*len(expected):raise ValueError('Unexpected final path rows')
    periods=[('early_descriptive','1981-01-31','2000-12-31'),('parent_selection','2001-01-31','2010-12-31'),
        ('extension4_selection',*config['validation_plan']['selection_returns']),
        ('primary',*config['validation_plan']['primary_returns']),('robustness',*config['validation_plan']['robustness_returns']),
        ('full_descriptive',str(expected[0].date()),str(expected[-1].date()))]
    metrics=[]
    for period,first,last in periods:
        subset=frame[frame.return_eom.between(first,last)]
        for seed,method in specs:
            r=subset[(subset.seed==seed)&(subset.method==method)]
            if r.empty:continue
            net=r.net_excess.to_numpy();vol=float(np.sqrt(12*np.var(net)))
            metrics.append(dict(period=period,seed=seed,method=method,months=len(r),annual_net_utility=float(utility(net)),
                annual_net_mean=float(12*net.mean()),annual_net_volatility=vol,
                net_sharpe=float(12*net.mean()/vol) if vol else None,annual_impact_cost=float(12*r.impact_cost.mean()),
                mean_turnover=float(r.turnover.mean()),mean_gross_exposure=float(r.gross_exposure.mean()),
                mean_net_exposure=float(r.net_exposure.mean()),maximum_absolute_weight=float(r.largest_absolute_weight.max()),
                mean_concentration=float(r.gross_weight_concentration.mean()),mean_modeled_annual_volatility=float(r.modeled_annual_volatility.mean())))
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    settings=config['inference'];intervals=[];all_primary=True;all_robust=True;r_checks=[]
    contrasts=config['primary_comparisons'];seeds=config['history']['seeds']
    rscript=shutil.which('Rscript')
    if not rscript:raise RuntimeError('Independent R verification is required and Rscript is unavailable')
    for period,first,last in [p for p in periods if p[0] in ('primary','robustness')]:
        subset=frame[frame.return_eom.between(first,last)]
        dates=pd.date_range(first,last,freq='ME')
        returns=np.column_stack([subset[(subset.seed==seed)&(subset.method==method)].sort_values('return_eom').net_excess.to_numpy() for seed,method in specs])
        if returns.shape!=(len(dates),len(specs)):raise ValueError('Unpaired inference sample')
        observed=utility(returns)
        for length in settings['expected_block_lengths']:
            case=folder/f'{period}_L{length}';case.mkdir()
            indices,_,_,_=stationary('extension4_v1',settings['seed'],length,last,len(dates),settings['draws'])
            draws=bootstrap_utility(returns,indices)
            np.savetxt(case/'returns.csv',returns,delimiter=',',fmt='%.17g')
            np.savetxt(case/'indices.csv',indices+1,delimiter=',',fmt='%d')
            # R receives raw returns and date indices, never Python utility inputs.
            run=subprocess.run([rscript,str(independent_script),str(case.resolve())],capture_output=True,text=True)
            (case/'R_console.txt').write_text(run.stdout+run.stderr)
            if run.returncode:raise RuntimeError('Independent R calculation failed')
            independent=np.loadtxt(case/'R_draws.csv',delimiter=',',ndmin=2)
            independent_point=np.loadtxt(case/'R_observed.csv',delimiter=',',ndmin=1)
            np.testing.assert_allclose(independent,draws,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(independent_point,observed,atol=1e-11,rtol=1e-10)
            # R computes all mean-seed intervals independently from an explicit index map.
            mapping=[];case_rows=[]
            for challenger,benchmark in contrasts:
                a=[specs.index((s,challenger)) for s in seeds];b=[specs.index((s,benchmark)) for s in seeds]
                point=observed[a]-observed[b];distribution=np.mean(draws[:,a]-draws[:,b],axis=1)
                lo,hi=basic_interval(float(point.mean()),distribution,alpha=settings['family_alpha'],family=len(contrasts))
                row=dict(period=period,block_months=length,challenger=challenger,benchmark=benchmark,
                    mean_delta=float(point.mean()),lower=lo,upper=hi,all_seed_deltas_positive=bool(np.all(point>0)),
                    seed_deltas=point.tolist())
                intervals.append(row);case_rows.append(row);mapping.append([*[i+1 for i in a],*[i+1 for i in b]])
                if period=='primary':all_primary &= bool(np.all(point>0) and lo>0)
                else:all_robust &= bool(point.mean()>0)
            np.savetxt(case/'contrast_columns.csv',mapping,fmt='%d',delimiter=',')
            run=subprocess.run([rscript,str(independent_script),str(case.resolve()),'intervals',str(settings['family_alpha']),str(len(contrasts))],capture_output=True,text=True)
            (case/'R_intervals_console.txt').write_text(run.stdout+run.stderr)
            if run.returncode:raise RuntimeError('Independent R interval verification failed')
            actual=np.loadtxt(case/'R_intervals.csv',delimiter=',',ndmin=2)
            np.testing.assert_allclose(actual,np.array([[r['mean_delta'],r['lower'],r['upper']] for r in case_rows]),atol=1e-11,rtol=1e-10)
            r_checks.append(dict(period=period,block=length,utilities_checked=int(draws.size),intervals_checked=len(contrasts),passed=True))
    write_json(folder/'intervals.json',intervals)
    result=dict(status='VERIFIED_NUMERICAL_ANALYSIS',prespecified_success=bool(all_primary and all_robust),
                primary_criteria_passed=bool(all_primary),robustness_mean_criteria_passed=bool(all_robust),
                independent_R_checks=r_checks,source_timing_certified=False,fresh_holdout=False,
                inference_conditional_on_fitted_models=True,seed_ensemble=False,scientific_study_complete=False,
                bootstrap_draws=settings['draws'],comparison_family=len(contrasts))
    write_json(folder/'analysis_result.json',result)
    return result
