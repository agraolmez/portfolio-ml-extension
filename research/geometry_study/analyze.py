"""Prespecified paired seed-average inference and a reviewable research report."""
import argparse,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from feature_geometry.history import sha
from inference.core import stationary,utility,bootstrap_utility,basic_interval
from .run import ROOT,OUT,PLAN,locked,write

LABELS={'fixed512':'Fixed 512','full':'Full economic geometry','risk_only':'Risk-only geometry','return_prediction':'Prediction geometry','joint':'Joint gradient training','joint_refit':'Joint geometry, exact refit','fixed1024':'Fixed 1,024','fixed1024_double_ridge':'Fixed 1,024, double ridge'}

def metrics(g):
    net=g.net_excess.to_numpy();gross=g.gross_excess.to_numpy();cost=g.impact_cost.to_numpy()
    vol=float(np.std(net)*np.sqrt(12));gvol=float(np.std(gross)*np.sqrt(12))
    return dict(months=len(g),annual_gross_excess_mean=float(12*gross.mean()),annual_net_excess_mean=float(12*net.mean()),annual_gross_volatility=gvol,annual_net_volatility=vol,annual_net_utility=float(utility(net)),annual_impact_bps=float(12*cost.mean()*10000),monthly_absolute_turnover=float(g.turnover.mean()),average_gross_exposure=float(g.gross_exposure.mean()),net_sharpe=float(12*net.mean()/vol) if vol else None,gross_sharpe=float(12*gross.mean()/gvol) if gvol else None)

def analyze():
    lock=locked();plan=lock['plan'];inf=plan['inference'];folder=OUT/'analysis';folder.mkdir()
    # Lock analysis code before opening any new portfolio return table.
    write(folder/'analysis_lock.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),study_lock_sha256=sha(OUT/'plan_lock.json'),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'inference/core.py',ROOT/'scripts/geometry_inference_reference.R']}))
    pm=json.loads((OUT/'paths_manifest.json').read_text())
    if pm['status']!='COMPLETE' or pm['months']!=540:raise ValueError('Complete paths required')
    for p,h in pm['artifact_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed completed path '+p)
    monthly=pd.read_parquet(OUT/'paths/monthly.parquet').sort_values(['return_eom','seed','method'])
    if monthly.duplicated(['return_eom','seed','method']).any():raise ValueError('Duplicate monthly path key')
    specs=[(s,m) for s in plan['frequency_seeds'] for m in plan['methods']+['fixed1024_double_ridge']]
    if len(monthly)!=540*len(specs):raise ValueError('Incomplete path records')
    rows=[];contrasts=[];decomposition=[];max_calc_error=0.;rchecks=[]
    methods=[c.split('-',1)[1] for c in plan['primary_contrasts']]
    for period,(first,last) in plan['periods'].items():
        sub=monthly[monthly.return_eom.dt.year.between(first,last)]
        n=(last-first+1)*12
        wide=sub.pivot(index='return_eom',columns=['seed','method'],values='net_excess').reindex(columns=pd.MultiIndex.from_tuples(specs))
        if len(wide)!=n or wide.isna().any().any() or list(wide.index)!=list(pd.date_range(f'{first}-01-31',f'{last}-12-31',freq='ME')):raise ValueError('Calendar/seed alignment failed')
        matrix=wide.to_numpy();point=utility(matrix)
        for (seed,method),g in sub.groupby(['seed','method']):
            result=metrics(g);rows.append(dict(period=period,seed=int(seed),method=method,**result))
            # Independent scalar moment identity rather than helper reuse.
            values=[float(x) for x in g.net_excess];mean=math.fsum(values)/len(values)
            variance=math.fsum((x-mean)**2 for x in values)/len(values)
            independent=12*mean-60*variance
            err=abs(independent-result['annual_net_utility']);max_calc_error=max(max_calc_error,err)
            if err>1e-12:raise ValueError('Independent utility calculation differs')
        for method in methods:
            by_seed=[];meanparts=[]
            for seed in plan['frequency_seeds']:
                a=sub[(sub.seed==seed)&(sub.method=='full')];b=sub[(sub.seed==seed)&(sub.method==method)]
                mu=12*(a.gross_excess.mean()-b.gross_excess.mean());cost=-12*(a.impact_cost.mean()-b.impact_cost.mean());risk=-60*(a.net_excess.var(ddof=0)-b.net_excess.var(ddof=0))
                delta=float(utility(a.net_excess)-utility(b.net_excess));by_seed.append(delta)
                if abs(delta-(mu+cost+risk))>1e-12:raise ValueError('Utility accounting decomposition differs')
                decomposition.append(dict(period=period,seed=seed,contrast='full-'+method,delta_utility=delta,gross_mean_contribution=mu,cost_mean_contribution=cost,net_variance_contribution=risk))
            contrasts.append(dict(period=period,contrast='full-'+method,mean_seed_delta=float(np.mean(by_seed)),all_seed_points_positive=bool(min(by_seed)>0),seed1=by_seed[0],seed2=by_seed[1],seed3=by_seed[2]))
        if period not in ('evaluation','exposed_robustness'):continue
        pd.DataFrame(matrix,columns=[f's{s}_{m}' for s,m in specs]).to_csv(folder/f'{period}_returns.csv',index=False,float_format='%.17g')
        for length in inf['block_months']:
            indices,starts,restart,digests=stationary(plan['id'],inf['master_seed'],length,f'{first}-01-31',n,replicas=inf['replicas'])
            boot=bootstrap_utility(matrix,indices)
            np.savez_compressed(folder/f'{period}_L{length}_bootstrap.npz',indices=indices,starts=starts,restart=restart,utility_draws=boot,seed_sha256=np.array(digests))
            indices.astype('<i4').tofile(folder/f'{period}_L{length}_indices.bin')
            pd.DataFrame(boot).to_csv(folder/f'{period}_L{length}_utility_python.csv',index=False,header=False,float_format='%.17g')
            for method in methods:
                pa=[specs.index((seed,'full')) for seed in plan['frequency_seeds']];pb=[specs.index((seed,method)) for seed in plan['frequency_seeds']]
                observed=float(np.mean(point[pa]-point[pb]));draws=np.mean(boot[:,pa]-boot[:,pb],axis=1)
                lo,hi=basic_interval(observed,draws,inf['alpha'],inf['family']);ulo,uhi=basic_interval(observed,draws,inf['alpha'],1)
                rchecks.append(dict(period=period,block_months=length,contrast='full-'+method,mean_seed_delta=observed,adjusted_lower=lo,adjusted_upper=hi,ordinary_lower=ulo,ordinary_upper=uhi,replicas=inf['replicas'],family=inf['family']))
            print(f'Bootstrap complete: {period}, mean block {length}, {inf["replicas"]} paired draws',flush=True)
    summary=pd.DataFrame(rows);summary.to_csv(folder/'strategy_summary.csv',index=False)
    means=summary.groupby(['period','method'],sort=False).mean(numeric_only=True).drop(columns='seed').reset_index();means.to_csv(folder/'mean_seed_metrics.csv',index=False)
    contrast=pd.DataFrame(contrasts);contrast.to_csv(folder/'seed_contrasts.csv',index=False)
    intervals=pd.DataFrame(rchecks);intervals.to_csv(folder/'primary_intervals.csv',index=False)
    pd.DataFrame(decomposition).to_csv(folder/'utility_decomposition.csv',index=False)
    diagnostics=[];scales=[]
    names=json.loads((ROOT/'output/step2_20260910/source_settings.json').read_text())['features']
    for path in sorted((OUT/'fits').glob('*/*/*/fit.json')):
        j=json.loads(path.read_text());diagnostics.append({k:j.get(k) for k in ['year','seed','method','training_months','training_first','training_last','latest_available_at','coefficient_ridge','geometry_ridge','objective_normalizer','status','profile_evaluations','seconds','solve_residual','scale_min','scale_max','final_training_value','plan_lock_sha256','optimizer_implementation']})
        if j['method'] in ['full','risk_only','return_prediction','joint']:
            with np.load(path.parent/'model.npz',allow_pickle=False) as z:theta=z['theta']
            for name,value in zip(names,theta):scales.append(dict(year=j['year'],seed=j['seed'],method=j['method'],characteristic=name,log_scale=float(value),scale=float(np.exp(value))))
    pd.DataFrame(diagnostics).to_csv(folder/'fit_diagnostics.csv',index=False);pd.DataFrame(scales).to_csv(folder/'learned_scales.csv',index=False)
    # Retained practical references use their original period initialization.
    prior=pd.read_csv(ROOT/'output/step10_20260913/strategy_summary.csv')
    prior=prior[prior.variant.eq('primary')&prior.method.isin(['R1','A0','A1','B2'])]
    prior.to_csv(folder/'retained_first_extension_references.csv',index=False)
    # Independent R recomputation on the same saved returns and resample indices.
    subprocess.run(['Rscript',str(ROOT/'scripts/geometry_inference_reference.R'),str(folder)],check=True)
    primary=intervals[intervals.period.eq('evaluation')];robust=contrast[contrast.period.eq('exposed_robustness')];seedprimary=contrast[contrast.period.eq('evaluation')]
    success=bool((primary.adjusted_lower>0).all() and (primary.mean_seed_delta>0).all() and seedprimary.all_seed_points_positive.all() and (robust.mean_seed_delta>0).all())
    result=dict(status='COMPLETE',historical_paths_complete=True,retrospective_evaluation_years=[2011,2020],robustness_years=[2021,2025],primary_family_improvement_established=success,originality_established=False,rpm_full_method_comparison_completed=False,independent_scalar_utility_max_error=max_calc_error,independent_R=json.loads((folder/'R_verification.json').read_text()),conditional_on_original_hyperparameters=True,all_outcomes_previously_exposed=True,bootstrap_refits_models=False,primary_contrasts=primary.to_dict('records'),all_seed_point_checks=seedprimary.to_dict('records'))
    write(folder/'analysis_result.json',result)
    write(folder/'analysis_manifest.json',dict(status='COMPLETE',artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir() if p.is_file()}))
    print(json.dumps({k:result[k] for k in ['status','primary_family_improvement_established','originality_established','independent_scalar_utility_max_error']}),flush=True)

if __name__=='__main__':analyze()
