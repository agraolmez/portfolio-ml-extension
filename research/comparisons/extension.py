"""Extend accepted baseline and confidence machinery without editing old files."""
import argparse,json
import numpy as np
import pandas as pd
from baseline import engine as machinery
from baseline.core import ridge_path,transition
from baseline.engine import feature_indices
from baseline.io import datekey
from confidence.core import replica_seed,moving_blocks,fit_coefficients,state_features
from .io import *


def configure():
    machinery.PREP, machinery.OUT=BASE,OUT/'base_engine'
    return machinery.Engine(end=DATES[-1])


def moments():
    begin('moments');BASE.mkdir(parents=True,exist_ok=True)
    link(ROOT/'data/prepared/step4_v1/risk',BASE/'risk')
    for name in ['matrices','moments/g0','moments/g1','coefficients/g0','coefficients/g1']:
        (BASE/name).mkdir(parents=True,exist_ok=True)
        for p in (OLD_BASE/name).glob('*.npz'):link(p,BASE/name/p.name)
    for g in [0,1]:link(ROOT/f'output/step6_20260911/reference/rff_g{g}.bin',OUT/f'base_engine/reference/rff_g{g}.bin')
    engine=configure();engine.build_moments(ADDED)
    finish('moments',[p for sub in ['matrices','moments'] for p in (BASE/sub).rglob('*.npz') if not p.is_symlink()],added_dates=60,bandwidths=2,accepted_prefix_read_only=True)


def base_fits():
    begin('base_fits');engine=configure();pvec=engine.spec['pf_ml']['p_vec'];lvec=np.asarray(engine.spec['pf_ml']['l_vec']);newval=[];fitmeta=[]
    for gi in [0,1]:
        dates=pd.date_range('1965-11-30',DATES[-1],freq='ME')
        # One bandwidth in memory; retain the original chronological summation.
        reals=[]
        for d in dates:
            with np.load(BASE/f'moments/g{gi}/{datekey(d)}.npz') as z:reals.append((z['r_tilde'].copy(),z['denom'].copy()))
        for year in range(2021,2026):
            n=sum(d<=pd.Timestamp(f'{year-1}-11-30') for d in dates)
            a=np.sum([v[1] for v in reals[:n]],axis=0)/n;r=np.sum([v[0] for v in reals[:n]],axis=0)/n
            for p in pvec:
                idx=feature_indices(p);coef,diag=ridge_path(a[np.ix_(idx,idx)],r[idx],lvec)
                np.savez_compressed(BASE/f'coefficients/g{gi}/{year}_p{p}.npz',coefficients=coef,lambdas=lvec,indices=idx,training_months=n,training_cutoff=np.array(f'{year-1}-11-30'))
                fitmeta.append(dict(g_index=gi,year=year,p=p,training_months=n,**diag))
                for d in pd.date_range(f'{year-1}-12-31',f'{year}-11-30',freq='ME'):
                    rr,aa=reals[dates.get_loc(d)];v=rr[idx]@coef-.5*np.sum(coef*(aa[np.ix_(idx,idx)]@coef),axis=0)
                    newval.append(pd.DataFrame(dict(decision_eom=d,return_eom=d+pd.offsets.MonthEnd(1),g_index=gi,g=engine.spec['pf_ml']['g_vec'][gi],p=p,lambda_index=np.arange(len(lvec)),**{'lambda':lvec},objective=v,fit_year=year)))
            print(f'Extended base grid: g{gi}, {year}',flush=True)
        del reals
    old=pd.read_parquet(OLD_BASE/'validation_scores.parquet');v=pd.concat([old.drop(columns='cumulative_objective'),*newval],ignore_index=True).sort_values(['g_index','p','lambda_index','return_eom'])
    v['cumulative_objective']=v.groupby(['g_index','p','lambda_index']).objective.transform(lambda s:s.expanding().mean())
    prefix=v[v.return_eom.le('2020-12-31')].reset_index(drop=True);pd.testing.assert_frame_equal(prefix,old.reset_index(drop=True),check_exact=True)
    choices=[];within=[]
    for year,g in v[v.return_eom.dt.month.eq(12)].groupby(v.return_eom.dt.year):
        best=g[g.cumulative_objective.eq(g.cumulative_objective.max())]
        if len(best)!=1:raise ValueError('Nonunique original base-grid selection')
        choices.append(best)
        for gi,gg in g.groupby('g_index'):
            b=gg[gg.cumulative_objective.eq(gg.cumulative_objective.max())]
            if len(b)!=1:raise ValueError('Nonunique source within-bandwidth choice')
            within.append(b)
    v.to_parquet(BASE/'validation_scores.parquet',index=False);pd.concat(choices).to_parquet(BASE/'annual_selection.parquet',index=False);pd.concat(within).to_parquet(BASE/'annual_selection_by_bandwidth.parquet',index=False)
    oldaim=pd.read_parquet(OLD_BASE/'baseline_aims.parquet');aims=pd.concat([oldaim,engine.aims(ADDED)],ignore_index=True);aims.to_parquet(BASE/'baseline_aims.parquet',index=False)
    oldpositions=pd.read_parquet(ROOT/'output/step6_baseline_1981_2020/PolicyDecision.parquet');oldmonthly=pd.read_parquet(ROOT/'output/step6_baseline_1981_2020/monthly_performance.parquet')
    last=oldpositions[oldpositions.eom.eq(DATES[479])].set_index('id');nextids=engine.months[ADDED[0]].query('valid').id.to_numpy()
    holdings=last.drifted_weight.reindex(nextids,fill_value=0).to_numpy();positions=[];monthly=[];exits=[]
    for i,d in enumerate(ADDED):
        ids,sigma,impact,wealth,m=engine.matrix(d);frame=engine.months[d].loc[ids];target=aims[aims.eom.eq(d)].set_index('id').reindex(ids).aim.to_numpy()
        following=engine.months[ADDED[i+1]].query('valid').id.to_numpy() if i+1<len(ADDED) else ids
        result=transition(ids,holdings,target,m,frame.tr_ld1.to_numpy(),frame.mu_ld1.iloc[0],impact,wealth,frame.ret_ld1.to_numpy(),following)
        positions.append(pd.DataFrame(dict(id=ids,eom=d,pretrade=holdings,aim=target,weight=result['weights'],trade=result['trades'],drifted_weight=result['drifted_weights'])))
        monthly.append(dict(decision_eom=d,return_eom=d+pd.offsets.MonthEnd(1),gross_excess=result['gross_excess'],impact_cost=result['impact_cost'],net_excess=result['gross_excess']-result['impact_cost'],turnover=result['turnover'],gross_exposure=result['gross_exposure']))
        for sid,value in zip(result['dropped_ids'],result['dropped_drifted_weights']):exits.append(dict(decision_eom=d,id=int(sid),omitted_next_weight=float(value),source_exit_trade_cost=0.))
        holdings=result['next_pretrade']
    RUN.mkdir();pd.concat([oldpositions,*positions],ignore_index=True).to_parquet(RUN/'PolicyDecision.parquet',index=False);pd.concat([oldmonthly,pd.DataFrame(monthly)],ignore_index=True).to_parquet(RUN/'monthly_performance.parquet',index=False);pd.DataFrame(exits).to_parquet(RUN/'extension_exit_ledger.parquet',index=False)
    pd.DataFrame(fitmeta).to_parquet(BASE/'extension_fit_diagnostics.parquet',index=False)
    finish('base_fits',[p for p in BASE.rglob('*') if p.is_file() and not p.is_symlink()]+list(RUN.glob('*.parquet')),added_coefficient_fits=len(fitmeta)*len(lvec),added_annual_cycles=5,complete_baseline_months=540,old_validation_prefix_exact=True)


def confidence():
    begin('confidence');CONF.mkdir(parents=True,exist_ok=True)
    for folder in ['coefficients/block12','coefficients/block24','aims/block12','aims/block24']:
        (CONF/folder).mkdir(parents=True,exist_ok=True)
        for p in (OLD_CONF/folder).iterdir():
            if p.is_file():link(p,CONF/folder/p.name)
    selection=pd.read_parquet(BASE/'annual_selection.parquet');positions=pd.read_parquet(RUN/'PolicyDecision.parquet');modelrows=[];seedrows=[];maximum=0.
    for gi in [0,1]:
        years=[]
        for year in range(2021,2026):
            row=selection[selection.return_eom.dt.year.eq(year-1)].iloc[0]
            if int(row.g_index)==gi:years.append((year,row))
        if not years:continue
        dates=pd.date_range('1965-11-30','2024-11-30',freq='ME');a=np.empty((len(dates),513,513));r=np.empty((len(dates),513))
        for j,d in enumerate(dates):
            with np.load(BASE/f'moments/g{gi}/{datekey(d)}.npz') as z:a[j]=z['denom'];r[j]=z['r_tilde']
        for year,row in years:
            p=int(row.p);li=int(row.lambda_index);penalty=float(row['lambda']);n=int(sum(dates<=pd.Timestamp(f'{year-1}-11-30')))
            with np.load(BASE/f'coefficients/g{gi}/{year}_p{p}.npz') as z:idx=z['indices'].copy();central=z['coefficients'][:,li].copy()
            matrices=a[:n][:,idx][:,:,idx];scores=r[:n,idx]
            center,res=fit_coefficients(matrices.mean(axis=0),scores.mean(axis=0),penalty)
            if not np.allclose(center,central,atol=1e-12,rtol=1e-9):raise ValueError('Original central coefficient mismatch')
            for length in [12,24]:
                coefs=[];startslist=[];samples=[];countslist=[];residuals=[]
                for replica in range(1,101):
                    number,digest=replica_seed(PROTOCOL,length,f'{year-1}-12-31',replica);starts,sampled,counts=moving_blocks(n,length,number)
                    ma=(counts@matrices.reshape(n,-1)/n).reshape(len(idx),len(idx));mr=counts@scores/n
                    coef,res=fit_coefficients(ma,mr,penalty);maximum=max(maximum,res);coefs.append(coef);residuals.append(res);startslist.append(starts);samples.append(sampled);countslist.append(counts)
                    seedrows.append(dict(year=year,block_months=length,replica=replica,seed_sha256=digest))
                path=CONF/f'coefficients/block{length}/{year}.npz'
                np.savez_compressed(path,coefficients=np.column_stack(coefs),indices=idx,starts=np.array(startslist),sampled_indices=np.array(samples),counts=np.array(countslist),training_dates=dates[:n].strftime('%Y-%m-%d').to_numpy(dtype=str),residuals=np.array(residuals))
                meta=dict(fit_year=year,g_index=gi,p=p,lambda_index=li,penalty=penalty,training_months=n,training_cutoff=f'{year-1}-11-30',latest_training_return=f'{year-1}-12-31',first_decision=f'{year-1}-12-31',block_months=length,replicas=100,array_sha256=sha(path))
                save_json(path.with_suffix('.json'),meta);modelrows.append(meta)
                for d in pd.date_range(f'{year-1}-12-31',f'{year}-11-30',freq='ME'):
                    with np.load(BASE/f'moments/g{gi}/{datekey(d)}.npz') as z:ids=z['ids'].copy();signals=z['signal_t'][:,idx].copy()
                    with np.load(BASE/f'matrices/{datekey(d)}.npz') as z:m=z['m'].copy()
                    with np.load(BASE/f'risk/{datekey(d)}.npz') as z:sigma=z['loadings']@z['factor_cov']@z['loadings'].T+np.diag(z['specific_variance'])
                    h=positions[positions.eom.eq(d)].set_index('id').loc[ids,'pretrade'].to_numpy()
                    frame=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=['id','eom','valid','lambda_diagonal','wealth_usd'],filters=[('eom','==',d)]).query('valid').set_index('id').loc[ids]
                    replica_aims=signals@np.column_stack(coefs);aim=signals@central
                    feature,q0,qbar=state_features(aim,replica_aims,h,m,sigma,frame.lambda_diagonal.to_numpy(),float(frame.wealth_usd.iloc[0]))
                    target=CONF/f'aims/block{length}/{datekey(d)}.npz';np.savez_compressed(target,ids=ids,replica_aims=replica_aims,central_aim=aim,mean_aim=replica_aims.mean(axis=1),baseline_mean_trade=qbar)
                    save_json(target.with_suffix('.json'),dict(features=feature,annual_model=meta,array_sha256=sha(target),current_signal_sha256=sha(BASE/f'moments/g{gi}/{datekey(d)}.npz')))
            print(f'Extended conditional replicas: {year}',flush=True)
        del a,r
    pd.DataFrame(modelrows).to_parquet(CONF/'extension_annual_models.parquet',index=False);pd.DataFrame(seedrows).to_parquet(CONF/'extension_replica_seeds.parquet',index=False)
    finish('confidence',[p for p in CONF.rglob('*') if p.is_file() and not p.is_symlink()],added_annual_variants=10,added_replica_fits=1000,added_aim_records=120,maximum_relative_solve_residual=maximum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['moments','base_fits','confidence']);args=p.parse_args();globals()[args.stage]()
