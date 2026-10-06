"""Independent extension checks and the prerequisite gate for comparison paths."""
import sys,json
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from comparisons.io import *
from comparisons.extended_learner import Sources,STATES,training_rows
from learner.core import A0,A1,VARIANTS,weights,split,seed,blocks,predict,fit_ridge,select_model
from learner.pipeline import risk

def main():
    begin('extension_review');checks=[];artifacts={};maximum={}
    def check(name,condition,**details):
        checks.append(dict(check=name,passed=bool(condition),**details))
        if not condition:save_json(OUT/'extension_failed_checks.json',checks);raise AssertionError(name)
    def close(name,actual,expected,atol=1e-12,rtol=1e-9):
        a,b=np.asarray(actual),np.asarray(expected);error=float(np.max(np.abs(a-b)))
        maximum[name]=error;check(name,a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),max_abs_error=error)
    for stage in ['moments','base_fits','confidence','states','labels','fits','extension_export']:
        path=OUT/f'{stage}_manifest.json';manifest=json.loads(path.read_text())
        for p,h in manifest['artifact_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed stage file '+p)
        for p,h in json.loads((OUT/f'{stage}_code.json').read_text()).items():
            if sha(ROOT/p)!=h:raise ValueError('Changed stage code '+p)
        artifacts.update(manifest['artifact_sha256']);artifacts[str(path.relative_to(ROOT))]=sha(path)
        check(stage+' source/code/output integrity',True)
    from step9_synthetic import run as synthetic_run
    synthetic=synthetic_run();save_json(OUT/'synthetic_review.json',synthetic);check('Comparison-specific synthetic boundaries',synthetic['status']=='PASS')
    ref=OUT/'extension_reference';reference=json.loads((ref/'summary.json').read_text());speed=json.loads((ref/'speed/summary.json').read_text())
    check('Original R extension and speed references completed',reference['status']=='PASS' and speed['status']=='PASS')
    for gi in [0,1]:
        for year in range(2021,2026):
            for p in [64,128,256,512]:
                with np.load(BASE/f'coefficients/g{gi}/{year}_p{p}.npz') as z:expected=z['coefficients'].copy()
                actual=np.fromfile(ref/f'fit/g{gi}/r_coef_{year}_p{p}.bin',dtype='<f8').reshape(expected.shape)
                close(f'R coefficient grid {gi}/{year}/{p}',actual,expected)
        for d in ADDED:
            if d.month not in [12,6] and d!=ADDED[-1]:continue
            with np.load(BASE/f'moments/g{gi}/{d:%Y-%m-%d}.npz') as z:
                for rkey,key in [('signal','signal_t'),('denom','denom'),('rtilde','r_tilde')]:
                    actual=np.fromfile(ref/f'sample_inputs/r_g{gi}_{d:%Y-%m-%d}_{rkey}.bin',dtype='<f8').reshape(z[key].shape)
                    # Same conditioned source-input tolerance as accepted Step 6.
                    close(f'Original R moment {gi}/{d:%Y-%m-%d}/{key}',actual,z[key],atol=1e-8,rtol=1e-6)
    rv=pd.read_csv(ref/'r_new_validation.csv');py=pd.read_parquet(BASE/'validation_scores.parquet');py=py[py.return_eom.ge('2021-01-31')]
    rv['return_eom']=pd.to_datetime(rv.eom_ret);rv=rv.rename(columns={'obj':'r_objective'})
    joined=py.merge(rv[['return_eom','g_index','p','lambda_index','r_objective']],on=['return_eom','g_index','p','lambda_index'],validate='one_to_one')
    check('All additional grid validation observations',len(joined)==48480);close('All R validation objectives',joined.r_objective,joined.objective,atol=1e-10,rtol=1e-8)
    rs=pd.read_csv(ref/'r_annual_selection.csv');ps=pd.read_parquet(BASE/'annual_selection.parquet');rs['year']=pd.to_datetime(rs.eom_ret).dt.year;ps['year']=ps.return_eom.dt.year
    check('All 55 annual original-source selections',rs.sort_values('year')[['g_index','p','lambda_index']].reset_index(drop=True).equals(ps.sort_values('year')[['g_index','p','lambda_index']].astype(int).reset_index(drop=True)))
    from confidence.core import replica_seed,moving_blocks
    conditional_records=pd.read_parquet(CONF/'extension_annual_models.parquet');conditional_seeds=pd.read_parquet(CONF/'extension_replica_seeds.parquet').set_index(['year','block_months','replica'])
    for row in conditional_records.itertuples():
        path=CONF/f'coefficients/block{row.block_months}/{row.fit_year}.npz';meta=json.loads(path.with_suffix('.json').read_text());selected=ps[ps.year.eq(row.fit_year-1)].iloc[0]
        check(f'Conditional fit maturity and frozen source grid {row.fit_year}/{row.block_months}',row.training_cutoff==f'{row.fit_year-1}-11-30' and int(selected.g_index)==row.g_index and int(selected.p)==row.p and int(selected.lambda_index)==row.lambda_index and sha(path)==meta['array_sha256'])
        with np.load(path) as z:
            check(f'Conditional complete finite coefficient set {row.fit_year}/{row.block_months}',z['coefficients'].shape==(row.p+1,100) and np.isfinite(z['coefficients']).all() and z['residuals'].max()<1e-10)
            for replica in range(1,101):
                number,digest=replica_seed(PROTOCOL,row.block_months,f'{row.fit_year-1}-12-31',replica);starts,sampled,counts=moving_blocks(row.training_months,row.block_months,number)
                assert conditional_seeds.loc[(row.fit_year,row.block_months,replica),'seed_sha256']==digest and np.array_equal(z['starts'][replica-1],starts) and np.array_equal(z['sampled_indices'][replica-1],sampled) and np.array_equal(z['counts'][replica-1],counts)
            check(f'All 100 conditional seeds blocks and counts {row.fit_year}/{row.block_months}',True)
            for d in pd.date_range(f'{row.fit_year-1}-12-31',f'{row.fit_year}-11-30',freq='ME'):
                with np.load(BASE/f'moments/g{row.g_index}/{d:%Y-%m-%d}.npz') as m:signals=m['signal_t'][:,z['indices']].copy()
                with np.load(CONF/f'aims/block{row.block_months}/{d:%Y-%m-%d}.npz') as a:
                    close(f'Every conditional aim {d:%Y-%m-%d}/{row.block_months}',a['replica_aims'],signals@z['coefficients'])
                    close(f'Ensemble arithmetic mean {d:%Y-%m-%d}/{row.block_months}',a['mean_aim'],a['replica_aims'].mean(axis=1))
    f=pd.read_parquet(LEARNER/'state_features.parquet');oldf=pd.read_parquet(OLD_LEARNER/'state_features.parquet')
    pd.testing.assert_frame_equal(f[f.decision_eom.le('2020-11-30')].reset_index(drop=True),oldf,check_exact=True);check('All accepted feature rows remain exactly unchanged',True)
    labels=pd.read_parquet(LEARNER/'DeviationOutcome.parquet');old=pd.read_parquet(OLD_LEARNER/'DeviationOutcome.parquet');keys=['decision_eom','state_id','horizon']
    overlap=labels.merge(old[keys],on=keys,validate='one_to_one').sort_values(keys).reset_index(drop=True);old=old.sort_values(keys).reset_index(drop=True)
    cols=[c for c in old if c!='endpoint_artifact'];pd.testing.assert_frame_equal(overlap[cols],old[cols],check_exact=True);check('All 2742 accepted label rows remain exactly unchanged',True)
    check('Full 2025 labels and immature boundary',labels.groupby('horizon').size().to_dict()=={12:1587,36:1515} and len(pd.read_parquet(LEARNER/'immature_labels.parquet'))==138)
    check('Label maturity and curvature',labels.label_available_at.equals(labels.outcome_end) and labels.c.ge(1-1e-12).all() and all(r.label_available_at==r.decision_eom+pd.offsets.MonthEnd(r.horizon) for r in labels.itertuples()))
    for d in DATES[:480]:
        with np.load(LEARNER/f'states/{d:%Y-%m-%d}.npz') as a,np.load(OLD_LEARNER/f'states/{d:%Y-%m-%d}.npz') as b:
            check(f'Accepted state arrays exact {d:%Y-%m-%d}',all(np.array_equal(a[k],b[k]) for k in b.files))
    replicas=0
    for variant in VARIANTS:
        frame=training_rows(variant)
        for year in range(1981,2026):
            path=LEARNER/f'fits/{variant}/{year}.json';fit=json.loads(path.read_text())
            if year<=2020:
                check(f'Accepted annual learner exact {variant}/{year}',fit==json.loads((OLD_LEARNER/f'fits/{variant}/{year}.json').read_text()));continue
            eligible,tr,va,info=split(frame,pd.Timestamp(f'{year-1}-12-31'))
            check(f'Annual maturity and purge {variant}/{year}',all(fit[k]==v for k,v in info.items()) and eligible.label_available_at.max()<=pd.Timestamp(f'{year-1}-12-31') and tr.outcome_end.max()<va.decision_eom.min())
            with np.load(LEARNER/f'fits/{variant}/{year}_training.npz') as z:data={k:z[k].copy() for k in z.files}
            check(f'Actual matured training inputs {variant}/{year}',np.array_equal(data['x1'],eligible[A1]) and np.array_equal(data['y'],eligible[['b','c']]))
            for method,cols in [('A0',A0),('A1',A1)]:check(f'All ridge candidates and refit {variant}/{year}/{method}',select_model(tr,va,eligible,cols,PROTOCOL['learning']['ridge_penalties'])==fit['models'][method])
            dates=pd.DatetimeIndex(data['dates']);di=dates.get_indexer(eligible.decision_eom);w=weights(eligible)
            check(f'Exactly 100 uncertainty fits {variant}/{year}',len(fit['bootstrap'])==100 and fit['uncertainty_status']=='FITTED')
            for replica in range(1,101):
                number,digest=seed(PROTOCOL,variant,f'{year-1}-12-31',replica);starts,sampled,counts=blocks(dates,VARIANTS[variant][2],number)
                assert np.array_equal(starts,data['bootstrap_starts'][replica-1]) and np.array_equal(sampled,data['bootstrap_sampled'][replica-1]) and np.array_equal(counts,data['bootstrap_counts'][replica-1])
                rep=fit_ridge(eligible[A1],eligible.b,w*counts[di],fit['models']['A1']['penalty']);assert rep==fit['bootstrap'][replica-1];replicas+=1
            check(f'Every new uncertainty replica and scaling replay {variant}/{year}',True)
    source=Sources()
    @lru_cache(maxsize=40)
    def month(i):return source.month(DATES[i],confidence=True)
    # Fixed starts cover labels crossing the old endpoint and added observations.
    records=[];dexits=pd.read_parquet(LEARNER/'deviation_exit_ledger.parquet');caps=pd.read_parquet(LEARNER/'matrix_feasibility.parquet').set_index('decision_eom')
    for start in [456,468,479,480,492,504,528]:
        with np.load(LEARNER/f'states/{DATES[start]:%Y-%m-%d}.npz') as z:h0=z['pretrade'].copy()
        positive=.8*min(.2,caps.loc[DATES[start],'kappa_max']-1);deltas=np.array([0.,-.3,positive]);h=np.tile(h0,3);total=np.zeros((3,3))
        for age in range(1,min(36,len(DATES)-start)+1):
            x=month(start+age-1);n=len(x['ids'])
            q=(np.eye(n)-x['m'])@(x['aim'][:,None]-h)
            w=h+q if age>1 else h+q*np.repeat(1+deltas,3)
            trade=w-h;u=x['returns']@w-5*np.sum(w*risk(x,w),axis=0)-x['wealth']/2*np.sum(x['impact'][:,None]*trade**2,axis=0);total+=u.reshape(3,3)
            next_h=np.zeros((len(x['next_ids']),9));mapping={int(v):j for j,v in enumerate(x['next_ids'])};drop=[]
            for j,sid in enumerate(x['ids']):
                if int(sid) in mapping:next_h[mapping[int(sid)]]=w[j]*x['growth'][j]
                else:drop.append(j)
            if age in [12,36]:
                saved=labels[labels.decision_eom.eq(DATES[start])&labels.horizon.eq(age)].set_index('state_id').loc[STATES]
                for j,delta in enumerate(deltas[1:],1):
                    expected=delta*saved.B.to_numpy()-.5*delta**2*saved.C.to_numpy();actual=total[j]-total[0]
                    close(f'Explicit extension rollout {start}/H{age}/{delta}',actual,expected,atol=1e-9,rtol=1e-8)
                    for s in range(3):records.append(dict(start=str(DATES[start].date()),horizon=age,state=STATES[s],delta=delta,explicit=float(actual[s]),quadratic=float(expected[s]),error=float(abs(actual[s]-expected[s]))))
                with np.load(ROOT/saved.endpoint_artifact.iloc[0]) as z:
                    close(f'Explicit terminal derivative {start}/H{age}',(w[:,6:9]-w[:,:3])/positive,z['weight_derivative'],atol=1e-10,rtol=1e-8)
                    close(f'Explicit mapped terminal derivative {start}/H{age}',(next_h[:,6:9]-next_h[:,:3])/positive,z['next_derivative'],atol=1e-10,rtol=1e-8)
            ledger=dexits[dexits.start_decision.eq(DATES[start])&dexits.continuation_decision.eq(DATES[start+age-1])];ok=len(ledger)==3*len(drop)
            for row in ledger.itertuples():
                j=np.flatnonzero(x['ids']==row.id)[0];s=STATES.index(row.state_id)
                ok=ok and np.isclose(row.omitted_reference_weight,w[j,s]*x['growth'][j],atol=1e-10) and np.isclose(row.omitted_derivative,(w[j,6+s]-w[j,s])*x['growth'][j]/positive,atol=1e-10)
            check(f'Explicit omitted derivative ledger {start}/{age}',ok);h=next_h
    pd.DataFrame(records).to_csv(OUT/'extension_rollout_comparison.csv',index=False)
    # All added state transitions and source baseline aims are checked directly.
    previous=None
    for d in ADDED:
        x=source.month(d,confidence=True)
        with np.load(LEARNER/f'states/{d:%Y-%m-%d}.npz') as z:state={k:z[k].copy() for k in z.files}
        if previous is not None:close(f'Added state recurrence {d:%Y-%m-%d}',state['pretrade'],previous,atol=1e-12)
        q=(np.eye(len(x['ids']))-x['m'])@(x['aim'][:,None]-state['pretrade']);close(f'Added state weights {d:%Y-%m-%d}',state['weight'],state['pretrade']+q*state['kappa'])
        base=source.baseline[d];close(f'Added R1 state {d:%Y-%m-%d}',state['pretrade'][:,1],base.pretrade)
        from learner.core import remap
        previous,_=remap(state['weight']*x['growth'][:,None],x['ids'],x['next_ids'])
        selection=ps[ps.year.eq(d.year if d.month==12 else d.year-1)].iloc[0];year=d.year+(d.month==12)
        with np.load(BASE/f'moments/g{int(selection.g_index)}/{d:%Y-%m-%d}.npz') as z:signals=z['signal_t'].copy()
        with np.load(BASE/f'coefficients/g{int(selection.g_index)}/{year}_p{int(selection.p)}.npz') as z:expected=signals[:,z['indices']]@z['coefficients'][:,int(selection.lambda_index)]
        close(f'Current original aim {d:%Y-%m-%d}',x['aim'],expected)
    check('Final unavailable next universe is not invented',np.array_equal(source.month(DATES[-1])['next_ids'],source.month(DATES[-1])['ids']))
    for p in [*ref.rglob('*'),OUT/'extension_rollout_comparison.csv']:
        if p.is_file():artifacts[str(p.relative_to(ROOT))]=sha(p)
    result=dict(status='PASS',checks_total=len(checks),checks=checks,reference=reference,speed_reference=speed,
        new_uncertainty_refits_replayed=replicas,explicit_rollout_comparisons=len(records),maximum_rollout_error=max(v['error'] for v in records),
        accepted_prefix_exact=True,artifact_sha256=artifacts,main_period_unchanged=['1981-01','2020-12'],additional_period=['2021-01','2025-12'])
    save_json(OUT/'extension_review.json',result);finish('extension_review',[OUT/'extension_review.json',OUT/'extension_rollout_comparison.csv'],checks=len(checks))
    print(f'PASS: {len(checks)} extension checks, {replicas} new uncertainty refits replayed.',flush=True)
if __name__=='__main__':main()
