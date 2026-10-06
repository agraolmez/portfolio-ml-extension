"""Independent chronological, accounting, rollout and regression review."""
import json,sys
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import ROOT,PROTOCOL,sha,save_json,datekey
from confidence.core import state_features
from learner.core import A0,A1,VARIANTS,weights,split,seed,blocks,predict,decide,fit_ridge,select_model
from learner.pipeline import OUT,PREP,CONF,DATES,STATES,Sources,preflight,training_rows
from learner.policy import DecisionInputs,act
from step8_synthetic import run as synthetic_run


def main():
    if (OUT/'acceptance_manifest.json').exists():raise ValueError('Accepted review is immutable')
    preflight();checks=[];rollout_records=[];max_feature=0.;max_prediction=0.;max_replica=0.
    def check(name,condition,**details):
        item=dict(check=name,passed=bool(condition),**details);checks.append(item)
        if not condition:
            save_json(OUT/'failed_review_checks.json',checks);raise AssertionError(name)
    for stage in ['states','labels','fit','predictions']:
        manifest=json.loads((OUT/f'{stage}_manifest.json').read_text())
        for rel,digest in manifest['artifact_sha256'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Changed stage artifact '+rel)
        for rel,digest in json.loads((OUT/f'{stage}_code.json').read_text()).items():
            if sha(ROOT/rel)!=digest:raise ValueError('Changed producing code '+rel)
        check(f'{stage} artifact and producing code integrity',True,artifacts=len(manifest['artifact_sha256']))
    synthetic=synthetic_run();check('All synthetic boundary and optimizer checks pass',synthetic['status']=='PASS',count=synthetic['checks_total'])
    source=Sources()
    @lru_cache(maxsize=48)
    def month(i):return source.month(DATES[i],confidence=True)
    features=pd.read_parquet(PREP/'state_features.parquet')
    labels=pd.read_parquet(PREP/'DeviationOutcome.parquet')
    immature=pd.read_parquet(PREP/'immature_labels.parquet')
    predictions=pd.read_parquet(PREP/'state_prediction_diagnostics.parquet')
    caps=pd.read_parquet(PREP/'matrix_feasibility.parquet').set_index('decision_eom')
    exits=pd.read_parquet(PREP/'state_exit_ledger.parquet')
    dexits=pd.read_parquet(PREP/'deviation_exit_ledger.parquet')
    check('Complete unique state-date-feature records',len(features)==2880 and not features.duplicated(['decision_eom','state_id','aim_block']).any())
    check('Complete fully matured horizon coverage',labels.groupby('horizon').size().to_dict()=={12:1407,36:1335} and len(immature)==138)
    check('No clipped normalized labels or negative curvature',np.isfinite(labels[['B','C','S','b','c']]).all().all() and labels.c.ge(1-1e-12).all() and np.allclose(labels.b*labels.S,labels.B) and np.allclose(labels.c*labels.S,labels.C))
    check('Availability dates equal complete horizon outcome end',labels.label_available_at.equals(labels.outcome_end) and all(row.outcome_end==row.decision_eom+pd.offsets.MonthEnd(row.horizon) for row in labels.itertuples()))
    check('Exactly the required end-of-history immature starts',all(row.required_outcome_end==row.decision_eom+pd.offsets.MonthEnd(row.horizon) and row.required_outcome_end>pd.Timestamp('2020-12-31') for row in immature.itertuples()))
    check('Omissions carry no invented liquidation trade',not exits.liquidation_trade_added.any() and not dexits.liquidation_trade_added.any())
    direct_sample=set([0,11,12,60,72,119,120,179,239,299,359,419,444,468,479])
    expected_exit_rows=0;max_state_difference=0.
    for i,d in enumerate(DATES):
        x=month(i)
        with np.load(PREP/f'states/{datekey(d)}.npz') as z:state={k:z[k].copy() for k in z.files}
        h=state['pretrade'];sigma=x['loadings']@x['fc']@x['loadings'].T+np.diag(x['specific'])
        adjustment=np.eye(len(h))-x['m'];q=adjustment@(x['aim'][:,None]-h)
        expected=h+q*state['kappa'];check(f'Implemented state weights and costs input {datekey(d)}',np.allclose(expected,state['weight'],atol=1e-12) and np.allclose(q*state['kappa'],state['trade'],atol=1e-12) and np.all(state['kappa']<=caps.loc[d,'kappa_max']))
        old_ids=state['ids'];nextids=x['next_ids'];where={int(v):j for j,v in enumerate(nextids)};next_h=np.zeros((len(nextids),3));drop=[]
        for j,id_value in enumerate(old_ids):
            if int(id_value) in where:next_h[where[int(id_value)]]=state['weight'][j]*x['growth'][j]
            else:drop.append(j)
        ledger=exits[exits.decision_eom.eq(d)]
        expected_exit_rows+=3*len(drop)
        check(f'All omitted state positions explicitly ledgered {datekey(d)}',len(ledger)==3*len(drop) and all(np.isclose(r.omitted_drifted_weight,state['weight'][np.where(old_ids==r.id)[0][0],STATES.index(r.state_id)]*x['growth'][np.where(old_ids==r.id)[0][0]],atol=1e-12) for r in ledger.itertuples()))
        if i<479:
            with np.load(PREP/f'states/{datekey(DATES[i+1])}.npz') as z:err=float(np.max(abs(next_h-z['pretrade'])))
            max_state_difference=max(max_state_difference,err);check(f'Own-holdings recurrence and zero entries {datekey(d)}',err<1e-12)
        base=source.baseline[d];check(f'Continuous speed-one path reproduces R1 {datekey(d)}',np.allclose(state['pretrade'][:,1],base.pretrade,atol=1e-8,rtol=1e-6) and np.allclose(state['weight'][:,1],base.weight,atol=1e-8,rtol=1e-6))
        # All initial curvatures are checked using an independently assembled dense covariance.
        curvature=10*np.sum(q*(sigma@q),axis=0)+x['wealth']*np.sum(x['impact'][:,None]*q*q,axis=0)
        actual=features[features.decision_eom.eq(d)&features.aim_block.eq(12)].set_index('state_id').loc[STATES]
        check(f'Independent dense-risk initial curvature {datekey(d)}',np.allclose(curvature,actual.initial_curvature,atol=1e-12,rtol=1e-10))
        if i in direct_sample:
            for block in [12,24]:
                with np.load(CONF/f'aims/block{block}/{datekey(d)}.npz') as z:replicas=z['replica_aims'].copy()
                for s,identity in enumerate(STATES):
                    value,_,_=state_features(x['aim'],replicas,h[:,s],x['m'],sigma,x['impact'],x['wealth'])
                    saved=features[features.decision_eom.eq(d)&features.aim_block.eq(block)&features.state_id.eq(identity)].iloc[0]
                    err=max(abs(value[k]-saved[k]) for k in A1+['initial_curvature','dispersion','dispersion_numerator','mean_trade_risk'])
                    max_feature=max(max_feature,err)
                    check(f'Direct 100-replica own-state features {datekey(d)} / {block} / {identity}',err<1e-9)
            # Use the public current-input interface on an actual inherited state.
            fit=json.loads((PREP/f'fits/primary/{d.year+1 if d.month==12 else d.year}.json').read_text())
            inputs=DecisionInputs(str(d.date()),x['ids'],x['aim'],x['m'],x['loadings'],x['fc'],x['specific'],x['impact'],x['wealth'],x['confidence'][12]['mean'],x['confidence'][12]['numerator'])
            result=act(fit,'A2',inputs,h[:,0]);saved=predictions[predictions.variant.eq('primary')&predictions.method.eq('A2')&predictions.decision_eom.eq(d)&predictions.state_id.eq(STATES[0])].iloc[0]
            check(f'Public own-holdings decision interface {datekey(d)}',abs(result['decision']['kappa']-saved.kappa)<1e-12 and np.allclose(result['weight'],h[:,0]+saved.kappa*q[:,0],atol=1e-12))
        if i%120==0:print(f'Review state month {i+1}/480',flush=True)
    check('All source state exit rows accounted for',expected_exit_rows==len(exits),rows=len(exits))
    # Independent explicit utility rollouts. No derivative recursion is used here.
    for start in sorted(direct_sample):
        maximum=min(36,480-start);x0=month(start)
        with np.load(PREP/f'states/{datekey(DATES[start])}.npz') as z:h0=z['pretrade'].copy()
        positive=min(.2,float(caps.loc[DATES[start],'kappa_max'])-1)*.8
        deltas=np.array([0.,-.3,positive]);h=np.tile(h0,3);total=np.zeros((3,3))
        for age in range(1,maximum+1):
            x=month(start+age-1);sigma=x['loadings']@x['fc']@x['loadings'].T+np.diag(x['specific']);n=len(x['ids'])
            if age==1:
                q0=(np.eye(n)-x['m'])@(x['aim'][:,None]-h0)
                w=h+np.column_stack([(1+delta)*q0 for delta in deltas])
            else:w=x['m']@h+(np.eye(n)-x['m'])@np.repeat(x['aim'][:,None],9,axis=1)
            trade=w-h
            u=x['returns']@w-5*np.sum(w*(sigma@w),axis=0)-x['wealth']/2*np.sum(x['impact'][:,None]*trade*trade,axis=0)
            total+=u.reshape(3,3)
            next_h=np.zeros((len(x['next_ids']),9));mapping={int(v):j for j,v in enumerate(x['next_ids'])};drop=[]
            for j,id_value in enumerate(x['ids']):
                if int(id_value) in mapping:next_h[mapping[int(id_value)]]=w[j]*x['growth'][j]
                else:drop.append(j)
            if age in [1,12,36]:
                if age==1:
                    ref=h0+q0;bb=x['returns']@q0-10*np.sum(q0*(sigma@ref),axis=0)-x['wealth']*np.sum(x['impact'][:,None]*q0*q0,axis=0)
                    cc=10*np.sum(q0*(sigma@q0),axis=0)+x['wealth']*np.sum(x['impact'][:,None]*q0*q0,axis=0)
                else:
                    saved=labels[labels.decision_eom.eq(DATES[start])&labels.horizon.eq(age)].set_index('state_id').loc[STATES];bb=saved.B.to_numpy();cc=saved.C.to_numpy()
                for j,delta in enumerate(deltas[1:],1):
                    actual=total[j]-total[0];expected=delta*bb-.5*delta**2*cc;err=float(np.max(abs(actual-expected)))
                    check(f'Explicit finite-horizon utility difference {start} / H{age} / delta {delta}',err<1e-9)
                    for s,identity in enumerate(STATES):rollout_records.append(dict(decision_eom=DATES[start],horizon=age,state_id=identity,delta=float(delta),explicit_gain=float(actual[s]),quadratic_gain=float(expected[s]),absolute_error=float(abs(actual[s]-expected[s]))))
                if age>1:
                    with np.load(ROOT/saved.endpoint_artifact.iloc[0]) as z:
                        derivative=(w[:,6:9]-w[:,:3])/positive;next_derivative=(next_h[:,6:9]-next_h[:,:3])/positive
                        check(f'Explicit endpoint derivative {start} / H{age}',np.allclose(derivative,z['weight_derivative'],atol=1e-10,rtol=1e-8) and np.allclose(next_derivative,z['next_derivative'],atol=1e-10,rtol=1e-8))
            # Every omission in these independent rollouts is matched to the derivative ledger.
            led=dexits[dexits.start_decision.eq(DATES[start])&dexits.continuation_decision.eq(DATES[start+age-1])]
            ok=len(led)==3*len(drop)
            for r in led.itertuples():
                j=np.where(x['ids']==r.id)[0][0];s=STATES.index(r.state_id)
                ok=ok and np.isclose(r.omitted_reference_weight,w[j,s]*x['growth'][j],atol=1e-10) and np.isclose(r.omitted_derivative,(w[j,6+s]-w[j,s])*x['growth'][j]/positive,atol=1e-10)
            check(f'Explicit rollout omitted derivative ledger {start} / age {age}',ok)
            h=next_h
    pd.DataFrame(rollout_records).to_csv(OUT/'explicit_rollout_comparison.csv',index=False)
    # Every annual fit, candidate and bootstrap replay is reviewed.
    annual=pd.read_parquet(PREP/'annual_fit_records.parquet');candidate_rows=[];bootstrap_n=0
    for variant in VARIANTS:
        frame=training_rows(variant)
        for year in range(1981,2021):
            saved=json.loads((PREP/f'fits/{variant}/{year}.json').read_text());cutoff=pd.Timestamp(f'{year-1}-12-31')
            eligible,tr,va,info=split(frame,cutoff)
            check(f'Annual maturity/purge/fallback record {variant} / {year}',all(saved[k]==v for k,v in info.items()))
            if info['status']!='FITTED':continue
            check(f'Strict annual chronology {variant} / {year}',eligible.label_available_at.max()<=cutoff and tr.outcome_end.max()<va.decision_eom.min() and tr.label_available_at.max()<va.decision_eom.min() and tr.decision_eom.nunique()>=120 and va.decision_eom.nunique()==36)
            w=weights(eligible);tot=pd.Series(w,index=eligible.index).groupby(eligible.decision_eom).sum()
            check(f'Date total training weight {variant} / {year}',np.allclose(tot,1))
            with np.load(PREP/f'fits/{variant}/{year}_training.npz') as z:data={k:z[k].copy() for k in z.files}
            check(f'Saved training inputs equal matured source labels {variant} / {year}',np.array_equal(data['x1'],eligible[A1].to_numpy()) and np.array_equal(data['y'],eligible[['b','c']].to_numpy()) and np.array_equal(data['decision_dates'],eligible.decision_eom.dt.strftime('%Y-%m-%d').to_numpy(dtype=str)))
            for method,cols in [('A0',A0),('A1',A1)]:
                repeated=select_model(tr,va,eligible,cols,PROTOCOL['learning']['ridge_penalties']);model=saved['models'][method]
                check(f'All penalty candidates and selection replay {variant} / {year} / {method}',repeated==model)
                for c in model['candidates']:candidate_rows.append(dict(variant=variant,year=year,method=method,selected=c['penalty']==model['penalty'],**c))
                # Independent augmented least squares includes an intercept column.
                x=eligible[cols].to_numpy();wn=w/w.sum();mean=wn@x;sd=np.sqrt(wn@((x-mean)**2));zero=sd<=1e-12;sd[zero]=1;z=(x-mean)/sd;z[:,zero]=0
                design=np.c_[np.ones(len(z)),z];penalty=np.c_[np.zeros(len(cols)),np.eye(len(cols))*np.sqrt(model['penalty'])]
                coef=np.linalg.lstsq(np.r_[design*np.sqrt(wn)[:,None],penalty],np.r_[eligible[['b','c']].to_numpy()*np.sqrt(wn)[:,None],np.zeros((len(cols),2))],rcond=None)[0]
                actual=design@coef;expected=predict(model,x);err=float(np.max(abs(actual-expected))/max(1,np.max(abs(expected))))
                check(f'Independent augmented least-squares full ridge {variant} / {year} / {method}',err<1e-9)
            dates=pd.DatetimeIndex(data['dates']);lookup=dates.get_indexer(eligible.decision_eom)
            check(f'All 100 uncertainty replicas required {variant} / {year}',saved['uncertainty_status']=='FITTED' and len(saved['bootstrap'])==100)
            for replica in range(1,101):
                number,digest=seed(PROTOCOL,variant,str(cutoff.date()),replica);starts,sampled,counts=blocks(dates,VARIANTS[variant][2],number)
                if not (np.array_equal(starts,data['bootstrap_starts'][replica-1]) and np.array_equal(sampled,data['bootstrap_sampled'][replica-1]) and np.array_equal(counts,data['bootstrap_counts'][replica-1])):raise AssertionError('Bootstrap replay')
                rep=fit_ridge(eligible[A1].to_numpy(),eligible.b.to_numpy(),w*counts[lookup],saved['models']['A1']['penalty'])
                err=float(np.max(abs(predict(rep,eligible[A1].to_numpy())-predict(saved['bootstrap'][replica-1],eligible[A1].to_numpy()))));max_replica=max(max_replica,err)
                if err>1e-10:raise AssertionError('Bootstrap prediction replay')
                bootstrap_n+=1
            check(f'All replica blocks, counts and refitted scaling replay {variant} / {year}',True,replicas=100)
        print(f'Reviewed annual models {variant}',flush=True)
    pd.DataFrame(candidate_rows).to_parquet(OUT/'penalty_validation_review.parquet',index=False)
    # All saved monthly decisions, range flags and failure reasons must replay exactly.
    feature_lookup=features.set_index(['decision_eom','state_id','aim_block'])
    for (variant,year),group in predictions.groupby(['variant','fit_year']):
        fitted=json.loads((PREP/f'fits/{variant}/{year}.json').read_text())
        for row in group.itertuples():
            feature=feature_lookup.loc[(row.decision_eom,row.state_id,VARIANTS[variant][1])]
            action=decide(fitted,row.method,feature,row.kappa_min,row.kappa_max);max_prediction=max(max_prediction,abs(action['kappa']-row.kappa))
            if abs(action['kappa']-row.kappa)>1e-12 or action['out_of_range']!=row.out_of_range or action['out_of_range_features']!=list(row.out_of_range_features) or action['fallback_reason']!=row.fallback_reason:raise AssertionError('Decision replay')
        check(f'All monthly own-state diagnostic decisions replay {variant} / {year}',True,rows=len(group))
    check('All monthly predictions are feasible',predictions.kappa.ge(predictions.kappa_min).all() and predictions.kappa.le(predictions.kappa_max).all() and len(predictions)==12960)
    check('A2 weakly shrinks deviations toward baseline on identical states',all(np.all(abs(g.pivot(index=['decision_eom','state_id'],columns='method',values='delta')['A2'])<=abs(g.pivot(index=['decision_eom','state_id'],columns='method',values='delta')['A1'])+1e-12) for _,g in predictions.groupby('variant')))
    reference=json.loads((OUT/'reference/summary.json').read_text());check('Independent R tuning/full refits/bootstrap samples pass',reference['status']=='PASS')
    label_summary=labels.groupby('horizon').agg(rows=('b','size'),decision_months=('decision_eom','nunique'),median_b=('b','median'),minimum_b=('b','min'),maximum_b=('b','max'),median_c=('c','median'),minimum_c=('c','min'),maximum_c=('c','max'),median_terminal_risk_ratio=('terminal_derivative_risk_ratio','median'),maximum_terminal_risk_ratio=('terminal_derivative_risk_ratio','max')).reset_index()
    label_summary.to_csv(OUT/'label_summary.csv',index=False)
    prediction_summary=[]
    for (variant,method),g in predictions[predictions.decision_eom.ge('2000-12-31')].groupby(['variant','method']):
        prediction_summary.append(dict(variant=variant,method=method,rows=len(g),distinct_decision_months=g.decision_eom.nunique(),median_kappa=float(g.kappa.median()),minimum_kappa=float(g.kappa.min()),maximum_kappa=float(g.kappa.max()),baseline_fraction=float(g.kappa.eq(1).mean()),lower_bound_fraction=float(np.isclose(g.kappa,g.kappa_min).mean()),upper_bound_fraction=float(np.isclose(g.kappa,g.kappa_max).mean()),out_of_range_fraction=float(g.out_of_range.mean()),fallback_rows=int(g.fallback_reason.notna().sum())))
    pd.DataFrame(prediction_summary).to_csv(OUT/'prediction_summary.csv',index=False)
    first={v:int(g.loc[g.status.eq('FITTED'),'fit_year'].min()) for v,g in annual.groupby('variant')}
    result=dict(status='PASS',checks_total=len(checks),checks_passed=sum(c['passed'] for c in checks),checks=checks,
        synthetic_checks=synthetic['checks_total'],reference=reference,state_months=1440,feature_rows=len(features),outcome_rows=len(labels),immature_rows=len(immature),
        bootstrap_fits_replayed=bootstrap_n,maximum_direct_feature_absolute_error=max_feature,maximum_state_recurrence_error=max_state_difference,
        maximum_replica_prediction_replay_error=max_replica,maximum_decision_replay_error=max_prediction,
        explicit_rollout_comparisons=len(rollout_records),maximum_explicit_rollout_absolute_error=max(r['absolute_error'] for r in rollout_records),
        fitted_annual_variants=int(annual.status.eq('FITTED').sum()),annual_fallback_variants=int(annual.status.ne('FITTED').sum()),first_fitted_trading_year=first,
        capped_months=int(caps.kappa_max.lt(1.5).sum()),minimum_upper_cap=float(caps.kappa_max.min()),label_summary=label_summary.to_dict('records'),prediction_summary=prediction_summary,
        adaptive_portfolio_performance_run=False,protected_science_unchanged=True,
        limitations=['Retrospective source reconstruction; vendor vintage, eligibility, imputation and omitted-ID accounting limits remain.',
        'Finite horizons leave residual holdings/derivatives; no terminal payoff or universal stability guarantee.',
        'One-time-deviation labels use baseline continuation; repeated adaptive deployment is untested until Step 9.',
        'Aim and advantage bootstraps are conditional sensitivity measures, not calibrated probabilities or performance guarantees.',
        'Three states share each date; they are not independent market observations. Future-information checks concern the computation under accepted source availability conventions.',
        '2021–2025 robustness inputs and later comparison/inference stages remain outside this Step 8 package.'])
    save_json(OUT/'review_summary.json',result)
    print(f'PASS: {len(checks)} review checks, {synthetic["checks_total"]} synthetic checks, {len(rollout_records)} explicit rollout comparisons.',flush=True)
if __name__=='__main__':main()
