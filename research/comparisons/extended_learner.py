"""Replay the fixed Step 8 learner on the extended available calendar.

Formulas and annual maturity rules are copied from the accepted pipeline;
all outputs are isolated and overlapping numerical records are verified.
"""
import argparse,json
import numpy as np
import pandas as pd
from baseline.io import datekey
from learner.core import A0,A1,VARIANTS,FLOOR,feasible,risk_apply,remap,weights,split,select_model,seed,blocks,fit_ridge,decide
from .io import ROOT,DATA,PROTOCOL,OUT,LEARNER as PREP,BASE,CONF,RUN,DATES,save_json,begin as stage_begin,finish as stage_end
STATES=['speed_0.5','speed_1.0','speed_1.5']
class Sources:
    def __init__(self):
        columns=['id','eom','valid','me','ret_ld1','tr_ld1','mu_ld1','lambda_diagonal','wealth_usd']
        frame=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=columns)
        frame=frame[frame.valid & frame.eom.ge(DATES[0]) & frame.eom.le(DATES[-1]+pd.offsets.MonthEnd(1))]
        self.groups={d:g.sort_values('id').reset_index(drop=True) for d,g in frame.groupby('eom')}
        self.baseline={d:g.sort_values('id').reset_index(drop=True) for d,g in pd.read_parquet(RUN/'PolicyDecision.parquet').groupby('eom')}
    def month(self,d,confidence=False):
        key=datekey(d);rows=self.groups[d];ids=rows.id.to_numpy(np.int64)
        with np.load(BASE/f'matrices/{key}.npz') as z:
            assert np.array_equal(ids,z['ids']);m=z['m'].copy()
        with np.load(BASE/f'risk/{key}.npz') as z:
            assert np.array_equal(ids,z['ids']);loadings=z['loadings'].copy();fc=z['factor_cov'].copy();specific=z['specific_variance'].copy()
        with np.load(CONF/f'aims/block12/{key}.npz') as z:
            assert np.array_equal(ids,z['ids']);aim=z['central_aim'].copy()
        next_date=d+pd.offsets.MonthEnd(1)
        if next_date not in self.groups and d != DATES[-1]:raise ValueError('Missing next-universe accounting input')
        result=dict(date=d,ids=ids,m=m,loadings=loadings,fc=fc,specific=specific,aim=aim,
            impact=rows.lambda_diagonal.to_numpy(),wealth=float(rows.wealth_usd.iloc[0]),
            returns=rows.ret_ld1.to_numpy(),growth=(1+rows.tr_ld1.to_numpy())/(1+float(rows.mu_ld1.iloc[0])),
            next_ids=self.groups[next_date].id.to_numpy(np.int64) if next_date in self.groups else ids.copy())
        for name in ['impact','returns','growth','aim','m','loadings','fc','specific']:
            if not np.isfinite(result[name]).all():raise ValueError(f'Nonfinite monthly input: {name}')
        if confidence:
            result['confidence']={}
            for block in [12,24]:
                path=CONF/f'aims/block{block}/{key}.npz'
                with np.load(path) as z:
                    assert np.array_equal(ids,z['ids']) and np.array_equal(aim,z['central_aim'])
                    mean=z['mean_aim'].copy()
                meta=json.loads(path.with_suffix('.json').read_text())
                result['confidence'][block]=dict(mean=mean,numerator=meta['features']['dispersion_numerator'])
        return result


def risk(month,x):return risk_apply(x,month['loadings'],month['fc'],month['specific'])


def features(month,h,q,block):
    item=month['confidence'][block];m=month['m'];qbar=item['mean']-h-m@(item['mean']-h)
    qr=float(q@risk(month,q));hr=float(h@risk(month,h));ir=float(month['wealth']*np.sum(month['impact']*q*q));den=float(qbar@risk(month,qbar))
    if min(qr,hr,ir,den,item['numerator'])<0:raise ValueError('Negative quadratic')
    alignfloor=qr<=FLOOR or hr<=FLOOR
    alignment=0. if alignfloor else float(q@risk(month,h)/np.sqrt(qr*hr));disp=item['numerator']/max(den,FLOOR);logdisp=float(np.log1p(disp))
    return dict(log_trade_risk=float(np.log(max(qr,FLOOR))),log_trade_impact=float(np.log(max(ir,FLOOR))),
        log_holdings_risk=float(np.log(max(hr,FLOOR))),risk_alignment=alignment,log1p_aim_dispersion=logdisp,
        alignment_times_log1p_dispersion=alignment*logdisp,initial_curvature=10*qr+ir,trade_risk=qr,
        holdings_risk=hr,trade_impact_curvature=ir,dispersion=disp,mean_trade_risk=den,
        dispersion_numerator=item['numerator'],denominator_floor=den<=FLOOR,alignment_floor=alignfloor)


def states():
    stage_begin('states');source=Sources();h=None;rows=[];exits=[];caps=[];maxerror=0.
    folder=PREP/'states';folder.mkdir()
    for i,d in enumerate(DATES):
        x=source.month(d,confidence=True);ids=x['ids'];base=source.baseline[d]
        if h is None:h=np.repeat(base.pretrade.to_numpy()[:,None],3,axis=1)
        cap=feasible(x['m'],x['impact']);caps.append(dict(decision_eom=d,**cap))
        q=(x['aim'][:,None]-h)-x['m']@(x['aim'][:,None]-h)
        implemented=np.minimum(np.array([.5,1.,1.5]),cap['kappa_max']);w=h+q*implemented
        drift=w*x['growth'][:,None];next_h,dropped=remap(drift,ids,x['next_ids'])
        err=float(np.max(np.abs(h[:,1]-base.pretrade.to_numpy())));maxerror=max(maxerror,err)
        if not np.allclose(h[:,1],base.pretrade,atol=1e-8,rtol=1e-6):raise ValueError('State speed 1 differs from accepted R1')
        np.savez_compressed(folder/f'{datekey(d)}.npz',ids=ids,pretrade=h,baseline_trade=q,weight=w,trade=q*implemented,drifted_weight=drift,kappa=implemented)
        for s,state in enumerate(STATES):
            for block in [12,24]:rows.append(dict(decision_eom=d,state_id=state,aim_block=block,implemented_kappa=float(implemented[s]),
                kappa_min=cap['kappa_min'],kappa_max=cap['kappa_max'],**features(x,h[:,s],q[:,s],block)))
            for j in dropped:exits.append(dict(decision_eom=d,next_decision=d+pd.offsets.MonthEnd(1),state_id=state,id=int(ids[j]),omitted_drifted_weight=float(drift[j,s]),terminal_boundary=i==len(DATES)-1,liquidation_trade_added=False))
        h=next_h
        if i%60==0:print(f'States {i+1}/{len(DATES)}, R1 max difference {maxerror:.2e}',flush=True)
    pd.DataFrame(rows).to_parquet(PREP/'state_features.parquet',index=False)
    pd.DataFrame(exits).to_parquet(PREP/'state_exit_ledger.parquet',index=False)
    pd.DataFrame(caps).to_parquet(PREP/'matrix_feasibility.parquet',index=False)
    stage_end('states',list(folder.glob('*.npz'))+[PREP/p for p in ['state_features.parquet','state_exit_ledger.parquet','matrix_feasibility.parquet']],
        months=len(DATES),states=3,feature_rows=len(rows),exit_rows=len(exits),maximum_R1_state_difference=maxerror,
        capped_months=sum(c['kappa_max']<1.5 for c in caps),minimum_upper_cap=min(c['kappa_max'] for c in caps))


def labels():
    stage_begin('labels');source=Sources();labels=[];exits=[];active=[];href=dpre=None;B=C=np.zeros(0);initialS=initialrisk=np.zeros(0)
    endpoint=PREP/'endpoints';endpoint.mkdir()
    for i,d in enumerate(DATES):
        x=source.month(d);n=len(x['ids'])
        with np.load(PREP/f'states/{datekey(d)}.npz') as z:new=z['pretrade'].copy()
        if href is None:href=np.empty((n,0));dpre=np.empty((n,0))
        old=len(active);href=np.column_stack([href,new]);dpre=np.column_stack([dpre,np.zeros_like(new)])
        active+= [(i,s) for s in range(3)];B=np.r_[B,np.zeros(3)];C=np.r_[C,np.zeros(3)]
        q=(x['aim'][:,None]-href)-x['m']@(x['aim'][:,None]-href);w=href+q
        e=x['m']@dpre;e[:,old:]=q[:,old:];v=e-dpre
        re=risk(x,e);s_term=10*np.sum(e*re,axis=0)+x['wealth']*np.sum(x['impact'][:,None]*v*v,axis=0)
        initialS=np.r_[initialS,s_term[old:]];initialrisk=np.r_[initialrisk,np.sum(e[:,old:]*re[:,old:],axis=0)]
        B+=x['returns']@e-10*np.sum(e*risk(x,w),axis=0)-x['wealth']*np.sum(x['impact'][:,None]*v*q,axis=0)
        C+=s_term
        next_h,dropped=remap(w*x['growth'][:,None],x['ids'],x['next_ids'])
        next_e,_=remap(e*x['growth'][:,None],x['ids'],x['next_ids'])
        for j in dropped:
            for k,(start,s) in enumerate(active):
                exits.append(dict(start_decision=DATES[start],state_id=STATES[s],continuation_decision=d,age=i-start+1,id=int(x['ids'][j]),
                    omitted_reference_weight=float(w[j,k]*x['growth'][j]),omitted_derivative=float(e[j,k]*x['growth'][j]),liquidation_trade_added=False))
        for H in [12,36]:
            indices=[k for k,(start,s) in enumerate(active) if i-start+1==H]
            if indices:
                start=active[indices[0]][0];path=endpoint/f'H{H}_{datekey(DATES[start])}.npz'
                np.savez_compressed(path,ids=x['ids'],weight_derivative=e[:,indices],reference_weight=w[:,indices],next_ids=x['next_ids'],next_derivative=next_e[:,indices])
                for k in indices:
                    st,s=active[k];usable=initialS[k]>FLOOR
                    if C[k]<initialS[k]*(1-1e-12):raise ValueError('Accumulated curvature below its initial term')
                    labels.append(dict(decision_eom=DATES[st],state_id=STATES[s],horizon=H,outcome_start=DATES[st]+pd.offsets.MonthEnd(1),
                        outcome_end=d+pd.offsets.MonthEnd(1),label_available_at=d+pd.offsets.MonthEnd(1),B=float(B[k]),C=float(C[k]),S=float(initialS[k]),
                        b=float(B[k]/initialS[k]) if usable else np.nan,c=float(C[k]/initialS[k]) if usable else np.nan,usable=bool(usable),
                        exclusion_reason=None if usable else 'initial_curvature_floor',continuation_policy='original_baseline_from_same_initial_state',
                        terminal_derivative_l2=float(np.linalg.norm(e[:,k])),terminal_next_derivative_l2=float(np.linalg.norm(next_e[:,k])),
                        terminal_derivative_risk_ratio=float(np.sum(e[:,k]*re[:,k])/max(initialrisk[k],FLOOR)),
                        endpoint_artifact=str(path.relative_to(ROOT)),terminal_payoff_added=False))
        keep=np.array([i-start+1<36 for start,s in active]);active=[x for x,k in zip(active,keep) if k]
        href=next_h[:,keep];dpre=next_e[:,keep];B=B[keep];C=C[keep];initialS=initialS[keep];initialrisk=initialrisk[keep]
        if i%60==0:print(f'Labels through month {i+1}/{len(DATES)}; completed {len(labels)}',flush=True)
    table=pd.DataFrame(labels).sort_values(['horizon','decision_eom','state_id']);table.to_parquet(PREP/'DeviationOutcome.parquet',index=False)
    pd.DataFrame(exits).to_parquet(PREP/'deviation_exit_ledger.parquet',index=False)
    immature=[dict(decision_eom=d,state_id=s,horizon=H,required_outcome_end=d+pd.offsets.MonthEnd(H),status='IMMATURE_AT_END_OF_DATA') for H in [12,36] for d in DATES[-(H-1):] for s in STATES]
    pd.DataFrame(immature).to_parquet(PREP/'immature_labels.parquet',index=False)
    stage_end('labels',list(endpoint.glob('*.npz'))+[PREP/p for p in ['DeviationOutcome.parquet','deviation_exit_ledger.parquet','immature_labels.parquet']],
        outcome_rows=len(table),horizon_counts={str(k):int(v) for k,v in table.groupby('horizon').size().items()},immature_rows=len(immature),
        curvature_floor_rows=int((~table.usable).sum()),minimum_normalized_curvature=float(table.c.min()),exit_rows=len(exits))


def training_rows(variant):
    H,aimblock,_=VARIANTS[variant]
    outcomes=pd.read_parquet(PREP/'DeviationOutcome.parquet');outcomes=outcomes[outcomes.horizon.eq(H)]
    features=pd.read_parquet(PREP/'state_features.parquet');features=features[features.aim_block.eq(aimblock)]
    rows=outcomes.merge(features,on=['decision_eom','state_id'],validate='one_to_one').sort_values(['decision_eom','state_id']).reset_index(drop=True)
    if not np.allclose(rows.S,rows.initial_curvature,rtol=1e-10,atol=1e-12):raise ValueError('Label/feature initial curvature mismatch')
    return rows


def fits():
    stage_begin('fits');annual=[];seed_rows=[];folder=PREP/'fits';folder.mkdir();fit_count=0;replica_count=0
    for variant,(_,_,length) in VARIANTS.items():
        rows=training_rows(variant);(folder/variant).mkdir()
        for year in range(1981,2026):
            cutoff=pd.Timestamp(f'{year-1}-12-31');all_rows,train,validation,info=split(rows,cutoff)
            info.update(variant=variant,fit_year=year,first_decision=str(cutoff.date()),last_decision=f'{year}-11-30')
            fitted=dict(**info,models={},bootstrap=[],uncertainty_status='BASELINE_FALLBACK')
            if info['status']=='FITTED':
                for method,cols in [('A0',A0),('A1',A1)]:fitted['models'][method]=select_model(train,validation,all_rows,cols,PROTOCOL['learning']['ridge_penalties'])
                dates=pd.DatetimeIndex(sorted(all_rows.decision_eom.unique()));date_index=dates.get_indexer(all_rows.decision_eom)
                count_matrix=[];start_matrix=[];sample_matrix=[]
                try:
                    for replica in range(1,101):
                        number,digest=seed(PROTOCOL,variant,str(cutoff.date()),replica);starts,sampled,counts=blocks(dates,length,number)
                        model=fit_ridge(all_rows[A1].to_numpy(),all_rows.b.to_numpy(),weights(all_rows)*counts[date_index],fitted['models']['A1']['penalty'])
                        fitted['bootstrap'].append(model);count_matrix.append(counts);start_matrix.append(starts);sample_matrix.append(sampled)
                        seed_rows.append(dict(variant=variant,fit_year=year,replica=replica,seed_sha256=digest,block_months=length))
                except ValueError as error:
                    fitted['bootstrap']=[];fitted['uncertainty_status']='FAILED';fitted['uncertainty_failure']=str(error)
                else:fitted['uncertainty_status']='FITTED';replica_count+=100
                np.savez_compressed(folder/variant/f'{year}_training.npz',decision_dates=all_rows.decision_eom.dt.strftime('%Y-%m-%d').to_numpy(dtype=str),
                    states=all_rows.state_id.to_numpy(dtype=str),dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),x0=all_rows[A0].to_numpy(),x1=all_rows[A1].to_numpy(),
                    y=all_rows[['b','c']].to_numpy(),weights=weights(all_rows),bootstrap_counts=np.array(count_matrix),bootstrap_starts=np.array(start_matrix),bootstrap_sampled=np.array(sample_matrix),
                    inner_train_dates=train.decision_eom.dt.strftime('%Y-%m-%d').to_numpy(dtype=str),validation_dates=validation.decision_eom.dt.strftime('%Y-%m-%d').to_numpy(dtype=str))
                fit_count+=1
            save_json(folder/variant/f'{year}.json',fitted)
            annual.append(dict(**info,uncertainty_status=fitted['uncertainty_status'],A0_penalty=fitted['models'].get('A0',{}).get('penalty'),A1_penalty=fitted['models'].get('A1',{}).get('penalty')))
        print(f'{variant}: annual fits complete',flush=True)
    pd.DataFrame(annual).to_parquet(PREP/'annual_fit_records.parquet',index=False)
    pd.DataFrame(seed_rows).to_parquet(PREP/'advantage_replica_seeds.parquet',index=False)
    stage_end('fits',[p for p in folder.rglob('*') if p.is_file()]+[PREP/'annual_fit_records.parquet',PREP/'advantage_replica_seeds.parquet'],
        annual_records=len(annual),fitted_annual_variants=fit_count,bootstrap_fits=replica_count,baseline_fallback_annual_variants=len(annual)-fit_count,
        uncertainty_failures=sum(r['status']=='FITTED' and r['uncertainty_status']!='FITTED' for r in annual))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['states','labels','fits']);args=parser.parse_args();PREP.mkdir(parents=True,exist_ok=True);globals()[args.stage]()
