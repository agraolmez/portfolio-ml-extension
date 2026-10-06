"""Fresh independent numerical/accounting evidence for the Step 4 machinery."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.io import *
from baseline.core import *
from baseline.wealth import reconstruct,validate
from baseline.engine import Engine,feature_indices


def compare(name,a,b,atol=1e-8,rtol=1e-6):
    a=np.asarray(a);b=np.asarray(b)
    same_shape=a.shape==b.shape
    ok=same_shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)
    return {'check':name,'passed':bool(ok),'values':int(a.size),'max_abs_difference':float(np.max(abs(a-b))) if same_shape and a.size else None,'atol':atol,'rtol':rtol}


def historical():
    engine=Engine();checks=[]
    for d in ['1965-11-30','1980-12-31','2025-11-30']:
        ids,sigma,impact,wealth,m=engine.matrix(d);n=len(ids)
        ref=np.fromfile(OUT/f'reference/r_m_{d}.bin',dtype='<f8').reshape(n,n)
        checks.append(compare(f'{d}: ten-iteration inertia',m,ref))
        for gi in [0,1]:
            if d!='2025-11-30':
                with np.load(PREP/f'moments/g{gi}/{d}.npz') as z:result={f:z[f].copy() for f in ['r_tilde','denom','risk','tc','signal_t']}
            else:
                result=engine.inputs(d,gi)
                folder=OUT/'historical_python';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'g{gi}_{d}.npz',**result)
            for field in ['r_tilde','denom','risk','tc','signal_t']:
                a=result[field];b=np.fromfile(OUT/f'reference/r_moment_g{gi}_{d}_{field}.bin',dtype='<f8').reshape(a.shape)
                checks.append(compare(f'{d}: g{gi} {field}',a,b))
    dates=pd.date_range('1980-11-30','1981-02-28',freq='ME')
    aims=pd.read_csv(OUT/'reference/mechanics_aims.csv',parse_dates=['eom'],float_precision='round_trip').rename(columns={'w_aim':'aim'})
    monthly,positions,exits=engine.run(dates,aims,require_step4=False)
    monthly.to_csv(OUT/'mechanics_monthly.csv',index=False);positions.to_parquet(OUT/'mechanics_positions.parquet',index=False);exits.to_parquet(OUT/'mechanics_exits.parquet',index=False)
    ref=pd.read_csv(OUT/'reference/r_mechanics_weights.csv',parse_dates=['eom'],float_precision='round_trip')
    joined=positions.merge(ref,on=['id','eom'],validate='one_to_one')
    checks.append(compare('Historical holdings: pretrade',joined.pretrade,joined.w_start))
    checks.append(compare('Historical holdings: posttrade',joined.weight,joined.w))
    pf=pd.read_csv(OUT/'reference/r_mechanics_returns.csv',parse_dates=['eom_ret'],float_precision='round_trip').rename(columns={'eom_ret':'return_eom'})
    merged=monthly.merge(pf,on='return_eom',validate='one_to_one')
    for p,r in [('gross_excess','r'),('impact_cost','tc'),('turnover','turnover_y'),('gross_exposure','inv')]:
        p='turnover_x' if p=='turnover' else p
        checks.append(compare('Historical accounting: '+p,merged[p],merged[r],atol=1e-8,rtol=0))
    identities=[];entries=0
    for j,d in enumerate(dates):
        p=positions[positions.eom.eq(d)];f=engine.months[d].loc[p.id]
        next_wealth=f.wealth_usd.iloc[0]*(1+f.mu_ld1.iloc[0])
        identities.append(compare(f'{datekey(d)}: dollar/weight drift identity',p.drifted_weight.to_numpy()*next_wealth,p.weight.to_numpy()*f.wealth_usd.iloc[0]*(1+f.tr_ld1.to_numpy()),atol=1e-7,rtol=1e-12))
        if j:
            previous=positions[positions.eom.eq(dates[j-1])];new=~p.id.isin(previous.id);entries+=int(new.sum())
            checks.append({'check':f'{datekey(d)}: new IDs start at zero','passed':bool(p.loc[new,'pretrade'].eq(0).all()),'entries':int(new.sum())})
    checks.extend(identities)
    ledger=pd.read_parquet(DATA/'outcome_and_exit_ledger.parquet')
    expected=ledger[ledger.eom.isin(dates[:-1])&ledger.exit_after_return]
    checks.append({'check':'Every bounded-fixture exit is inventoried with source treatment','passed':set(zip(exits.id,exits.decision_eom))==set(zip(expected.id,expected.eom)) and bool(exits.source_exit_trade_cost.eq(0).all()),'exits':len(exits),'entries':entries})
    checks.append({'check':'Terminal drift retained without invented liquidation','passed':len(positions[positions.eom.eq(dates[-1])])==len(engine.months[dates[-1]].query('valid'))})
    save_json(OUT/'historical_parity.json',{'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,'moment_cross_sections':3,'bandwidths':2,'mechanics_months':4,'mechanics_positions':len(positions),'entries':entries,'exits':len(exits),'scope':'Historical full-universe machinery fixtures with explicitly given targets; no selected-strategy performance pilot','matrix_diagnostics':engine.matrix_diagnostics,'resources':guard()})
    print('Historical checks:',sum(x['passed'] for x in checks),'/',len(checks),flush=True)
    for x in checks:
        if not x['passed']:print(x,flush=True)


def fits():
    checks=[];coef_values=0
    for gi in [0,1]:
        for year in range(1971,1983):
            for p in SETTINGS['settings']['pf_ml']['p_vec']:
                with np.load(PREP/f'coefficients/g{gi}/{year}_p{p}.npz') as z:a=z['coefficients'];cutoff=str(z['training_cutoff']);n=int(z['training_months'])
                b=np.fromfile(OUT/f'reference/fit/g{gi}/r_coef_{year}_p{p}.bin',dtype='<f8').reshape(a.shape)
                checks.append(compare(f'g{gi} {year} p{p}: coefficients including lambda zero',a,b));coef_values+=a.size
                expected=len(pd.date_range('1965-11-30',f'{year-1}-11-30',freq='ME'))
                checks.append({'check':f'g{gi} {year} p{p}: matured training cutoff','passed':cutoff==f'{year-1}-11-30' and n==expected,'training_months':n})
    a=pd.read_parquet(PREP/'validation_scores.parquet')
    b=pd.read_csv(OUT/'reference/r_validation.csv',parse_dates=['eom','eom_ret'],float_precision='round_trip').rename(columns={'eom':'decision_eom','eom_ret':'return_eom'})
    keys=['g_index','p','lambda_index','decision_eom','return_eom'];joined=a.merge(b,on=keys,validate='one_to_one')
    for p,r in [('objective','obj'),('cumulative_objective','cum_obj')]:checks.append(compare('All original-grid '+p,joined[p],joined[r],atol=1e-8,rtol=0))
    choice=pd.read_parquet(PREP/'annual_selection.parquet')
    ref=pd.read_csv(OUT/'reference/r_annual_selection.csv',parse_dates=['eom_ret'],float_precision='round_trip').rename(columns={'eom_ret':'return_eom'})
    checks.append({'check':'Exact global annual selection agreement','passed':set(map(tuple,choice[['return_eom','g_index','p','lambda_index']].to_numpy()))==set(map(tuple,ref[['return_eom','g_index','p','lambda_index']].to_numpy())),'years':len(choice)})
    # Independent R-selected coefficient plus R-normalized signal produces the same first pilot target.
    d='1980-12-31';row=ref[ref.return_eom.dt.year.eq(1980)].iloc[0];gi=int(row.g_index);p=int(row.p);li=int(row.lambda_index)
    bcoef=np.fromfile(OUT/f'reference/fit/g{gi}/r_coef_1981_p{p}.bin',dtype='<f8').reshape(p+1,101)[:,li]
    rs=np.fromfile(OUT/f'reference/r_moment_g{gi}_{d}_signal_t.bin',dtype='<f8').reshape(-1,513)
    target=rs[:,feature_indices(p)]@bcoef
    engine=Engine(end='1982-11-30');aims=engine.aims(pd.date_range('1980-12-31','1982-11-30',freq='ME'))
    checks.append(compare('First selected aim: independent R selection/coefficient/signal',aims[aims.eom.eq(d)].aim.to_numpy(),target))
    aims.to_parquet(PREP/'pilot_aims_preflight.parquet',index=False)
    checks.append({'check':'All 24 pilot targets finite with lagged annual selection','passed':aims.eom.nunique()==24 and bool(np.isfinite(aims.aim).all()) and bool((aims.fit_year==aims.selection_return_year+1).all()),'stock_months':len(aims)})
    save_json(OUT/'fit_parity.json',{'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,'coefficient_values':coef_values,'coefficient_fits':9696,'validation_rows':len(joined),'annual_selections':len(choice),'pilot_target_stock_months':len(aims),'portfolio_performance_pilot_run':False})
    print('Annual fitting checks:',sum(x['passed'] for x in checks),'/',len(checks),flush=True)
    for x in checks:
        if not x['passed']:print(x,flush=True)


def contracts():
    checks=[]
    def reject(name,fn):
        try:fn();passed=False
        except (ValueError,FileNotFoundError,KeyError):passed=True
        checks.append({'check':name,'passed':passed})
    dates=pd.date_range('2020-10-31','2020-12-31',freq='ME');r=np.array([0.,.1,.2])
    exact=reconstruct(dates,r,dates[-1],100.)
    checks.append(compare('Independent exact reciprocal example',exact,[100/1.2/1.1,100/1.2,100],atol=1e-12,rtol=0))
    full_dates=pd.date_range('1952-12-31','2025-12-31',freq='ME');rr=.02*np.sin(np.arange(len(full_dates)))
    w=reconstruct(full_dates,rr,'2020-12-31',1e10)
    checks.append(compare('Synthetic full-history adjacent round trip',w[1:]/w[:-1],1+rr[1:],atol=1e-12,rtol=0))
    bad=[('Missing month',[dates[0],dates[2]],r[[0,2]]),('Duplicate month',[dates[0],dates[0],dates[2]],r),('Non month-end',dates-pd.Timedelta(days=1),r),('Unsorted months',dates[::-1],r),('Missing return',dates,[0,np.nan,.1]),('Minus 100 percent return',dates,[0,-1,.1]),('Below minus 100 percent return',dates,[0,-1.1,.1]),('Infinite return',dates,[0,np.inf,.1])]
    for label,d,x in bad:reject(label,lambda d=d,x=x:reconstruct(d,x,dates[-1],100.))
    reject('Missing anchor',lambda:reconstruct(dates,r,'2019-12-31',100.))
    reject('Nonpositive anchor',lambda:reconstruct(dates,r,dates[-1],0.))
    reject('Overflow',lambda:reconstruct(dates,[0,1e308,1e308],dates[0],100.))
    reject('Underflow',lambda:reconstruct(dates,[0,1e308,1e308],dates[-1],1e-300))
    frame=pd.read_parquet(DATA/'market_wealth.parquet');validated=validate(frame)
    checks.append({'check':'All historical wealth values independently reconstructed','passed':True,**validated})
    wrong=frame.copy();wrong.loc[wrong.eom.eq('2020-12-31'),'wealth_usd']+=1
    reject('Incorrect exact anchor rejected',lambda:validate(wrong))
    wrong=frame.copy();wrong.loc[10,'wealth_usd']*=1.01
    reject('Broken historical wealth link rejected',lambda:validate(wrong))
    # Verify fail-closed accepted-data contract with isolated scratch manifests; never mutate the real bundle.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);src=root/'output/step3_author_conventions_20260911';src.mkdir(parents=True);d=root/'data/prepared/step3_author_v3';d.mkdir(parents=True)
        accepted={'status':'ACCEPTED_AUTHOR_CONVENTION_RECONSTRUCTION','step3_complete':True,'gates':[{'passed':True}]}
        save_json(src/'acceptance_manifest.json',accepted);save_json(d/'STATUS.json',{'status':accepted['status']});(d/'input').write_text('accepted')
        save_json(src/'bundle_checksums.json',{'data/prepared/step3_author_v3/input':sha(d/'input')})
        verify_data(root);(d/'input').write_text('changed');reject('Stale input evidence rejected',lambda:verify_data(root))
        (d/'input').unlink();reject('Missing accepted input rejected',lambda:verify_data(root))
        accepted['status']='NOT_ACCEPTED';save_json(src/'acceptance_manifest.json',accepted);reject('Mislabeled acceptance rejected',lambda:verify_data(root))
    m=np.array([[.4,.1],[.05,.3]]);h=np.array([.6,.4]);a=np.array([.2,.3]);total=np.array([.1,-.2]);impact=np.array([.001,.002])
    result=transition([1,2],h,a,m,total,.05,impact,100,[.09,-.21],[2,3])
    expected=m@h+(np.eye(2)-m)@a
    checks.append(compare('Dollar accounting and entry/exit synthetic fixture',result['next_pretrade'],[expected[1]*.8/1.05,0],atol=1e-14,rtol=0))
    checks.append(compare('Trade-only cost includes half wealth',result['impact_cost'],50*np.sum(impact*(expected-h)**2),atol=1e-14,rtol=0))
    checks.append({'check':'Exit convention explicit; no fictitious liquidation cost','passed':result['dropped_ids'].tolist()==[1] and result['source_exit_trade_cost']==0})
    reject('Invalid portfolio dimensions',lambda:transition([1,2],h[:1],a,m,total,.05,impact,100,[.09,-.21],[2,3]))
    reject('Duplicate portfolio IDs',lambda:transition([1,1],h,a,m,total,.05,impact,100,[.09,-.21],[2,3]))
    reject('Nonfinite impact rejected',lambda:transition([1,2],h,a,m,total,.05,[np.nan,.002],100,[.09,-.21],[2,3]))
    reject('Nonpositive wealth growth rejected',lambda:transition([1,2],h,a,m,total,-1,impact,100,[.09,-.21],[2,3]))
    reject('Nonpositive definite covariance',lambda:inertia(np.array([[1,2],[2,1]]),impact,100,.01))
    reject('Asymmetric economic matrix',lambda:symmetric(m))
    reject('Incomplete moment lookback',lambda:moment([],[],m,m,impact,100,h))
    reject('Zero random-feature norm',lambda:normalize_signal(np.ones((2,2)),np.ones(2)))
    reject('Zero-ridge singular matrix fails without silent regularization',lambda:ridge_path(np.zeros((2,2)),h,[0,.1]))
    gross=np.array([.02,-.01,.03]);cost=np.array([.001,.002,.003]);net=gross-cost
    report=report_returns(gross,cost)
    direct=12*(sum(net)/3-5*sum((net-sum(net)/3)**2)/3)
    checks.append(compare('Primary utility uses net returns and divisor T',report['net_population_utility'],direct,atol=1e-15,rtol=0))
    checks.append({'check':'Uncorrected source utility kept distinct','passed':abs(report['net_population_utility']-report['source_gross_sample_variance_utility'])>1e-6})
    before=json.loads((ROOT/'output/step3_20260911/start.json').read_text())['protected']
    checks.append({'check':'Approved science/protocol protected bytes unchanged','passed':all(sha(ROOT/p)==s for p,s in before.items()),'files':len(before)})
    checks.append({'check':'Accepted Step 3 input hashes reverified','passed':verify_data()['files_verified']==226})
    save_json(OUT/'contract_checks.json',{'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,'resources':guard()})
    print('Contract checks:',sum(x['passed'] for x in checks),'/',len(checks),flush=True)
    for x in checks:
        if not x['passed']:print(x,flush=True)

if __name__=='__main__':{'historical':historical,'fits':fits,'contracts':contracts}[sys.argv[1]]()
