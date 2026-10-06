"""Independent full-baseline accounting, chronology, R parity and reporting review."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.full import OUT, RUN, PREP, PRIOR, DATES, MOMENT_DATES, preflight
from baseline.io import ROOT, DATA, PROTOCOL, SETTINGS, sha, save_json


def statistics(frame):
    gross = frame.gross_excess.to_numpy(); cost = frame.impact_cost.to_numpy()
    net = gross - cost; n = len(net); gamma = PROTOCOL['economic']['risk_aversion']
    mean = float(sum(net)/n); variance = float(sum((net-mean)**2)/n)
    gm = float(sum(gross)/n); gv = float(sum((gross-gm)**2)/(n-1))
    return {'months':n,'annual_mean_net_excess':12*mean,'annual_net_volatility':float(np.sqrt(12*variance)),
        'net_population_utility':12*(mean-gamma/2*variance),'net_sharpe':float(np.sqrt(12)*mean/np.sqrt(variance)),
        'source_gross_sample_variance_utility':12*(mean-gamma/2*gv),
        'source_net_over_gross_sharpe':float(np.sqrt(12)*mean/np.sqrt(gv)),
        'annual_mean_gross_excess':12*gm,'annual_gross_sample_volatility':float(np.sqrt(12*gv)),
        'gross_sample_sharpe':float(np.sqrt(12)*gm/np.sqrt(gv)),
        'annualized_mean_impact_cost':12*float(sum(cost)/n),'mean_monthly_turnover':float(frame.turnover.mean()),
        'mean_gross_exposure':float(frame.gross_exposure.mean()),'minimum_monthly_net_excess':float(net.min()),
        'maximum_monthly_net_excess':float(net.max()),'sum_monthly_net_excess':float(sum(net))}


def main():
    if (OUT/'acceptance_manifest.json').exists():
        raise ValueError('Accepted Step 6 review evidence is immutable; use a separately versioned replay')
    checks = []
    def boolean(name, ok, evidence=None):
        checks.append({'check':name,'passed':bool(ok),'evidence':evidence})
    def compare(name, a, b, atol=1e-8, rtol=0.):
        a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        matched = a.shape == b.shape
        ok = matched and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)
        checks.append({'check':name,'passed':bool(ok),'values':int(a.size),
            'max_abs_difference':float(np.max(abs(a-b))) if matched and a.size else None,'atol':atol,'rtol':rtol})
    receipt, locked_aims = preflight()
    boolean('All accepted and prepared dependencies verify',receipt['months']==480,receipt)
    start = json.loads((OUT/'start.json').read_text())
    boolean('Accepted copied seeds remain identical',all(sha(ROOT/p)==x['sha256'] and sha(ROOT/x['source'])==x['sha256']
        for p,x in start['accepted_seed_copies'].items()),{'copied_files':len(start['accepted_seed_copies'])})
    ref = OUT/'reference'
    export = json.loads((ref/'export_manifest.json').read_text())
    boolean('Exact exported R interchange inputs unchanged',all(sha(ROOT/p)==h for p,h in export['artifact_sha256'].items()))
    rf = json.loads((ref/'fit_status.json').read_text()); rp = json.loads((ref/'path_status.json').read_text())
    boolean('Actual R full-grid fitting completed',rf['status']=='ACTUAL_R_FULL_GRID_COMPLETE' and rf['coefficient_fits']==40400 and rf['validation_rows']==484800,rf)
    boolean('Actual R full holdings path completed',rp['status']=='ACTUAL_R_FULL_PATH_COMPLETE' and rp['months']==480 and rp['sample_input_months']==81,rp)
    m = pd.read_parquet(RUN/'monthly_performance.parquet')
    p = pd.read_parquet(RUN/'PolicyDecision.parquet')
    aims = pd.read_parquet(RUN/'AimArtifact.parquet')
    exits = pd.read_parquet(RUN/'source_exit_ledger.parquet')
    snapshots = pd.read_parquet(RUN/'MarketSnapshot.parquet')
    boolean('Exact 480-month decision/return calendar',list(m.decision_eom)==list(DATES) and list(m.return_eom)==list(DATES+pd.offsets.MonthEnd(1)))
    boolean('Unique security-month records',not p.duplicated(['id','eom']).any() and not aims.duplicated(['id','eom']).any())
    boolean('Snapshot calendar matches all decisions',list(snapshots.decision_eom)==list(DATES) and list(snapshots.return_eom)==list(DATES+pd.offsets.MonthEnd(1)))
    cols=['id','eom','valid','me','ret_ld1','tr_ld1','mu_ld1','lambda_diagonal','wealth_usd']
    raw = pd.read_parquet(DATA/'monthly_inputs.parquet',columns=cols,filters=[('eom','>=',DATES[0]),('eom','<=',DATES[-1])])
    active = raw[raw.valid]
    membership = set(zip(active.id,active.eom))
    boolean('All portfolios and targets match accepted membership',set(zip(p.id,p.eom))==membership==set(zip(aims.id,aims.eom)),{'positions':len(p),'securities':p.id.nunique()})
    combo = p.merge(active,on=['id','eom'],validate='one_to_one')
    boolean('All portfolio/economic numeric values finite',np.isfinite(combo.select_dtypes('number')).all().all())
    compare('Locked aims equal run aims',locked_aims.sort_values(['eom','id']).aim,aims.sort_values(['eom','id']).aim,atol=0)
    compare('Policy aims equal target records',p.sort_values(['eom','id']).aim,aims.sort_values(['eom','id']).aim,atol=0)
    by_date = {d:f.sort_values('id') for d,f in combo.groupby('eom')}
    rows, entries, independent_exits = [], [], []
    for i,d in enumerate(DATES):
        x=by_date[d];ids=x.id.to_numpy();w=x.weight.to_numpy();h=x.pretrade.to_numpy();a=x.aim.to_numpy();q=w-h
        snap=snapshots.iloc[i]
        boolean(f'{d:%Y-%m}: market snapshot',snap.eligible_stocks==len(x) and snap.wealth_usd==x.wealth_usd.iloc[0]
            and snap.market_total_growth==x.mu_ld1.iloc[0] and snap.risk_sha256==sha(PRIOR/f'risk/{d:%Y-%m-%d}.npz'))
        with np.load(PREP/f'matrices/{d:%Y-%m-%d}.npz') as z: matrix=z['m']; mids=z['ids']
        boolean(f'{d:%Y-%m}: matrix membership',np.array_equal(ids,mids))
        compare(f'{d:%Y-%m}: independent trade rule',w,h+(np.eye(len(h))-matrix)@(a-h),rtol=1e-6)
        if i==0: expected=x.me.to_numpy()/x.me.sum()
        else:
            prev=by_date[DATES[i-1]].set_index('id')
            expected=(prev.weight*(1+prev.tr_ld1)/(1+prev.mu_ld1)).reindex(ids).fillna(0).to_numpy()
            new=~x.id.isin(prev.index);entries.append(x.loc[new,['id','eom','pretrade']])
        compare(f'{d:%Y-%m}: inherited holdings',h,expected,rtol=1e-6)
        compare(f'{d:%Y-%m}: trade records',x.trade,q,rtol=1e-6)
        expected_drift=w*(1+x.tr_ld1.to_numpy())/(1+x.mu_ld1.to_numpy())
        compare(f'{d:%Y-%m}: total-return drift',x.drifted_weight,expected_drift,rtol=1e-6)
        if i<len(DATES)-1:
            dropped=~np.isin(ids,by_date[DATES[i+1]].id)
            independent_exits.extend({'id':int(j),'decision_eom':d,'omitted_next_weight':float(v)} for j,v in zip(ids[dropped],expected_drift[dropped]))
        rows.append({'decision_eom':d,'gross_excess':float(sum(w*x.ret_ld1)),
            'impact_cost':float(x.wealth_usd.iloc[0]/2*sum(x.lambda_diagonal*q**2)),
            'turnover':float(sum(abs(q))),'gross_exposure':float(sum(abs(w))),
            'net_exposure':float(sum(w)),'short_exposure':float(-sum(w[w<0])),
            'largest_absolute_position':float(max(abs(w))),'eligible_stocks':len(x),'wealth_usd':float(x.wealth_usd.iloc[0])})
    direct=pd.DataFrame(rows)
    for field in ['gross_excess','impact_cost','turnover','gross_exposure']:
        compare('Independent monthly '+field,m[field],direct[field])
    compare('Net return charges impact once',m.net_excess,m.gross_excess-m.impact_cost,atol=1e-12)
    boolean('Nonnegative costs; initial trade charged',m.impact_cost.ge(0).all() and m.impact_cost.iloc[0]>0)
    entrants=pd.concat(entries,ignore_index=True)
    boolean('Every entrant begins at zero',entrants.pretrade.eq(0).all(),{'entries':len(entrants)})
    entrants.to_parquet(OUT/'entry_ledger.parquet',index=False)
    expected_exits=pd.DataFrame(independent_exits)
    merged=exits.merge(expected_exits,on=['id','decision_eom'],validate='one_to_one',suffixes=('_run','_independent'))
    boolean('All exits and zero extra exit costs recorded',len(merged)==len(exits)==len(expected_exits) and exits.source_exit_trade_cost.eq(0).all())
    compare('Omitted source exit holdings match drift',merged.omitted_next_weight_run,merged.omitted_next_weight_independent,rtol=1e-6)
    ledger=pd.read_parquet(DATA/'outcome_and_exit_ledger.parquet');ledger=ledger[ledger.eom.isin(DATES)].copy()
    boolean('Full outcome ledger membership',set(zip(ledger.id,ledger.eom))==membership)
    planned_exits=ledger[ledger.exit_after_return&ledger.eom.isin(DATES[:-1])]
    boolean('Universe exits agree with accepted event ledger',set(zip(exits.id,exits.decision_eom))==set(zip(planned_exits.id,planned_exits.eom)))
    ledger=ledger.merge(p[['id','eom','weight']],on=['id','eom'],validate='one_to_one')
    ledger['r1_minus_r0_return']=ledger.ret_ld1-ledger.ret_ld1_r0
    ledger['fixed_weights_outcome_contribution']=ledger.weight*ledger.r1_minus_r0_return
    ledger.to_parquet(OUT/'outcome_convention_ledger.parquet',index=False)
    outcome=ledger.groupby('eom').agg(changed_returns=('r1_minus_r0_return',lambda x:int(x.abs().gt(1e-12).sum())),
        fixed_weights_R1_minus_R0_gross=('fixed_weights_outcome_contribution','sum')).reset_index()
    outcome.to_csv(OUT/'outcome_only_diagnostic.csv',index=False)
    # Prefix comparison checks that extending history never changes an earlier decision.
    pilot=ROOT/'output/step5_pilot_1981_1982'
    for name, date_col in [('monthly_performance.parquet','decision_eom'),('PolicyDecision.parquet','eom'),('AimArtifact.parquet','eom'),('MarketSnapshot.parquet','decision_eom')]:
        old=pd.read_parquet(pilot/name); new=pd.read_parquet(RUN/name);new=new[new[date_col]<=old[date_col].max()].reset_index(drop=True)
        try: pd.testing.assert_frame_equal(old.reset_index(drop=True),new,check_exact=True); ok=True
        except AssertionError: ok=False
        boolean('Unchanged complete pilot prefix: '+name,ok)
    # Independent original R coefficients, all candidate validation scores and selections.
    validations=pd.read_parquet(PREP/'validation_scores.parquet')
    rv=pd.read_csv(ref/'r_validation.csv',parse_dates=['eom','eom_ret'],float_precision='round_trip')
    v=validations.merge(rv,left_on=['decision_eom','g_index','p','lambda_index'],right_on=['eom','g_index','p','lambda_index'],validate='one_to_one')
    boolean('Complete original-grid validation membership',len(v)==len(validations)==len(rv)==484800)
    compare('All validation objectives versus original R',v.objective,v.obj,atol=1e-8,rtol=1e-6)
    compare('All cumulative objectives versus original R',v.cumulative_objective,v.cum_obj,atol=1e-8,rtol=1e-6)
    selections=pd.read_parquet(PREP/'annual_selection.parquet');rs=pd.read_csv(ref/'r_annual_selection.csv',parse_dates=['eom_ret'],float_precision='round_trip')
    sel=selections.merge(rs,left_on='return_eom',right_on='eom_ret',validate='one_to_one',suffixes=('_py','_r'))
    boolean('All 50 validation-year selections match original R',len(sel)==50 and all(sel[f+'_py'].eq(sel[f+'_r']).all() for f in ['g_index','p','lambda_index']))
    for gi in [0,1]:
        for year in range(1971,2021):
            for feature_count in SETTINGS['settings']['pf_ml']['p_vec']:
                with np.load(PREP/f'coefficients/g{gi}/{year}_p{feature_count}.npz') as z:
                    coefs=z['coefficients'];cutoff=str(z['training_cutoff']);n=int(z['training_months'])
                original=np.fromfile(ref/f'fit/g{gi}/r_coef_{year}_p{feature_count}.bin',dtype='<f8').reshape(coefs.shape)
                compare(f'{year} g{gi} p{feature_count}: all original R coefficients',coefs,original,atol=1e-8,rtol=1e-6)
                boolean(f'{year} g{gi} p{feature_count}: matured training window',cutoff==f'{year-1}-11-30' and n==len(MOMENT_DATES[MOMENT_DATES<=pd.Timestamp(cutoff)]))
    choices=[]
    for year in range(1981,2021):
        block=aims[aims.fit_year.eq(year)]; row=block.iloc[0];gi=int(row.g_index);pc=int(row.p);li=int(row.lambda_index)
        chosen=selections[selections.return_eom.dt.year.eq(year-1)].iloc[0]
        boolean(f'{year}: prior-December choice controls every monthly aim',block[['g_index','p','lambda_index']].drop_duplicates().shape[0]==1
            and gi==chosen.g_index and pc==chosen.p and li==chosen.lambda_index and block.selection_return_year.eq(year-1).all())
        with np.load(PREP/f'coefficients/g{gi}/{year}_p{pc}.npz') as z:cutoff=str(z['training_cutoff']);n=int(z['training_months'])
        choices.append({'return_year':year,'selection_return_year':year-1,'g_index':gi,
            'g':SETTINGS['settings']['pf_ml']['g_vec'][gi],'p':pc,'lambda_index':li,
            'lambda':SETTINGS['settings']['pf_ml']['l_vec'][li],'training_months':n,'training_decision_cutoff':cutoff})
    pd.DataFrame(choices).to_csv(OUT/'annual_model_choices.csv',index=False)
    meta=json.loads((ref/'metadata.json').read_text())
    for gi in [0,1]:
        for ds in meta['sample_input_dates']:
            with np.load(PREP/f'moments/g{gi}/{ds}.npz') as z:
                for pyname,rname in [('signal_t','signal'),('denom','denom'),('r_tilde','rtilde')]:
                    a=z[pyname]; b=np.fromfile(ref/f'sample_inputs/r_g{gi}_{ds}_{rname}.bin',dtype='<f8').reshape(a.shape)
                    compare(f'{ds} g{gi}: sampled actual R {pyname}',a,b,atol=1e-8,rtol=1e-6)
    ra=pd.read_csv(ref/'r_aims.csv',parse_dates=['eom'],float_precision='round_trip')
    rw=pd.read_csv(ref/'r_weights.csv',parse_dates=['eom'],float_precision='round_trip')
    rm=pd.read_csv(ref/'r_performance.csv',parse_dates=['eom_ret'],float_precision='round_trip')
    a=aims.merge(ra,on=['id','eom'],validate='one_to_one',suffixes=('_py','_r'))
    boolean('Complete R aim membership',len(a)==len(aims)==len(ra))
    compare('All 480 months of R aims',a.aim,a.w_aim,atol=1e-8,rtol=1e-6)
    boolean('Every R target model choice matches',all(a[f+'_py'].eq(a[f+'_r']).all() for f in ['g_index','p','lambda_index']))
    w=p.merge(rw,on=['id','eom'],validate='one_to_one');boolean('Complete R holdings membership',len(w)==len(p)==len(rw))
    compare('All R pretrade holdings',w.pretrade,w.w_start,atol=1e-8,rtol=1e-6)
    compare('All R posttrade holdings',w.weight,w.w,atol=1e-8,rtol=1e-6)
    r=m.merge(rm,left_on='return_eom',right_on='eom_ret',validate='one_to_one',suffixes=('_py','_r'))
    boolean('Full 480-month R return membership',len(r)==480==len(rm))
    for pyname,rname in [('gross_excess','r'),('impact_cost','tc'),('turnover_py','turnover_r'),('gross_exposure','inv')]:
        compare('Full original R path: '+pyname,r[pyname],r[rname])
    independent=statistics(m); summary=json.loads((RUN/'summary.json').read_text())
    for key,value in summary.items():compare('Independent reported '+key,value,independent[key],atol=1e-12)
    annual=[{'year':int(y),**statistics(g)} for y,g in m.groupby(m.return_eom.dt.year)]
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False)
    periods=[]
    for label,(first,last) in {'Full baseline':('1981-01','2020-12'),'Development':('1981-01','2000-12'),
            'Selection':('2001-01','2010-12'),'Retrospective evaluation':('2011-01','2020-12')}.items():
        g=m[(m.return_eom>=pd.Timestamp(first))&(m.return_eom<=pd.Timestamp(last)+pd.offsets.MonthEnd(0))]
        periods.append({'period':label,'first_return':str(g.return_eom.min().date()),'last_return':str(g.return_eom.max().date()),**statistics(g)})
    pd.DataFrame(periods).to_csv(OUT/'subperiod_metrics.csv',index=False)
    details=m.merge(direct[['decision_eom','net_exposure','short_exposure','largest_absolute_position','eligible_stocks','wealth_usd']],on='decision_eom',validate='one_to_one')
    details['cumulative_sum_net_excess']=details.net_excess.cumsum();details.to_csv(OUT/'monthly_review.csv',index=False)
    original_manifest=json.loads((RUN/'ExperimentManifest.json').read_text())
    boolean('Original run hashes unchanged',all(sha(ROOT/p)==h for p,h in original_manifest['outputs_sha256'].items()))
    for stage in ['prepare','run','export','reference_fit','reference_path']:
        attempts=[json.loads(path.read_text()) for path in OUT.glob(f'{stage}_attempt*_resources.json')]
        boolean(stage+': completed measured job within limits',bool(attempts) and any(x['exit_code']==0 and x['within_limits'] for x in attempts)
            and all(x['within_limits'] for x in attempts),{'attempts':len(attempts)})
    diagnostics={'positions':len(p),'securities':int(p.id.nunique()),'entries':len(entrants),'exits':len(exits),
        'terminal_source_exits':int(ledger[ledger.eom.eq(DATES[-1])].exit_after_return.sum()),
        'imputed_next_returns':int(ledger.imputed_ld1.sum()),'changed_next_returns':int(ledger.r1_minus_r0_return.abs().gt(1e-12).sum()),
        'fixed_R1_weights_outcome_only_annualized_effect':float(12*outcome.fixed_weights_R1_minus_R0_gross.mean()),
        'outcome_effect_scope':'R1 weights fixed; not the performance of a refitted R0 strategy',
        'mean_net_exposure':float(direct.net_exposure.mean()),'mean_short_exposure':float(direct.short_exposure.mean()),
        'universe_min':int(direct.eligible_stocks.min()),'universe_max':int(direct.eligible_stocks.max()),
        'first_rebalance_cost':float(m.impact_cost.iloc[0]),'first_cost_share':float(m.impact_cost.iloc[0]/m.impact_cost.sum()),
        'feature_counts_selected':sorted(set(int(x['p']) for x in choices)),
        'annual_choices_at_pmax':sum(x['p']==max(SETTINGS['settings']['pf_ml']['p_vec']) for x in choices),
        'worst_month':str(m.loc[m.net_excess.idxmin(),'return_eom'].date()),
        'best_month':str(m.loc[m.net_excess.idxmax(),'return_eom'].date()),
        'full_original_vintage_reproduced':False,'confidence_extension_tested':False}
    # Published numbers are rounded table entries, not an exact reference time series.
    table=[('Annual gross excess return','annual_mean_gross_excess',.15,.005),
        ('Annual gross volatility','annual_gross_sample_volatility',.11,.005),
        ('Gross Sharpe','gross_sample_sharpe',1.38,.005),('Annual impact cost','annualized_mean_impact_cost',.006,.0005),
        ('Annual net excess return','annual_mean_net_excess',.15,.005),
        ('Net / gross-volatility Sharpe','source_net_over_gross_sharpe',1.33,.005),
        ('Source gross-variance utility','source_gross_sample_variance_utility',.086,.0005),
        ('Mean monthly turnover','mean_monthly_turnover',.25,.005),('Mean gross exposure','mean_gross_exposure',3.,.005)]
    reconciliation=pd.DataFrame([{'measure':label,'field':key,'published_displayed':value,'R1_reconstruction':independent[key],
        'difference_from_displayed':independent[key]-value,'published_rounding_half_unit':precision,
        'inside_displayed_rounding_interval':abs(independent[key]-value)<=precision}
        for label,key,value,precision in table])
    reconciliation.to_csv(OUT/'published_table2_comparison.csv',index=False)
    save_json(OUT/'review_summary.json',{'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
        'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'checks':checks,
        'independent_metrics':independent,'annual_metrics':annual,'subperiod_metrics':periods,'diagnostics':diagnostics,
        'published_source':{'file':'output/step2_20260910/paper_from_user.pdf','sha256':sha(ROOT/'output/step2_20260910/paper_from_user.pdf'),
            'table':2,'pdf_page_one_based':28,'paper_page':28,'printed_values_are_rounded':True},
        'input_rebuild_coverage':'81 prespecified decision months × 2 bandwidths, not all estimation months',
        'full_R_fit_and_holdings_coverage':True})
    print(f'Full baseline review: {sum(c["passed"] for c in checks)}/{len(checks)} checks pass.',flush=True)
    for c in checks:
        if not c['passed']:print(c,flush=True)
    if not all(c['passed'] for c in checks):raise SystemExit(1)


if __name__=='__main__': main()
