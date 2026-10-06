"""Independent complete-pilot numerical, chronology, accounting and reporting audit."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import ROOT,DATA,PREP,PROTOCOL,SETTINGS,save_json,sha
from baseline.acceptance import verify_baseline
from baseline.wealth import validate
OUT=ROOT/'output/step5_20260911';RUN=ROOT/'output/step5_pilot_1981_1982';REF=OUT/'reference'


def stats(frame):
    gross=frame.gross_excess.to_numpy();cost=frame.impact_cost.to_numpy();net=gross-cost;n=len(net);gamma=PROTOCOL['economic']['risk_aversion']
    mean=float(sum(net)/n);variance=float(sum((net-mean)**2)/n)
    gross_var=float(sum((gross-gross.mean())**2)/(n-1))
    return {'months':n,'annual_mean_net_excess':12*mean,'annual_net_volatility':np.sqrt(12*variance),'net_population_utility':12*(mean-gamma/2*variance),'net_sharpe':np.sqrt(12)*mean/np.sqrt(variance),'source_gross_sample_variance_utility':12*(mean-gamma/2*gross_var),'source_net_over_gross_sharpe':np.sqrt(12)*mean/np.sqrt(gross_var),'annual_mean_gross_excess':12*gross.mean(),'annualized_mean_impact_cost':12*cost.mean(),'mean_monthly_turnover':frame.turnover.mean(),'mean_gross_exposure':frame.gross_exposure.mean(),'minimum_monthly_net_excess':net.min(),'maximum_monthly_net_excess':net.max(),'sum_monthly_net_excess':net.sum()}


def main():
    checks=[]
    def boolean(name,ok,evidence=None):checks.append({'check':name,'passed':bool(ok),'evidence':evidence})
    def compare(name,a,b,atol=1e-8,rtol=0):
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        ok=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)
        checks.append({'check':name,'passed':bool(ok),'values':int(a.size),'max_abs_difference':float(np.max(abs(a-b))) if a.shape==b.shape and a.size else None,'atol':atol,'rtol':rtol})
    baseline=verify_baseline()
    cache=json.loads((OUT/'reference_cache_receipt.json').read_text())
    boolean('Reused R input calculations retain their recorded hashes',len(cache['files'])==72 and all(sha(ROOT/p)==h for p,h in cache['files'].items()),{'cached_files':len(cache['files'])})
    rstatus=json.loads((REF/'status.json').read_text())
    boolean('Actual pinned R pilot completed',rstatus['status']=='ACTUAL_PINNED_R_PILOT_COMPLETE' and rstatus['months']==24,rstatus)
    m=pd.read_parquet(RUN/'monthly_performance.parquet');p=pd.read_parquet(RUN/'PolicyDecision.parquet');aims=pd.read_parquet(RUN/'AimArtifact.parquet');exits=pd.read_parquet(RUN/'source_exit_ledger.parquet');snapshot=pd.read_parquet(RUN/'MarketSnapshot.parquet')
    dates=pd.date_range('1980-12-31','1982-11-30',freq='ME');returns=dates+pd.offsets.MonthEnd(1)
    boolean('Exact 24 decision and return months',list(m.decision_eom)==list(dates) and list(m.return_eom)==list(returns),{'first_return':str(returns[0].date()),'last_return':str(returns[-1].date())})
    boolean('Unique stock/month records',not p.duplicated(['id','eom']).any() and not aims.duplicated(['id','eom']).any())
    columns=['id','eom','valid','me','ret_ld1','tr_ld1','mu_ld1','lambda_diagonal','wealth_usd']
    raw=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=columns,filters=[('eom','>=',dates[0]),('eom','<=',dates[-1])]);active=raw[raw.valid].copy()
    boolean('Market snapshots cover each exact decision and return month',list(snapshot.decision_eom)==list(dates) and list(snapshot.return_eom)==list(returns))
    for d in dates:
        s=snapshot[snapshot.decision_eom.eq(d)].iloc[0];x=active[active.eom.eq(d)]
        boolean(f'{d:%Y-%m}: market snapshot matches accepted economic inputs',s.eligible_stocks==len(x) and s.wealth_usd==x.wealth_usd.iloc[0] and s.market_total_growth==x.mu_ld1.iloc[0] and s.risk_sha256==sha(PREP/f'risk/{d:%Y-%m-%d}.npz'))
    boolean('Exact accepted security/month membership',set(zip(p.id,p.eom))==set(zip(active.id,active.eom)),{'positions':len(p),'securities':p.id.nunique()})
    comb=p.merge(active,on=['id','eom'],validate='one_to_one');boolean('All position/economic inputs finite',np.isfinite(comb.select_dtypes('number')).all().all())
    combo=p.merge(aims,on=['id','eom'],validate='one_to_one',suffixes=('_policy','_artifact'))
    compare('Policy targets match accepted aim artifact',combo.aim_policy,combo.aim_artifact,atol=1e-8,rtol=1e-6)
    compare('Aim artifacts match locked preflight targets',aims.sort_values(['eom','id']).aim,pd.read_parquet(PREP/'pilot_aims_preflight.parquet').sort_values(['eom','id']).aim,atol=0)
    rows=[];entry_rows=[];continuation=[];trade_errors=[];drift_errors=[]
    for i,d in enumerate(dates):
        x=comb[comb.eom.eq(d)].sort_values('id');ids=x.id.to_numpy();w=x.weight.to_numpy();h=x.pretrade.to_numpy();a=x.aim.to_numpy();q=w-h
        with np.load(PREP/f'matrices/{d:%Y-%m-%d}.npz') as z:matrix=z['m'];mids=z['ids']
        boolean(f'{d:%Y-%m}: cached matrix security alignment',np.array_equal(ids,mids))
        compare(f'{d:%Y-%m}: independent trade rule',w,h+(np.eye(len(h))-matrix)@(a-h),atol=1e-8,rtol=1e-6)
        if i==0:expected=x.me.to_numpy()/x.me.sum()
        else:
            prev=comb[comb.eom.eq(dates[i-1])].set_index('id')
            expected=(prev.weight*(1+prev.tr_ld1)/(1+prev.mu_ld1)).reindex(ids).fillna(0).to_numpy()
            new=~x.id.isin(prev.index)
            entry_rows.append(x.loc[new,['id','eom','pretrade']])
        compare(f'{d:%Y-%m}: independent inherited holdings',h,expected,atol=1e-8,rtol=1e-6)
        compare(f'{d:%Y-%m}: recorded trades',x.trade,q,atol=1e-8,rtol=1e-6)
        expected_drift=w*(1+x.tr_ld1.to_numpy())/(1+x.mu_ld1.to_numpy())
        compare(f'{d:%Y-%m}: total-return wealth drift',x.drifted_weight,expected_drift,atol=1e-8,rtol=1e-6)
        wealth=x.wealth_usd.iloc[0]
        rows.append({'decision_eom':d,'gross_excess':float(np.sum(w*x.ret_ld1)),'impact_cost':float(wealth/2*np.sum(x.lambda_diagonal*q**2)),'turnover':float(np.sum(abs(q))),'gross_exposure':float(np.sum(abs(w))),'net_exposure':float(np.sum(w)),'short_exposure':float(-np.sum(w[w<0])),'largest_absolute_position':float(np.max(abs(w))),'eligible_stocks':len(x),'wealth_usd':wealth})
    direct=pd.DataFrame(rows);merged=m.merge(direct,on='decision_eom',validate='one_to_one',suffixes=('_run','_recomputed'))
    for field in ['gross_excess','impact_cost','turnover','gross_exposure']:compare('Independent monthly '+field,merged[field+'_run'],merged[field+'_recomputed'])
    compare('Net return deducts cost exactly once',m.net_excess,m.gross_excess-m.impact_cost,atol=1e-12)
    boolean('Nonnegative costs and first rebalance charged',m.impact_cost.ge(0).all() and m.impact_cost.iloc[0]>0 and m.turnover.iloc[0]>0)
    entries=pd.concat(entry_rows,ignore_index=True);boolean('Every entrant starts at zero',entries.pretrade.eq(0).all(),{'entrants':len(entries)});entries.to_parquet(OUT/'entry_ledger.parquet',index=False)
    ledger=pd.read_parquet(DATA/'outcome_and_exit_ledger.parquet');ledger=ledger[ledger.eom.isin(dates)].copy()
    expected_exits=ledger[ledger.eom.isin(dates[:-1])&ledger.exit_after_return]
    boolean('All in-pilot exits recorded with source zero exit trade cost',set(zip(expected_exits.id,expected_exits.eom))==set(zip(exits.id,exits.decision_eom)) and exits.source_exit_trade_cost.eq(0).all(),{'exits':len(exits)})
    terminal=ledger[ledger.eom.eq(dates[-1])&ledger.exit_after_return]
    boolean('Terminal holdings retained without invented liquidation',len(p[p.eom.eq(dates[-1])])==len(active[active.eom.eq(dates[-1])]),{'source_exits_after_final_return':len(terminal),'terminal_policy':'Retain final drift; no additional trade beyond pilot horizon'})
    enriched=ledger.merge(p[['id','eom','weight']],on=['id','eom'],validate='one_to_one')
    enriched['r1_minus_r0_return']=enriched.ret_ld1-enriched.ret_ld1_r0
    enriched['fixed_weights_outcome_contribution']=enriched.weight*enriched.r1_minus_r0_return
    enriched.to_parquet(OUT/'outcome_convention_ledger.parquet',index=False)
    outcome_diff=enriched.groupby('eom').agg(changed_returns=('r1_minus_r0_return',lambda x:int(x.abs().gt(1e-12).sum())),fixed_weights_R1_minus_R0_gross=('fixed_weights_outcome_contribution','sum')).reset_index()
    outcome_diff.to_csv(OUT/'outcome_only_diagnostic.csv',index=False)
    boolean('No hidden pilot return imputations',len(ledger)==len(p) and not ledger.imputed_ld1.any(),{'imputed_next_excess_returns':int(ledger.imputed_ld1.sum()),'included_value_corrections':int(enriched.r1_minus_r0_return.abs().gt(1e-12).sum())})
    refaim=pd.read_csv(REF/'r_aims.csv',parse_dates=['eom'],float_precision='round_trip');refw=pd.read_csv(REF/'r_weights.csv',parse_dates=['eom'],float_precision='round_trip');refr=pd.read_csv(REF/'r_performance.csv',parse_dates=['eom_ret'],float_precision='round_trip')
    ar=aims.merge(refaim,on=['id','eom'],validate='one_to_one',suffixes=('_py','_r'))
    boolean('Complete actual R target membership',len(ar)==len(aims)==len(refaim))
    compare('All selected targets versus actual R aims',ar.aim,ar.w_aim,atol=1e-8,rtol=1e-6)
    boolean('Actual R model selections match',all(ar[f+'_py'].eq(ar[f+'_r']).all() for f in ['g_index','p','lambda_index']))
    wr=p.merge(refw,on=['id','eom'],validate='one_to_one');boolean('Complete actual R holdings membership',len(wr)==len(p)==len(refw))
    compare('All pretrade weights versus R',wr.pretrade,wr.w_start,atol=1e-8,rtol=1e-6);compare('All posttrade weights versus R',wr.weight,wr.w,atol=1e-8,rtol=1e-6)
    mr=m.merge(refr,left_on='return_eom',right_on='eom_ret',validate='one_to_one',suffixes=('_py','_r'))
    for x,y in [('gross_excess','r'),('impact_cost','tc'),('gross_exposure','inv'),('turnover_py','turnover_r')]:compare('Full pilot R performance: '+x,mr[x],mr[y])
    for gi in [0,1]:
        for d in dates:
            with np.load(PREP/f'moments/g{gi}/{d:%Y-%m-%d}.npz') as z:
                for pf,rf in [('signal_t','signal'),('denom','denom'),('r_tilde','rtilde')]:
                    x=z[pf];y=np.fromfile(REF/f'r_g{gi}_{d:%Y-%m-%d}_{rf}.bin',dtype='<f8').reshape(x.shape)
                    compare(f'{d:%Y-%m}: g{gi} full R {pf}',x,y,atol=1e-8,rtol=1e-6)
    summary=json.loads((RUN/'summary.json').read_text());independent=stats(m)
    for key in summary:compare('Independent summary: '+key,summary[key],independent[key],atol=1e-12)
    annual=[]
    for year,group in m.groupby(m.return_eom.dt.year):annual.append({'year':int(year),**stats(group)})
    pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False)
    choice=pd.read_parquet(PREP/'annual_selection.parquet');selection_rows=[]
    for year in [1981,1982]:
        block=aims[aims.fit_year.eq(year)];g=int(block.g_index.iloc[0]);pcount=int(block.p.iloc[0]);li=int(block.lambda_index.iloc[0]);ch=choice[choice.return_eom.dt.year.eq(year-1)].iloc[0]
        with np.load(PREP/f'coefficients/g{g}/{year}_p{pcount}.npz') as z:cutoff=str(z['training_cutoff']);ntrain=int(z['training_months'])
        boolean(f'{year}: correct prior-December selection and matured fit',block.selection_return_year.eq(year-1).all() and ch.g_index==g and ch.p==pcount and ch.lambda_index==li and cutoff==f'{year-1}-11-30',{'training_months':ntrain,'training_decision_cutoff':cutoff})
        selection_rows.append({'return_year':year,'selection_return_year':year-1,'g':SETTINGS['settings']['pf_ml']['g_vec'][g],'p':pcount,'lambda_index':li,'lambda':SETTINGS['settings']['pf_ml']['l_vec'][li],'training_months':ntrain,'training_decision_cutoff':cutoff})
    pd.DataFrame(selection_rows).to_csv(OUT/'annual_model_choices.csv',index=False)
    wealth=validate(pd.read_parquet(DATA/'market_wealth.parquet'));boolean('Full-anchor wealth remains valid',wealth['anchor_usd']==1e10,wealth)
    manifest=json.loads((RUN/'ExperimentManifest.json').read_text())
    boolean('Original run output hashes still match',all(sha(ROOT/p)==h for p,h in manifest['outputs_sha256'].items()),{'outputs':len(manifest['outputs_sha256'])})
    boolean('Run uses exactly the accepted Step 4 version',manifest['preflight']['baseline']['acceptance_sha256']==baseline['acceptance_sha256'])
    protected=json.loads((OUT/'start.json').read_text())['protected'];boolean('Protected scientific files unchanged',all(sha(ROOT/p)==h for p,h in protected.items()))
    for stage in ['pilot','reference']:
        record=json.loads((OUT/f'{stage}_resources.json').read_text());boolean(stage+' execution stays within approved resource limits',record['within_limits'],record)
    first_attempt=json.loads((OUT/'reference_attempt_1/reference_resources.json').read_text());limits=first_attempt['limits']
    boolean('Preserved first R attempt stayed within resource boundaries before import failure',first_attempt['aborted'] is None and max(first_attempt['sampled_process_group_plus_supervisor_peak_gib'],first_attempt['os_peak_main_job_gib'])<=limits['aggregate_memory_gib'] and first_attempt['minimum_free_disk_gib']>=limits['minimum_free_disk_gib'] and first_attempt['maximum_fresh_storage_gib']<=limits['new_storage_gib'],{'exit_code':first_attempt['exit_code'],'failure':'Decimal lambda import; repaired by exact saved grid index'})
    details=m.merge(direct[['decision_eom','net_exposure','short_exposure','largest_absolute_position','eligible_stocks','wealth_usd']],on='decision_eom',validate='one_to_one');details['cumulative_sum_net_excess']=details.net_excess.cumsum();details.to_csv(OUT/'monthly_review.csv',index=False)
    diagnostics={'first_month_impact_cost':float(m.impact_cost.iloc[0]),'first_cost_share_of_total':float(m.impact_cost.iloc[0]/m.impact_cost.sum()),'total_monthly_cost_sum':float(m.impact_cost.sum()),'p_at_original_grid_maximum_both_years':all(row['p']==max(SETTINGS['settings']['pf_ml']['p_vec']) for row in selection_rows),'original_grid_retained':True,'entrants':len(entries),'exits':len(exits),'terminal_source_exits':len(terminal),'securities':p.id.nunique(),'positions':len(p),'universe_min':int(direct.eligible_stocks.min()),'universe_max':int(direct.eligible_stocks.max()),'mean_net_exposure':float(direct.net_exposure.mean()),'mean_short_exposure':float(direct.short_exposure.mean()),'included_value_rows':int(enriched.r1_minus_r0_return.abs().gt(1e-12).sum()),'outcome_only_diagnostic_annualized_R1_minus_R0':float(12*outcome_diff.fixed_weights_R1_minus_R0_gross.mean()),'outcome_only_interpretation':'Holds R1 weights fixed and substitutes R0 next excess returns; not a refitted or independently executed R0 strategy','full_R0_performance_effect_measured':False,'published_statistics_reproduced':False,'confidence_extension_tested':False}
    save_json(OUT/'review_summary.json',{'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','independent_metrics':independent,'annual_metrics':annual,'diagnostics':diagnostics,'checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),'checks':checks})
    print(f'Pilot review: {sum(x["passed"] for x in checks)}/{len(checks)} checks pass.',flush=True)
    for item in checks:
        if not item['passed']:print(item,flush=True)
    if not all(x['passed'] for x in checks):raise SystemExit(1)

if __name__=='__main__':main()
