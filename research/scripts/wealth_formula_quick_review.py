"""Bounded diagnostic review: full independent accounting; repeated full R fit/path deferred."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wealth_formula_study import ROOT, OUT, PREP, REF, OLD_RUN, OLD, DATA, DATES, MOMENT_DATES, SAMPLES, SETTINGS, sha, save_json, verify, wealth
import numpy as np
import pandas as pd
from step6_review import statistics

checks=[]
def boolean(name,ok,**detail):
    checks.append(dict(check=name,passed=bool(ok),**detail))
def compare(name,a,b,atol=1e-8,rtol=1e-6):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    same=a.shape==b.shape
    boolean(name,same and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol),
        values=a.size,max_abs=float(np.max(abs(a-b))) if same and a.size else None,atol=atol,rtol=rtol)

def main():
    start=json.loads((OUT/'start.json').read_text());verify(start['sources']);verify(start['protected'])
    for stage in ['preflight','neutral','moments','fit','path']:
        cp=json.loads((OUT/f'checkpoints/{stage}.json').read_text());verify(cp['artifact_sha256'])
    boolean('All completed stage hashes and protected decisions unchanged',True)
    verify(json.loads((ROOT/'output/step6_20260911/acceptance_manifest.json').read_text())['artifact_sha256'])
    boolean('Accepted original baseline remains unchanged',True)
    m=pd.read_parquet(OUT/'monthly.parquet');p=pd.read_parquet(OUT/'positions.parquet');aims=pd.read_parquet(OUT/'aims.parquet');exits=pd.read_parquet(OUT/'exits.parquet')
    boolean('Exactly 480 consecutive realized months',list(m.decision_eom)==list(DATES) and list(m.return_eom)==list(DATES+pd.offsets.MonthEnd(1)))
    boolean('Unique positions and aims',not p.duplicated(['id','eom']).any() and not aims.duplicated(['id','eom']).any())
    raw=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=['id','eom','valid','me','ret_ld1','tr_ld1','mu_ld1','lambda_diagonal','wealth_usd'],filters=[('eom','>=',DATES[0]),('eom','<=',DATES[-1])]).query('valid').copy()
    w=wealth().set_index('eom');raw['wealth_usd']=raw.eom.map(w.author_formula_wealth)
    boolean('All original universe memberships retained',set(zip(p.id,p.eom))==set(zip(raw.id,raw.eom))==set(zip(aims.id,aims.eom)))
    combo=p.merge(raw,on=['id','eom'],validate='one_to_one');by={d:g.sort_values('id') for d,g in combo.groupby('eom')}
    rows=[];expected_exits=[];entry_count=0
    for i,d in enumerate(DATES):
        b=by[d];h=b.pretrade.to_numpy();post=b.weight.to_numpy();target=b.aim.to_numpy();trade=post-h
        with np.load(PREP/f'matrices/{d:%Y-%m-%d}.npz') as z:
            boolean(f'{d}: risk/matrix membership',np.array_equal(z['ids'],b.id))
            compare(f'{d}: unchanged trading equation',post,h+(np.eye(len(h))-z['m'])@(target-h))
        if i==0:inherited=b.me.to_numpy()/b.me.sum()
        else:
            prev=by[DATES[i-1]].set_index('id')
            inherited=(prev.weight*(1+prev.tr_ld1)/(1+prev.mu_ld1)).reindex(b.id).fillna(0).to_numpy()
            entrants=b[~b.id.isin(prev.index)];entry_count+=len(entrants)
            boolean(f'{d}: entrants start at zero',entrants.pretrade.eq(0).all())
        compare(f'{d}: inherited holdings',h,inherited)
        compare(f'{d}: trades',b.trade,trade)
        drift=post*(1+b.tr_ld1.to_numpy())/(1+b.mu_ld1.to_numpy())
        compare(f'{d}: unchanged market-relative drift',b.drifted_weight,drift)
        if i<len(DATES)-1:
            dropped=~b.id.isin(by[DATES[i+1]].id).to_numpy()
            expected_exits.extend(dict(id=int(j),decision_eom=d,omitted_next_weight=float(v)) for j,v in zip(b.id.to_numpy()[dropped],drift[dropped]))
        rows.append(dict(decision_eom=d,gross_excess=float(sum(post*b.ret_ld1)),impact_cost=float(b.wealth_usd.iloc[0]/2*sum(b.lambda_diagonal*trade**2)),turnover=float(sum(abs(trade))),gross_exposure=float(sum(abs(post))),wealth_usd=float(b.wealth_usd.iloc[0])))
    independent=pd.DataFrame(rows);independent.to_parquet(OUT/'independent_monthly.parquet',index=False)
    for key in ['gross_excess','impact_cost','turnover','gross_exposure']:compare('Independent accounting '+key,m[key],independent[key],rtol=0)
    compare('Costs charged exactly once',m.net_excess,m.gross_excess-m.impact_cost,atol=1e-12,rtol=0)
    boolean('Positive first trade cost and nonnegative costs',m.impact_cost.iloc[0]>0 and m.impact_cost.ge(0).all())
    xx=pd.DataFrame(expected_exits).merge(exits,on=['id','decision_eom'],validate='one_to_one',suffixes=('_check','_run'))
    boolean('All exits retained with source zero exit-cost convention',len(xx)==len(expected_exits)==len(exits) and exits.source_exit_trade_cost.eq(0).all())
    compare('Exit weights',xx.omitted_next_weight_check,xx.omitted_next_weight_run)
    selections=pd.read_parquet(PREP/'annual_selection.parquet')
    validation=pd.read_parquet(PREP/'validation_scores.parquet')
    boolean('Complete original-grid validation rows',len(validation)==484800 and not validation.duplicated(['decision_eom','g_index','p','lambda_index']).any())
    boolean('Finite validation scores',np.isfinite(validation[['objective','cumulative_objective']]).all().all())
    boolean('All 50 annual choices present',len(selections)==50)
    for year,block in aims.groupby('fit_year'):
        chosen=selections[selections.return_eom.dt.year.eq(year-1)].iloc[0]
        boolean(f'{year}: prior-year selection only',block.selection_return_year.eq(year-1).all() and all(block[k].eq(chosen[k]).all() for k in ['g_index','p','lambda_index']))
        with np.load(PREP/f'coefficients/g{int(chosen.g_index)}/{year}_p{int(chosen.p)}.npz') as z:
            boolean(f'{year}: matured coefficient training',str(z['training_cutoff'])==f'{year-1}-11-30' and int(z['training_months'])==sum(MOMENT_DATES<=pd.Timestamp(year-1,11,30)))
    bound=json.loads((OUT/'memory_recovery/bounded_sum_checks.json').read_text())
    exact=json.loads((OUT/'memory_recovery/fit_equivalence.json').read_text())
    boolean('Bounded sum and all retained coefficient blocks agree exactly',bound['passed'] and exact['passed'])
    old=pd.read_parquet(OLD_RUN/'monthly_performance.parquet');baseline=statistics(old);alternative=statistics(m)
    old_report=json.loads((ROOT/'output/step6_20260911/review_summary.json').read_text())['independent_metrics']
    for k,value in old_report.items():compare('Corrected baseline metric '+k,baseline[k],value,atol=1e-12,rtol=0)
    key='source_gross_sample_variance_utility';delta=alternative[key]-baseline[key];gap=.086-baseline[key]
    effects=dict(utility_difference=delta,baseline_to_rounded_paper_gap=gap,remaining_gap=.086-alternative[key],fraction_of_rounded_gap_closed=delta/gap,
        gross_return_effect=alternative['annual_mean_gross_excess']-baseline['annual_mean_gross_excess'],cost_effect=baseline['annualized_mean_impact_cost']-alternative['annualized_mean_impact_cost'],
        risk_penalty_effect=5*(baseline['annual_gross_sample_volatility']**2-alternative['annual_gross_sample_volatility']**2))
    compare('Utility decomposition reconciles',delta,sum(effects[k] for k in ['gross_return_effect','cost_effect','risk_penalty_effect']),atol=1e-12,rtol=0)
    table=pd.DataFrame([dict(arm='corrected_reciprocal',**baseline),dict(arm='author_multiply_1_minus_return',**alternative)])
    table.to_csv(OUT/'comparison.csv',index=False)
    old_selection=pd.read_parquet(OLD/'annual_selection.parquet');sel=old_selection.merge(selections,on='return_eom',suffixes=('_old','_new'),validate='one_to_one')
    sel=sel[sel.return_eom.dt.year.between(1980,2019)]
    changed=int((~np.column_stack([sel[k+'_old'].eq(sel[k+'_new']) for k in ['g_index','p','lambda_index']]).all(axis=1)).sum())
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,checks_passed=sum(c['passed'] for c in checks),checks_total=len(checks),
        baseline=baseline,author_formula=alternative,effects=effects,changed_traded_year_model_choices=changed,entry_count=entry_count,
        source_input_rebuild_dates=[],source_full_grid_and_holdings=False,only_changed_input='backward wealth recurrence',
        scientific_baseline_changed=False,paper_vintage_reproduced=False,reviewer_source_sha256=sha(Path(__file__)))
    save_json(OUT/'quick_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)
    if result['status']!='PASS':
        for c in checks:
            if not c['passed']:print(c,flush=True)
        raise SystemExit(1)

if __name__=='__main__':main()
