"""Verify complete comparison paths, frozen choices and economic reporting."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from comparisons.io import *
from comparisons.markets import Markets
from comparisons.core import metrics,select
from learner.core import VARIANTS

def main():
    begin('review');source=Markets();checks=[];summaries=[];contrasts=[];max_recurrence=0.
    code={str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('comparisons/*.py'),*ROOT.glob('scripts/step9_*')] if p.is_file() and p.name not in ['step9_document.py','step9_notebook.py','step9_finalize.py']}
    save_json(OUT/'verification_context.json',dict(verification_code_sha256=code))
    def check(name,ok,**details):
        checks.append(dict(check=name,passed=bool(ok),**details))
        if not ok:save_json(OUT/'failed_review_checks.json',checks);raise AssertionError(name)
    for stage in ['extension_review','selection','evaluation','robustness','path_export']:
        item=json.loads((OUT/f'{stage}_manifest.json').read_text())
        for p,h in item['artifact_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed stage output '+p)
        for p,h in json.loads((OUT/f'{stage}_code.json').read_text()).items():
            if sha(ROOT/p)!=h:raise ValueError('Changed producing code '+p)
        check(stage+' input/code integrity',True)
    extension=json.loads((OUT/'extension_review.json').read_text());r=json.loads((OUT/'path_reference/summary.json').read_text());frozen=json.loads((OUT/'frozen_selection.json').read_text())
    check('Independent complete R paths and selection',r['status']=='PASS' and r['monthly_records']==8460 and r['independent_selection'] and r['complete_own_holdings_recursion'])
    rchoices=json.loads((OUT/'path_reference/r_frozen_selection.json').read_text())
    for variant in VARIANTS:check('R control selection '+variant,all(frozen['choices'][variant][k]==v for k,v in rchoices[variant].items()))
    diagnostic=[]
    for period,(first,last) in PERIODS.items():
        folder=PREP/'paths'/period;monthly=pd.read_parquet(folder/'monthly.parquet');saved_summary=pd.read_parquet(folder/'summary.parquet');ledger=pd.read_parquet(folder/'position_exit_ledger.parquet');sledger=pd.read_parquet(folder/'smoothing_exit_ledger.parquet');meta=json.loads((folder/'path_metadata.json').read_text())
        dates=pd.date_range(first,last,freq='ME');ns=39 if period=='selection' else 21
        check(period+' complete unique monthly rows',len(monthly)==len(dates)*ns and not monthly.duplicated(['variant','method','decision_eom']).any())
        check(period+' correct economic calendar',monthly.decision_eom.min()==dates[0] and monthly.decision_eom.max()==dates[-1] and (monthly.decision_eom+pd.offsets.MonthEnd(1)).equals(monthly.return_eom))
        check(period+' no omitted fallback months or infeasible speed',monthly.net_excess.notna().all() and monthly.kappa.ge(monthly.kappa_min).all() and monthly.kappa.le(monthly.kappa_max).all())
        if period!='selection':check(period+' preserved selection hash',meta['selection_sha256']==sha(OUT/'frozen_selection.json'))
        previous=None;previous_ids=previous_growth=previous_target=None;position_exits=smoothing_exits=0
        for d in dates:
            x,y,cap=source.current(d)
            with np.load(folder/f'{d:%Y-%m-%d}.npz') as z:a={k:z[k].copy() for k in z.files}
            ids=a['ids'];assert np.array_equal(ids,x['ids']);h=a['pretrade'];w=a['weight'];q=a['trade'];n=len(ids)
            if previous is None:
                initial=source.baseline[d].set_index('id').loc[ids,'pretrade'].to_numpy()
                check(period+' common archived R1 block initialization',np.array_equal(h,np.repeat(initial[:,None],ns,axis=1)))
            else:
                error=float(np.max(abs(h-previous)));max_recurrence=max(max_recurrence,error);check(f'{period}/{d:%Y-%m-%d} inherited own holdings without resets',error<1e-12)
            table=monthly[monthly.decision_eom.eq(d)].set_index(['variant','method']);table=table.loc[list(zip(a['variants'],a['methods']))]
            check(f'{period}/{d:%Y-%m-%d} actual weights and trades',np.allclose(w-h,q,atol=1e-12) and np.allclose(q,((np.eye(n)-x['m'])@(a['target']-h))*table.kappa.to_numpy(),atol=1e-12))
            costs=.5*np.sum(x['impact'][:,None]*(x['wealth']*q)**2,axis=0)/x['wealth'];gross=y['returns']@w
            check(f'{period}/{d:%Y-%m-%d} independent dollar costs and excess returns',np.allclose(costs,table.impact_cost,atol=1e-12) and np.allclose(gross,table.gross_excess,atol=1e-12) and np.allclose(gross-costs,table.net_excess,atol=1e-12))
            check(f'{period}/{d:%Y-%m-%d} total return drift',np.allclose(a['drifted_weight'],w*y['growth'][:,None],atol=1e-12))
            ix={int(sid):j for j,sid in enumerate(y['next_ids'])};next_h=np.zeros((len(ix),ns));drop=[]
            for j,sid in enumerate(ids):
                if int(sid) in ix:next_h[ix[int(sid)]]=w[j]*y['growth'][j]
                else:drop.append(j)
            check(f'{period}/{d:%Y-%m-%d} zero entrants and ID mapping',np.array_equal(a['next_ids'],y['next_ids']) and np.allclose(next_h,a['next_pretrade'],atol=1e-12));previous=next_h
            rows=ledger[ledger.decision_eom.eq(d)];ok=len(rows)==len(drop)*ns;position_exits+=len(drop)*ns
            cols={(v,m):j for j,(v,m) in enumerate(zip(a['variants'],a['methods']))};where={int(sid):j for j,sid in enumerate(ids)}
            for row in rows.itertuples():ok=ok and row.id not in ix and not row.liquidation_trade_added and np.isclose(row.omitted_drifted_weight,w[where[row.id],cols[row.variant,row.method]]*y['growth'][where[row.id]],atol=1e-12)
            check(f'{period}/{d:%Y-%m-%d} complete position omission ledger',ok)
            smoothcols=[j for j,s in enumerate(meta['columns']) if s['policy']=='B3'];rows=sledger[sledger.decision_eom.eq(d)]
            dropped_previous=[] if previous_ids is None else [j for j,sid in enumerate(previous_ids) if sid not in set(ids)]
            ok=len(rows)==len(dropped_previous)*len(smoothcols);smoothing_exits+=len(rows)
            for row in rows.itertuples():
                j=int(np.flatnonzero(previous_ids==row.id)[0]);k=cols[row.variant,row.method];ok=ok and np.isclose(row.omitted_smoothed_aim,previous_target[j,k]*previous_growth[j],atol=1e-12)
            check(f'{period}/{d:%Y-%m-%d} separate smoothing omission ledger',ok)
            for variant in VARIANTS:
                j=cols[variant,'R1'];base=source.baseline[d].set_index('id').loc[ids]
                check(f'{period}/{d:%Y-%m-%d}/{variant} R1 benchmark identity',np.allclose(w[:,j],base.weight,atol=1e-8,rtol=1e-6))
                if period=='selection':check(f'{d:%Y-%m-%d}/{variant} speed-one identity',np.array_equal(w[:,j],w[:,cols[variant,'B1_1']]))
            check(f'{period}/{d:%Y-%m-%d} A0 identical across matched 12-month variants',np.array_equal(w[:,cols['primary','A0']],w[:,cols['blocks24','A0']]))
            controlnames=[s['method'] for s in meta['columns'] if s['variant']=='primary' and not s['policy'].startswith('A')]
            check(f'{period}/{d:%Y-%m-%d} equivalent controls across horizon variants',all(np.array_equal(w[:,cols['primary',name]],w[:,cols['horizon36',name]]) for name in controlnames))
            previous_ids=ids;previous_growth=y['growth'];previous_target=a['target']
        check(period+' all position exits accounted for',position_exits==len(ledger))
        check(period+' all smoothing exits accounted for',smoothing_exits==len(sledger))
        if period=='robustness':check('No invented terminal December 2025 universe',np.array_equal(a['ids'],a['next_ids']))
        rebuilt=pd.DataFrame([dict(period=period,variant=v,method=m,**metrics(g)) for (v,m),g in monthly.groupby(['variant','method'],sort=False)])
        pd.testing.assert_frame_equal(rebuilt,saved_summary,check_exact=True);check(period+' reported descriptive metrics replay',True);summaries.append(rebuilt)
        if period=='selection':
            for variant in VARIANTS:check(variant+' frozen choice uses selection utility only',select(rebuilt[rebuilt.variant.eq(variant)])==frozen['choices'][variant])
        for variant,g in rebuilt.groupby('variant'):
            scores=g.set_index('method');a1=scores.loc['A1']
            control=frozen['choices'][variant]['C_star_selection_path'] if period=='selection' else frozen['choices'][variant]['C_star']
            for label,other in [('A0','A0'),('R1','R1'),('C_star',control)]:
                contrasts.append(dict(period=period,variant=variant,contrast='A1 minus '+label,comparator_path=other,
                    annual_net_utility_difference=float(a1.annual_net_utility-scores.loc[other,'annual_net_utility']),annual_net_mean_difference=float(a1.annual_net_excess_mean-scores.loc[other,'annual_net_excess_mean']),
                    inference_status='NOT_RUN_STEP_10'))
        for (variant,method),g in monthly.groupby(['variant','method']):
            diagnostic.append(dict(period=period,variant=variant,method=method,months=len(g),out_of_range_months=int(g.out_of_range.sum()),fallback_months=int(g.fallback_reason.notna().sum()),upper_bound_months=int(g.upper_bound.sum()),lower_bound_months=int(g.lower_bound.sum()),mean_kappa=float(g.kappa.mean())))
        print('Reviewed complete path block:',period,flush=True)
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(OUT/'strategy_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'primary_contrasts.csv',index=False);pd.DataFrame(diagnostic).to_csv(OUT/'decision_diagnostics.csv',index=False)
    for p,h in code.items():
        if sha(ROOT/p)!=h:raise ValueError('Verification code changed during review')
    result=dict(status='PASS',checks_total=len(checks),checks_passed=sum(c['passed'] for c in checks),checks=checks,extension_checks=extension['checks_total'],
        synthetic_checks=json.loads((OUT/'synthetic_review.json').read_text())['checks_total'],independent_R=r,maximum_own_holdings_recurrence_error=max_recurrence,
        monthly_records=8460,period_months=dict(selection=120,evaluation=120,robustness=60),paths=dict(selection=39,evaluation=21,robustness=21),
        choices=frozen['choices'],primary_contrasts=contrasts,main_study=['1981-01','2020-12'],exposed_robustness=['2021-01','2025-12'],
        performance_inference_run=False,step10_complete=False,protected_science_unchanged=True,
        limitations=['Historical source reconstruction and source eligibility/imputation/omitted-ID assumptions remain.',
            'The main study is 1981–2020; 2021–2025 is separately initialized previously exposed robustness.',
            'One-time-deviation training and the fixed three-state bank are approximations for repeated adaptive deployment.',
            'Costs are modeled source impact only; financing, borrow, spread, taxes and capacity are not established.',
            'Drawdown uses a compounded net-excess-return diagnostic index; it is not the exogenous-AUM account NAV.',
            'Point estimates do not establish statistical improvement. Paired simultaneous inference is Step 10.',
            'No historical period becomes an untouched holdout through this reconstruction.'])
    save_json(OUT/'review_summary.json',result)
    finish('review',[OUT/p for p in ['review_summary.json','strategy_summary.csv','primary_contrasts.csv','decision_diagnostics.csv','verification_context.json']],checks=len(checks))
    print(f'PASS: {len(checks)} complete-path checks and {r["checks"]} independent R checks.',flush=True)
if __name__=='__main__':main()
