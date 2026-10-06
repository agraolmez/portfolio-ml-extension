"""Run block-specific own-holdings paths under the frozen comparison rules."""
import argparse,json
import numpy as np
import pandas as pd
from baseline.io import datekey
from learner.core import VARIANTS,remap
from learner.pipeline import risk
from .io import *
from .core import family,smoothing,action,select,metrics
from .markets import Markets

def run(period):
    begin(period)
    extension=json.loads((OUT/'extension_review.json').read_text())
    if extension['status']!='PASS':raise ValueError('Extension review has not passed')
    for p,h in extension['artifact_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed verified extension '+p)
    frozen=None
    if period!='selection':
        manifest=json.loads((OUT/'selection_manifest.json').read_text());path=OUT/'frozen_selection.json'
        if sha(path)!=manifest['artifact_sha256'][str(path.relative_to(ROOT))]:raise ValueError('Changed frozen selection')
        frozen=json.loads(path.read_text())
    source=Markets();dates=pd.date_range(*PERIODS[period],freq='ME');folder=PREP/'paths'/period;folder.mkdir(parents=True)
    specs=[(variant,name,method,parameter) for variant in VARIANTS for name,method,parameter in family(None if frozen is None else frozen['choices'][variant])]
    h=None;previous_ids=previous_growth=None;stored={};monthly=[];exits=[];smooth_exits=[];initial=None
    for i,d in enumerate(dates):
        x,outcome,cap=source.current(d);ids=x['ids'];count=len(specs)
        if h is None:
            initial=source.baseline[d].set_index('id').loc[ids,'pretrade'].to_numpy()
            h=np.repeat(initial[:,None],count,axis=1)
        targets=np.empty_like(h);trades=np.empty_like(h);weights=np.empty_like(h)
        for j,(variant,name,method,parameter) in enumerate(specs):
            block=VARIANTS[variant][1];smoothed=None
            if method=='B3':
                smoothed,dropped,values=smoothing(x['aim'],stored.get(j),previous_ids,previous_growth,ids,parameter)
                smooth_exits += [dict(decision_eom=d,variant=variant,method=name,id=int(sid),omitted_smoothed_aim=float(v)) for sid,v in zip(dropped,values)]
            result=action(x,h[:,j],method,parameter,source.fit(variant,d) if method.startswith('A') else None,block,cap,smoothed)
            target,q,w=result['target'],result['trade'],result['weight'];targets[:,j]=target;trades[:,j]=q;weights[:,j]=w
            if method=='B3':stored[j]=target.copy()
            gross=float(w@outcome['returns']);cost=float(x['wealth']/2*np.sum(x['impact']*q*q));exposure=float(np.abs(w).sum())
            choice=result['decision']
            monthly.append(dict(period=period,variant=variant,method=name,policy=method,parameter=parameter,decision_eom=d,return_eom=d+pd.offsets.MonthEnd(1),
                gross_excess=gross,impact_cost=cost,net_excess=gross-cost,turnover=float(np.abs(q).sum()),gross_exposure=exposure,net_exposure=float(w.sum()),
                modeled_annual_volatility=float(np.sqrt(12*(w@risk(x,w)))),max_absolute_weight=float(np.abs(w).max()),absolute_weight_hhi=float(w@w/max(exposure**2,1e-30)),
                kappa_min=cap['kappa_min'],kappa_max=cap['kappa_max'],upper_bound=abs(choice['kappa']-cap['kappa_max'])<=1e-12,lower_bound=abs(choice['kappa']-cap['kappa_min'])<=1e-12,
                fit_year=d.year+(d.month==12) if method.startswith('A') else None,**choice,**result['features']))
        drift=weights*outcome['growth'][:,None];next_h,dropped=remap(drift,ids,outcome['next_ids'])
        for j,(variant,name,method,parameter) in enumerate(specs):
            for k in dropped:exits.append(dict(decision_eom=d,variant=variant,method=name,id=int(ids[k]),omitted_drifted_weight=float(drift[k,j]),liquidation_trade_added=False))
        np.savez_compressed(folder/f'{datekey(d)}.npz',ids=ids,pretrade=h,target=targets,trade=trades,weight=weights,drifted_weight=drift,
            next_ids=outcome['next_ids'],next_pretrade=next_h,variants=np.array([s[0] for s in specs]),methods=np.array([s[1] for s in specs]))
        h=next_h;previous_ids=ids;previous_growth=outcome['growth']
        if i%24==0:print(f'{period}: {i+1}/{len(dates)} months, {count} paths',flush=True)
    table=pd.DataFrame(monthly);table.to_parquet(folder/'monthly.parquet',index=False)
    summary=pd.DataFrame([dict(period=period,variant=v,method=m,**metrics(g)) for (v,m),g in table.groupby(['variant','method'],sort=False)])
    summary.to_parquet(folder/'summary.parquet',index=False);summary.to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(exits).to_parquet(folder/'position_exit_ledger.parquet',index=False);pd.DataFrame(smooth_exits).to_parquet(folder/'smoothing_exit_ledger.parquet',index=False)
    save_json(folder/'path_metadata.json',dict(period=period,decision_first=str(dates[0].date()),decision_last=str(dates[-1].date()),months=len(dates),
        columns=[dict(variant=v,method=n,policy=m,parameter=p) for v,n,m,p in specs],initial_state_source=str((RUN/'PolicyDecision.parquet').relative_to(ROOT)),
        first_trade_charged=True,annual_holdings_reset=False,selection_sha256=sha(OUT/'frozen_selection.json') if frozen else None,terminal_retains_drift=period=='robustness'))
    files=list(folder.iterdir())
    if period=='selection':
        choices={variant:select(summary[summary.variant.eq(variant)]) for variant in VARIANTS}
        save_json(OUT/'frozen_selection.json',dict(selection_period=['2001-01-31','2010-12-31'],criterion='annualized net empirical utility with population variance',
            choices=choices,summary_sha256=sha(folder/'summary.parquet'),A1_is_primary=True,performance_inference_run=False))
        files.append(OUT/'frozen_selection.json')
    finish(period,files,months=len(dates),portfolio_paths=len(specs),monthly_records=len(table),selection_sha256=sha(OUT/'frozen_selection.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('period',choices=list(PERIODS));run(p.parse_args().period)
