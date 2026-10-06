"""Step 5 runner, gated on Step 4; --preflight never computes a strategy path."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from .io import ROOT,OUT,PREP,DATA,PROTOCOL,sha,save_json,verify_data,guard
from .acceptance import verify_baseline
from .wealth import validate
from .engine import Engine
from .core import report_returns


def preflight():
    baseline=verify_baseline();source=verify_data();wealth=validate(pd.read_parquet(DATA/'market_wealth.parquet'))
    first,last=PROTOCOL['periods']['pilot']
    returns=pd.date_range(pd.Timestamp(first)+pd.offsets.MonthEnd(0),pd.Timestamp(last)+pd.offsets.MonthEnd(0),freq='ME')
    decisions=returns-pd.offsets.MonthEnd(1)
    aims=pd.read_parquet(PREP/'pilot_aims_preflight.parquet')
    if set(aims.eom)!=set(decisions) or aims.duplicated(['id','eom']).any() or not np.isfinite(aims.aim).all():raise ValueError('Incomplete, duplicate or nonfinite pilot targets')
    if not (aims.fit_year.eq((aims.eom+pd.offsets.MonthEnd(1)).dt.year)&aims.selection_return_year.eq(aims.fit_year-1)).all():raise ValueError('Pilot target chronology mismatch')
    return {'baseline':baseline,'source':source,'wealth':wealth,'return_first':str(returns[0].date()),'return_last':str(returns[-1].date()),'months':len(returns),'decision_first':str(decisions[0].date()),'decision_last':str(decisions[-1].date()),'target_stock_months':len(aims),'financial_path_run':False},decisions,aims


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    start=time.monotonic();receipt,dates,aims=preflight()
    if args.preflight:
        print(json.dumps(receipt,indent=2));return
    folder=ROOT/'output/step5_pilot_1981_1982'
    if (folder/'ExperimentManifest.json').exists():raise ValueError('Existing pilot results must be reviewed before a new run')
    folder.mkdir(parents=True,exist_ok=True)
    engine=Engine(end=dates[-1]);monthly,positions,exits=engine.run(dates,aims)
    monthly.to_parquet(folder/'monthly_performance.parquet',index=False)
    positions.to_parquet(folder/'PolicyDecision.parquet',index=False);aims.to_parquet(folder/'AimArtifact.parquet',index=False)
    exits.to_parquet(folder/'source_exit_ledger.parquet',index=False)
    snapshots=[]
    for d in dates:
        frame=engine.months[d].query('valid')
        snapshots.append({'decision_eom':d,'return_eom':d+pd.offsets.MonthEnd(1),'eligible_stocks':len(frame),'wealth_usd':frame.wealth_usd.iloc[0],'market_total_growth':frame.mu_ld1.iloc[0],'rf_french_for_matrix':engine.rf[d],'risk_sha256':sha(PREP/f'risk/{d:%Y-%m-%d}.npz')})
    pd.DataFrame(snapshots).to_parquet(folder/'MarketSnapshot.parquet',index=False)
    summary=report_returns(monthly.gross_excess,monthly.impact_cost,PROTOCOL['economic']['risk_aversion'])
    save_json(folder/'summary.json',summary)
    save_json(folder/'ExperimentManifest.json',{'status':'PILOT_COMPUTED_PENDING_STEP5_REVIEW','protocol_id':PROTOCOL['protocol_id'],'preflight':receipt,'runtime_seconds':time.monotonic()-start,'resources':guard(),'continuous_holdings':True,'initialization':'source first-month value weights; first trades charged','accounting':'author conventions; entrants zero; exits omitted without separate liquidation trade; terminal drift retained','source_differences':['Approved reciprocal wealth','Accepted R1 included-value/risk-free reconstruction corrections','Primary utility uses net population variance'],'input_aims_sha256':sha(PREP/'pilot_aims_preflight.parquet'),'outputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir() if p.is_file()},'scope':'Conditional historical benchmark; not original-vintage equality, causal execution or prospective evidence'})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
