"""Versioned inference inputs, complete results and traceable exports."""
import argparse
import itertools
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from baseline.io import ROOT, PROTOCOL, sha, save_json
from baseline.full import PROTECTED, preflight as financial_preflight
from .core import stationary, utility, bootstrap_utility, basic_interval, success, acf

OUT = ROOT / 'output/step10_20260913'
PRIOR = ROOT / 'output/step9_20260913'
PATHS = ROOT / 'data/prepared/step9_v1/paths'
METHODS = ['R1', 'B1', 'B2', 'B3', 'A0', 'A1', 'A2']
VARIANTS = ['primary', 'horizon36', 'blocks24']
PERIODS = {'selection': ('2001-01-31', '2010-12-31', 120),
           'evaluation': ('2011-01-31', '2020-12-31', 120),
           'robustness': ('2021-01-31', '2025-12-31', 60)}


def initialize():
    if (OUT/'start.json').exists():
        raise ValueError('Step 10 already initialized')
    layout = json.loads((ROOT/'config/project_layout.json').read_text())
    unique, accepted = {}, {}
    for stage in layout['accepted_stages']:
        p = ROOT/stage['manifest']; a = json.loads(p.read_text())
        assert a[f"step{stage['step']}_complete"] and all(g['passed'] for g in a['gates'])
        hashes = json.loads((ROOT/stage['checksum_file']).read_text()) if 'checksum_file' in stage else a[stage['checksum_key']]
        for rel, digest in hashes.items():
            if rel in unique and unique[rel] != digest:
                raise ValueError('Conflicting accepted hash '+rel)
            unique[rel] = digest
        accepted[stage['manifest']] = sha(p)
    for rel, digest in unique.items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Changed accepted file '+rel)
    receipt, _ = financial_preflight()
    sources = [PATHS/p/'monthly.parquet' for p in PERIODS]
    sources += [PRIOR/f for f in ['acceptance_manifest.json', 'frozen_selection.json', 'strategy_summary.csv', 'primary_contrasts.csv', 'decision_diagnostics.csv']]
    machinery = [ROOT/'inference/core.py', ROOT/'inference/pipeline.py', ROOT/'inference/__init__.py', ROOT/'docs/STEP_10_PLAN.md']
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT/'start.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization="User: let's start and finish step 10", protocol_id=PROTOCOL['protocol_id'],
        prior_acceptances=accepted, accepted_unique_files_verified=len(unique), financial_preflight=receipt,
        protected={p:sha(ROOT/p) for p in PROTECTED}, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in machinery},
        quantile_convention='linear interpolation: NumPy linear / R type 7',
        period_roles=PERIODS, claim_family='primary/evaluation/L12: A1-A0, A1-R1, A1-C*',
        prospective_started=False))
    print('Source and scientific lock verified.', flush=True)


def preflight():
    a = json.loads((OUT/'start.json').read_text())
    for section in ['protected','source_sha256','code_sha256','prior_acceptances']:
        for rel, digest in a[section].items():
            if sha(ROOT/rel) != digest: raise ValueError('Changed locked input '+rel)
    return a


def load_returns(period, variant):
    d = pd.read_parquet(PATHS/period/'monthly.parquet')
    d = d[d.variant.eq(variant)].copy()
    choice = json.loads((PRIOR/'frozen_selection.json').read_text())['choices'][variant]
    if period == 'selection':
        mapping = {f"B1_{choice['B1_kappa']:g}":'B1', f"B3_{choice['B3_rho']:g}":'B3'}
        d['method'] = d.method.replace(mapping)
        d = d[d.method.isin(METHODS)]
    if d.duplicated(['return_eom','method']).any(): raise ValueError('Duplicate date/method')
    w = d.pivot(index='return_eom', columns='method', values='net_excess').reindex(columns=METHODS).sort_index()
    dates = pd.date_range(PERIODS[period][0], PERIODS[period][1], freq='ME')
    if not w.index.equals(dates.rename('return_eom')) or not np.isfinite(w.to_numpy()).all():
        raise ValueError('Incomplete/nonfinite interval')
    if not ((d.decision_eom+pd.offsets.MonthEnd(1)) == d.return_eom).all(): raise ValueError('Decision/outcome mismatch')
    if not np.allclose(d.gross_excess-d.impact_cost,d.net_excess,atol=1e-14,rtol=0): raise ValueError('Net accounting')
    return w, d, choice


def run():
    preflight()
    if (OUT/'run_manifest.json').exists() or (OUT/'acceptance_manifest.json').exists(): raise ValueError('Completed inference is immutable')
    if (OUT/'bootstrap').exists(): raise ValueError('Partial computation exists; diagnose without overwrite')
    (OUT/'bootstrap').mkdir(); (OUT/'inputs').mkdir(); (OUT/'reference_inputs').mkdir()
    primary, secondary, mechanism, dependence, subperiod, sampling, seeds = [], [], [], [], [], [], []
    for period, (start,end,n) in PERIODS.items():
        samples = {}
        for length in [12,24]:
            idx, starts, restart, digests = stationary(PROTOCOL['protocol_id'],PROTOCOL['seeds']['extension'],length,start,n)
            tag=f'{period}_L{length}'
            np.savez_compressed(OUT/'bootstrap'/f'{tag}_indices.npz', indices=idx, starts=starts, restart=restart)
            for name, array in [('indices',idx),('starts',starts),('restart',restart)]:
                np.asarray(array,dtype='<i4').tofile(OUT/'reference_inputs'/f'{tag}_{name}.bin')
            seeds.extend(dict(period=period,block_months=length,replica=b+1,seed_sha256=v) for b,v in enumerate(digests))
            sampling.append(dict(period=period,block_months=length,months=n,replicas=len(idx),
                restart_fraction_after_first=float(restart[:,1:].mean()),expected_restart_fraction=1/length,
                mean_recorded_segments=float(restart.sum(axis=1).mean()),expected_recorded_segments=1+(n-1)/length,
                months_divided_by_block_length=n/length,uniform_start_min_count=int(np.bincount(starts[:,0],minlength=n).min()),
                uniform_start_max_count=int(np.bincount(starts[:,0],minlength=n).max())))
            samples[length]=idx
        for variant in VARIANTS:
            w,d,choice=load_returns(period,variant); x=w.to_numpy(); obs=utility(x)
            tag=f'{period}_{variant}'
            w.to_csv(OUT/'inputs'/f'{tag}_returns.csv',float_format='%.17g')
            for length,idx in samples.items():
                boot=bootstrap_utility(x,idx)
                np.savez_compressed(OUT/'bootstrap'/f'{tag}_L{length}_utility.npz', utility=boot, observed=obs, methods=np.array(METHODS))
                for a,b,label in [('A1','A0','A1-A0'),('A1','R1','A1-R1'),('A1',choice['C_star'],'A1-C*')]:
                    i,j=METHODS.index(a),METHODS.index(b); point=float(obs[i]-obs[j]); lo,hi=basic_interval(point,boot[:,i]-boot[:,j],family=3)
                    primary.append(dict(period=period,variant=variant,block_months=length,months=n,contrast=label,comparator=b,
                        estimate=point,lower=lo,upper=hi,point_positive=point>0,lower_positive=lo>0,
                        primary_claim=period=='evaluation' and variant=='primary' and length==12,
                        interval='Bonferroni 3 basic; nominal simultaneous 95% within this three-contrast family'))
                for a,b in itertools.combinations(METHODS,2):
                    i,j=METHODS.index(a),METHODS.index(b); point=float(obs[i]-obs[j]); lo,hi=basic_interval(point,boot[:,i]-boot[:,j])
                    secondary.append(dict(period=period,variant=variant,block_months=length,contrast=f'{a}-{b}',estimate=point,lower=lo,upper=hi,
                        interval='Ordinary 95% basic; descriptive, no multiplicity adjustment'))
            for a,b,label in [('A1','A0','A1-A0'),('A1','R1','A1-R1'),('A1',choice['C_star'],'A1-C*')]:
                i,j=METHODS.index(a),METHODS.index(b); xa,xb=x[:,i],x[:,j]
                da,db=[d[d.method.eq(m)].sort_values('return_eom') for m in [a,b]]
                meanpart=float(12*(xa.mean()-xb.mean())); riskpart=float(-60*(xa.var()-xb.var()))
                mechanism.append(dict(period=period,variant=variant,contrast=label,utility_difference=float(obs[i]-obs[j]),
                    net_mean_contribution=meanpart,variance_penalty_contribution=riskpart,
                    gross_mean_contribution=float(12*(da.gross_excess.mean()-db.gross_excess.mean())),
                    impact_contribution=float(-12*(da.impact_cost.mean()-db.impact_cost.mean())),
                    monthly_turnover_difference=float(da.turnover.mean()-db.turnover.mean()),
                    gross_exposure_difference=float(da.gross_exposure.mean()-db.gross_exposure.mean())))
                influence=12*((xa-xb)-5*((xa-xa.mean())**2-(xb-xb.mean())**2))
                for lag in range(1,25): dependence.append(dict(period=period,variant=variant,series=label,kind='utility_contrast_influence',lag=lag,acf=acf(influence,lag)))
            for i,method in enumerate(METHODS):
                for kind,y in [('net_return',x[:,i]),('squared_centered_net_return',(x[:,i]-x[:,i].mean())**2)]:
                    for lag in range(1,25):dependence.append(dict(period=period,variant=variant,series=method,kind=kind,lag=lag,acf=acf(y,lag)))
                for half,sl in [('first',slice(0,n//2)),('second',slice(n//2,n))]:
                    y=x[sl,i]; subperiod.append(dict(period=period,variant=variant,method=method,half=half,months=len(y),
                        start=str(w.index[sl][0].date()),end=str(w.index[sl][-1].date()),annual_net_mean=12*y.mean(),annual_net_utility=float(utility(y))))
            print(f'Computed {period}/{variant}: {n} observed months, 2 × 10,000 paired draws.',flush=True)
    tables=dict(primary_intervals=primary,secondary_intervals=secondary,utility_decomposition=mechanism,dependence_diagnostics=dependence,
                subperiod_diagnostics=subperiod,sampling_diagnostics=sampling,seeds=seeds)
    for name,records in tables.items():pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False,float_format='%.17g')
    p=pd.DataFrame(primary);main=p[p.primary_claim]
    save_json(OUT/'inference_summary.json',dict(status='COMPUTED_PENDING_REVIEW',primary_success=success(main.estimate,main.lower),
        primary_family=main.to_dict('records'),primary_interval_rows=len(primary),secondary_interval_rows=len(secondary),
        interval_lengths=[12,24],replicas_per_interval=10000,unique_index_sets=6,utility_arrays=18,
        original_market_months=dict(selection=120,evaluation=120,robustness=60),
        interpretation='Failure of the prescribed positive-success criterion is not proof of equivalence or universal inferiority.',
        formal_significance_claim=False,prospective_started=False))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['heavy_job.lock'] and not p.name.endswith('.log') and 'resources' not in p.name and 'system_time' not in p.name]
    save_json(OUT/'run_manifest.json',dict(status='COMPLETE',start_sha256=sha(OUT/'start.json'),
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','run']);args=parser.parse_args()
    initialize() if args.stage=='initialize' else run()
