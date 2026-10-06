"""Review conditional refits and independently recompute all monthly features."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import ROOT, DATA, PROTOCOL, sha, save_json, datekey
from confidence.pipeline import OUT, PREP, BASE, RUN, DATES, BLOCKS, preflight
from confidence.core import state_features


def main():
    if (OUT / 'review_summary.json').exists():
        raise ValueError('Review already recorded; preserve prior evidence')
    start = preflight(); checks = []
    def boolean(name, value, detail=None):
        checks.append(dict(check=name, passed=bool(value), detail=detail))
        if not value: raise AssertionError(name)
    def close(name, actual, expected, atol=1e-12, rtol=1e-9):
        a,b = np.asarray(actual),np.asarray(expected)
        err=float(np.max(np.abs(a-b)))
        boolean(name, a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol),
                dict(max_abs_error=err,atol=atol,rtol=rtol))
    synthetic=json.loads((OUT/'synthetic_checks.json').read_text())
    boolean('Synthetic algebra and boundary checks',synthetic['status']=='PASS' and len(synthetic['checks'])==127 and all(c['passed'] for c in synthetic['checks']))
    for name in ['fit_manifest.json','feature_manifest.json']:
        manifest=json.loads((OUT/name).read_text())
        boolean(f'Complete immutable {name}',bool(manifest['artifact_sha256']) and all(sha(ROOT/p)==h for p,h in manifest['artifact_sha256'].items()))
    reference=json.loads((OUT/'reference_status.json').read_text())
    rc=pd.read_csv(OUT/'reference_comparison.csv')
    expected={(y,l,b) for y in range(1981,2021) for l in BLOCKS for b in [1,50,100]}
    boolean('Independent R verification covers prespecified 240 refits',reference['status']=='PASS' and reference['fits']==240 and len(rc)==240 and rc.passed.all() and set(zip(rc.fit_year,rc.block_months,rc.replica))==expected)
    models=json.loads((OUT/'annual_models.json').read_text())
    seed_records=pd.read_parquet(PREP/'replica_seeds.parquet')
    boolean('All 8000 unique seeds retained',len(seed_records)==8000 and seed_records.sha256.nunique()==8000 and not seed_records.duplicated(['fit_year','block_months','replica']).any())
    max_residual=0.; min_unique=10000; max_unique=0; all_coef={}
    for row in models:
        year,n=row['fit_year'],row['training_months']
        dates=pd.date_range(row['training_start'],row['training_cutoff'],freq='ME')
        boolean(f'Annual chronology {year}',len(dates)==n and dates[-1]+pd.offsets.MonthEnd(1)==pd.Timestamp(row['latest_training_return'])==pd.Timestamp(row['first_decision']) and row['selection_available_at']==row['first_decision'])
        for length in BLOCKS:
            path=PREP/f'coefficients/block{length}/{year}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as z:
                coef,starts,samples,counts,residuals=(z[k] for k in ['coefficients','starts','sampled_indices','counts','residuals'])
                boolean(f'Full finite coefficient ensemble {year}/{length}',coef.shape==(513,100) and np.isfinite(coef).all() and len(residuals)==100 and np.isfinite(residuals).all() and np.max(residuals)<1e-10)
                boolean(f'Exact ordered training dates {year}/{length}',np.array_equal(z['training_dates'],dates.strftime('%Y-%m-%d')) and np.array_equal(z['indices'],np.arange(513)))
                boolean(f'Fixed selected model {year}/{length}',all(meta[k]==row[k] for k in ['p','g_index','lambda_index','penalty','central_coefficients_sha256','training_cutoff']) and meta['uniform_central_max_abs_error']<1e-12)
                ref=OUT/f'reference/block{length}'
                boolean(f'R input counts equal saved resamples {year}/{length}',np.array_equal(
                    np.loadtxt(ref/f'{year}_counts.csv',delimiter=',',dtype=int),counts[[0,49,99]]))
                close(f'R comparison coefficients equal saved replicas {year}/{length}',
                    np.loadtxt(ref/f'{year}_coefficients.csv',delimiter=','),coef[:,[0,49,99]],atol=1e-18,rtol=1e-14)
                selected=seed_records[(seed_records.fit_year==year)&(seed_records.block_months==length)].set_index('replica')
                for b in range(1,101):
                    payload=dict(protocol_id=PROTOCOL['protocol_id'],master_seed=20260905,
                                 component=f'aim_moving_blocks_{length}',date=row['first_decision'],replica=b)
                    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    rng=np.random.Generator(np.random.PCG64(int(digest[:32],16)))
                    reference_starts=rng.integers(0,n-length+1,size=(n+length-1)//length)
                    expanded=[]
                    for first in reference_starts:expanded.extend(range(int(first),int(first)+length))
                    expanded=np.asarray(expanded[:n])
                    independent_counts=np.zeros(n,dtype=int)
                    for month in expanded:independent_counts[month]+=1
                    boolean(f'Seed, whole-month blocks and exact counts {year}/{length}/{b}',
                        digest==selected.loc[b,'sha256'] and digest[:32]==selected.loc[b,'seed_hex'] and
                        np.array_equal(starts[b-1],reference_starts) and np.array_equal(samples[b-1],expanded) and
                        np.array_equal(counts[b-1],independent_counts) and counts[b-1].sum()==n and samples[b-1].max()<n)
                all_coef[(year,length)]=coef.copy()
                max_residual=max(max_residual,float(residuals.max()))
                min_unique=min(min_unique,int((counts>0).sum(axis=1).min()))
                max_unique=max(max_unique,int((counts>0).sum(axis=1).max()))
    table=pd.read_parquet(PREP/'monthly_confidence.parquet').set_index(['decision_eom','block_months'])
    boolean('All 960 date and block records',len(table)==960 and table.index.is_unique and set(table.index)=={(d,l) for d in DATES for l in BLOCKS})
    boolean('Finite full predictor dictionary',np.isfinite(table.select_dtypes(include='number')).all().all())
    lineage=pd.read_parquet(PREP/'AimArtifact.parquet').set_index(['decision_eom','block_months'])
    positions=pd.read_parquet(RUN/'PolicyDecision.parquet')
    inputs=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=['id','eom','lambda_diagonal','wealth_usd'],
                           filters=[('eom','>=',DATES[0]),('eom','<=',DATES[-1])])
    groups={d:f.set_index('id') for d,f in inputs.groupby('eom')}
    modelmap={x['fit_year']:x for x in models}
    maximum_monthly_error=0.; stock_values=0
    for d,state in positions.groupby('eom',sort=True):
        row=modelmap[(d+pd.offsets.MonthEnd(1)).year];year,g=row['fit_year'],row['g_index']
        with np.load(BASE/f'moments/g{g}/{datekey(d)}.npz') as z:ids,signal=z['ids'],z['signal_t']
        with np.load(BASE/f'risk/{datekey(d)}.npz') as z:
            risk_ids=z['ids'];x,fc,sp=z['loadings'],z['factor_cov'],z['specific_variance']
        sigma=x@fc@x.T+np.diag(sp)
        with np.load(BASE/f'matrices/{datekey(d)}.npz') as z:matrix_ids,m=z['ids'],z['m']
        frame=groups[d].reindex(ids);state=state.set_index('id').reindex(ids)
        boolean(f'Aligned input and state IDs {datekey(d)}',np.array_equal(ids,risk_ids) and np.array_equal(ids,matrix_ids) and state.notna().all().all() and frame.notna().all().all())
        h=state.pretrade.to_numpy();central=state.aim.to_numpy();adjustment=np.eye(len(ids))-m
        q0=adjustment@(central-h)
        close(f'Unchanged original central trade {datekey(d)}',q0,state.trade.to_numpy())
        for length in BLOCKS:
            key=f'{datekey(d)}/{length}';saved=table.loc[(d,length)]
            path=PREP/f'aims/block{length}/{datekey(d)}.npz'
            with np.load(path) as z:
                boolean(f'100 aligned monthly replicas {key}',np.array_equal(ids,z['ids']) and z['replica_aims'].shape==(len(ids),100))
                aim_replicas=z['replica_aims'];saved_central=z['central_aim'];saved_mean=z['mean_aim'];saved_qbar=z['baseline_mean_trade']
            close(f'Every replica applies fixed annual coefficients {key}',aim_replicas,signal@all_coef[(year,length)])
            close(f'Original aim preserved {key}',saved_central,central)
            mean_aim=aim_replicas.sum(axis=1)/100
            close(f'Ensemble mean retained separately {key}',saved_mean,mean_aim)
            # Independent algebra: transformed deviations of the AIMS are state independent.
            qbar=adjustment@(mean_aim-h)
            delta=adjustment@(aim_replicas-mean_aim[:,None])
            # Factor-plus-specific covariance route, not the production dense quadratic.
            factor_delta=x.T@delta
            numerator=float((np.sum(factor_delta*(fc@factor_delta))+np.sum(sp[:,None]*delta**2))/100)
            def quadratic(v):return float((x.T@v)@fc@(x.T@v)+np.sum(sp*v*v))
            denominator=quadratic(qbar); qrisk=quadratic(q0); hrisk=quadratic(h)
            impact=float(frame.wealth_usd.iloc[0]*np.sum(frame.lambda_diagonal.to_numpy()*q0*q0))
            alignment=0. if min(qrisk,hrisk)<=1e-12 else float(q0@sigma@h/np.sqrt(qrisk*hrisk))
            dispersion=numerator/max(denominator,1e-12);logd=np.log1p(dispersion)
            expected_values=dict(dispersion=dispersion,dispersion_numerator=numerator,mean_trade_risk=denominator,
                log1p_aim_dispersion=logd,trade_risk=qrisk,trade_impact_curvature=impact,holdings_risk=hrisk,
                risk_alignment=alignment,log_trade_risk=np.log(max(qrisk,1e-12)),log_trade_impact=np.log(max(impact,1e-12)),
                log_holdings_risk=np.log(max(hrisk,1e-12)),alignment_times_log1p_dispersion=alignment*logd,
                initial_curvature=10*qrisk+impact,mean_aim_distance_from_central=np.linalg.norm(mean_aim-central))
            expected_array=np.array(list(expected_values.values()));actual_array=saved[list(expected_values)].to_numpy(dtype=float)
            close(f'Independent factor-risk calculation of every predictor {key}',actual_array,expected_array)
            close(f'Ensemble mean trade from same state {key}',saved_qbar,qbar)
            maximum_monthly_error=max(maximum_monthly_error,float(np.max(np.abs(actual_array-expected_array))))
            boolean(f'Chronology, state and numerical floors {key}',saved.latest_training_return<=d and saved.selection_available_at<=d
                and saved.state_id=='R1_continuous' and saved.replicas==100 and saved.denominator_floor==(denominator<=1e-12)
                and saved.alignment_floor==(min(qrisk,hrisk)<=1e-12) and saved.fit_year==year and saved.securities==len(ids))
            source=lineage.loc[(datekey(d),length)]
            boolean(f'Dated source lineage {key}',source.coefficient_sha256==sha(PREP/f'coefficients/block{length}/{year}.npz')
                    and source.current_signals_sha256==sha(BASE/f'moments/g{g}/{datekey(d)}.npz')
                    and source.risk_sha256==sha(BASE/f'risk/{datekey(d)}.npz')
                    and source.matrix_sha256==sha(BASE/f'matrices/{datekey(d)}.npz')
                    and source.central_coefficients_sha256==row['central_coefficients_sha256'])
            if d.month==12:
                alternate=h*.5
                alternate_features,_,_=state_features(central,aim_replicas,alternate,m,sigma,
                    frame.lambda_diagonal.to_numpy(),frame.wealth_usd.iloc[0])
                alternate_denominator=quadratic(adjustment@(mean_aim-alternate))
                close(f'Own-state confidence recomputation {key}',
                    [alternate_features['dispersion_numerator'],alternate_features['dispersion']],
                    [numerator,numerator/max(alternate_denominator,1e-12)])
            stock_values+=len(ids)*100
        if d.month==11:print(f'Independent monthly feature review: through {datekey(d)}.',flush=True)
    boolean('All stored replica aim values reviewed',stock_values==65360000)
    summary=[]
    flat_table=table.reset_index()
    for length,group in flat_table.groupby('block_months'):
        summary.append(dict(block_months=int(length),months=len(group),mean_dispersion=float(group.dispersion.mean()),
            median_dispersion=float(group.dispersion.median()),minimum_dispersion=float(group.dispersion.min()),
            maximum_dispersion=float(group.dispersion.max()),p05_dispersion=float(group.dispersion.quantile(.05)),
            p95_dispersion=float(group.dispersion.quantile(.95)),denominator_floor_months=int(group.denominator_floor.sum()),
            maximum_month=datekey(group.loc[group.dispersion.idxmax(),'decision_eom']),
            mean_log1p_dispersion=float(group.log1p_aim_dispersion.mean())))
    pivot=flat_table.pivot(index='decision_eom',columns='block_months',values='dispersion')
    sensitivity=dict(pearson_correlation=float(pivot[12].corr(pivot[24])),
        spearman_correlation=float(pivot[12].corr(pivot[24],method='spearman')),
        median_24_to_12_ratio=float((pivot[24]/pivot[12]).median()),
        months_24_above_12=int((pivot[24]>pivot[12]).sum()))
    pd.DataFrame(summary).to_csv(OUT/'confidence_summary.csv',index=False)
    annual=flat_table.groupby(['fit_year','block_months']).agg(months=('dispersion','size'),
        mean_dispersion=('dispersion','mean'),median_dispersion=('dispersion','median'),
        mean_log1p_dispersion=('log1p_aim_dispersion','mean'),mean_numerator=('dispersion_numerator','mean'),
        mean_denominator=('mean_trade_risk','mean')).reset_index()
    annual.to_csv(OUT/'annual_confidence.csv',index=False)
    save_json(OUT/'review_summary.json',dict(status='PASS',checks_total=len(checks),checks_passed=len(checks),checks=checks,
        synthetic_checks=synthetic['count'],annual_coefficient_fits=8000,independent_R_sampled_fits=240,
        monthly_confidence_rows=960,monthly_replica_aim_values=stock_values,maximum_relative_solve_residual=max_residual,
        maximum_monthly_predictor_error=maximum_monthly_error,minimum_unique_resampled_months=min_unique,
        maximum_unique_resampled_months=max_unique,summary=summary,sensitivity=sensitivity,reference=reference,
        original_aim_unchanged=True,speed_learner_run=False,performance_claim=False,
        limitations=['Conditional coefficient sensitivity only; fixed feature draw, selected penalty, risk and impact models.',
            'Monthly D is evaluated on R1 pretrade holdings; other strategies must recompute from their own states.',
            'The accepted historical vendor, eligibility, timing and exit conventions remain; causal execution is not certified.',
            'Neither replicas nor two block lengths add independent observed months; no calibrated confidence intervals.',
            'Step 8 labels, learner and predictive/economic usefulness are not yet tested.']))
    print(f'Step 7 review PASS: {len(checks)} checks plus {synthetic["count"]} synthetic checks.',flush=True)


if __name__=='__main__':main()
