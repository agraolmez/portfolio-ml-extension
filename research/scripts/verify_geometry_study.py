"""Verify completed study evidence; presentation review is a separate acceptance gate."""
import sys,json,math
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from geometry_study.run import ROOT,OUT,locked,known_locks,feature_map,write

def verify():
    plan=locked()['plan'];digests={};gates=[]
    def require(condition,message):
        if not condition:raise ValueError(message)
    def inventory(path):
        j=json.loads(path.read_text())
        require(bool(j['artifact_sha256']),'Empty evidence inventory')
        for rel,h in j['artifact_sha256'].items():
            require(sha(ROOT/rel)==h,'Changed study evidence '+rel);digests[rel]=h
        digests[str(path.relative_to(ROOT))]=sha(path)
        return j
    fits=inventory(OUT/'fits_manifest.json');paths=inventory(OUT/'paths_manifest.json');analysis=inventory(OUT/'analysis/analysis_manifest.json');report=inventory(OUT/'report/report_manifest.json')
    require(fits['status']==paths['status']==analysis['status']=='COMPLETE','Incomplete scientific stages')
    require(report['report_code_sha256']==sha(ROOT/'geometry_study/report.py'),'Report implementation changed after generation')
    analysis_lock=json.loads((OUT/'analysis/analysis_lock.json').read_text())
    for rel,h in analysis_lock['code_sha256'].items():require(sha(ROOT/rel)==h,'Analysis implementation changed after its lock')
    for rel,h in fits['consumed_input_sha256'].items():require(sha(ROOT/rel)==h,'Changed training input '+rel)
    for old in (OUT/'engineering_versions').glob('*/plan_lock.json'):
        j=json.loads(old.read_text())
        for rel,h in j['code_sha256'].items():require(sha(old.parent/'code'/rel)==h,'Changed retained execution source '+rel)
    locks=known_locks();methods=['fixed512','full','risk_only','return_prediction','joint','fixed1024','fixed1024_double_ridge']
    expected={(year,seed,method) for year in range(1981,2026) for seed in [1,2,3] for method in methods};seen=set();max_residual=0.;max_identity=0.;learned=0
    selections=pd.read_parquet(ROOT/'data/prepared/step9_v1/baseline/annual_selection.parquet')
    for file in sorted((OUT/'fits').glob('*/*/*/fit.json')):
        j=json.loads(file.read_text());year,seed,method=j['year'],j['seed'],j['method'];key=(year,seed,method)
        require(key in expected and key not in seen,'Unexpected or duplicate annual fit');seen.add(key)
        cutoff=f'{year-1}-12-31';last=f'{year-1}-11-30';row=selections[selections.return_eom.eq(cutoff)]
        require(len(row)==1,'Nonunique original annual choice');row=row.iloc[0]
        require(j['cutoff']==cutoff and j['training_first']=='1965-11-30' and j['training_last']==last,'Wrong annual training dates')
        require(j['training_months']==181+12*(year-1981) and j['latest_available_at']<=cutoff,'Unmatured or shortened training history')
        require(j['bandwidth_index']==int(row.g_index) and j['coefficient_ridge']==float(row['lambda'])*(2 if method=='fixed1024_double_ridge' else 1),'Original tuning rule changed')
        require(j['plan_lock_sha256'] in locks,'Unknown execution version')
        residual=j['solve_residual'];require(math.isfinite(residual) and residual<1e-8,'Coefficient solve failed');max_residual=max(max_residual,residual)
        p=1024 if method.startswith('fixed1024') else 512;fmap=feature_map(seed,int(row.g_index),p)
        for name,digest in j['files_sha256'].items():
            model=GeometryModel.load(file.parent/name)
            require(sha(file.parent/name)==digest,'Changed frozen model')
            np.testing.assert_array_equal(model.feature_map.frequencies,fmap.frequencies)
            require(model.cutoff==cutoff and model.source_fingerprint==j['source_fingerprint'],'Model provenance differs from its fit receipt')
            require(len(model.beta)==p+1 and len(model.theta)==115,'Wrong model dimensions')
            if method.startswith('fixed'):np.testing.assert_array_equal(model.theta,np.zeros(115))
        if method=='fixed512' and seed==1:
            with np.load(ROOT/f'data/prepared/step9_v1/baseline/coefficients/g{int(row.g_index)}/{year}_p512.npz') as z:beta=z['coefficients'][:,int(row.lambda_index)]
            model=GeometryModel.load(file.parent/'model.npz');max_identity=max(max_identity,float(max(abs(model.beta-beta))))
            np.testing.assert_allclose(model.beta,beta,atol=1e-9,rtol=1e-8)
        if method in ['full','risk_only','return_prediction','joint']:
            learned+=1;settings=j['settings'];n=len(j['trace'])-1
            require(settings['cutoff']==cutoff and n<=8,'Geometry date/update budget changed')
            for name in ['bound','max_steps','max_backtracks','initial_step','armijo','gradient_tolerance']:
                require(settings[name]==plan['geometry_settings'][name],'Geometry setting changed: '+name)
            require(settings['scale_penalty']==.01*j['objective_normalizer'],'Relative scale penalty changed')
            require(j['status'] in ['ITERATION_BUDGET_REACHED','PROJECTED_GRADIENT_TOLERANCE','LINE_SEARCH_STALLED'],'Unknown optimization status')
            values=np.array([r['value'] for r in j['trace']]);require(np.isfinite(values).all() and (np.diff(values)>=-1e-14).all(),'Nonmonotone accepted optimization')
    require(seen==expected and learned==540,'Missing annual models')
    gates.append(dict(name='All 945 annual fits; complete expanding training history, original tuning, models and provenance',passed=True))
    monthly=pd.read_parquet(OUT/'paths/monthly.parquet');specs={(s,m) for s in [1,2,3] for m in plan['methods']+['fixed1024_double_ridge']}
    require(len(monthly)==12960 and not monthly.duplicated(['seed','method','return_eom']).any(),'Incomplete or duplicate monthly paths')
    require(set(zip(monthly.seed,monthly.method))==specs,'Unexpected path set')
    for key,frame in monthly.groupby(['seed','method']):
        require(list(frame.sort_values('return_eom').return_eom)==list(pd.date_range('1981-01-31','2025-12-31',freq='ME')),'Wrong path calendar')
    require(np.isfinite(monthly.select_dtypes('number').to_numpy()).all(),'Nonfinite performance values')
    require((monthly.impact_cost>=0).all(),'Negative quadratic impact')
    np.testing.assert_allclose(monthly.net_excess,monthly.gross_excess-monthly.impact_cost,atol=1e-14,rtol=1e-13)
    require(paths['months']==540 and paths['paths']==24 and paths['independent_position_accounting_all_months'] and paths['first_trade_charged'] and not paths['annual_or_period_holdings_reset'],'Invalid path execution receipt')
    gates.append(dict(name='All 24 continuous paths and 12,960 monthly records; costs and original-rule recovery',passed=True))
    result=json.loads((OUT/'analysis/analysis_result.json').read_text());intervals=pd.read_csv(OUT/'analysis/primary_intervals.csv');contrasts=pd.read_csv(OUT/'analysis/seed_contrasts.csv')
    require(len(intervals)==24 and not intervals.duplicated(['period','block_months','contrast']).any(),'Incomplete or duplicate inference family')
    require(set(intervals.contrast)==set(plan['primary_contrasts']) and set(intervals.block_months)=={12,24},'Inference specification differs')
    require((intervals.replicas==10000).all() and (intervals.family==6).all(),'Inference replication count or family differs')
    require(np.isfinite(intervals.select_dtypes('number').to_numpy()).all() and (intervals.adjusted_lower<=intervals.adjusted_upper).all(),'Invalid interval values')
    primary=intervals[intervals.period=='evaluation'];seedprimary=contrasts[contrasts.period=='evaluation'];robust=contrasts[contrasts.period=='exposed_robustness']
    success=bool((primary.adjusted_lower>0).all() and (primary.mean_seed_delta>0).all() and (seedprimary[['seed1','seed2','seed3']]>0).all().all() and (robust.mean_seed_delta>0).all())
    require(success==result['primary_family_improvement_established'],'Success conclusion differs from fixed rule')
    require(result['independent_R']['status']=='PASS' and result['independent_scalar_utility_max_error']<1e-12,'Independent statistical verification failed')
    require(result['originality_established'] is False and result['rpm_full_method_comparison_completed'] is False and result['all_outcomes_previously_exposed'] is True,'Unsupported scope claim')
    gates.append(dict(name='Paired seed-average inference, independent R checks, fixed success rule and disclosed limits',passed=True))
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());require(total<10*1024**3,'Study storage exceeds configured limit')
    require(fits['resources']['peak_memory_gib']<24,'Recorded fit memory exceeds configured limit')
    record=dict(status='NUMERICAL_STUDY_VERIFIED_PENDING_PRESENTATION',checked_at_utc=datetime.now(timezone.utc).isoformat(),historical_comparison_complete=True,accepted=False,annual_fits=len(seen),learned_geometry_fits=learned,paths=24,monthly_records=len(monthly),maximum_coefficient_solve_residual=max_residual,original_coefficient_max_error=max_identity,original_path_max_weight_error=paths['original_recovery_max_weight_error'],study_storage_gib=total/1024**3,primary_family_improvement_established=success,originality_established=False,missing_comparison='Full RPM methods unavailable',gates=gates,artifact_sha256=digests,verification_code_sha256=sha(Path(__file__)))
    write(OUT/'numerical_completion_verification.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='artifact_sha256'}))

if __name__=='__main__':verify()
