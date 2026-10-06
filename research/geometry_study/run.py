"""Immutable, resumable annual fits and original-rule portfolio paths."""
import argparse,json,os,resource,shutil,time
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_geometry.features import fingerprint
from .features import CachedFeatureMap as FeatureMap
from feature_geometry.history import sha
from feature_geometry.training import FitSettings,_solve
from feature_geometry.policy import GeometryModel
from baseline.full import preflight
from baseline.core import transition
from .history import History
from .kernel import LeanBasis,evaluate,sums,fit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/feature_geometry_study_20260914'
PLAN=ROOT/'config/feature_geometry_study_v1.json'
BASE=ROOT/'data/prepared/step9_v1/baseline'
RUN=ROOT/'data/prepared/step9_v1/continuous_R1'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def code_paths():
    return sorted([*(p for p in ROOT.glob('geometry_study/*.py') if p.name not in ('analyze.py','report.py')),*ROOT.glob('feature_geometry/*.py'),PLAN,ROOT/'docs/FEATURE_GEOMETRY_STUDY_PLAN_20260914.md',ROOT/'tests/test_geometry_study.py',ROOT/'scripts/geometry_directions.R',ROOT/'scripts/feature_geometry_study.py'])

def guard():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    free=shutil.disk_usage(ROOT).free/1024**3
    if peak>24 or free<30:raise RuntimeError(f'Resource limit: peak {peak:.3f} GiB, free {free:.3f} GiB')
    return dict(peak_memory_gib=peak,free_disk_gib=free)

def locked():
    lock=json.loads((OUT/'plan_lock.json').read_text())
    for path,h in lock['code_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Study code/config changed after lock: '+path)
    for path,h in lock['directions_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Frequency draw changed after lock')
    return lock

def initialize():
    if (OUT/'plan_lock.json').exists():raise ValueError('Study already locked')
    OUT.mkdir(exist_ok=True);receipt,_=preflight();write(OUT/'financial_preflight.json',receipt)
    acceptance=ROOT/'output/step9_20260913/acceptance_manifest.json';j=json.loads(acceptance.read_text())
    # Relevant extended baseline inputs must still match their accepted receipts.
    verified=0
    for rel,h in j['artifact_sha256'].items():
        if rel.startswith(('data/prepared/step9_v1/baseline/','data/prepared/step9_v1/continuous_R1/')):
            if sha(ROOT/rel)!=h:raise ValueError('Changed accepted extended baseline input '+rel)
            verified+=1
    for gi in [0,1]:
        source=np.fromfile(ROOT/f'output/step6_20260911/reference/rff_g{gi}.bin',dtype='<f8').reshape(115,256)
        draw=np.fromfile(OUT/f'directions/seed1_g{gi}.bin',dtype='<f8').reshape(115,512)
        np.testing.assert_array_equal(source,draw[:,:256])
    write(OUT/'plan_lock.json',dict(status='LOCKED_BEFORE_COMPLETE_HISTORICAL_EVALUATION',created_at_utc=datetime.now(timezone.utc).isoformat(),plan=json.loads(PLAN.read_text()),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code_paths()},directions_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'directions').iterdir()},
        financial_preflight_sha256=sha(OUT/'financial_preflight.json'),step9_acceptance_sha256=sha(acceptance),extended_files_verified=verified))
    print('Study plan/source lock complete',flush=True)

def feature_map(seed,gi,p=512):
    path=OUT/f'directions/seed{seed}_g{gi}.bin'
    all_directions=np.fromfile(path,dtype='<f8').reshape(115,512)
    return FeatureMap(all_directions[:,:p//2],f'R_MASS_seed_{seed}_g{gi}_nested_{p}_source_{sha(path)}')

def complete_fit(folder,result,meta,fmap):
    folder.mkdir(parents=True,exist_ok=True)
    for name,key in [('model','beta'),('geometry_model','geometry_beta')]:
        if name=='geometry_model' and key not in result:continue
        model=GeometryModel(fmap,result['theta'],result[key],meta['cutoff'],meta['bound'],meta['source_fingerprint'])
        model.save(folder/f'{name}.npz')
    stats={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    stats.update(meta);stats['scale_min']=float(np.exp(result['theta']).min());stats['scale_max']=float(np.exp(result['theta']).max())
    stats['files_sha256']={p.name:sha(p) for p in folder.glob('*.npz')}
    write(folder/'fit.json',stats)

def known_locks():
    # Older equivalent execution versions retain their original receipts.
    paths=[OUT/'plan_lock.json',*sorted((OUT/'engineering_versions').glob('*/plan_lock.json'))]
    current=json.loads((OUT/'plan_lock.json').read_text())
    result=set()
    for path in paths:
        old=json.loads(path.read_text())
        if old['plan']!=current['plan'] or old['directions_sha256']!=current['directions_sha256']:
            raise ValueError('Scientific configuration differs across retained execution versions')
        result.add(sha(path))
    return result

def fit_done(folder,expected=None):
    if not (folder/'fit.json').exists():
        if folder.exists() and any(folder.iterdir()):raise ValueError('Incomplete fit requires preserving its attempt before retry: '+str(folder))
        return False
    j=json.loads((folder/'fit.json').read_text())
    if expected is not None:
        for key in ['year','seed','bandwidth_index','coefficient_ridge','cutoff','bound','training_first','training_last','training_months','latest_available_at','source_fingerprint']:
            if j.get(key)!=expected[key]:raise ValueError('Completed fit metadata mismatch: '+key+' in '+str(folder))
        if j.get('method')!=folder.name:raise ValueError('Completed fit method differs')
        if j.get('plan_lock_sha256') not in known_locks():raise ValueError('Unrecognized completed fit code lock')
    for p,h in j['files_sha256'].items():
        if sha(folder/p)!=h:raise ValueError('Changed completed fit '+str(folder/p))
    return True

def annual_fits(end_year=2025):
    if (OUT/'fits_manifest.json').exists():
        j=json.loads((OUT/'fits_manifest.json').read_text())
        for p,h in j['artifact_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed complete fit artifact '+p)
        locked();return
    lock=locked();plan=lock['plan'];receipt=json.loads((OUT/'financial_preflight.json').read_text())
    fresh,_=preflight()
    if fresh!=receipt:raise ValueError('Financial source preflight changed before resume')
    source=History(ROOT,receipt);selection=pd.read_parquet(BASE/'annual_selection.parquet')
    bases=[];next_date=pd.Timestamp(plan['training_first_decision']);settings_plan=plan['geometry_settings'];start=time.monotonic()
    annual_records=[]
    with threadpool_limits(limits=1):
        for year in range(plan['first_return_year'],end_year+1):
            cutoff=f'{year-1}-12-31';last=pd.Timestamp(f'{year-1}-11-30')
            while next_date<=last:
                bases.append(LeanBasis(source.training_sample(next_date,cutoff)));next_date+=pd.offsets.MonthEnd(1)
                if len(bases)%60==0:print(f'Training inputs ready through {bases[-1].sample.market.decision_date}: {len(bases)} months; {guard()}',flush=True)
            rows=selection[selection.return_eom.eq(cutoff)]
            if len(rows)!=1 or int(rows.iloc[0].p)!=512:raise ValueError('Expected unique original 512-feature annual choice')
            row=rows.iloc[0];gi=int(row.g_index);ridge=float(row['lambda'])
            source_key=fingerprint(*(b.sample.fingerprint for b in bases))
            settings=FitSettings(cutoff,ridge=ridge,scale_penalty=0.,bound=settings_plan['bound'],max_steps=settings_plan['max_steps'],max_backtracks=settings_plan['max_backtracks'],initial_step=settings_plan['initial_step'],armijo=settings_plan['armijo'],gradient_tolerance=settings_plan['gradient_tolerance'])
            for seed in plan['frequency_seeds']:
                fmap=feature_map(seed,gi);zero=np.zeros(115)
                meta=dict(year=year,seed=seed,bandwidth_index=gi,coefficient_ridge=ridge,cutoff=cutoff,bound=settings.bound,training_first=bases[0].sample.market.decision_date,training_last=bases[-1].sample.market.decision_date,training_months=len(bases),latest_available_at=bases[-1].sample.available_at,source_fingerprint=source_key,plan_lock_sha256=sha(OUT/'plan_lock.json'))
                root=OUT/f'fits/{year}/seed{seed}'
                wanted=['fixed512','full','risk_only','return_prediction','joint','fixed1024','fixed1024_double_ridge']
                if all(fit_done(root/m,dict(meta,coefficient_ridge=ridge*(2 if m=='fixed1024_double_ridge' else 1))) for m in wanted):print(f'Reused verified completed {year} seed {seed}',flush=True);continue
                identity=evaluate(bases,fmap,zero,settings,'full')
                if seed==1:
                    with np.load(BASE/f'coefficients/g{gi}/{year}_p512.npz',allow_pickle=False) as z:expected=z['coefficients'][:,int(row.lambda_index)]
                    error=float(np.max(abs(identity['beta']-expected)))
                    np.testing.assert_allclose(identity['beta'],expected,atol=1e-9,rtol=1e-8)
                    meta['identity_beta_max_error']=error
                if not fit_done(root/'fixed512',meta):
                    complete_fit(root/'fixed512',dict(theta=zero,beta=identity['beta'],status='EXACT_FIXED_FEATURE_SOLVE',solve_residual=identity['solve_residual']),dict(meta,method='fixed512'),fmap)
                for objective,joint in [('full',False),('risk_only',False),('return_prediction',False),('full',True)]:
                    method='joint' if joint else objective;folder=root/method
                    if fit_done(folder,meta):continue
                    obj_settings=settings
                    initial=identity
                    if objective=='risk_only':
                        beta,res=_solve(identity['risk'],identity['score'],ridge)
                        initial=dict(value=float(.5*identity['score']@beta))
                    elif objective=='return_prediction':
                        score,gram,_=sums(bases,fmap,zero,'return_prediction')
                        prediction_ridge=ridge*float(np.trace(gram)/np.trace(identity['risk']+identity['cost']))
                        beta,res=_solve(gram,score,prediction_ridge)
                        initial=dict(value=float(.5*score@beta))
                        obj_settings=replace(settings,ridge=prediction_ridge)
                    normalizer=float(initial['value'])
                    if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('Nonpositive objective scale')
                    obj_settings=replace(obj_settings,scale_penalty=settings_plan['relative_scale_penalty']*normalizer)
                    print(f'Fitting {year} seed {seed} {method}: {len(bases)} matured months',flush=True)
                    def progress(item):
                        print(json.dumps(dict(year=year,seed=seed,method=method,**item,**guard())),flush=True)
                    result=fit(bases,fmap,obj_settings,objective,joint=joint,progress=progress,objective_scale=normalizer,coefficient_ridge=ridge)
                    complete_fit(folder,result,dict(meta,method=method,geometry_ridge=obj_settings.ridge,objective_normalizer=normalizer,settings=asdict(obj_settings)),fmap)
                folder=root/'fixed1024'
                if not fit_done(folder,meta) or not fit_done(root/'fixed1024_double_ridge',dict(meta,coefficient_ridge=2*ridge)):
                    bigmap=feature_map(seed,gi,1024);big=evaluate(bases,bigmap,zero,settings,'full')
                    for name,mult in [('fixed1024',1),('fixed1024_double_ridge',2)]:
                        if not fit_done(root/name,dict(meta,coefficient_ridge=ridge*mult)):
                            beta,res=_solve(big['risk']+big['cost'],big['score'],ridge*mult)
                            complete_fit(root/name,dict(theta=zero,beta=beta,status='EXACT_FIXED_FEATURE_SOLVE',solve_residual=res),dict(meta,method=name,coefficient_ridge=ridge*mult),bigmap)
                print(f'Completed all fits for {year} seed {seed}; elapsed {(time.monotonic()-start)/3600:.2f} hours',flush=True)
            guard();locked()
        if end_year==plan['last_return_year']:
            files=sorted((OUT/'fits').rglob('*'));files=[p for p in files if p.is_file()]
            write(OUT/'fits_manifest.json',dict(status='COMPLETE',years=[plan['first_return_year'],end_year],code_lock_sha256=sha(OUT/'plan_lock.json'),consumed_input_sha256=source.consumed,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},resources=guard(),seconds=time.monotonic()-start))

def portfolio_paths():
    if (OUT/'paths_manifest.json').exists():
        j=json.loads((OUT/'paths_manifest.json').read_text())
        for p,h in j['artifact_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed complete path artifact '+p)
        locked();return
    lock=locked();plan=lock['plan']
    manifest=json.loads((OUT/'fits_manifest.json').read_text())
    for rel,h in manifest['artifact_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Changed fitted model '+rel)
    source=History(ROOT,json.loads((OUT/'financial_preflight.json').read_text()))
    baseline={d:g.sort_values('id').set_index('id') for d,g in pd.read_parquet(RUN/'PolicyDecision.parquet').groupby('eom')}
    specs=[(s,m) for s in plan['frequency_seeds'] for m in plan['methods']+['fixed1024_double_ridge']]
    dates=pd.date_range('1980-12-31','2025-11-30',freq='ME');h=None;monthly=[];exit_rows=[];models={};active_year=None
    folder=OUT/'paths';folder.mkdir();max_identity=0.
    with threadpool_limits(limits=4):
        for d in dates:
            year=(d+pd.offsets.MonthEnd(1)).year
            if year!=active_year:
                models={}
                for s,method in specs:
                    fit_method='joint' if method=='joint_refit' else method
                    filename='geometry_model' if method=='joint' else 'model'
                    models[s,method]=GeometryModel.load(OUT/f'fits/{year}/seed{s}/{fit_method}/{filename}.npz')
                active_year=year
            market=source.market(d);ids=market.ids
            if h is None:h=np.repeat(baseline[d].loc[ids,'pretrade'].to_numpy()[:,None],len(specs),axis=1)
            targets=np.column_stack([models[s,m].aim(market.characteristics[0],market.volatility[0],str(d.date())) for s,m in specs])
            next_date=d+pd.offsets.MonthEnd(1)
            nextids=source.months[next_date].query('valid').id.to_numpy() if next_date in source.months else ids.copy()
            outcomes=source.outcomes[d].loc[ids];excess=source.months[d].loc[ids,'ret_ld1'].to_numpy()
            weights=np.empty_like(h);trades=np.empty_like(h);drift=np.empty_like(h);next_h=np.empty((len(nextids),len(specs)))
            for j,(seed,method) in enumerate(specs):
                r=transition(ids,h[:,j],targets[:,j],market.m,outcomes.tr_ld1.to_numpy(),float(outcomes.mu_ld1.iloc[0]),market.impact,market.wealth,excess,nextids)
                weights[:,j]=r['weights'];trades[:,j]=r['trades'];drift[:,j]=r['drifted_weights'];next_h[:,j]=r['next_pretrade']
                # Independent direct accounting from realized positions, not transition outputs.
                gross=float(sum(float(w)*float(y) for w,y in zip(weights[:,j],excess)))
                cost=float(.5*market.wealth*sum(float(l)*float(q)**2 for l,q in zip(market.impact,trades[:,j])))
                if not np.allclose([gross,cost],[r['gross_excess'],r['impact_cost']],atol=1e-12,rtol=1e-11):raise ValueError('Independent position accounting differs')
                monthly.append(dict(seed=seed,method=method,decision_eom=d,return_eom=next_date,gross_excess=gross,impact_cost=cost,net_excess=gross-cost,turnover=r['turnover'],gross_exposure=r['gross_exposure'],modeled_annual_volatility=float(np.sqrt(12*weights[:,j]@market.covariance@weights[:,j]))))
                for sid,v in zip(r['dropped_ids'],r['dropped_drifted_weights']):exit_rows.append(dict(seed=seed,method=method,decision_eom=d,id=int(sid),omitted_drifted_weight=float(v),liquidation_cost_added=False))
            j=specs.index((1,'fixed512'));expected=baseline[d].loc[ids,'weight'].to_numpy()
            err=float(np.max(abs(weights[:,j]-expected)));max_identity=max(max_identity,err)
            np.testing.assert_allclose(weights[:,j],expected,atol=2e-8,rtol=1e-7)
            np.savez_compressed(folder/f'{d.date()}.npz',ids=ids,next_ids=nextids,pretrade=h,target=targets,weight=weights,trade=trades,drifted_weight=drift,next_pretrade=next_h,seeds=np.array([s for s,m in specs]),methods=np.array([m for s,m in specs]))
            h=next_h
            if d.month==11:print(f'Portfolio paths complete through return year {year}; original recovery error {max_identity:.3g}',flush=True)
        pd.DataFrame(monthly).to_parquet(folder/'monthly.parquet',index=False);pd.DataFrame(exit_rows).to_parquet(folder/'exit_ledger.parquet',index=False)
        write(OUT/'paths_manifest.json',dict(status='COMPLETE',months=len(dates),paths=len(specs),monthly_records=len(monthly),original_recovery_max_weight_error=max_identity,independent_position_accounting_all_months=True,first_trade_charged=True,annual_or_period_holdings_reset=False,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir() if p.is_file()}))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','fits','paths','all']);p.add_argument('--end-year',type=int,default=2025);args=p.parse_args()
    if args.stage=='initialize':initialize()
    elif args.stage=='fits':annual_fits(args.end_year)
    elif args.stage=='paths':portfolio_paths()
    else:
        if not (OUT/'plan_lock.json').exists():initialize()
        annual_fits(args.end_year)
        if args.end_year==2025:portfolio_paths()

if __name__=='__main__':main()
