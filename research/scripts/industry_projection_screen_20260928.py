"""One fixed industry-target projection and amplitude control; no training."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import sys
import uuid
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import portfolio_ensemble_screen_20260928 as shared
from adaptive_policy_screen_20260928 import verify_stage,stage_commit
from industry_projection_kernels_20260928 import project_industry_targets,FACTORS,INDUSTRIES
from baseline.core import transition
from inference.core import basic_interval

CONFIG=ROOT/'config/industry_projection_screen_20260928.json'
PARENT='output/portfolio_ensemble_screen_20260928'
ADAPTIVE='output/adaptive_policy_screen_20260928'
GEOMETRY='output/feature_geometry_study_20260914'
METHODS=('B1','BENS','CENS','IND','AMPL')
PAIRS=(('IND','B1'),('IND','BENS'),('IND','CENS'),('IND','AMPL'))
sha,read,save_json=shared.sha,shared.read,shared.save_json
ATOL,RTOL=1e-11,1e-10


def validate_config(config):
    if (config['decision_dates']!=['2015-12-31','2025-11-30'] or tuple(config['paths'])!=METHODS
            or tuple(map(tuple,config['comparisons']))!=PAIRS or config['gamma']!=10
            or config['training_authorized'] is not False
            or tuple(config['projection']['industry_names'])!=INDUSTRIES
            or config['projection']['parameter_or_subset_search'] is not False):
        raise ValueError('Frozen no-training industry projection has changed')
    if config['periods']!={'primary':['2016-01-31','2020-12-31'],'robustness':['2021-01-31','2025-12-31']}:
        raise ValueError('Frozen evaluation periods differ')
    for key,value in dict(draws=10000,expected_block_lengths=[12,24],alpha=.05,family=32).items():
        if config['inference'][key]!=value:raise ValueError('Frozen inference settings differ')


def prepare(config,out,guard=lambda:None):
    validate_config(config)
    if read(out/'protocol_freeze.json')['config_sha256']!=sha(CONFIG):raise ValueError('Pre-outcome configuration freeze differs')
    if read(out/'financial_preflight.json')!=read(ROOT/GEOMETRY/'financial_preflight.json'):
        raise ValueError('Fresh financial preflight differs from accepted parent')
    if (out/'lock.json').exists():
        lock=read(out/'lock.json')
        if lock['config']!=config:raise ValueError('Configuration differs from frozen lock')
        shared.verify_inventory(lock['artifact_sha256'],guard);return lock
    accepted={};consumed={};inventories=[]
    for parent in (PARENT,ADAPTIVE):
        path=parent+'/acceptance_manifest.json';manifest=read(ROOT/path)
        if manifest.get('accepted') is not True or not all(g['passed'] for g in manifest['gates']):
            raise ValueError('Accepted preceding diagnostic required')
        consumed[path]=sha(ROOT/path,guard)
        complete_path=parent+'/execution_complete.json'
        expected=manifest['source_sha256'][complete_path]
        if sha(ROOT/complete_path,guard)!=expected:raise ValueError('Parent execution inventory changed')
        consumed[complete_path]=expected
        inventories += [manifest['source_sha256'],manifest['artifact_sha256'],read(ROOT/complete_path)['artifact_sha256']]
    for relative in (GEOMETRY+'/acceptance_manifest.json',
                     'output/step4_20260911/acceptance_manifest.json',
                     'output/extension3_full_study_20260919/acceptance_manifest.json',
                     'output/extension4_full_study_20260924/acceptance_manifest.json',
                     'output/step3_author_conventions_20260911/bundle_checksums.json'):
        value=read(ROOT/relative)
        if 'gates' in value and not all(g['passed'] for g in value['gates']):raise ValueError('Failed prior acceptance')
        inventories.append(value.get('artifact_sha256',value));consumed[relative]=sha(ROOT/relative,guard)
    for inventory in inventories:
        for relative,digest in inventory.items():accepted.setdefault(relative,set()).add(digest)
    required=[GEOMETRY+'/financial_preflight.json','data/prepared/step4_v1/risk/factor_covariances.npz',
              PARENT+'/lock.json',PARENT+'/chronology.json','scripts/portfolio_ensemble_screen_20260928.py',
              'scripts/adaptive_policy_screen_20260928.py','scripts/adaptive_overlay_kernels_20260928.py',
              'baseline/core.py','baseline/io.py','baseline/risk.py','inference/core.py','extension4/runtime.py',
              'extension4/storage.py','temporal_fast_v4/memory.py','GOAL.md','docs/STEP_1_THEORY.md',
              'docs/RECONSTRUCTION_AMENDMENT.md','config/protocol_v1.json','config/reconstruction_amendment.json']
    for date in pd.date_range(*config['decision_dates'],freq='ME'):
        key=str(date.date())
        required += [f'{PARENT}/checkpoints/{key}/{name}' for name in ('arrays.npz','commit.json','monthly.json')]
        required.append(f'data/prepared/step9_v1/baseline/risk/{key}.npz')
    for parent in (PARENT,ADAPTIVE):
        required += [parent+'/intervals.json']
        for period in config['periods']:
            for length in (12,24):required.append(f'{parent}/analysis/bootstrap/{period}_L{length}_draws.csv')
    for period in config['periods']:
        for length in (12,24):required.append(f'{PARENT}/analysis/bootstrap/{period}_L{length}_indices.csv')
    for relative in required:
        choices=accepted.get(relative)
        if choices is None and relative.endswith('.py'):
            choices=accepted.get('output/extension4_full_study_20260924/lock/source_snapshot/'+relative)
        if choices is None or len(choices)!=1:raise ValueError('Missing/conflicting accepted input: '+relative)
        consumed[relative]=next(iter(choices))
    for path in (CONFIG,Path(__file__).resolve(),ROOT/'scripts/industry_projection_kernels_20260928.py',
                 ROOT/'tests/test_industry_projection.py',ROOT/config['projection']['factor_names_source'],
                 out/'financial_preflight.json',out/'protocol_freeze.json'):
        consumed[str(path.relative_to(ROOT))]=sha(path,guard)
    shared.verify_inventory(consumed,guard)
    factor_names=read(ROOT/config['projection']['factor_names_source'])['factors']
    if tuple(factor_names)!=FACTORS:raise ValueError('Accepted factor metadata/order does not match frozen names')
    # The historical calendar is not itself listed in an old acceptance
    # inventory. Verify its ordering against explicitly labelled, accepted
    # covariance artifacts instead of representing its current hash as an old
    # acceptance. The configured calendar remains a locked metadata dependency.
    with np.load(ROOT/'data/prepared/step4_v1/risk/factor_covariances.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['factors'],np.array(factor_names))
    lock=dict(config=config,created_at_utc=shared.now(),artifact_sha256=consumed,
              factor_names=factor_names,
              accepted_factor_label_source='data/prepared/step4_v1/risk/factor_covariances.npz',
              calendar_matches_accepted_factor_labels=True,
              scope='Target industry neutrality only; actual holdings unconstrained',new_models_fitted=0)
    save_json(out/'lock.json',lock);return lock


def replay_month(parent,risk,holdings,factor_names):
    np.testing.assert_array_equal(parent['ids'],risk['ids'])
    projection=project_industry_targets(parent['targets'][:,6],parent['targets'][:,7],risk['loadings'],
        risk['factor_cov'],risk['specific_variance'],parent['impact'],float(parent['wealth']),factor_names)
    industry=risk['loadings'][:,:12]
    np.testing.assert_allclose(projection['industry_exposure_after'],0,atol=1e-10,rtol=0)
    np.testing.assert_allclose(projection['q_times_correction'],industry@projection['multipliers'],atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(projection['qnorm_before'],projection['qnorm_after']+projection['qnorm_correction'],atol=1e-10,rtol=1e-9)
    targets=np.column_stack([parent['targets'][:,0],parent['targets'][:,6],parent['targets'][:,7],projection['IND'],projection['AMPL']])
    arrays=dict(ids=parent['ids'].copy(),next_ids=parent['next_ids'].copy(),methods=np.array(METHODS),
                targets=targets,pretrade=holdings.copy(),weights=np.empty_like(holdings),trades=np.empty_like(holdings),
                drifted_weights=np.empty_like(holdings),next_pretrade=np.empty((len(parent['next_ids']),5)),
                loadings=risk['loadings'],factor_cov=risk['factor_cov'],specific_variance=risk['specific_variance'],
                factor_names=np.array(factor_names))
    arrays.update({key:np.asarray(value) for key,value in projection.items() if key not in ('IND','AMPL','BENS','CENS')})
    rows=[]
    for j,method in enumerate(METHODS):
        result=transition(parent['ids'],holdings[:,j],targets[:,j],parent['m'],parent['total_returns'],
                          float(parent['market_growth']),parent['impact'],float(parent['wealth']),
                          parent['excess_returns'],parent['next_ids'])
        for key in ('weights','trades','drifted_weights','next_pretrade'):arrays[key][:,j]=result[key]
        rows.append(dict(method=method,gross_excess=result['gross_excess'],impact_cost=result['impact_cost'],
                         net_excess=result['gross_excess']-result['impact_cost'],turnover=result['turnover'],gross_exposure=result['gross_exposure']))
    for j,parent_j in ((0,0),(1,6),(2,7)):
        for key in ('pretrade','targets','weights','trades','drifted_weights','next_pretrade'):
            np.testing.assert_allclose(arrays[key][:,j],parent[key][:,parent_j],atol=ATOL,rtol=RTOL)
    neutral=project_industry_targets(parent['targets'][:,6],parent['targets'][:,7],risk['loadings'],risk['factor_cov'],
        risk['specific_variance'],parent['impact'],float(parent['wealth']),factor_names,force_neutral=True)
    np.testing.assert_array_equal(neutral['IND'],parent['targets'][:,7])
    result=transition(parent['ids'],holdings[:,2],neutral['IND'],parent['m'],parent['total_returns'],
                      float(parent['market_growth']),parent['impact'],float(parent['wealth']),parent['excess_returns'],parent['next_ids'])
    for key in ('weights','trades','drifted_weights','next_pretrade'):
        np.testing.assert_allclose(result[key],arrays[key][:,2],atol=ATOL,rtol=RTOL)
    arrays['actual_industry_exposure']=industry.T@arrays['weights']
    arrays['actual_industry_increment_vs_bens']=arrays['actual_industry_exposure']-arrays['actual_industry_exposure'][:,1,None]
    return arrays,rows


def run_paths(config,out,lock,runtime):
    directory=out/'checkpoints';directory.mkdir(exist_ok=True);lock_hash=sha(out/'lock.json')
    previous=None;ids=None;holdings=None;all_rows=[]
    for count,date in enumerate(pd.date_range(*config['decision_dates'],freq='ME'),1):
        runtime.guard();shared.native_pressure();key=str(date.date());folder=directory/key
        parent_relative=f'{PARENT}/checkpoints/{key}/arrays.npz';risk_relative=f'data/prepared/step9_v1/baseline/risk/{key}.npz'
        if folder.exists():
            commit=verify_stage(folder,lock_hash,runtime.guard)
            if commit['previous_commit_sha256']!=previous or commit['decision_eom']!=key:raise ValueError('Broken checkpoint chain')
            with np.load(folder/'arrays.npz',allow_pickle=False) as z:
                if ids is not None:
                    np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(holdings,z['pretrade'])
                ids=z['next_ids'].copy();holdings=z['next_pretrade'].copy()
            rows=read(folder/'monthly.json')
        else:
            with np.load(ROOT/parent_relative,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
            with np.load(ROOT/risk_relative,allow_pickle=False) as z:risk={k:z[k].copy() for k in z.files}
            if ids is None:ids=parent['ids'];holdings=np.repeat(parent['pretrade'][:,0,None],5,axis=1)
            np.testing.assert_array_equal(ids,parent['ids'])
            arrays,rows=replay_month(parent,risk,holdings,lock['factor_names'])
            for row in rows:row.update(decision_eom=key,return_eom=str((date+pd.offsets.MonthEnd(1)).date()))
            temporary=directory/('.'+key+'.partial_'+uuid.uuid4().hex);temporary.mkdir()
            with (temporary/'arrays.npz').open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            save_json(temporary/'monthly.json',rows)
            stage_commit(temporary,lock_hash,decision_eom=key,previous_commit_sha256=previous,
                         parent_packet=parent_relative,parent_packet_sha256=lock['artifact_sha256'][parent_relative],
                         risk_source=risk_relative,risk_source_sha256=lock['artifact_sha256'][risk_relative],
                         neutral_control_checked=True,target_neutrality_scope_only=True)
            runtime.guard();shared.native_pressure();temporary.rename(folder)
            ids,holdings=arrays['next_ids'],arrays['next_pretrade']
        previous=sha(folder/'commit.json');all_rows.extend(rows)
        runtime.event('month_complete',months=count,total_months=120,decision_eom=key)
    return pd.DataFrame(all_rows),previous


def gate(metrics,intervals):
    lookup={(x['period'],x['method']):x['source_utility'] for x in metrics}
    point=all(lookup[period,'IND']>lookup[period,control] for period in ('primary','robustness') for _,control in PAIRS)
    primary=[x for x in intervals if x['period']=='primary']
    if len(primary)!=8 or {(x['candidate'],x['control'],x['block_months']) for x in primary}!={('IND',b,l) for _,b in PAIRS for l in (12,24)}:
        raise ValueError('Missing fixed primary contrasts')
    inference=all(x['lower']>0 for x in primary);passed=bool(point and inference)
    return dict(candidates={'IND':dict(positive_vs_all_controls_both_periods=bool(point),
         adjusted_primary_lower_bounds_positive=bool(inference),passed=passed)},
         exploratory_gate_passed=passed,candidate_for_larger_study_consideration='IND' if passed else None,
         automatic_training_authorized=False,fresh_confirmation=False,published_paper_outperformance_established=False)


def analyze(config,frame,destination,guard=lambda:None):
    destination.mkdir(exist_ok=False);(destination/'bootstrap').mkdir();frame.to_csv(destination/'monthly.csv',index=False)
    metrics=[];intervals=[];prior_intervals=[]
    for period,endpoints in config['periods'].items():
        gross=[];cost=[];dates=list(pd.date_range(*endpoints,freq='ME'))
        for method in METHODS:
            rows=frame[frame.method==method].copy();rows['return_eom']=pd.to_datetime(rows.return_eom)
            rows=rows[rows.return_eom.between(*endpoints)].sort_values('return_eom')
            if list(rows.return_eom)!=dates:raise ValueError('Incomplete evaluation period')
            np.testing.assert_allclose(rows.net_excess,rows.gross_excess-rows.impact_cost,atol=ATOL,rtol=RTOL)
            gross.append(rows.gross_excess.to_numpy());cost.append(rows.impact_cost.to_numpy())
        gross=np.column_stack(gross);cost=np.column_stack(cost);net=gross-cost;utility=shared.source_utility(gross,net)
        for j,method in enumerate(METHODS):
            metrics.append(dict(period=period,method=method,months=60,source_utility=float(utility[j]),
                net_population_utility=float(12*(net[:,j].mean()-5*net[:,j].var(ddof=0))),
                annual_gross_volatility=float(np.sqrt(12*gross[:,j].var(ddof=1))),
                annual_net_mean=float(12*net[:,j].mean()),annual_cost=float(12*cost[:,j].mean())))
        for length in (12,24):
            guard();prefix=f'{period}_L{length}'
            indices=np.loadtxt(ROOT/PARENT/'analysis/bootstrap'/(prefix+'_indices.csv'),delimiter=',',dtype=np.int32)-1
            if indices.shape!=(10000,60) or indices.min()<0 or indices.max()>=60:raise ValueError('Invalid reused bootstrap indices')
            draws=np.empty((10000,4))
            for start in range(0,10000,250):
                guard();ix=indices[start:start+250];u=shared.source_utility(gross[ix],net[ix],axis=1)
                for j,(a,b) in enumerate(PAIRS):draws[start:start+250,j]=u[:,METHODS.index(a)]-u[:,METHODS.index(b)]
            pd.DataFrame(draws,columns=[a+'_vs_'+b for a,b in PAIRS]).to_csv(destination/'bootstrap'/(prefix+'_draws.csv'),index=False)
            for j,(a,b) in enumerate(PAIRS):
                observed=float(utility[METHODS.index(a)]-utility[METHODS.index(b)])
                lower,upper=basic_interval(observed,draws[:,j],alpha=.05,family=32)
                intervals.append(dict(period=period,candidate=a,control=b,block_months=length,
                                      observed=observed,lower=lower,upper=upper,family=32,draws=10000))
            for prior in (PARENT,ADAPTIVE):
                prior_draws=pd.read_csv(ROOT/prior/'analysis/bootstrap'/(prefix+'_draws.csv'))
                for old in read(ROOT/prior/'intervals.json'):
                    if old['period']!=period or old['block_months']!=length:continue
                    column=old['candidate']+'_vs_'+old['control']
                    lower,upper=basic_interval(old['observed'],prior_draws[column].to_numpy(),alpha=.05,family=32)
                    prior_intervals.append(dict(old,lower=lower,upper=upper,family=32,source_study=prior))
    if len(intervals)!=16 or len(prior_intervals)!=48:raise ValueError('Combined fixed inference family incomplete')
    pd.DataFrame(metrics).to_csv(destination/'metrics.csv',index=False)
    save_json(destination/'intervals.json',intervals);save_json(destination/'combined_prior_intervals.json',prior_intervals)
    save_json(destination/'result.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',gate=gate(metrics,intervals),
        periods=config['periods'],months=120,paths=list(METHODS),no_models_refitted=True,independent_review_required=True,
        combined_family=32,prior_results_modified=False,target_neutrality_scope_only=True))


def run(config,out):
    if config.get('execution_authorized') is not True:raise ValueError('Execution not authorized')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'execution_complete.json').exists():
            shared.verify_inventory(read(out/'execution_complete.json')['artifact_sha256'])
            print('Completed industry screen verified without rewriting.');return
        shared.native_pressure();runtime=shared.GuardedRuntime(out,config,'industry_target_projection_screen')
        try:
            lock=prepare(config,out,runtime.guard);runtime.event('lock_verified',no_fitting=True)
            frame,final_checkpoint=run_paths(config,out,lock,runtime);analysis=out/'analysis';lock_hash=sha(out/'lock.json')
            if analysis.exists():
                commit=verify_stage(analysis,lock_hash,runtime.guard)
                if commit['final_checkpoint_sha256']!=final_checkpoint:raise ValueError('Analysis inputs changed')
            else:
                temporary=out/('.analysis.partial_'+uuid.uuid4().hex);analyze(config,frame,temporary,runtime.guard)
                stage_commit(temporary,lock_hash,final_checkpoint_sha256=final_checkpoint)
                runtime.guard();shared.native_pressure();temporary.rename(analysis)
            names=('monthly.csv','metrics.csv','intervals.json','combined_prior_intervals.json','result.json')
            for name in names:
                target=out/name
                if target.exists():
                    if sha(target)!=sha(analysis/name):raise ValueError('Existing public summary differs')
                else:
                    with target.open('xb') as stream:stream.write((analysis/name).read_bytes());stream.flush();os.fsync(stream.fileno())
            shared.verify_inventory(lock['artifact_sha256'],runtime.guard)
            artifacts=[out/'lock.json',*[p for d in ('checkpoints','analysis') for p in (out/d).rglob('*') if p.is_file() and '.partial_' not in str(p)],
                       *[out/name for name in names]]
            runtime.guard();shared.native_pressure()
            save_json(out/'execution_complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
                artifact_sha256={str(p.relative_to(ROOT)):sha(p,runtime.guard) for p in artifacts}))
            runtime.event('screen_computed_pending_review',**read(out/'result.json')['gate']);runtime.close('COMPUTED_PENDING_INDEPENDENT_REVIEW')
        except BaseException as error:runtime.close('STOPPED',error);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare==args.run:parser.error('Choose exactly one of --prepare or --run')
    config=read(CONFIG);validate_config(config);out=ROOT/config['output_root']
    if out.resolve()!=ROOT/'output/industry_projection_screen_20260928':raise ValueError('Unexpected output root')
    if args.prepare:
        out.mkdir(parents=True,exist_ok=True);prepare(config,out);print('Industry inputs and code locked; no new historical path calculated.')
    else:run(config,out)


if __name__=='__main__':main()
