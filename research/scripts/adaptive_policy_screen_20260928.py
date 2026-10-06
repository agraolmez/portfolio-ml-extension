"""Frozen agreement/risk overlays of accepted targets; no model training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import portfolio_ensemble_screen_20260928 as shared
from adaptive_overlay_kernels_20260928 import agreement_targets, risk_scale
from baseline.core import transition
from inference.core import basic_interval

CONFIG=ROOT/'config/adaptive_policy_screen_20260928.json'
PARENT='output/portfolio_ensemble_screen_20260928'
GEOMETRY='output/feature_geometry_study_20260914'
METHODS=('B1','BENS','CENS','AGREE','ASIZE','RISK','RBASE')
PAIRS=(('AGREE','B1'),('AGREE','BENS'),('AGREE','CENS'),('AGREE','ASIZE'),
       ('RISK','B1'),('RISK','BENS'),('RISK','CENS'),('RISK','RBASE'))
sha,read,save_json=shared.sha,shared.read,shared.save_json
ATOL,RTOL=1e-11,1e-10


def validate_config(config):
    if (config['decision_dates']!=['2015-12-31','2025-11-30']
            or tuple(config['paths'])!=METHODS or tuple(map(tuple,config['comparisons']))!=PAIRS
            or config['gamma']!=10 or config['training_authorized'] is not False
            or config['risk_calibration']['window_months']!=60
            or config['agreement']['variance_ddof']!=1):
        raise ValueError('Frozen target-only experiment changed')
    if config['periods']!={'primary':['2016-01-31','2020-12-31'],
                           'robustness':['2021-01-31','2025-12-31']}:
        raise ValueError('Frozen evaluation dates changed')
    for key,value in dict(draws=10000,expected_block_lengths=[12,24],alpha=.05,family=24).items():
        if config['inference'][key]!=value:raise ValueError('Frozen inference changed')


def prepare(config,out,guard=lambda:None):
    validate_config(config)
    freeze=read(out/'protocol_freeze.json')
    if freeze['config_sha256']!=sha(CONFIG):raise ValueError('Configuration differs from pre-outcome freeze')
    if read(out/'financial_preflight.json')!=read(ROOT/GEOMETRY/'financial_preflight.json'):
        raise ValueError('Fresh financial preflight differs from accepted parent')
    if (out/'lock.json').exists():
        lock=read(out/'lock.json')
        if lock['config']!=config:raise ValueError('Frozen configuration differs')
        shared.verify_inventory(lock['artifact_sha256'],guard)
        return lock
    acceptance=read(ROOT/config['parent_screen'])
    if acceptance.get('accepted') is not True or not all(x['passed'] for x in acceptance['gates']):
        raise ValueError('Completed accepted source screen required')
    complete_path=PARENT+'/execution_complete.json'
    if sha(ROOT/complete_path,guard)!=acceptance['source_sha256'][complete_path]:
        raise ValueError('Parent execution inventory differs from acceptance')
    parent_complete=read(ROOT/complete_path)
    accepted={}
    consumed={config['parent_screen']:sha(ROOT/config['parent_screen'],guard),
              complete_path:acceptance['source_sha256'][complete_path]}
    manifests=[GEOMETRY+'/acceptance_manifest.json',
               'output/extension3_full_study_20260919/acceptance_manifest.json',
               'output/extension4_full_study_20260924/acceptance_manifest.json']
    inventories=[parent_complete['artifact_sha256'],acceptance['artifact_sha256'],acceptance['source_sha256']]
    for relative in manifests:
        manifest=read(ROOT/relative)
        if not all(x['passed'] for x in manifest['gates']):raise ValueError('Prior failed acceptance')
        inventories.append(manifest['artifact_sha256']);consumed[relative]=sha(ROOT/relative,guard)
    for inventory in inventories:
        for relative,digest in inventory.items():accepted.setdefault(relative,set()).add(digest)
    required=[GEOMETRY+'/paths/monthly.parquet',GEOMETRY+'/financial_preflight.json',
              PARENT+'/lock.json',PARENT+'/chronology.json',PARENT+'/monthly.csv',
              PARENT+'/intervals.json',PARENT+'/analysis/commit.json',
              'scripts/portfolio_ensemble_screen_20260928.py','baseline/core.py','baseline/io.py',
              'inference/core.py','extension4/runtime.py','extension4/storage.py','temporal_fast_v4/memory.py',
              'GOAL.md','docs/STEP_1_THEORY.md','docs/RECONSTRUCTION_AMENDMENT.md',
              'config/protocol_v1.json','config/reconstruction_amendment.json']
    for date in pd.date_range(*config['decision_dates'],freq='ME'):
        key=str(date.date());required+=[f'{PARENT}/checkpoints/{key}/{name}' for name in ('arrays.npz','commit.json','monthly.json')]
    for date in pd.date_range(*config['risk_calibration']['reference_decisions'],freq='ME'):
        key=str(date.date());required += [f'{GEOMETRY}/paths/{key}.npz',f'data/prepared/step9_v1/baseline/risk/{key}.npz']
    for period in config['periods']:
        for length in (12,24):
            required += [f'{PARENT}/analysis/bootstrap/{period}_L{length}_{name}.csv' for name in ('indices','draws')]
    for relative in required:
        choices=accepted.get(relative)
        if choices is None and relative.endswith('.py'):
            choices=accepted.get('output/extension4_full_study_20260924/lock/source_snapshot/'+relative)
        if choices is None or len(choices)!=1:raise ValueError('Missing/conflicting accepted input: '+relative)
        consumed[relative]=next(iter(choices))
    for path in [CONFIG,Path(__file__).resolve(),ROOT/'scripts/adaptive_overlay_kernels_20260928.py',
                 ROOT/'tests/test_adaptive_overlays.py',out/'protocol_freeze.json',out/'financial_preflight.json']:
        consumed[str(path.relative_to(ROOT))]=sha(path,guard)
    shared.verify_inventory(consumed,guard)
    lock=dict(config=config,created_at_utc=shared.now(),artifact_sha256=consumed,
              new_models_fitted=0,scope='Common-start target overlays; frozen baseline shadow risk reference')
    save_json(out/'lock.json',lock)
    return lock


def verify_stage(folder,lock_hash,guard=lambda:None):
    commit=read(folder/'commit.json')
    if commit['lock_sha256']!=lock_hash:raise ValueError('Stage protocol differs')
    for relative,digest in commit['files_sha256'].items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path,guard)!=digest:
            raise ValueError('Changed committed stage file: '+relative)
    return commit


def stage_commit(folder,lock_hash,**extra):
    files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    save_json(folder/'commit.json',dict(lock_sha256=lock_hash,files_sha256=files,**extra))


def make_reference(config,out,lock,runtime):
    destination=out/'reference';lock_hash=sha(out/'lock.json')
    if destination.exists():
        verify_stage(destination,lock_hash,runtime.guard)
        return pd.read_csv(destination/'observations.csv')
    temporary=out/('.reference.partial_'+uuid.uuid4().hex);temporary.mkdir()
    packets=temporary/'packets';packets.mkdir()
    frame=pd.read_parquet(ROOT/GEOMETRY/'paths/monthly.parquet')
    frame=frame[(frame.method=='fixed512')&(frame.seed==1)].copy()
    frame['decision_eom']=pd.to_datetime(frame.decision_eom)
    if frame.duplicated('decision_eom').any():raise ValueError('Duplicate shadow-reference observations')
    frame=frame.set_index('decision_eom')
    rows=[];sources={}
    for date in pd.date_range(*config['risk_calibration']['reference_decisions'],freq='ME'):
        runtime.guard();shared.native_pressure();key=str(date.date())
        position=f'{GEOMETRY}/paths/{key}.npz';risk=f'data/prepared/step9_v1/baseline/risk/{key}.npz'
        with np.load(ROOT/position,allow_pickle=False) as z:
            column=np.flatnonzero((z['methods']=='fixed512')&(z['seeds']==1))
            if len(column)!=1:raise ValueError('Missing/duplicate baseline shadow portfolio')
            ids=z['ids'].copy();weights=z['weight'][:,column[0]].copy()
        with np.load(ROOT/risk,allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids'])
            loadings=z['loadings'].copy();factor=z['factor_cov'].copy();specific=z['specific_variance'].copy()
        if (not all(np.isfinite(x).all() for x in (weights,loadings,factor,specific))
                or np.any(specific<0)):
            raise ValueError('Invalid accepted risk-factor components')
        expected_return=date+pd.offsets.MonthEnd(1)
        if pd.Timestamp(frame.loc[date,'return_eom'])!=expected_return:
            raise ValueError('Reference return was assigned to an incorrect maturity date')
        factor_weight=weights@loadings
        predicted=float(factor_weight@factor@factor_weight+np.sum(weights*weights*specific))
        gross=float(frame.loc[date,'gross_excess'])
        if not np.isfinite(predicted) or predicted<0 or not np.isfinite(gross):
            raise ValueError('Invalid reference variance or gross return')
        with (packets/(key+'.npz')).open('xb') as stream:
            np.savez_compressed(stream,ids=ids,weights=weights,loadings=loadings,factor_cov=factor,
                                specific_variance=specific,gross_excess=np.array(gross))
        rows.append(dict(decision_eom=key,return_eom=str(expected_return.date()),
                         gross_excess=gross,predicted_variance=predicted))
        sources[key]={position:lock['artifact_sha256'][position],risk:lock['artifact_sha256'][risk]}
    reference=pd.DataFrame(rows)
    if len(reference)!=179:raise ValueError('Expected 179 fixed reference months')
    reference.to_csv(temporary/'observations.csv',index=False)
    save_json(temporary/'source_hashes.json',dict(monthly_file=GEOMETRY+'/paths/monthly.parquet',
        monthly_sha256=lock['artifact_sha256'][GEOMETRY+'/paths/monthly.parquet'],months=sources))
    stage_commit(temporary,lock_hash,reference_months=179)
    runtime.guard();shared.native_pressure();temporary.rename(destination)
    runtime.event('reference_complete',months=179)
    return reference


def replay_month(parent,holdings,agreement,scale):
    targets=np.column_stack([parent['targets'][:,0],parent['targets'][:,6],parent['targets'][:,7],
                             agreement['AGREE'],agreement['ASIZE'],
                             scale['k']*parent['targets'][:,7],scale['k']*parent['targets'][:,6]])
    arrays=dict(ids=parent['ids'].copy(),next_ids=parent['next_ids'].copy(),methods=np.array(METHODS),
                component_baseline=parent['targets'][:,:3].copy(),component_c4=parent['targets'][:,3:6].copy(),
                targets=targets,pretrade=holdings.copy(),weights=np.empty_like(holdings),
                trades=np.empty_like(holdings),drifted_weights=np.empty_like(holdings),
                next_pretrade=np.empty((len(parent['next_ids']),7)),
                agreement_g=agreement['g'],dbar=agreement['dbar'],v=agreement['v'],
                h=np.array(agreement['h']),risk_k=np.array(scale['k']),
                predicted_variance_mean=np.array(scale['predicted_variance_mean']),
                realized_variance=np.array(scale['realized_variance']),reference_indices=scale['indices'])
    rows=[]
    for column,method in enumerate(METHODS):
        result=transition(parent['ids'],holdings[:,column],targets[:,column],parent['m'],
                          parent['total_returns'],float(parent['market_growth']),parent['impact'],
                          float(parent['wealth']),parent['excess_returns'],parent['next_ids'])
        for key in ('weights','trades','drifted_weights','next_pretrade'):arrays[key][:,column]=result[key]
        rows.append(dict(method=method,gross_excess=result['gross_excess'],impact_cost=result['impact_cost'],
                         net_excess=result['gross_excess']-result['impact_cost'],turnover=result['turnover'],
                         gross_exposure=result['gross_exposure']))
    for column,parent_column in ((0,0),(1,6),(2,7)):
        for key in ('pretrade','targets','weights','trades','drifted_weights','next_pretrade'):
            np.testing.assert_allclose(arrays[key][:,column],parent[key][:,parent_column],atol=ATOL,rtol=RTOL)
    # Neutral overlays are replayed from the CENS path's own state, avoiding a
    # false comparison with the active overlay's different inherited holdings.
    neutral=agreement_targets(parent['targets'][:,:3],parent['targets'][:,3:6],force_neutral=True)
    for aim in (neutral['AGREE'],1.*parent['targets'][:,7]):
        result=transition(parent['ids'],holdings[:,2],aim,parent['m'],parent['total_returns'],
                          float(parent['market_growth']),parent['impact'],float(parent['wealth']),
                          parent['excess_returns'],parent['next_ids'])
        for key in ('weights','trades','drifted_weights','next_pretrade'):
            np.testing.assert_allclose(result[key],arrays[key][:,2],atol=ATOL,rtol=RTOL)
    return arrays,rows


def run_paths(config,out,reference,runtime):
    directory=out/'checkpoints';directory.mkdir(exist_ok=True)
    lock_hash=sha(out/'lock.json');previous=None;ids=None;holdings=None;all_rows=[]
    for count,date in enumerate(pd.date_range(*config['decision_dates'],freq='ME'),1):
        runtime.guard();shared.native_pressure();key=str(date.date());folder=directory/key
        parent_relative=f'{PARENT}/checkpoints/{key}/arrays.npz'
        if folder.exists():
            commit=verify_stage(folder,lock_hash,runtime.guard)
            if commit['previous_commit_sha256']!=previous or commit['decision_eom']!=key:
                raise ValueError('Broken adaptive checkpoint chain')
            with np.load(folder/'arrays.npz',allow_pickle=False) as z:
                if ids is not None:
                    np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(holdings,z['pretrade'])
                ids=z['next_ids'].copy();holdings=z['next_pretrade'].copy()
            rows=read(folder/'monthly.json')
        else:
            with np.load(ROOT/parent_relative,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
            if ids is None:
                ids=parent['ids'];holdings=np.repeat(parent['pretrade'][:,0,None],7,axis=1)
            np.testing.assert_array_equal(ids,parent['ids'])
            agreement=agreement_targets(parent['targets'][:,:3],parent['targets'][:,3:6])
            scale=risk_scale(key,reference.return_eom.to_numpy(),reference.gross_excess.to_numpy(),
                             reference.predicted_variance.to_numpy())
            arrays,rows=replay_month(parent,holdings,agreement,scale)
            for row in rows:row.update(decision_eom=key,return_eom=str((date+pd.offsets.MonthEnd(1)).date()))
            temporary=directory/('.'+key+'.partial_'+uuid.uuid4().hex);temporary.mkdir()
            with (temporary/'arrays.npz').open('xb') as stream:
                np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            save_json(temporary/'monthly.json',rows)
            stage_commit(temporary,lock_hash,decision_eom=key,previous_commit_sha256=previous,
                         parent_packet=parent_relative,parent_packet_sha256=sha(ROOT/parent_relative),
                         reference_commit_sha256=sha(out/'reference/commit.json'),
                         neutral_controls_checked=True,window_first=scale['window_first'],window_last=scale['window_last'])
            runtime.guard();shared.native_pressure();temporary.rename(folder)
            ids,holdings=arrays['next_ids'],arrays['next_pretrade']
        previous=sha(folder/'commit.json');all_rows.extend(rows)
        runtime.event('month_complete',months=count,total_months=120,decision_eom=key)
    return pd.DataFrame(all_rows),previous


def gate(metrics,intervals):
    lookup={(r['period'],r['method']):r['source_utility'] for r in metrics};decisions={}
    for candidate in ('AGREE','RISK'):
        controls=[b for a,b in PAIRS if a==candidate]
        point=all(lookup[p,candidate]>lookup[p,b] for p in ('primary','robustness') for b in controls)
        relevant=[x for x in intervals if x['period']=='primary' and x['candidate']==candidate]
        if len(relevant)!=8 or {(x['control'],x['block_months']) for x in relevant}!={(b,l) for b in controls for l in (12,24)}:
            raise ValueError('Missing primary candidate comparisons')
        inference=all(x['lower']>0 for x in relevant)
        decisions[candidate]=dict(positive_vs_all_controls_both_periods=bool(point),
            adjusted_primary_lower_bounds_positive=bool(inference),passed=bool(point and inference))
    selected=next((x for x in ('AGREE','RISK') if decisions[x]['passed']),None)
    return dict(candidates=decisions,candidate_for_larger_study_consideration=selected,
                exploratory_gate_passed=selected is not None,automatic_training_authorized=False,
                fresh_confirmation=False,published_paper_outperformance_established=False)


def analyze(config,frame,destination,guard=lambda:None):
    destination.mkdir(exist_ok=False);frame.to_csv(destination/'monthly.csv',index=False)
    metrics=[];intervals=[];prior_intervals=[];(destination/'bootstrap').mkdir()
    prior_points=read(ROOT/PARENT/'intervals.json')
    for period,endpoints in config['periods'].items():
        gross=[];cost=[];expected=list(pd.date_range(*endpoints,freq='ME'))
        for method in METHODS:
            rows=frame[frame.method==method].copy();rows['return_eom']=pd.to_datetime(rows.return_eom)
            rows=rows[rows.return_eom.between(*endpoints)].sort_values('return_eom')
            if list(rows.return_eom)!=expected:raise ValueError('Incomplete monthly evaluation')
            np.testing.assert_allclose(rows.net_excess,rows.gross_excess-rows.impact_cost,atol=ATOL,rtol=RTOL)
            gross.append(rows.gross_excess.to_numpy());cost.append(rows.impact_cost.to_numpy())
        gross=np.column_stack(gross);cost=np.column_stack(cost);net=gross-cost
        utility=shared.source_utility(gross,net)
        for j,method in enumerate(METHODS):
            metrics.append(dict(period=period,method=method,months=60,source_utility=float(utility[j]),
                net_population_utility=float(12*(net[:,j].mean()-5*net[:,j].var(ddof=0))),
                annual_gross_volatility=float(np.sqrt(12*gross[:,j].var(ddof=1))),
                annual_net_mean=float(12*net[:,j].mean()),annual_cost=float(12*cost[:,j].mean())))
        for length in (12,24):
            guard();prefix=f'{period}_L{length}'
            index_path=ROOT/PARENT/'analysis/bootstrap'/(prefix+'_indices.csv')
            indices=np.loadtxt(index_path,delimiter=',',dtype=np.int32)-1
            if indices.shape!=(10000,60) or indices.min()<0 or indices.max()>=60:
                raise ValueError('Reused index stream differs from fixed contract')
            draws=np.empty((10000,len(PAIRS)))
            for start in range(0,10000,250):
                guard();ix=indices[start:start+250];u=shared.source_utility(gross[ix],net[ix],axis=1)
                for j,(a,b) in enumerate(PAIRS):draws[start:start+250,j]=u[:,METHODS.index(a)]-u[:,METHODS.index(b)]
            pd.DataFrame(draws,columns=[a+'_vs_'+b for a,b in PAIRS]).to_csv(
                destination/'bootstrap'/(prefix+'_draws.csv'),index=False)
            for j,(a,b) in enumerate(PAIRS):
                observed=float(utility[METHODS.index(a)]-utility[METHODS.index(b)])
                lower,upper=basic_interval(observed,draws[:,j],alpha=.05,family=24)
                intervals.append(dict(period=period,candidate=a,control=b,block_months=length,
                                      observed=observed,lower=lower,upper=upper,family=24,draws=10000))
            prior_draws=pd.read_csv(ROOT/PARENT/'analysis/bootstrap'/(prefix+'_draws.csv'))
            for old in prior_points:
                if old['period']!=period or old['block_months']!=length:continue
                contrast=old['candidate']+'_vs_'+old['control']
                lower,upper=basic_interval(old['observed'],prior_draws[contrast].to_numpy(),alpha=.05,family=24)
                prior_intervals.append(dict(old,lower=lower,upper=upper,family=24,
                    source='Preserved previous screen draws; new combined-family interval only'))
    pd.DataFrame(metrics).to_csv(destination/'metrics.csv',index=False)
    save_json(destination/'intervals.json',intervals)
    save_json(destination/'combined_prior_intervals.json',prior_intervals)
    save_json(destination/'result.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',gate=gate(metrics,intervals),
        periods=config['periods'],months=120,paths=list(METHODS),no_models_refitted=True,
        independent_review_required=True,combined_family=24,prior_results_modified=False))


def run(config,out):
    if config.get('execution_authorized') is not True:raise ValueError('Execution not authorized')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'execution_complete.json').exists():
            shared.verify_inventory(read(out/'execution_complete.json')['artifact_sha256'])
            print('Completed adaptive screen verified without rewriting.');return
        shared.native_pressure();runtime=shared.GuardedRuntime(out,config,'adaptive_saved_target_screen')
        try:
            lock=prepare(config,out,runtime.guard);runtime.event('lock_verified',no_fitting=True)
            reference=make_reference(config,out,lock,runtime)
            frame,final_checkpoint=run_paths(config,out,reference,runtime)
            analysis=out/'analysis';lock_hash=sha(out/'lock.json')
            if analysis.exists():
                commit=verify_stage(analysis,lock_hash,runtime.guard)
                if commit['final_checkpoint_sha256']!=final_checkpoint:raise ValueError('Analysis checkpoint differs')
            else:
                temporary=out/('.analysis.partial_'+uuid.uuid4().hex)
                analyze(config,frame,temporary,runtime.guard)
                stage_commit(temporary,lock_hash,final_checkpoint_sha256=final_checkpoint)
                runtime.guard();shared.native_pressure();temporary.rename(analysis)
            names=('monthly.csv','metrics.csv','intervals.json','combined_prior_intervals.json','result.json')
            for name in names:
                target=out/name
                if target.exists():
                    if sha(target)!=sha(analysis/name):raise ValueError('Existing public summary differs')
                else:
                    with target.open('xb') as stream:
                        stream.write((analysis/name).read_bytes());stream.flush();os.fsync(stream.fileno())
            shared.verify_inventory(lock['artifact_sha256'],runtime.guard)
            artifacts=[out/'lock.json',*[p for d in ('reference','checkpoints','analysis') for p in (out/d).rglob('*') if p.is_file() and '.partial_' not in str(p)],
                       *[out/name for name in names]]
            runtime.guard();shared.native_pressure()
            save_json(out/'execution_complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
                artifact_sha256={str(p.relative_to(ROOT)):sha(p,runtime.guard) for p in artifacts}))
            runtime.event('screen_computed_pending_review',**read(out/'result.json')['gate'])
            runtime.close('COMPUTED_PENDING_INDEPENDENT_REVIEW')
        except BaseException as error:
            runtime.close('STOPPED',error);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare==args.run:parser.error('Choose exactly one of --prepare or --run')
    config=read(CONFIG);validate_config(config);out=ROOT/config['output_root']
    if out.resolve()!=ROOT/'output/adaptive_policy_screen_20260928':raise ValueError('Unexpected output root')
    if args.prepare:
        out.mkdir(parents=True,exist_ok=True);prepare(config,out)
        print('Inputs and code locked; no adaptive historical performance calculated.')
    else:run(config,out)


if __name__=='__main__':main()
