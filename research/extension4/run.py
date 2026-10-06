"""Approved pilot -> full chronological study -> independent numerical report.

No command creates approval. A general implementation request cannot silently
approve the scientific exceptions or a historical run. Existing results are
resumed by verified commits, never overwritten.
"""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
from baseline.full import preflight
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_study.paths import AcceptedPathSource
from .authority import require_approval
from .parent import ParentCatalog, PARENT
from .state import StateBook, build_states
from .history import TrainingBatches
from .design import capacity_map
from .training import collect, fit_moments
from .model import ConditionalModel
from .storage import commit_stage, verify_stage, write_json
from .runtime import Runtime
from .selection import select, candidate
from .paths import replay
from .analysis import analyze
from .report import render

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def implementation_hashes(root):
    files=[*root.glob('extension4/*.py'),root/'scripts/extension4_study.py',root/'scripts/extension4_independent.R']
    return {str(p.relative_to(root)):sha(p) for p in files}


def require_engineering(root):
    path=root/'output/extension4_engineering_20260925/engineering_manifest.json'
    evidence=read(path)
    if (evidence.get('status')!='ENGINEERING_VERIFIED_HISTORICAL_NOT_RUN'
            or evidence.get('implementation_sha256')!=implementation_hashes(root)
            or not all(evidence.get('checks',{}).get(k) is True for k in
                       ('unit_integration','independent_R_artificial','actual_parent_neutral','financial_preflight'))):
        raise ValueError('Verified engineering checkpoint does not match the execution code')
    for rel,digest in evidence['artifact_sha256'].items():
        artifact=(root/rel).resolve()
        if not artifact.is_relative_to(path.parent) or sha(artifact)!=digest:
            raise ValueError('Engineering evidence changed: '+rel)
    return sha(path)


def validate_protocol(config):
    d=config['state_definitions'];c=config['interaction_columns_policy'];s=config['size_control']
    if (len(d)!=1 or d[0]['field']!='be_me' or d[0]['weights']!='equal'
            or d[0]['universe']!='parent_eligible_at_state_date' or not d[0]['positive_finite']
            or not d[0]['positive_finite_market_equity'] or d[0]['timing_certified']
            or c['indices']!=list(range(1,513)) or c['include_parent_constant']
            or c['normalize_interactions'] or not c['interact_before_dynamic_operator']
            or config['history']['first_return_year']!=1981 or config['history']['last_return_year']!=2025
            or config['history']['training_first_decision']!='1965-11-30'
            or config['history']['exit_liquidation_cost']!='omitted_as_in_parent_with_explicit_ledger'
            or not s['parent_beta_frozen'] or s['a_bounds']!=[-1/3,1/3]
            or config['controls']!=['P3','N4','C4','CAP4','SIZE4','E2','F1024']):
        raise ValueError('Protocol requires a scientific/implementation revision; unsupported choices cannot be ignored')
    if config['inference']['bootstrap']!='paired_circular_stationary' or config['inference']['interval']!='basic_two_sided':
        raise ValueError('Unsupported inference method')
    # Explicitly reject changes to fixed settings that this runner does not generalize.
    fixed={
        'pilot':dict(year=1981,seed=1,training_months=181,economic_fits=9,restricted_parent_check=1,
                     replay_return_months=12,late_probe_parent_year=2025,economic_performance_gate=False),
        'penalty_search':dict(base_ridge_unchanged=True,base_constant_penalized=True,criterion='annual_net_utility',per_seed_and_family=True),
        'capacity_control':dict(extra_nonlinear_features=512,frozen_parent_transform=True,source='accepted_nested_1024_dictionary',constant_duplicate=False),
        'history':dict(initial_holdings='accepted_1980_12_pretrade',continuous_own_holdings=True),
        'inference':dict(seed_average='mean_of_statistics_not_ensemble',refit_in_bootstrap=False,independent_language='R'),
        'compute_budget':dict(blas_threads=1,one_heavy_job=True,full_launch_projection_contingency=2.)}
    for section,values in fixed.items():
        if any(config[section].get(k)!=v for k,v in values.items()):raise ValueError('Unsupported fixed setting in '+section)
    if (config['primary_comparisons']!=[['C4','P3'],['C4','CAP4'],['C4','SIZE4']]
            or config['validation_plan']['selection_returns']!=['2011-01-31','2015-12-31']
            or config['validation_plan']['primary_returns']!=['2016-01-31','2020-12-31']
            or config['validation_plan']['robustness_returns']!=['2021-01-31','2025-12-31']
            or config['inference']['expected_block_lengths']!=[12,24]):
        raise ValueError('Selection, evaluation or comparison family requires reviewed implementation changes')
    if (config['state_standardization']['clip']!=3. or len(config['penalty_search']['multipliers_in_tie_order'])!=3
            or config['counts']!=dict(new_economic_fits=1035,size_scalar_fits=345,restricted_parent_checks=135,
                total_coefficient_solves=1170,candidate_paths=27,candidate_months=420,final_paths=21,final_months=540,final_rows=11340)):
        raise ValueError('Exposure bound or experiment counts require an implementation revision')


class Study:
    def __init__(self,root,config_path,approval_path):
        self.root=Path(root).resolve();self.config_path=Path(config_path);self.approval_path=Path(approval_path)
        self.config,self.approval=require_approval(root,config_path,approval_path)
        validate_protocol(self.config)
        engineering_sha256=require_engineering(self.root)
        self.destination=self.root/self.config['paths']['proposed_full_root']
        if not self.destination.resolve().is_relative_to(self.root/'output'):raise ValueError('Study must remain in project output')
        self.catalog=ParentCatalog(root)
        for relative in self.catalog.inventory:self.catalog.path(relative)
        self.destination.mkdir(parents=True,exist_ok=True)
        self.identity_context=dict(config_sha256=sha(config_path),approval_sha256=sha(approval_path),engineering_sha256=engineering_sha256,
            study_plan_sha256=sha(root/self.config['paths']['study_plan']),parent_acceptance_sha256=self.catalog.acceptance_sha256,
            implementation_sha256=implementation_hashes(self.root),python=sys.version,
            numerical_versions=dict(numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__),platform=platform.platform())
        def writer(folder):
            write_json(folder/'lock.json',self.identity_context)
            for rel in self.identity_context['implementation_sha256']:
                p=folder/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.root/rel,p)
        commit_stage(self.destination/'lock',self.identity_context,writer)
        self.lock=sha(self.destination/'lock'/'commit.json')
        self.model_cache={};self.geometry={};self.extra_cache={};self.parents={}

    def parent(self,year,seed):
        key=(year,seed)
        if key not in self.parents:self.parents[key]=self.catalog.load(year,seed)
        parent=self.parents[key]
        self.catalog.path(parent.model_path);parent.model._check_unchanged()
        return parent

    def setup(self,runtime):
        self.runtime=runtime
        receipt,_=preflight()
        if receipt!=self.catalog.read(PARENT+'/financial_preflight.json'):raise ValueError('Parent financial dependencies changed')
        commit_stage(self.destination/'preflight',dict(lock=self.lock),lambda p:write_json(p/'receipt.json',receipt))
        self.history=StudyHistory(self.root,receipt,weights=[1/6]*6)
        self.source=AcceptedPathSource(self.history)
        self.initial=self.source.common_initial()
        self.cache=TrainingBatches(self.history,int(self.config['compute_budget']['max_memory_cache_gib']*2**30),runtime.guard)
        state_dir=self.destination/'states'
        def writer(folder):
            d=self.config['state_definitions'][0]
            raw_path=self.catalog.path(d['source'])
            rank_rel='data/prepared/step3_author_v3/monthly_inputs.parquet'
            eligible=pd.read_parquet(self.catalog.path(rank_rel),columns=['id','eom','valid'])
            raw=pd.read_parquet(raw_path,columns=['id','eom','be_me','me'])
            self.book=build_states(raw,eligible,definition=d,normalizer=self.config['state_standardization'],
                source_metadata=dict(unranked=True,timing_mode='SOURCE_LAG_PROXY_UNCERTIFIED',
                    source_sha256=self.catalog.consumed[d['source']],eligibility_sha256=self.catalog.consumed[rank_rel],
                    source_lag_months=d['source_lag_months'],source_max_age_months=d['source_max_age_months'],
                    actual_accounting_age_observed=False,revision_vintages_available=False),
                first=str(raw.eom.min().date()),last=f"{self.config['history']['last_return_year']}-11-30")
            self.book.frame.to_parquet(folder/'history.parquet',index=False)
            write_json(folder/'metadata.json',self.book.metadata)
            write_json(folder/'identity.json',dict(identity=self.book.identity))
        commit_stage(state_dir,dict(lock=self.lock),writer)
        self.book=StateBook(pd.read_parquet(state_dir/'history.parquet'),read(state_dir/'metadata.json'))
        if self.book.identity!=read(state_dir/'identity.json')['identity']:raise ValueError('State serialization changed its identity')
        runtime.event('setup_verified',state_months=len(self.book.frame),state_identity=self.book.identity)

    def extra(self,parent):
        key=(parent.year,parent.seed)
        if key not in self.extra_cache:
            meta=self.catalog.read(f'output/feature_geometry_study_20260914/fits/{parent.year}/seed{parent.seed}/full/fit.json')
            rel=f"output/feature_geometry_study_20260914/directions/seed{parent.seed}_g{meta['bandwidth_index']}.bin"
            self.extra_cache[key]=capacity_map(parent.model,np.fromfile(self.catalog.path(rel),dtype='<f8').reshape(115,512))
        return self.extra_cache[key]

    def fit(self,year,seed,choices=None):
        if implementation_hashes(self.root)!=self.identity_context['implementation_sha256']:
            raise ValueError('Locked implementation changed during execution')
        if sha(self.config_path)!=self.identity_context['config_sha256'] or sha(self.approval_path)!=self.identity_context['approval_sha256']:
            raise ValueError('Locked protocol or authority changed during execution')
        parent=self.parent(year,seed);extra=self.extra(parent)
        grid=self.config['penalty_search']['multipliers_in_tie_order']
        families={m:([choices[seed,m]] if choices else grid) for m in ('C4','CAP4','SIZE4')}
        folder=self.destination/'fits'/str(year)/f'seed{seed}'
        context=dict(lock=self.lock,year=year,seed=seed,parent=parent.model._content_hash(),states=self.book.identity,multipliers=families)
        def writer(folder):
            start=time.monotonic();dates=pd.date_range(self.config['history']['training_first_decision'],f'{year-1}-11-30',freq='ME')
            timings=[];previous=time.monotonic()
            def progress(count,date):
                nonlocal previous
                elapsed=time.monotonic()-previous;previous=time.monotonic();timings.append(elapsed)
                if count%12==0 or count==len(dates):self.runtime.event('fit_progress',year=year,seed=seed,month=count,total=len(dates),decision=date)
                self.runtime.guard()
            collected=collect(self.cache.batches(dates,parent.model.cutoff),parent,self.book,extra,
                              expected_dates=dates,progress=progress)
            fits=fit_moments(collected,parent,grid,size_bounds=self.config['size_control']['a_bounds'],
                atol=self.config['tolerances']['neutral_atol'],rtol=self.config['tolerances']['neutral_rtol'],family_multipliers=families)
            with (folder/'moments.npz').open('xb') as f:
                np.savez_compressed(f,**{method+'_'+k:v for method,m in collected['moments'].items() for k,v in m.items()})
            receipts=[]
            for method,multiplier,beta,diagnostics in fits:
                name=method if multiplier is None else candidate(method,multiplier)
                model=ConditionalModel(parent,method,beta,state_identity=self.book.identity,protocol_identity=self.lock,
                    extras=extra if method=='CAP4' else None,multiplier=multiplier,
                    size_bounds=self.config['size_control']['a_bounds'] if method=='SIZE4' else None)
                model.save(folder/(name+'.npz'))
                restored=ConditionalModel.load(folder/(name+'.npz'),parent,state_identity=self.book.identity,
                    protocol_identity=self.lock,extras=extra if method=='CAP4' else None)
                if restored.identity!=model.identity:raise ValueError('Fit serialization changed model')
                receipts.append(dict(method=method,multiplier=multiplier,model_file=name+'.npz',model_identity=model.identity,diagnostics=diagnostics))
            write_json(folder/'fit.json',dict(context=context,models=receipts,training_months=len(dates),first=str(dates[0].date()),
                last=str(dates[-1].date()),cutoff=parent.model.cutoff,seconds=time.monotonic()-start,
                month_seconds=timings,affine=collected['affine'],cache=self.cache.diagnostics(),
                input_sha256=dict(self.history.files.consumed),parent_ridge=parent.settings.ridge,
                parent_scale_status=parent.receipt['parent_scale_optimizer_status'],parent_selection_available_at=parent.selection_available_at))
        created=commit_stage(folder,context,writer)
        self.runtime.event('fit_complete' if created else 'fit_reused',year=year,seed=seed)
        return read(folder/'fit.json')

    def model(self,year,seed,method,multiplier):
        key=(year,seed,method,multiplier)
        if key not in self.model_cache:
            folder=self.destination/'fits'/str(year)/f'seed{seed}';verify_stage(folder)
            parent=self.parent(year,seed)
            name=method if method=='N4' else candidate(method,multiplier)
            self.model_cache[key]=ConditionalModel.load(folder/(name+'.npz'),parent,state_identity=self.book.identity,
                protocol_identity=self.lock,extras=self.extra(parent) if method=='CAP4' else None)
        return self.model_cache[key]

    def aim(self,seed,method,market,bound,choices):
        year=(pd.Timestamp(market.decision_date)+pd.offsets.MonthEnd(1)).year
        if method=='P3':return self.parent(year,seed).model.aim(market.decision_date,market.ids,market.characteristics[0],market.volatility[0],bound)
        if method in ('E2','F1024'):
            key=(year,seed,method)
            if key not in self.geometry:
                label='full' if method=='E2' else 'fixed1024'
                self.geometry[key]=GeometryModel.load(self.catalog.path(f'output/feature_geometry_study_20260914/fits/{year}/seed{seed}/{label}/model.npz'))
            return self.geometry[key].aim(market.characteristics[0],market.volatility[0],market.decision_date)
        if '__m' in method:
            kind,value=method.split('__m');multiplier=float(value)
        else:
            kind=method;multiplier=None if method=='N4' else choices[seed,method]
        model=self.model(year,seed,kind,multiplier)
        return model.aim(market,bound,None if kind=='N4' else self.book.bind(market.decision_date))

    def compare(self,date,arrays,rows):
        tolerance=self.config['tolerances'];old={}
        for col,(seed,method) in enumerate(zip(arrays['seeds'],arrays['methods'])):
            if method not in ('P3','N4','E2','F1024'):continue
            relative=(PARENT+'/selected_paths/'+date+'.npz' if method in ('P3','N4')
                      else 'output/feature_geometry_study_20260914/paths/'+date+'.npz')
            if relative not in old:
                with np.load(self.catalog.path(relative),allow_pickle=False) as saved:old[relative]={k:saved[k].copy() for k in saved.files}
            saved=old[relative];label='selected_temporal' if method in ('P3','N4') else ('full' if method=='E2' else 'fixed1024')
            indices=np.flatnonzero((saved['seeds']==seed)&(saved['methods']==label))
            if len(indices)!=1:raise ValueError('Archived benchmark path missing')
            for key in ('ids','next_ids'):np.testing.assert_array_equal(arrays[key],saved[key])
            for key in ('pretrade','target','weight','trade','drifted_weight','next_pretrade'):
                np.testing.assert_allclose(arrays[key][:,col],saved[key][:,indices[0]],
                    atol=tolerance['neutral_atol'],rtol=tolerance['neutral_rtol'],err_msg='Archived control replay differs')

    def paths(self,name,last,specs,choices):
        dates=pd.date_range(self.initial.decision_date,last,freq='ME')
        policy_identity=fingerprint(self.lock,self.book.identity,sorted((s,m,v) for (s,m),v in choices.items()))
        return replay(self.source,dates,specs,self.initial,lambda s,m,market,b:self.aim(s,m,market,b,choices),
            destination=self.destination/name,lock_identity=self.lock,policy_identity=policy_identity,
            atol=self.config['tolerances']['accounting_atol'],rtol=self.config['tolerances']['accounting_rtol'],
            compare=self.compare,progress=lambda n,d:self.runtime.event('path_progress',path=name,months=n,decision=d))

    def pilot(self):
        gate=self.destination/'pilot_gate'
        if gate.exists():verify_stage(gate,dict(lock=self.lock));return read(gate/'result.json')
        self.setup(self.runtime)
        receipt=self.fit(1981,1)
        grid=self.config['penalty_search']['multipliers_in_tie_order']
        specs=[(1,'P3'),(1,'N4')]+[(1,candidate(m,p)) for m in ('C4','CAP4','SIZE4') for p in grid]
        start=time.monotonic();paths=self.paths('pilot_paths','1981-11-30',specs,{})
        replay_seconds=time.monotonic()-start
        # Resume must consume the same committed path without recomputing trades.
        pd.testing.assert_frame_equal(paths,self.paths('pilot_paths','1981-11-30',specs,{}))
        parent=self.parent(2025,1);dates=self.config['pilot']['late_probe_decisions'];start=time.monotonic()
        collect(self.cache.batches(dates,parent.model.cutoff),parent,self.book,self.extra(parent),expected_dates=dates,
                progress=lambda n,d:self.runtime.event('late_probe',month=n,decision=d))
        late_seconds=time.monotonic()-start
        max_universe=int(self.book.frame.eligible_count.max())
        late_counts=self.book.frame[self.book.frame.eom.isin(pd.to_datetime(dates))].eligible_count
        size_factor=max(1.,max_universe/float(late_counts.max()))**3
        seconds_per_month=max(receipt['seconds']/receipt['training_months'],late_seconds/len(dates))*size_factor
        fitting_months=sum(181+12*(y-1981) for y in range(1981,2026))*3
        fits_projection=fitting_months*seconds_per_month
        path_projection=replay_seconds/(12*len(specs))*(27*420+21*540)
        verification_reserve=max(3600.,2*path_projection)
        projection=2*(fits_projection+path_projection+verification_reserve)
        remaining=self.config['compute_budget']['total_active_seconds']-self.runtime.previous-(time.monotonic()-self.runtime.start)
        result=dict(status='PASS' if projection<=remaining else 'RESOURCE_PROJECTION_EXCEEDS_BUDGET',
            lock=self.lock,pilot_fit_count=9,parent_coefficient_checks=1,replay_months=12,replay_paths=len(specs),
            saved_resume_equal=True,late_probe_months=dates,late_probe_seconds=late_seconds,
            pilot_fit_seconds=receipt['seconds'],pilot_replay_seconds=replay_seconds,
            full_projected_seconds_with_contingency=projection,remaining_active_seconds=remaining,
            projection_basis=dict(total_training_months=fitting_months,conservative_size_multiplier=size_factor,
                fitting_seconds=fits_projection,path_seconds=path_projection,
                unmeasured_verification_reserve_seconds=verification_reserve,contingency_factor=2.),
            economic_performance_used_as_gate=False,technical_checks_passed=True)
        commit_stage(gate,dict(lock=self.lock),lambda p:write_json(p/'result.json',result))
        if result['status']!='PASS':raise RuntimeError('Measured pilot projects beyond approved resource budget')
        return result

    def full(self):
        self.setup(self.runtime)
        if read(self.destination/'pilot_gate'/'result.json')['status']!='PASS':raise RuntimeError('Passing technical pilot required')
        grid=self.config['penalty_search']['multipliers_in_tie_order'];seeds=self.config['history']['seeds']
        end=pd.Timestamp(self.config['validation_plan']['selection_returns'][1]).year
        for year in range(1981,end+1):
            for seed in seeds:self.fit(year,seed)
        specs=[(s,candidate(m,p)) for s in seeds for m in ('C4','CAP4','SIZE4') for p in grid]
        frame=self.paths('candidate_paths',f'{end}-11-30',specs,{})
        selection_folder=self.destination/'selection'
        def writer(folder):
            selected=select(frame,first=self.config['validation_plan']['selection_returns'][0],
                last=self.config['validation_plan']['selection_returns'][1],seeds=seeds,multipliers=grid,
                tie_tolerance=self.config['penalty_search']['tie_tolerance'],
                parent_selection_available_at=self.catalog.selection['selection_available_at'])
            write_json(folder/'selected.json',selected)
        commit_stage(selection_folder,dict(lock=self.lock),writer)
        selected=read(selection_folder/'selected.json');choices={(r['seed'],r['method']):r['multiplier'] for r in selected}
        self.runtime.event('selection_frozen',choices=selected)
        for year in range(end+1,2026):
            for seed in seeds:self.fit(year,seed,choices)
        specs=[(s,m) for s in seeds for m in self.config['controls']]
        frame=self.paths('final_paths','2025-11-30',specs,choices)
        def final_writer(folder):frame.to_parquet(folder/'monthly.parquet',index=False)
        commit_stage(self.destination/'final_monthly',dict(lock=self.lock),final_writer)
        from .verification import verify_financial_paths, verify_all_fits, verify_states
        def verification_writer(folder):
            states=verify_states(self.book,pd.read_parquet(self.catalog.path(self.config['state_definitions'][0]['source']),
                columns=['id','eom','be_me','me']),pd.read_parquet(self.catalog.path('data/prepared/step3_author_v3/monthly_inputs.parquet'),
                columns=['id','eom','valid']))
            fits=verify_all_fits(self)
            paths=verify_financial_paths(self)
            write_json(folder/'result.json',dict(status='PASS',fits=fits,paths=paths,states=states))
        commit_stage(self.destination/'independent_accounting',dict(lock=self.lock),verification_writer)
        def analysis_writer(folder):analyze(frame,self.config,folder,independent_script=self.root/'scripts/extension4_independent.R')
        commit_stage(self.destination/'analysis',dict(lock=self.lock),analysis_writer)
        def report_writer(folder):
            receipt=render(self.destination/'analysis',folder/'completion_report.html')
            write_json(folder/'render_receipt.json',receipt)
        commit_stage(self.destination/'report',dict(lock=self.lock),report_writer)
        self.history.files.unchanged()
        for relative in tuple(self.catalog.consumed):self.catalog.path(relative)
        self.runtime.sample()
        self.runtime.event('numerics_complete_report_review_pending',economic_fits=1035,neutral_fit_checks=135,paths=21,months=540)
        commit_stage(self.destination/'numerical_completion',dict(lock=self.lock),lambda p:write_json(p/'result.json',
            dict(status='NUMERICS_COMPLETE_REPORT_REVIEW_PENDING',study_complete=False,full_report_review_pending=True,
                 config_sha256=sha(self.config_path),approval_sha256=sha(self.approval_path),lock=self.lock)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['status','run'])
    parser.add_argument('--config',type=Path,default=ROOT/'config/extension4_study_v1_proposed.json')
    parser.add_argument('--approval',type=Path,default=ROOT/'config/extension4_end_to_end_approval.json')
    args=parser.parse_args()
    if args.command=='status':
        config=read(args.config);folder=ROOT/config['paths']['proposed_full_root']
        print(json.dumps(dict(proposal_status=config['approval_status'],approval_receipt_exists=args.approval.exists(),
            study_directory_exists=folder.exists(),financial_run_launched=any((folder/'attempts').glob('*_start.json')),
            pilot_gate=read(folder/'pilot_gate/result.json')['status'] if (folder/'pilot_gate/result.json').exists() else None,
            numerical_completion=(folder/'numerical_completion/commit.json').exists(),
            scientific_acceptance=(folder/'acceptance_manifest.json').exists()),indent=2));return
    # Fail before creating any study output when the required authority is absent.
    config,_=require_approval(ROOT,args.config,args.approval)
    validate_protocol(config)
    destination=(ROOT/config['paths']['proposed_full_root']).resolve()
    if not destination.is_relative_to(ROOT/'output'):raise ValueError('Study must remain in project output')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with (ROOT/'output/.extension3_heavy_job.lock').open('a+') as heavy,single_thread_blas():
        fcntl.flock(heavy,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Account for parent hashing and lock construction as well as fitting.
        phase='full' if (destination/'pilot_gate').exists() else 'pilot'
        runtime=Runtime(destination,config,phase)
        try:
            study=Study(ROOT,args.config,args.approval);study.runtime=runtime
            runtime.event('attempt_started')
            if phase=='full':
                verify_stage(study.destination/'pilot_gate',dict(lock=study.lock))
                if read(study.destination/'pilot_gate'/'result.json')['status']!='PASS':raise RuntimeError('Pilot resource gate requires review')
            else:
                study.pilot();runtime.close('COMPLETE')
                runtime=None
                runtime=Runtime(study.destination,study.config,'full');study.runtime=runtime
                runtime.event('attempt_started')
            study.full();runtime.close('COMPLETE')
        except BaseException as error:
            if runtime is not None:runtime.close('STOPPED',error)
            raise


if __name__=='__main__':main()
