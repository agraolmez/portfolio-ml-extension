"""Approved staged one-seed 1981–2020 study and 2020–2025 extension; resumable and local."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import fcntl,gc,time,traceback,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from temporal_fast.backend import single_thread_blas
from annual_staged.common import *
from annual_staged.runtime import Runtime
from annual_staged.setup import setup,verify_prefix
from annual_staged.baseline import initial_parent,parent_choices,check_original_selections,select_extensions
from annual_staged.fitting import fit_year
from annual_staged.evaluation import score_year,evaluate_paths
from annual_staged.verification import verify_system
from annual_staged.report import write_report
from annual_ind.checkpoints import CheckpointStore
from annual_ind.runtime_expanded import prior_seconds
from extension4.storage import write_json,verify_stage


def verify_lock():
    lock=read(OUT/'lock.json')
    if lock['configuration_sha256']!=sha(CONFIG):raise ValueError('Approved settings changed')
    for rel,digest in lock['code_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Locked code changed: '+rel)
    for rel,digest in lock['source_manifest_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Accepted source manifest changed: '+rel)
    for rel,digest in lock['required_checks_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Required integration check changed: '+rel)
    return lock


def project_remaining(out,annual,full_scores,runtime):
    # Conservative proxies retained from the accepted assessment, updated by
    # measured integrated candidates. This is a gate/estimate, not a guarantee.
    late=[x['temporal_seconds'] for y,a in annual.items() if y>=2011 for x in a['summary']['candidates'] if x['temporal_new']]
    late_per=max([2108.9/14*20]+late)
    conditional=[x['conditional_seconds'] for a in annual.values() for x in a['summary']['candidates'] if x['new_moments']]
    cond_per=max([330.]+conditional)
    early=[x['temporal_seconds'] for y,a in annual.items() if y<1981 for x in a['summary']['candidates'] if x['temporal_new']]
    early_per=max([171.83]+early)
    remaining=set(range(1971,2026))-set(annual)
    early_n=3*sum(y<1981 for y in remaining);late_n=2*sum(y>=2011 for y in remaining)
    groups=3*sum(y<1981 for y in remaining)+2*sum(y>=1981 for y in remaining)
    # Preserve the original complete 1h preparation + 4h scoring/review allowance.
    expected=early_n*early_per+late_n*late_per+groups*cond_per+5*3600
    spent=runtime.previous+time.monotonic()-runtime.start;available=runtime.budget['total_active_seconds']-spent
    receipt=dict(at_utc=now(),remaining_hours=expected/3600,budget_remaining_hours=available/3600,
        basis=dict(early_temporal=early_n,late_temporal=late_n,conditional_groups=groups,
                   early_temporal_seconds=early_per,late_temporal_seconds=late_per,conditional_seconds=cond_per,
                   preparation_validation_report_reserve_seconds=5*3600),passed=expected<=available)
    atomic(out/'projection.json',receipt)
    if not receipt['passed']:raise TimeoutError('Conservative updated runtime projection exceeds remaining approved 48-hour budget')
    return receipt


def main():
    lock=verify_lock();config=read(CONFIG);identity=sha(OUT/'lock.json')
    guardfile=(ROOT/'output/.annual_ind_probe.lock').open('a')
    try:fcntl.flock(guardfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another annual research calculation already holds the computation lock')
    runtime=Runtime(OUT,config,'staged');store=None;error=None;status='STOPPED'
    try:
        state(current_status='STAGED_INTEGRATION_RUNNING',background_computation_currently_running=True,
              next_work='Complete integrated checks, then automatically run all approved years if the timing gate passes.')
        atomic(OUT/'development.json',dict(at_utc=now(),status='numerical',label='Approved integrated study is running'))
        with threadpool_limits(limits=1),single_thread_blas():
            catalog=Catalog();history,cache,inputs,book=setup(config,catalog,OUT,runtime)
            startup=initial_parent(catalog,OUT/'initial_parent',sha(CONFIG),runtime)
            choices=parent_choices(catalog,startup)
            check_original_selections(catalog,choices,runtime)
            store=CheckpointStore(OUT/'models.sqlite',run_identity=identity)
            store.inventory()
            annual={};scores={}
            # Integrated trial includes an early 64-column parent, archived 1981
            # candidates/neutral reconstruction and a late missing-candidate year.
            order=[1971,1981,2020]+[y for y in range(1971,2026) if y not in (1971,1981,2020)]
            for index,year in enumerate(order):
                if index==3:
                    projection=project_remaining(OUT,annual,scores,runtime)
                    state(current_status='STAGED_HISTORICAL_RUNNING',full_study_launched=True,full_run_resource_gate_passed=True,
                          next_work='Complete annual fitting and selection through 2025, then evaluate and verify the continuous 540-month portfolios.')
                    runtime.event('full_run_gate_passed',projection=projection)
                annual[year]=fit_year(choices[year],history,cache,inputs,book,catalog,store,runtime,OUT,verify_system,
                                     check_archived_neutral=year==1981 and not (OUT/'archived_neutral.json').exists())
                if year==1981 and not (OUT/'archived_neutral.json').exists():write_json(OUT/'archived_neutral.json',dict(passed=True,lock=identity))
                scores[year]=score_year(year,annual[year]['models'],history,cache,inputs,book,OUT,identity,runtime)
                atomic(OUT/'year_progress.json',dict(completed=len(annual),total=55,years=sorted(annual)))
                atomic(OUT/'annual_summaries.json',{str(y):a['summary'] for y,a in annual.items()})
                runtime.event('annual_candidate_bank_complete',year=year,completed_years=len(annual),total_years=55)
                gc.collect()
            all_scores=pd.concat([scores[y] for y in sorted(scores)],ignore_index=True)
            all_scores.to_parquet(OUT/'validation_scores.parquet',index=False)
            selections={y:select_extensions(all_scores,y) for y in range(1981,2026)}
            # A second reconstruction from individually truncated histories proves
            # that later rows cannot influence any chosen year.
            for year,s in selections.items():
                subset=all_scores[pd.to_datetime(all_scores.return_eom)<=pd.Timestamp(year-1,12,31)]
                if select_extensions(subset,year)!=s:raise ValueError('Future-score selection invariance failed')
            atomic(OUT/'selections.json',{str(y):s for y,s in selections.items()})
            state(current_status='STAGED_PORTFOLIOS_RUNNING',next_work='Evaluate continuous monthly portfolios and independently verify accounting and inference.')
            frame=evaluate_paths(history,book,choices,selections,{y:a['models'] for y,a in annual.items()},catalog,OUT,identity,runtime)
            runtime.event('phase_progress',label='Verifying results and preparing report',completed=None,total=None)
            inventory=store.inventory();atomic(OUT/'checkpoint_inventory.json',inventory)
            catalog.unchanged();history.files.unchanged();runtime.guard()
            verification=dict(annual_candidates=55,monthly_validation_scores=len(all_scores),annual_selections=45,
                continuous_returns=540,portfolios=3,baseline_reproduction=True,independent_accounting=True,
                independent_coefficient_solves=True,chronological_selection=True,convergence_gates=True)
            result=write_report(frame,OUT,verification,identity)
            atomic(OUT/'consumed_sources.json',dict(catalog=catalog.consumed,history=history.files.consumed))
            store.close();store=None
            files={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('live.json','progress.jsonl','stdout.log','resource_ledger.json') and 'attempts' not in p.parts and not p.name.endswith(('-wal','-shm'))}
            atomic(OUT/'numerical_completion_manifest.json',dict(status='NUMERICALLY_COMPLETE_REVIEW_PENDING',artifact_sha256=files,verification=verification))
            runtime.event('numerical_study_complete',months=540)
            status='COMPLETE'
    except BaseException as exc:
        error=exc;atomic(OUT/f'stop_{time.time_ns()}.json',dict(error=str(exc),type=type(exc).__name__,traceback=traceback.format_exc()));raise
    finally:
        if store is not None:store.close()
        runtime.close(status,error)
        state(current_status='STAGED_NUMERICALLY_COMPLETE_REVIEW_PENDING' if status=='COMPLETE' else 'STAGED_STOPPED_CHECKPOINTS_PRESERVED',
            background_computation_currently_running=False,numerical_study_complete=status=='COMPLETE',
            next_work='Review the generated report and complete final project acceptance.' if status=='COMPLETE' else 'Inspect the saved failure and resume from verified checkpoints after resolving it.',
            last_stop_reason=None if error is None else str(error))
        guardfile.close()

if __name__=='__main__':main()
