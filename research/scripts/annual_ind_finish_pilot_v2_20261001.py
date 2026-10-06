"""Prepared continuation of the remaining pilot layers; explicit added-time approval required.

Never launches historical validation/deployment. The original candidate grid is
unchanged; two mechanically chosen timing coordinates do not select a model.
Run with --resume only after an interrupted/resource-stopped attempt, using the
same lock and completed-layer checkpoints. All attempts consume the same budget.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import sqlite3
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from baseline.full import preflight
from feature_geometry.history import sha
from temporal_fast_v4.history import StudyHistory
from temporal_fast.backend import single_thread_blas
from extension4.storage import write_json,verify_stage
from extension4.runtime import prior_seconds
from annual_ind.inputs import OperatorCache
from annual_ind.panels import CalendarSourcePanels
from annual_ind.checkpoints import CheckpointStore
from annual_ind.coordinate_cached import fit_coordinate
from annual_ind.cached_inputs import ReusedInputCache, open_parent_store
from annual_ind.preparation_checks import PreparationFiles, PreparationCheckedInputCache
from annual_ind.preparation_verification import verify_preparation
from annual_ind.cache_verification import verify_existing_inputs
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime

OUT=ROOT/'output/annual_ind_runner_pilot_v4_20261001'
PREVIOUS=ROOT/'output/annual_ind_runner_pilot_v3_20260930'
WORK=ROOT/'output/annual_ind_finish_launch_20261001'
APPROVAL=ROOT/'config/annual_ind_finish_approval_20261001.json'
STATE=ROOT/'config/annual_ind_execution_state_20260929.json'


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def record_state(**values):
    current=json.loads(STATE.read_text());current.update(values);atomic(STATE,current)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    lockfile=(ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reference=ROOT/'output/annual_ind_fresh_probe_pipeline_r50_20260929'
    if args.resume:
        lock=json.loads((OUT/'lock.json').read_text())
        if (OUT/'result.json').exists():raise ValueError('Pilot already completed; nothing to resume')
    else:
        if OUT.exists():raise ValueError('Existing pilot requires explicit --resume; never overwrite')
        ledger_path=WORK/'previous_resource_ledger.json'
        ledger=json.loads(ledger_path.read_text())
        tests=json.loads((WORK/'unit_tests_final.json').read_text())
        preparation_checks=json.loads((WORK/'preparation_checks.json').read_text())
        approval=json.loads(APPROVAL.read_text())
        if approval.get('status')!='APPROVED':raise PermissionError('Additional pilot time has not been approved')
        if tests['exit_code']!=0:raise ValueError('Streaming numerical tests must pass')
        old=json.loads((reference/'lock.json').read_text())
        for block in ('code_sha256','protected_sha256'):
            for rel,expected in old[block].items():
                if sha(ROOT/rel)!=expected:raise ValueError('Accepted parent/source changed: '+rel)
        # Retain a conservative 120-second review allowance plus measured tests.
        reserved=120.+tests['seconds']+tests['preliminary_test_seconds']+preparation_checks['financial_preflight_seconds']+2.
        seconds=approval['additional_compute_seconds']-reserved
        if approval['prior_charged_seconds']!=ledger['charged_seconds']:raise ValueError('Prior budget carry changed')
        if seconds<=0:raise TimeoutError('No engineering time remains')
        limits=dict(total_active_seconds=seconds,pilot_active_seconds=seconds,
            max_physical_footprint_gib=16.,min_system_headroom_gib=4.,
            max_additional_disk_gib=2.,min_free_disk_gib=50.)
        code=list((ROOT/'annual_ind').glob('*.py'))+[Path(__file__),
            ROOT/'scripts/annual_ind_pipeline_probe_20260929.py',ROOT/'config/annual_ind_seed1_20260929.json',APPROVAL]
        lock=dict(at_utc=datetime.now(timezone.utc).isoformat(),
            purpose=__doc__,status='BOUNDED_FITTING_PILOT_NOT_FULL_STUDY',
            years=[1971,2020],seed=1,width=512,bandwidth_index=0,ridge_index=50,
            temporal_restraints=[.1],conditional_multipliers=[100.,10.,1.],
            coordinate_rule='first and final annual fitting windows; widest author features; first bandwidth; middle ridge; strongest temporal restraint',
            historical_selection=False,portfolio_paths=False,AI_calls=False,training_months=649,
            workers=8,blas_threads=1,operator_cache_bytes=5*2**30,
            temporal_intermediate_bytes=128*2**20,temporal_profile_bytes=128*2**20,
            calendar_source_bytes=256*2**20,input_cache_bytes=4*2**30,budget=limits,
            previous_pilot_lock_sha256=sha(PREVIOUS/'lock.json'),
            previous_pilot_database_sha256=sha(PREVIOUS/'models.sqlite'),
            prior_ledger=str(ledger_path.relative_to(ROOT)),prior_ledger_sha256=sha(ledger_path),
            prior_charged_seconds=ledger['charged_seconds'],review_and_test_reservation_seconds=reserved,
            additional_allowance_seconds=approval['additional_compute_seconds'],
            cumulative_engineering_cap_seconds=approval['cumulative_engineering_cap_seconds'],
            input_cache_study_lock_sha256=sha(ROOT/'output/extension3_full_study_20260919/study_lock.json'),
            input_cache_read_only=True,completed_2020_geometry_reused=True,
            unfinished_temporal_layer_starts_from_neutral=True,source_checks_before_and_after_month=True,
            full_run_runtime_limit_seconds=None,
            code_sha256={**old['code_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in code}},
            protected_sha256=old['protected_sha256'])
        OUT.mkdir();write_json(OUT/'lock.json',lock)
    for block in ('code_sha256','protected_sha256'):
        for rel,expected in lock[block].items():
            if sha(ROOT/rel)!=expected:raise ValueError('Locked pilot dependency changed: '+rel)
    for stage in ('fresh_parent','fresh_temporal','fresh_conditional'):verify_stage(reference/stage)
    runtime=None;store=None;error=None;status='STOPPED'
    try:
        runtime=ProbeRuntime(OUT,{'compute_budget':lock['budget']},'pilot')
        record_state(current_status='REMAINING_PILOT_LAYERS_RUNNING',
            current_attempt_directory=str(OUT.relative_to(ROOT)),
            current_resource_ledger=str((WORK/'resource_ledger.json').relative_to(ROOT)),
            implementation_status_file=str((WORK/'development.json').relative_to(ROOT)),
            pilot_additional_time_approval_pending=False,
            background_computation_currently_running=True,
            next_work='Complete temporal and conditional pilot layers using saved scales; review the result before budgeting any full historical experiment.')
        atomic(WORK/'development.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
            status='numerical',label='Fresh fitting pilot is running',completed=False))
        runtime.event('phase_progress',label='Checking financial inputs and checkpoints',completed=None,total=None)
        receipt,_=preflight()
        if not (OUT/'financial_preflight.json').exists():write_json(OUT/'financial_preflight.json',receipt)
        history=StudyHistory(ROOT,receipt,weights=[1/6]*6)
        history.panels=CalendarSourcePanels(history.files,history.panels.columns,lock['calendar_source_bytes'])
        preparation=ROOT/'output/step6_20260911/preparation_manifest.json'
        if sha(preparation)!=receipt['preparation_manifest_sha256']:raise ValueError('Preparation identity changed')
        history.files.inventory.update(json.loads(preparation.read_text())['artifact_sha256'])
        history.files=PreparationFiles(history.files)
        history.panels.files=history.files
        cache=OperatorCache(maximum_bytes=lock['operator_cache_bytes'],guard=runtime.guard)
        store=CheckpointStore(OUT/'models.sqlite',run_identity=sha(OUT/'lock.json'))
        if not (OUT/'checkpoint_import.json').exists():
            prior_lock=json.loads((PREVIOUS/'lock.json').read_text())
            if sha(PREVIOUS/'lock.json')!=lock['previous_pilot_lock_sha256'] or sha(PREVIOUS/'models.sqlite')!=lock['previous_pilot_database_sha256']:
                raise ValueError('Prior pilot checkpoint source changed')
            for rel,expected in prior_lock['code_sha256'].items():
                if sha(ROOT/rel)!=expected:raise ValueError('Prior pilot dependency changed: '+rel)
            prior=CheckpointStore.__new__(CheckpointStore)
            prior.path=PREVIOUS/'models.sqlite'
            prior.run_identity=sha(PREVIOUS/'lock.json')
            prior.connection=sqlite3.connect(prior.path.resolve().as_uri()+'?mode=ro',uri=True)
            if prior.connection.execute('SELECT identity FROM run').fetchall()!=[(prior.run_identity,)]:
                raise ValueError('Prior checkpoint run identity changed')
            imported=prior.inventory()
            for row in imported:
                store.put(row['key'],dict(row['context'],checkpoint_origin=lock['previous_pilot_lock_sha256']),prior.get(row['key']))
            prior.close();write_json(OUT/'checkpoint_import.json',dict(prior_lock=lock['previous_pilot_lock_sha256'],bundles=imported))
        inventory=store.inventory()  # Every saved checkpoint is verified before continuation.
        if '2020/p512/g0/r50/E2' not in {x['key'] for x in inventory}:
            raise ValueError('Completed 2020 scale checkpoint required; this continuation may not refit it')
        disk_store=open_parent_store(ROOT,history,lock['input_cache_study_lock_sha256'])
        input_cache=PreparationCheckedInputCache(lock['input_cache_bytes'],runtime.guard,disk_store)
        results=[]
        with threadpool_limits(limits=1),single_thread_blas():
            verify_existing_inputs(history,disk_store,runtime,OUT)
            preparation_check=verify_preparation(history,disk_store,runtime,OUT)
            if not preparation_check['faster']:
                input_cache=ReusedInputCache(lock['input_cache_bytes'],runtime.guard,disk_store)
                runtime.event('preparation_fallback',reason='No paired speed benefit; use previous checked input path')
            for year in lock['years']:
                result=fit_coordinate(ROOT,OUT,history,cache,store,runtime,year=year,width=512,
                    gi=0,ridge_index=50,restraints=(.1,),input_cache=input_cache,workers=8,reference=reference if year==1971 else None)
                results.append(result['summary']);del result
        runtime.guard();history.files.unchanged()
        inventory=store.inventory();store.close();store=None
        write_json(OUT/'result.json',dict(status='COMPLETE_FITTING_PILOT_NOT_HISTORICAL_STUDY',
            summary='Both early and maximum-history fitting cycles completed. Full historical validation and deployment have not run.',
            coordinates=results,checkpoints=inventory,source_sha256=history.files.consumed,
            historical_experiment_started=False,new_utility_or_pvalue=False))
        runtime.event('pilot_complete',coordinates=2)
        status='COMPLETE'
    except BaseException as exc:
        error=exc
        write_json(OUT/f'stop_{time.time_ns()}.json',dict(error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        if store is not None:store.close()
        if runtime is not None:runtime.close(status,error)
        seconds,_=prior_seconds(OUT)
        charged=lock['prior_charged_seconds']+lock['review_and_test_reservation_seconds']+seconds
        atomic(WORK/'resource_ledger.json',dict(prior_resource_ledger=lock['prior_ledger'],
            prior_charged_seconds=lock['prior_charged_seconds'],
            review_and_test_reservation_seconds=lock['review_and_test_reservation_seconds'],
            pilot_active_seconds=seconds,charged_seconds=charged,
            remaining_engineering_seconds=max(0.,lock['cumulative_engineering_cap_seconds']-charged),
            cumulative_engineering_cap_seconds=lock['cumulative_engineering_cap_seconds'],
            additional_allowance_seconds=lock['additional_allowance_seconds'],
            full_study_runtime_limit_seconds=None,full_study_started=False))
        record_state(current_status='PILOT_COMPLETE_AWAITING_REVIEW' if status=='COMPLETE' else 'REMAINING_PILOT_LAYERS_STOPPED',
            background_computation_currently_running=False,
            current_resource_ledger=str((WORK/'resource_ledger.json').relative_to(ROOT)),
            maximum_history_learned_pipeline_tested=status=='COMPLETE',
            next_work='Review completed pilot and establish a full-run envelope.' if status=='COMPLETE' else
                'Review the stop reason and completed checkpoints. No historical experiment has started.')


if __name__=='__main__':main()
