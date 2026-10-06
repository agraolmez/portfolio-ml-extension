"""Bounded early regression then maximum-history fresh E2/E3/C4 pilot.

Never launches historical validation/deployment. The original candidate grid is
unchanged; two mechanically chosen timing coordinates do not select a model.
Run with --resume only after an interrupted/resource-stopped attempt, using the
same lock and completed-layer checkpoints. All attempts consume the same budget.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
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
from annual_ind.coordinate import fit_coordinate
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime

OUT=ROOT/'output/annual_ind_runner_pilot_20260929'
WORK=ROOT/'output/annual_ind_runner_20260929'
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
        ledger_path=ROOT/'output/annual_ind_seed1_review_20260929/resource_ledger.json'
        ledger=json.loads(ledger_path.read_text())
        tests=json.loads((WORK/'tests_initial.json').read_text())
        if tests['exit_code']!=0:raise ValueError('Streaming numerical tests must pass')
        old=json.loads((reference/'lock.json').read_text())
        for block in ('code_sha256','protected_sha256'):
            for rel,expected in old[block].items():
                if sha(ROOT/rel)!=expected:raise ValueError('Accepted parent/source changed: '+rel)
        # Retain a conservative 180-second review allowance plus measured tests.
        reserved=180.+tests['seconds']
        seconds=min(3900.,ledger['remaining_engineering_seconds']-reserved)
        if seconds<=0:raise TimeoutError('No engineering time remains')
        limits=dict(total_active_seconds=seconds,pilot_active_seconds=seconds,
            max_physical_footprint_gib=16.,min_system_headroom_gib=4.,
            max_additional_disk_gib=2.,min_free_disk_gib=50.)
        code=list((ROOT/'annual_ind').glob('*.py'))+[Path(__file__),
            ROOT/'scripts/annual_ind_pipeline_probe_20260929.py',ROOT/'config/annual_ind_seed1_20260929.json']
        lock=dict(at_utc=datetime.now(timezone.utc).isoformat(),
            purpose=__doc__,status='BOUNDED_FITTING_PILOT_NOT_FULL_STUDY',
            years=[1971,2020],seed=1,width=512,bandwidth_index=0,ridge_index=50,
            temporal_restraints=[.1],conditional_multipliers=[100.,10.,1.],
            coordinate_rule='first and final annual fitting windows; widest author features; first bandwidth; middle ridge; strongest temporal restraint',
            historical_selection=False,portfolio_paths=False,AI_calls=False,training_months=649,
            workers=8,blas_threads=1,operator_cache_bytes=5*2**30,
            temporal_intermediate_bytes=128*2**20,temporal_profile_bytes=128*2**20,
            calendar_source_bytes=256*2**20,budget=limits,
            prior_ledger=str(ledger_path.relative_to(ROOT)),prior_ledger_sha256=sha(ledger_path),
            prior_charged_seconds=ledger['charged_seconds'],review_and_test_reservation_seconds=reserved,
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
        record_state(current_status='STREAMING_FITTING_PILOT_RUNNING',
            current_attempt_directory=str(OUT.relative_to(ROOT)),
            background_computation_currently_running=True,
            next_work='Complete the bounded fitting pilot and review its measurements before agreeing a full-run time limit.')
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
        cache=OperatorCache(maximum_bytes=lock['operator_cache_bytes'],guard=runtime.guard)
        store=CheckpointStore(OUT/'models.sqlite',run_identity=sha(OUT/'lock.json'))
        store.inventory()  # Every saved checkpoint is verified before continuation.
        results=[]
        with threadpool_limits(limits=1),single_thread_blas():
            for year in lock['years']:
                result=fit_coordinate(ROOT,OUT,history,cache,store,runtime,year=year,width=512,
                    gi=0,ridge_index=50,restraints=(.1,),workers=8,reference=reference if year==1971 else None)
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
            remaining_engineering_seconds=max(0.,7200.-charged),cumulative_engineering_cap_seconds=7200,
            full_study_runtime_limit_seconds=None,full_study_started=False))
        record_state(current_status='STREAMING_PILOT_COMPLETE_AWAITING_REVIEW' if status=='COMPLETE' else 'STREAMING_PILOT_STOPPED',
            background_computation_currently_running=False,
            current_resource_ledger=str((WORK/'resource_ledger.json').relative_to(ROOT)),
            maximum_history_learned_pipeline_tested=status=='COMPLETE',
            next_work='Review completed pilot and establish a full-run envelope.' if status=='COMPLETE' else
                'Review the stop reason and completed checkpoints. No historical experiment has started.')


if __name__=='__main__':main()
