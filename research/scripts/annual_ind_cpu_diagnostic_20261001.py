"""Bounded CPU/input diagnostic requested by the user; no optimizer training."""
from pathlib import Path
import sys,json,time,fcntl,cProfile,pstats,io,sqlite3
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from baseline.full import preflight
from feature_geometry.history import sha
from feature_geometry.features import FeatureMap
from feature_geometry.training import FitSettings
from temporal_extension.features import TemporalFeatureMap
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_fast_v4.settings import VARIANTS
from annual_ind.panels import CalendarSourcePanels
from annual_ind.inputs import OperatorCache
from annual_ind.cached_inputs import ReusedInputCache,open_parent_store
from annual_ind.preparation_checks import PreparationFiles,PreparationCheckedInputCache
from annual_ind.preparation_verification import verify_preparation
from annual_ind.parallel_layers import ParallelTemporalObjective
from annual_ind.checkpoints import CheckpointStore
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime
from extension4.storage import write_json

OUT=ROOT/'output/annual_ind_cpu_optimization_20261001'

def main():
    mutex=(ROOT/'output/.annual_ind_probe.lock').open('a');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if OUT.exists():raise FileExistsError('Diagnostic output is immutable; preserve previous attempt')
    OUT.mkdir()
    limits=dict(total_active_seconds=600.,pilot_active_seconds=600.,max_physical_footprint_gib=16.,min_system_headroom_gib=4.,max_additional_disk_gib=2.,min_free_disk_gib=50.)
    old=ROOT/'output/annual_ind_runner_pilot_v3_20260930';oldlock=json.loads((old/'lock.json').read_text())
    for block in ('code_sha256','protected_sha256'):
        for name,digest in oldlock[block].items():
            if sha(ROOT/name)!=digest:raise ValueError('Locked source changed: '+name)
    code=[Path(__file__),ROOT/'annual_ind/preparation_checks.py',ROOT/'annual_ind/preparation_verification.py']
    write_json(OUT/'lock.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),user_request='bro use my cpu idk do something make this fast',scope='Bounded engineering measurement, not additional historical training',agent_selected_diagnostic_ceiling_seconds=600,budget=limits,workers=[1,8],blas_threads=1,paid_resources=False,full_pilot_restart=False,code_sha256={**oldlock['code_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in code}},protected_sha256=oldlock['protected_sha256']))
    runtime=None;error=None;status='STOPPED'
    try:
        runtime=ProbeRuntime(OUT,{'compute_budget':limits},'diagnostic')
        runtime.event('phase_progress',label='Measuring input preparation and CPU use',completed=None,total=None)
        receipt,_=preflight();write_json(OUT/'financial_preflight.json',receipt)
        history=StudyHistory(ROOT,receipt,weights=[1/6]*6)
        history.files.inventory.update(json.loads((ROOT/'output/step6_20260911/preparation_manifest.json').read_text())['artifact_sha256'])
        history.files=PreparationFiles(history.files)
        history.panels=CalendarSourcePanels(history.files,history.panels.columns,256*2**20)
        disk=open_parent_store(ROOT,history,oldlock['input_cache_study_lock_sha256'])
        with threadpool_limits(limits=1),single_thread_blas():
            # Restoring one accepted late month also installs its complete source
            # inventory, matching the previous repeated-check workload.
            restored=disk.get('2019-11-30','2019-12-31');del restored
            comparisons=verify_preparation(history,disk,runtime,OUT)
            profiles={}
            operators=OperatorCache(maximum_bytes=1,guard=runtime.guard)
            for label,cls in [('original',ReusedInputCache),('grouped',PreparationCheckedInputCache)]:
                cache=cls(1,runtime.guard,disk);profile=cProfile.Profile();profile.enable()
                batch=cache.get_batch(history,'2000-01-31','2019-12-31',operators)
                profile.disable();del batch
                stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(20)
                (OUT/f'{label}_profile.txt').write_text(stream.getvalue())
                profiles[label]=stream.getvalue()
            store=CheckpointStore.__new__(CheckpointStore);store.path=old/'models.sqlite';store.run_identity=sha(old/'lock.json');store.connection=sqlite3.connect(store.path.resolve().as_uri()+'?mode=ro',uri=True)
            row=next(x for x in store.inventory() if x['key']=='2020/p512/g0/r50/E2');arrays=store.get(row['key']);store.close();context=row['context']
            freq=np.fromfile(history.files.path('output/feature_geometry_study_20260914/directions/seed1_g0.bin'),dtype='<f8').reshape(115,512)[:,:256]
            dictionary=pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'));groups=dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
            fmap=TemporalFeatureMap(FeatureMap(freq,'cpu_diagnostic_20261001'),arrays['theta'],groups)
            settings=FitSettings(**dict(context['geometry_settings'],scale_penalty=.01*context['normalizer']))
            prepared=[];cache=PreparationCheckedInputCache(1,runtime.guard,disk)
            runtime.event('phase_progress',label='Preparing 12 fixed months for one-versus-eight CPU check',completed=0,total=12)
            for i,date in enumerate(pd.date_range('2000-01-31','2000-12-31',freq='ME'),1):
                prepared.append(cache.get_batch(history,str(date.date()),settings.cutoff,operators))
                runtime.event('phase_progress',label='Preparing 12 fixed months for one-versus-eight CPU check',completed=i,total=12)
            timings=[];reference=None
            for workers in (1,8,8,1):
                runtime.guard();runtime.event('phase_progress',label=f'Measuring {workers} CPU workers on identical saved inputs',completed=None,total=None)
                engine=ParallelTemporalObjective(lambda:iter(prepared),fmap,settings,workers=workers,rho=.1,objective_scale=context['normalizer'],trig_backend='accelerate',**dict(VARIANTS,reuse_bytes=128*2**20))
                t=time.monotonic();cpu=time.process_time();result=engine(np.ones(13));elapsed=time.monotonic()-t;cpu=time.process_time()-cpu
                if reference is None:reference=result
                for k in ('score','risk','cost','beta','gradient','value'):np.testing.assert_array_equal(result[k],reference[k])
                timings.append(dict(workers=workers,seconds=elapsed,cpu_seconds=cpu,average_cpu_cores=cpu/elapsed));del engine,result
            runtime.guard();history.files.unchanged()
            write_json(OUT/'result.json',dict(status='COMPLETE_BOUNDED_CPU_DIAGNOSTIC',summary='Input parity and one-versus-eight worker comparisons complete; no model optimization performed.',input_preparation=comparisons,cpu_comparison=timings,exact_numeric_parity=True,months=12,full_history_timing_certified=False,full_pilot_restart=False,consumed_inputs_sha256=history.files.consumed))
            status='COMPLETE';runtime.event('cpu_diagnostic_complete',cpu_comparison=timings)
    except BaseException as exc:
        error=exc;write_json(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if runtime is not None:runtime.close(status,error)

if __name__=='__main__':main()
