#!/usr/bin/env python3
"""Bounded fixed-profile engineering test: longest existing history, no fitting."""
from pathlib import Path
import sys,argparse,json,time,fcntl,shutil,statistics
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    import psutil,pyarrow as pa
    from baseline.full import preflight
    from feature_geometry.history import sha
    from temporal_extension.history import load_parent
    from temporal_study.history import StudyHistory as OldHistory
    from temporal_fast.backend import single_thread_blas
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_fast_v3.settings import VARIANTS,environment_identity
    from temporal_fast_v3.streaming import OrderedObjective as Previous
    from temporal_fast_v4.history import StudyHistory
    from temporal_fast_v4.cache import StorageBatchCache
    from temporal_fast_v4.deferred import DeferredMonth,OrderedObjective
    from temporal_fast_v4.source_checks import identity as source_identity
    from temporal_fast_v4.native import identity as kernel_identity
    from temporal_numerics.spill import OverflowStore
    from temporal_numerics.study import write_new
    from temporal_numerics.logging import DurableLog
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('New output required')
    lock=(ROOT/'output/.extension3_heavy_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True);start=time.monotonic();process=psutil.Process();peak=0
    logger=DurableLog((out/'progress.log').open('x',buffering=1))
    def event(kind,**values):logger.emit(dict(event=kind,at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,**values))
    def guard():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if peak>24*2**30:raise MemoryError('Process memory bound')
        if time.monotonic()-start>2400:raise TimeoutError('40-minute fixed benchmark bound')
        if shutil.disk_usage(ROOT).free<30*2**30:raise OSError('Free disk reserve')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    environments=dict(previous=environment_identity(),source_checks=source_identity(),kernels=kernel_identity())
    code=[Path(__file__).resolve(),*sorted((ROOT/'temporal_fast_v4').glob('*.py')),*sorted((ROOT/'temporal_fast_v4').glob('*.c'))]
    for record in (environments['previous']['native_build'],environments['source_checks'],environments['kernels']):
        code.extend(ROOT/name for name in record['artifact_sha256'])
    inventory={str(p.relative_to(ROOT)):sha(p) for p in code}
    for name in inventory:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_new(out/'benchmark_lock.json',dict(code_sha256=inventory,maximum_wall_seconds=2400,maximum_profile_calls=8,
        maximum_operator_storage_bytes=9*2**30,historical_gain_optimization_run=False,portfolio_evaluation_run=False))
    with single_thread_blas():
        receipt,_=preflight();write_new(out/'financial_preflight.json',receipt)
        samples,old_history,config=accepted_samples(ROOT,receipt)
        history=StudyHistory(ROOT,receipt,weights=config['reference_weights'])
        parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
        reference_checks=[]
        for basis,_ in samples:
            timing={'previous':[],'array':[]}
            for repeat in range(3):
                values={}
                for name,fn in ([('previous',old_history.innovations),('array',history.innovations)] if repeat%2==0 else [('array',history.innovations),('previous',old_history.innovations)]):
                    begin=time.monotonic();values[name]=fn(basis.sample.market);timing[name].append(time.monotonic()-begin);guard()
                for a,b in zip(values['previous'],values['array']):
                    if a.fingerprint!=b.fingerprint or a.innovation.reasons!=b.innovation.reasons:raise ValueError('Reference differs')
            reference_checks.append(dict(month=basis.sample.market.decision_date,timings=timing,exact=True))
            event('reference_equivalence_passed',**reference_checks[-1])
        del samples,old_history
        cache=StorageBatchCache(history,maximum_bytes=16*2**30,directory=out/'operators',maximum_disk_bytes=9*2**30,
            study_lock_sha256=sha(ROOT/'config/extension3_study_v3_proposed.json'))
        begin=time.monotonic()
        for count,value in enumerate(cache.factory(parent)(),1):
            guard()
            if count in (1,181,520,709) or count%60==0:
                event('history_prepared',months=count,cache=cache.diagnostics(),rss_gib=process.memory_info().rss/2**30)
            if count in (1,181,520,709):
                basis,innovations=value
                old=history.validate_market(basis.sample.market,OldHistory.innovations(history,basis.sample.market))
                for a,b in zip(old,innovations):
                    np.testing.assert_array_equal(a.values,b.values);np.testing.assert_array_equal(a.valid,b.valid)
                    if a.reasons!=b.reasons:raise ValueError('Full-history reference differs')
        preparation_seconds=time.monotonic()-begin
        def verify(name,expected):
            history.files.path(name)
            if history.files.consumed[name]!=expected:raise ValueError('Accepted input changed')
        old_store=OverflowStore(out/'operators',source_fingerprint=history.source_fingerprint,reference_fingerprint=history.reference_fingerprint,
            study_lock_sha256=sha(ROOT/'config/extension3_study_v3_proposed.json'),maximum_bytes=9*2**30,verify_input=verify)
        def previous_batches():
            for value in cache.factory(parent)():
                guard()
                yield old_store.get(value.day,value.cutoff) if type(value) is DeferredMonth else value
        def new_batches():
            for value in cache.factory(parent)():guard();yield value
        metrics=[];equivalence=[];calls=0
        for label,gains in (('neutral',np.ones(13)),('nonneutral',np.linspace(.7,1.3,13))):
            times={name:[] for name in ('previous','selected')};diagnostics=[]
            for repeat in range(2):
                outputs={}
                for name in (('previous','selected') if repeat%2==0 else ('selected','previous')):
                    common=dict(rho=.01,objective_scale=parent.objective_scale,workers=4,trig_backend='accelerate',**dict(VARIANTS))
                    objective=(Previous(previous_batches,parent.feature_map,parent.settings,**common) if name=='previous' else
                        OrderedObjective(new_batches,parent.feature_map,parent.settings,**common,deferred_loads=True,reuse_bytes=8*2**30,reuse_mode='trig'))
                    guard();begin=time.monotonic();result=objective(gains);seconds=time.monotonic()-begin;calls+=1;guard()
                    outputs[name]=result;times[name].append(seconds)
                    if name=='selected':diagnostics.append(result.get('computation_diagnostics'))
                    event('profile_complete',case=label,method=name,seconds_for_profile=seconds,rss_gib=process.memory_info().rss/2**30)
                reference=outputs['previous'];actual=outputs['selected'];errors={}
                for key in ('value','gradient','beta','score','risk','cost'):
                    np.testing.assert_allclose(actual[key],reference[key],atol=1e-11,rtol=1e-10)
                    errors[key]=float(np.max(abs(np.asarray(actual[key])-np.asarray(reference[key]))))
                if actual['coverage']!=reference['coverage'] or actual['training_source_fingerprint']!=reference['training_source_fingerprint']:
                    raise ValueError('Source identity or coverage differs')
                if label=='neutral':np.testing.assert_allclose(actual['beta'],parent.model.beta,atol=1e-11,rtol=1e-10)
                if max(actual['solve_residual'],reference['solve_residual'])>1e-9:raise ValueError('Solve residual gate')
                equivalence.append(dict(case=label,repeat=repeat,maximum_absolute_errors=errors,
                    archived_neutral_beta_error=float(np.max(abs(actual['beta']-parent.model.beta))) if label=='neutral' else None))
            metrics.append(dict(case=label,methods={name:dict(seconds=v,median_seconds=statistics.median(v)) for name,v in times.items()},diagnostics=diagnostics))
        history.files.unchanged()
        for name,value in inventory.items():
            if sha(ROOT/name)!=value:raise ValueError('Benchmark source changed')
        write_new(out/'benchmark_result.json',dict(status='PASS',scope='all_709_training_months_for_2025_seed1_fixed_gains',
            historical_gain_optimization_run=False,portfolio_evaluation_run=False,profile_calls=calls,measurements=metrics,equivalence=equivalence,
            reference_checks=reference_checks,preparation_seconds=preparation_seconds,cache=cache.diagnostics(),panels=history.panels.diagnostics(),
            accepted_inputs_sha256=dict(history.files.consumed),code_sha256=inventory,execution_environment=environments,
            peak_rss_gib=peak/2**30,seconds=time.monotonic()-start,numeric_tolerances=dict(atol=1e-11,rtol=1e-10)))
        event('benchmark_complete',profile_calls=calls,peak_rss_gib=peak/2**30)

if __name__=='__main__':main()
