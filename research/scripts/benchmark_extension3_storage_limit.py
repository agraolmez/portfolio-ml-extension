#!/usr/bin/env python3
"""Bounded real-array reload/provenance stress test; no historical fitting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse,fcntl,json,shutil,time,statistics
    from datetime import datetime,timezone
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import psutil,pyarrow as pa
    from baseline.full import preflight
    from feature_geometry.history import sha
    from geometry_study.kernel import LeanBasis
    from temporal_extension.history import load_parent
    from temporal_fast.backend import single_thread_blas,worker_init
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_fast_v3.streaming import OrderedObjective as Previous
    from temporal_fast_v3.settings import VARIANTS,environment_identity
    from temporal_fast_v4.deferred import DeferredMonth,OrderedObjective
    from temporal_fast_v4.source_checks import AcceptedBatchVerifier,identity as source_identity
    from temporal_fast_v4.native import identity as kernel_identity
    from temporal_fast_v4.storage import OverflowStore
    from temporal_numerics.spill import OverflowStore as OldStore,payload_fingerprint
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
        if time.monotonic()-start>1800:raise TimeoutError('30-minute benchmark bound')
        if shutil.disk_usage(ROOT).free<30*2**30:raise OSError('Free disk reserve')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    environments=dict(previous=environment_identity(),source_checks=source_identity(),kernels=kernel_identity())
    code=[Path(__file__).resolve(),*sorted((ROOT/'temporal_fast_v4').glob('*.py')),*sorted((ROOT/'temporal_fast_v4').glob('*.c'))]
    for record in (environments['previous']['native_build'],environments['source_checks'],environments['kernels']):
        code.extend(ROOT/name for name in record['artifact_sha256'])
    inventory={str(p.relative_to(ROOT)):sha(p) for p in code}
    for name in inventory:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_new(out/'benchmark_lock.json',dict(code_sha256=inventory,maximum_wall_seconds=1800,maximum_profile_calls=30,
        maximum_operator_storage_bytes=2**30,historical_gain_optimization_run=False,portfolio_evaluation_run=False))
    with single_thread_blas():
        receipt,_=preflight();write_new(out/'financial_preflight.json',receipt)
        samples,history,config=accepted_samples(ROOT,receipt);parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
        for day in ('2000-04-30','2000-05-31','2000-06-30','2024-08-31','2024-09-30','2024-10-31'):
            sample=history.training_sample(day,parent.settings.cutoff)
            innovations=history.validate_market(sample.market,history.innovations(sample.market))
            samples.append((LeanBasis(sample),innovations));guard();event('sample_prepared',month=day)
        samples.sort(key=lambda v:v[0].sample.market.decision_date)
        # Conservative, explicitly labeled metadata stress case: verify the entire
        # accepted 709-month source inventory before binding it to these 8 payloads.
        cache=json.loads((ROOT/'output/extension3_temporal_workflow_20260918/cache_feasibility/cache_feasibility.json').read_text())
        for row in cache['monthly_header_inventory']:
            day=row.get('decision_date',row.get('month',row.get('date')))
            if day is None:raise ValueError('Unknown inventory date field')
            for kind in ('matrices','risk'):history.files.path(f'data/prepared/step9_v1/baseline/{kind}/{day}.npz')
            guard()
        event('complete_source_inventory_verified',files=len(history.files.consumed))
        def verify(name,expected):
            history.files.path(name)
            if history.files.consumed[name]!=expected:raise ValueError('Accepted input changed')
        options=dict(source_fingerprint=history.source_fingerprint,reference_fingerprint=history.reference_fingerprint,
            study_lock_sha256=sha(ROOT/'config/extension3_study_v3_proposed.json'),maximum_bytes=2**30,verify_input=verify)
        old=OldStore(out/'operators',**options)
        new=OverflowStore(out/'operators',**options,verify_inputs=AcceptedBatchVerifier(history.files))
        expected={};days=[]
        for basis,innovations in samples:
            day=basis.sample.market.decision_date;days.append(day);expected[day]=payload_fingerprint(basis,innovations)
            old.put(day,basis,innovations,dict(history.files.consumed));guard()
        event('stress_payloads_written',months=len(days),accepted_inputs_per_payload=len(history.files.consumed))
        def restored(day):
            value=new.get(day,parent.settings.cutoff)
            if value is None or payload_fingerprint(*value)!=expected[day]:raise ValueError('Concurrent restore differs')
            return True
        with ThreadPoolExecutor(max_workers=4,initializer=worker_init) as pool:
            assert all(pool.map(restored,days*4))
        event('parallel_restore_equivalence_passed',loads=32)
        read_times={}
        for name,store in (('previous',old),('batch_checks',new)):
            times=[]
            for repeat in range(3):
                for day in days:
                    begin=time.monotonic();value=store.get(day,parent.settings.cutoff);times.append(time.monotonic()-begin)
                    if payload_fingerprint(*value)!=expected[day]:raise ValueError('Restored payload differs')
                    guard()
            read_times[name]=dict(seconds=times,median_seconds=statistics.median(times),mean_seconds=statistics.mean(times))
        def factory(method):
            def batches():
                history.files.unchanged()
                for day in days:
                    guard()
                    if method.startswith('deferred'):
                        yield DeferredMonth(day,parent.settings.cutoff,history.source_id,lambda day=day:new.get(day,parent.settings.cutoff))
                    else:yield (old if method=='serial_original' else new).get(day,parent.settings.cutoff)
                history.files.unchanged()
            return batches
        methods=['ram_reference','serial_original','serial_verified','deferred_verified','deferred_compact']
        metrics=[];equivalence=[];calls=0
        for label,gains in (('neutral',np.ones(13)),('nonneutral',np.linspace(.7,1.3,13))):
            times={name:[] for name in methods}
            for repeat in range(3):
                outputs={}
                for name in methods if repeat%2==0 else methods[::-1]:
                    common=dict(rho=.01,objective_scale=parent.objective_scale,workers=4,trig_backend='accelerate',**dict(VARIANTS))
                    if name=='ram_reference':objective=Previous(lambda:iter(samples),parent.feature_map,parent.settings,**common)
                    elif name=='serial_original':objective=Previous(factory(name),parent.feature_map,parent.settings,**common)
                    else:objective=OrderedObjective(factory(name),parent.feature_map,parent.settings,**common,
                        deferred_loads=name.startswith('deferred'),reuse_bytes=2*2**30 if name=='deferred_compact' else 0,reuse_mode='trig')
                    guard();begin=time.monotonic();result=objective(gains);seconds=time.monotonic()-begin;calls+=1;guard()
                    outputs[name]=result;times[name].append(seconds);event('profile_complete',case=label,method=name,seconds_for_profile=seconds)
                reference=outputs['ram_reference']
                for name,result in outputs.items():
                    errors={}
                    for key in ('value','gradient','beta','score','risk','cost'):
                        np.testing.assert_allclose(result[key],reference[key],atol=1e-11,rtol=1e-10)
                        errors[key]=float(np.max(abs(np.asarray(result[key])-np.asarray(reference[key]))))
                    if result['coverage']!=reference['coverage'] or result['training_source_fingerprint']!=reference['training_source_fingerprint']:
                        raise ValueError('Restored source identity or coverage differs')
                    equivalence.append(dict(case=label,repeat=repeat,method=name,maximum_absolute_errors=errors))
            metrics.append(dict(case=label,methods={name:dict(seconds=v,median_seconds=statistics.median(v)) for name,v in times.items()}))
        history.files.unchanged()
        for name,value in inventory.items():
            if sha(ROOT/name)!=value:raise ValueError('Benchmark source changed')
        write_new(out/'benchmark_result.json',dict(status='PASS',scope='eight_actual_months_with_full_709_month_source_inventory_metadata',
            historical_gain_optimization_run=False,portfolio_evaluation_run=False,profile_calls=calls,measurements=metrics,equivalence=equivalence,
            serial_reload_measurements=read_times,concurrent_restore_equivalence_loads=32,accepted_inputs_per_payload=len(history.files.consumed),
            accepted_inputs_sha256=dict(history.files.consumed),code_sha256=inventory,execution_environment=environments,
            peak_rss_gib=peak/2**30,seconds=time.monotonic()-start,numeric_tolerances=dict(atol=1e-11,rtol=1e-10)))
        event('benchmark_complete',profile_calls=calls,peak_rss_gib=peak/2**30)


if __name__=='__main__':main()
