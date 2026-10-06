#!/usr/bin/env python3
"""Bounded paired profile benchmark; never optimize gains or evaluate portfolios."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse,fcntl,json,os,shutil,statistics,time
    from datetime import datetime,timezone
    import numpy as np
    import psutil
    from baseline.full import preflight
    from feature_geometry.history import sha
    from geometry_study.kernel import LeanBasis
    from temporal_fast.backend import single_thread_blas,accelerate
    from temporal_fast_v2.settings import environment_identity
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_fast.streaming import OrderedObjective as PreviousObjective
    from temporal_fast_v2.streaming import OrderedObjective
    from temporal_extension.history import load_parent
    from temporal_extension.streaming import StreamingObjective
    from temporal_numerics.study import write_new
    from temporal_numerics.logging import DurableLog
    from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--scope',choices=['eight_months','pilot_profiles'],required=True)
    parser.add_argument('--variant',choices=['checks_rank','shared_all'],default='checks_rank')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('New output required')
    lock=(ROOT/'output/.extension3_heavy_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True);started=time.monotonic();process=psutil.Process();peak=0
    log=(out/'progress.log').open('x',buffering=1);logger=DurableLog(log)
    def event(kind,**values):logger.emit(dict(event=kind,at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,**values))
    def guard():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if peak>24*2**30:raise MemoryError('Benchmark process exceeds 24 GiB')
        if time.monotonic()-started>1800:raise TimeoutError('Bounded profile benchmark exceeds 30 minutes')
        if shutil.disk_usage(ROOT).free<30*2**30:raise OSError('Free-disk reserve reached')
    code=[*sorted((ROOT/'temporal_fast').glob('*.py')),*sorted((ROOT/'temporal_fast_v2').glob('*.py')),Path(__file__).resolve()]
    inventory={str(p.relative_to(ROOT)):sha(p) for p in code}
    for relative in inventory:
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    write_new(out/'benchmark_lock.json',dict(scope=args.scope,code_sha256=inventory,maximum_wall_seconds=1800,
        gain_optimization_run=False,portfolio_evaluation_run=False,maximum_profile_calls=60 if args.scope=='eight_months' else 16))
    with single_thread_blas():
        receipt,_=preflight();write_new(out/'financial_preflight.json',receipt);guard()
        if args.scope=='eight_months':
            samples,history,config=accepted_samples(ROOT,receipt)
            parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
            for day in ('2000-04-30','2000-05-31','2000-06-30','2024-08-31','2024-09-30','2024-10-31'):
                sample=history.training_sample(day,parent.settings.cutoff)
                innovations=history.validate_market(sample.market,history.innovations(sample.market))
                samples.append((LeanBasis(sample),innovations));guard();event('sample_prepared',month=day)
            samples.sort(key=lambda x:x[0].sample.market.decision_date)
            factory=lambda:iter(samples)
        else:
            from temporal_numerics.cache import RetainedBatchCache
            from temporal_study.history import StudyHistory
            config=json.loads((ROOT/'config/extension3_study_v2.json').read_text())
            history=StudyHistory(ROOT,receipt,weights=config['reference_weights'])
            parent=load_parent(ROOT,1981,1,expected_cutoff='1980-12-31')
            cache=RetainedBatchCache(history,maximum_bytes=16*2**30)
            source=cache.factory(parent)
            for index,_ in enumerate(source(),1):
                guard()
                if index%25==0:event('history_prepared',months=index,cache_gib=cache.retained_bytes/2**30)
            factory=source
        def checked_factory():
            for item in factory():guard();yield item
        metrics=[];equivalence=[];profile_calls=0
        settings=[('reference',None),
                  ('checks_rank',dict(parallel_checks=True,rank_one=True,packed=False,shared=False)),
                  ('shared_only',dict(parallel_checks=False,rank_one=False,packed=False,shared=True)),
                  ('shared_all',dict(parallel_checks=True,rank_one=True,packed=False,shared=True))]
        kinds=['temporal'] if args.scope=='eight_months' else ['temporal','current_only']
        for kind in kinds:
            fmap=parent.feature_map;batch_factory=checked_factory
            if kind=='current_only':
                fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
                batch_factory=current_only_batch_factory(lambda:(b for b,_ in checked_factory()))
            make=lambda kw: (PreviousObjective(batch_factory,fmap,parent.settings,rho=.01,objective_scale=parent.objective_scale,workers=4,trig_backend='accelerate')
                if kw is None else OrderedObjective(batch_factory,fmap,parent.settings,rho=.01,objective_scale=parent.objective_scale,workers=4,trig_backend='accelerate',**kw))
            if args.scope=='eight_months':
                cases=[('nonneutral',np.linspace(.7,1.3,13))];repeats=5;methods=settings
            else:
                pilot=json.loads((ROOT/f'output/extension3_numerical_pilot_20260919/run02/{kind}/fit.json').read_text())
                cases=[('neutral',np.ones(13)),('converged_pilot',np.array(pilot['gains']))]
                repeats=2;methods=[settings[0],(args.variant,dict(settings)[args.variant])]
            for label,gains in cases:
                reference=None;durations={name:[] for name,_ in methods}
                for repeat in range(repeats):
                    order=methods if repeat%2==0 else methods[::-1]
                    outputs={}
                    for name,kw in order:
                        objective=make(kw);guard();begin=time.monotonic();result=objective(gains);seconds=time.monotonic()-begin
                        profile_calls+=1;guard();durations[name].append(seconds);outputs[name]=result
                        event('profile_complete',model_kind=kind,case=label,method=name,seconds_for_profile=seconds,
                              projected_gradient=float(np.max(abs(np.clip(gains+result['gradient'],0,2)-gains))))
                    reference=outputs['reference']
                    for name,result in outputs.items():
                        errors={}
                        for key in ('value','gradient','beta','score','risk','cost'):
                            np.testing.assert_allclose(result[key],reference[key],atol=1e-11,rtol=1e-10)
                            errors[key]=float(np.max(np.abs(np.asarray(result[key])-np.asarray(reference[key]))))
                        if result['coverage']!=reference['coverage'] or result['training_source_fingerprint']!=reference['training_source_fingerprint']:
                            raise ValueError('Coverage or source identity differs')
                        if result['solve_residual']>1e-9:raise ValueError('Conditional solve gate failed')
                        if args.scope=='pilot_profiles':
                            expected=parent.model.beta if label=='neutral' else np.array(pilot['beta'])
                            np.testing.assert_allclose(result['beta'],expected,atol=1e-11,rtol=1e-10)
                            if label=='converged_pilot' and np.max(abs(np.clip(gains+result['gradient'],0,2)-gains))>1e-5:
                                raise ValueError('Saved converged pilot no longer meets unchanged accuracy')
                            errors['archived_beta']=float(np.max(abs(result['beta']-expected)))
                        equivalence.append(dict(kind=kind,case=label,repeat=repeat,method=name,maximum_absolute_errors=errors))
                metrics.append(dict(kind=kind,case=label,methods={name:dict(seconds=times,median_seconds=statistics.median(times)) for name,times in durations.items()}))
        history.files.unchanged()
        for relative,expected in inventory.items():
            if sha(ROOT/relative)!=expected:raise ValueError('Benchmark code changed during execution')
        event('benchmark_complete',profile_calls=profile_calls,peak_rss_gib=peak/2**30)
        write_new(out/'benchmark_result.json',dict(status='PASS',scope=args.scope,profile_calls=profile_calls,
            gain_optimization_run=False,portfolio_evaluation_run=False,measurements=metrics,equivalence=equivalence,
            parent_artifacts_sha256=dict(parent.hashes),accepted_inputs_sha256=dict(history.files.consumed),
            code_sha256=inventory,peak_rss_gib=peak/2**30,seconds=time.monotonic()-started,
            execution_environment=environment_identity(),numeric_tolerances=dict(atol=1e-11,rtol=1e-10),blas_policy='native thread-local SINGLE_THREADED on coordinator and every worker'))


if __name__=='__main__':main()
