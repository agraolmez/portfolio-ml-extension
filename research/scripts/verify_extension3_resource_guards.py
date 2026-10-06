"""Fixed actual-data profiles with normal and exhausted reuse headroom; no fitting."""
from pathlib import Path
import sys,json,time,fcntl
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    from baseline.full import preflight
    from feature_geometry.history import sha
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_fast.backend import single_thread_blas
    from temporal_extension.history import load_parent
    from temporal_fast_v3.streaming import OrderedObjective as Previous
    from temporal_fast_v3.settings import VARIANTS as OLD
    from temporal_fast_v4.deferred import OrderedObjective
    from temporal_fast_v4.settings import VARIANTS,environment_identity
    from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory
    from temporal_numerics.study import write_new
    out=ROOT/'output/extension3_limit_20260919/resource_checks'
    if out.exists():raise FileExistsError('New evidence destination required')
    lock=(ROOT/'output/.extension3_heavy_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir();start=time.monotonic()
    with single_thread_blas():
        receipt,_=preflight();write_new(out/'financial_preflight.json',receipt)
        samples,history,_=accepted_samples(ROOT,receipt);parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
        evidence=[]
        for current in (False,True):
            fmap=parent.feature_map;factory=lambda:iter(samples)
            if current:
                fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
                factory=current_only_batch_factory(lambda:(b for b,_ in samples))
            common=dict(rho=.01,objective_scale=parent.objective_scale,workers=4,trig_backend='accelerate')
            for label,gains in (('neutral',np.ones(13)),('nonneutral',np.linspace(.4,1.6,13))):
                expected=Previous(factory,fmap,parent.settings,**common,**dict(OLD))(gains)
                for mode in ('ordinary','exhausted_headroom'):
                    objective=OrderedObjective(factory,fmap,parent.settings,**common,**dict(VARIANTS))
                    if mode=='ordinary':actual=objective(gains)
                    else:
                        with patch('temporal_fast_v4.streaming.charged_bytes',return_value=21*2**30):actual=objective(gains)
                        if label!='neutral':assert actual['computation_diagnostics']['reused_months']==0
                    errors={}
                    for key in ('value','gradient','beta','score','risk','cost'):
                        np.testing.assert_allclose(actual[key],expected[key],atol=1e-11,rtol=1e-10)
                        errors[key]=float(np.max(abs(np.asarray(actual[key])-np.asarray(expected[key]))))
                    assert actual['coverage']==expected['coverage'] and actual['training_source_fingerprint']==expected['training_source_fingerprint']
                    evidence.append(dict(current_only=current,gains=label,memory_policy=mode,maximum_absolute_errors=errors))
        history.files.unchanged()
        files=[*sorted((ROOT/'temporal_fast_v4').glob('*.py')),*sorted((ROOT/'temporal_fast_v4').glob('*.c')),Path(__file__).resolve()]
        write_new(out/'verification.json',dict(status='PASS',historical_gain_optimization_run=False,portfolio_evaluation_run=False,
            fixed_profile_calls=12,equivalence=evidence,seconds=time.monotonic()-start,
            execution_environment=environment_identity(),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
        print('Actual-data memory-policy equivalence PASS',flush=True)

if __name__=='__main__':main()
