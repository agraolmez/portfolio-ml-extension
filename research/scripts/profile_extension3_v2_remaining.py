"""Bounded current-engine profiling on two accepted fixed monthly samples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse,cProfile,fcntl,io,json,pstats,statistics,time
    import numpy as np
    from baseline.full import preflight
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_extension.history import load_parent
    from temporal_fast_v2.streaming import OrderedObjective
    from temporal_fast_v2.settings import VARIANTS
    from temporal_fast.backend import single_thread_blas
    from temporal_numerics.study import write_new
    from feature_geometry.history import sha
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('New project output required')
    lock=(ROOT/'output/.extension3_heavy_job.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True)
    started=time.monotonic()
    with single_thread_blas():
        receipt,_=preflight();write_new(out/'financial_preflight.json',receipt)
        samples,history,config=accepted_samples(ROOT,receipt)
        parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
        gains=np.linspace(.7,1.3,13);measurements=[]
        for workers in (1,4):
            objective=OrderedObjective(lambda:iter(samples),parent.feature_map,parent.settings,
                rho=.01,objective_scale=parent.objective_scale,workers=workers,trig_backend='accelerate',**dict(VARIANTS))
            durations=[]
            for _ in range(5):
                begin=time.monotonic();objective(gains);durations.append(time.monotonic()-begin)
            measurements.append(dict(workers=workers,seconds=durations,median_seconds=statistics.median(durations)))
            if workers==1:
                profile=cProfile.Profile();profile.enable()
                for _ in range(5):objective(gains)
                profile.disable();profile.dump_stats(str(out/'serial.prof'))
                stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(50)
                (out/'serial_profile.txt').write_text(stream.getvalue())
        history.files.unchanged()
        write_new(out/'profile_result.json',dict(status='PASS',sample_months=2,
            historical_gain_optimization_run=False,portfolio_evaluation_run=False,
            profile_calls=15,seconds=time.monotonic()-started,measurements=measurements,
            code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*sorted((ROOT/'temporal_fast').glob('*.py')),*sorted((ROOT/'temporal_fast_v2').glob('*.py'))]},
            accepted_input_sha256=dict(history.files.consumed)))
    print(json.dumps(measurements,indent=2));print(stream.getvalue())


if __name__=='__main__':main()
