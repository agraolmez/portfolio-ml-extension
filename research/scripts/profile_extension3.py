#!/usr/bin/env python3
"""Profile two fixed monthly samples; no gain fitting or portfolio evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse,cProfile,fcntl,io,json,pstats,statistics,time
    import numpy as np
    from threadpoolctl import threadpool_limits,threadpool_info
    from baseline.full import preflight
    from temporal_fast.benchmark_data import accepted_samples
    from temporal_extension.history import load_parent
    from temporal_extension.streaming import StreamingObjective
    from temporal_numerics.study import write_new
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('New project output required')
    lock=(ROOT/'output/.extension3_heavy_job.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True);limits=threadpool_limits(limits=4)
    receipt,_=preflight();write_new(out/'financial_preflight.json',receipt)
    samples,history,config=accepted_samples(ROOT,receipt)
    parent=load_parent(ROOT,2025,1,expected_cutoff='2024-12-31')
    objective=StreamingObjective(lambda:iter(samples),parent.feature_map,parent.settings,rho=.01,objective_scale=parent.objective_scale)
    gains=np.linspace(.7,1.3,parent.feature_map.gain_count)
    measurements=[]
    for threads in (4,1,2):
        with threadpool_limits(limits=threads):
            objective(gains)
            durations=[]
            for _ in range(5):
                started=time.monotonic();objective(gains);durations.append(time.monotonic()-started)
            measurements.append(dict(blas_threads=threads,seconds=durations,median_seconds=statistics.median(durations)))
    with threadpool_limits(limits=4):
        profile=cProfile.Profile();profile.enable()
        for _ in range(5):objective(gains)
        profile.disable();profile.dump_stats(str(out/'baseline.prof'))
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(45)
        (out/'baseline_profile.txt').write_text(stream.getvalue())
    history.files.unchanged()
    write_new(out/'profile_result.json',dict(status='PROFILE_COMPLETE',sample_months=2,
        optimizer_run=False,portfolio_evaluation_run=False,measurements=measurements,
        backend=threadpool_info(),accepted_inputs_sha256=dict(history.files.consumed)))
    print(json.dumps(measurements,indent=2),flush=True)
    print(stream.getvalue(),flush=True)


if __name__=='__main__':main()
