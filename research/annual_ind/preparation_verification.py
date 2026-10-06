"""Paired real-input parity/timing gate before using grouped source checks."""
import time
import statistics
from annual_ind.inputs import OperatorCache
from annual_ind.cached_inputs import ReusedInputCache
from annual_ind.preparation_checks import PreparationCheckedInputCache
from temporal_numerics.spill import payload_fingerprint
from extension4.storage import write_json


def verify_preparation(history,store,runtime,out):
    rows=[]
    # Middle-window uncached months chosen without model performance results.
    for day in ('1990-01-31','2000-01-31','2008-11-30'):
        runtime.event('phase_progress',label='Checking faster monthly preparation',completed=len(rows),total=3)
        original=ReusedInputCache(1,runtime.guard,store)
        grouped=PreparationCheckedInputCache(1,runtime.guard,store)
        operators=OperatorCache(maximum_bytes=1,guard=runtime.guard)
        warm=original.get_batch(history,day,'2019-12-31',operators)
        identity=payload_fingerprint(*warm);del warm
        timings={'original':[],'grouped':[]}
        for label,cache in [('original',original),('grouped',grouped),('grouped',grouped),('original',original)]:
            runtime.guard();started=time.monotonic()
            batch=cache.get_batch(history,day,'2019-12-31',operators)
            elapsed=time.monotonic()-started
            if payload_fingerprint(*batch)!=identity:
                raise ValueError('Grouped monthly preparation changed numerical input: '+day)
            timings[label].append(elapsed);del batch
        rows.append(dict(day=day,exact_payload_match=True,seconds=timings,
            median_original=statistics.median(timings['original']),
            median_grouped=statistics.median(timings['grouped'])))
    original=sum(r['median_original'] for r in rows)
    grouped=sum(r['median_grouped'] for r in rows)
    history.files.unchanged();runtime.guard()
    result=dict(status='PASS',comparisons=rows,speed_ratio=original/grouped,
        faster=grouped<original,method='Same warmed month, original/grouped/grouped/original; exact payload comparison',
        full_history_timing_certified=False)
    write_json(out/f'preparation_equivalence_{runtime.stem.name}.json',result)
    runtime.event('preparation_verification_complete',speed_ratio=result['speed_ratio'],faster=result['faster'])
    return result
