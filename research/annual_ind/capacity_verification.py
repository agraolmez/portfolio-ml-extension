"""Real-source equivalence gate for the capacity-only execution amendment."""
from annual_ind.inputs import OperatorCache as OriginalOperators
from annual_ind.preparation_checks import PreparationCheckedInputCache
from annual_ind.cache_expanded import ExpandedInputCache,OperatorCache
from temporal_numerics.spill import payload_fingerprint
from extension4.storage import write_json


def verify_capacity(history,store,runtime,out):
    old=PreparationCheckedInputCache(256*2**20,runtime.guard,store)
    new=ExpandedInputCache(256*2**20,runtime.guard,store)
    old_ops=OriginalOperators(maximum_bytes=256*2**20,guard=runtime.guard)
    new_ops=OperatorCache(maximum_bytes=256*2**20,guard=runtime.guard)
    rows=[]
    for day in ('1990-01-31','2000-01-31','2019-11-30'):
        runtime.event('phase_progress',label='Checking expanded-cache equivalence',completed=len(rows),total=3)
        a=old.get_batch(history,day,'2019-12-31',old_ops)
        expected=payload_fingerprint(*a);del a
        for attempt in range(2):
            b=new.get_batch(history,day,'2019-12-31',new_ops)
            if payload_fingerprint(*b)!=expected:raise ValueError('Expanded cache changed inputs: '+day)
            del b
        rows.append(dict(day=day,payload_sha256=expected,first_and_repeat_exact=True))
    try:new.get_batch(history,'2019-11-30','2019-11-30',new_ops)
    except ValueError:pass
    else:raise ValueError('Cache allowed unmatured outcome')
    history.files.unchanged();runtime.guard()
    result=dict(status='PASS',comparisons=rows,causality_rejection_passed=True,
                new_cache_diagnostics=new.diagnostics(),new_operator_hits=new_ops.hits)
    write_json(out/f'capacity_equivalence_{runtime.stem.name}.json',result)
    runtime.event('expanded_capacity_equivalence_passed',comparisons=len(rows))
    return result
