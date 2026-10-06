"""Bounded real-input equivalence gate, executed inside the pilot resource guard."""
import time
import numpy as np
from geometry_study.kernel import LeanBasis
from temporal_numerics.spill import payload_fingerprint
from extension4.storage import write_json

def verify_existing_inputs(history,store,runtime,out):
    rows=[]
    # Predetermined first/middle/final cached months in the 2020 training window.
    for day in ('2009-03-31','2014-06-30','2019-11-30'):
        runtime.guard()
        runtime.event('phase_progress',label='Verifying saved inputs against original sources',completed=len(rows),total=3)
        t=time.monotonic();restored=store.get(day,'2019-12-31');cached_seconds=time.monotonic()-t
        if restored is None:raise ValueError('Required cache verification month missing: '+day)
        t=time.monotonic();sample=history.training_sample(day,'2019-12-31')
        innovation=history.validate_market(sample.market,history.innovations(sample.market))
        basis=LeanBasis(sample);fresh_seconds=time.monotonic()-t
        if payload_fingerprint(*restored)!=payload_fingerprint(basis,innovation):
            raise ValueError('Existing input operators differ from fresh reconstruction: '+day)
        for x,y in zip((*restored[0].current_lu,*restored[0].prior_lu),(*basis.current_lu,*basis.prior_lu)):
            np.testing.assert_array_equal(x,y)
        rows.append(dict(day=day,exact_payload_match=True,cached_seconds=cached_seconds,
                         fresh_seconds=fresh_seconds,payload_fingerprint=payload_fingerprint(basis,innovation)))
        del restored,sample,innovation,basis
    history.files.unchanged();runtime.guard()
    write_json(out/f'cache_equivalence_{runtime.stem.name}.json',dict(status='PASS',comparisons=rows,
        reused_trained_models=False,new_disk_cache_written=False))
    runtime.event('input_reuse_verified',comparisons=rows)
