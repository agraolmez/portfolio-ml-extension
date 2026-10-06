from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
import numpy as np
from annual_ind.cached_inputs import ReusedInputCache
from annual_ind.inputs import OperatorCache
from temporal_fast_v4.storage import OverflowStore
from temporal_numerics.spill import payload_fingerprint
import test_annual_ind_parallel_layers as fixtures

class CachedInputTests(unittest.TestCase):
    def problem(self,path):
        args,_=fixtures.ParallelLayerTests().problem(8)
        basis,innovation=next(iter(args[0]()));day=basis.sample.market.decision_date;calls=[]
        def verify(name,wanted):
            if (name,wanted)!=('accepted','sha'):raise ValueError('Unaccepted input')
        store=OverflowStore(path,source_fingerprint='source',reference_fingerprint='reference',
            study_lock_sha256='lock',maximum_bytes=2**24,verify_input=verify)
        def sample(day,cutoff):calls.append(day);return basis.sample
        h=SimpleNamespace(source_fingerprint='source',reference_fingerprint='reference',
            source_id=basis.sample.market.source_id,training_sample=sample,
            innovations=lambda market:innovation,validate_market=lambda market,bound:bound)
        cache=ReusedInputCache(2**20,lambda:None,store)
        return store,h,cache,OperatorCache(maximum_bytes=2**20,guard=lambda:None),basis,innovation,day,calls

    def test_existing_operator_roundtrip_is_exact_and_read_only(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path);store.put(day,b,i,{'accepted':'sha'})
            before={p.name:p.read_bytes() for p in Path(path).iterdir()}
            got,j=c.get_batch(h,day,'2000-12-31',o)
            self.assertEqual(payload_fingerprint(got,j),payload_fingerprint(b,i))
            self.assertEqual(calls,[]);self.assertEqual(o.misses,0);self.assertEqual(c.disk_hits,1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in Path(path).iterdir()})

    def test_absent_cache_falls_back_and_retains_without_disk_writes(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path)
            with patch('annual_ind.retained_inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=32*2**30)):
                first=c.get_batch(h,day,'2000-12-31',o);second=c.get_batch(h,day,'2000-12-31',o)
            self.assertEqual(payload_fingerprint(*first),payload_fingerprint(*second))
            self.assertEqual(len(calls),1);self.assertEqual(list(Path(path).iterdir()),[])

    def test_future_month_and_changed_reference_rejected_before_read(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path)
            with patch.object(store,'get',side_effect=AssertionError('Must not read future/source')):
                with self.assertRaisesRegex(ValueError,'cutoff'):c.get_batch(h,day,day,o)
                h.reference_fingerprint='changed'
                with self.assertRaisesRegex(ValueError,'source context'):c.get_batch(h,day,'2000-12-31',o)

    def test_payload_corruption_is_failure_not_silent_fallback(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path);store.put(day,b,i,{'accepted':'sha'})
            payload,_=store._paths(day)
            with payload.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'payload changed'):c.get_batch(h,day,'2000-12-31',o)
            self.assertEqual(calls,[])

    def test_partial_cache_and_unaccepted_inventory_are_rejected(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path);store.put(day,b,i,{'accepted':'sha'})
            with patch.object(store,'verify_input',side_effect=ValueError('Unaccepted input')):
                with self.assertRaisesRegex(ValueError,'Unaccepted input'):c.get_batch(h,day,'2000-12-31',o)
            store._paths(day)[1].unlink()
            with self.assertRaisesRegex(ValueError,'Partial overflow'):c.get_batch(h,day,'2000-12-31',o)

    def test_wrong_stored_market_source_is_rejected(self):
        with TemporaryDirectory() as path:
            store,h,c,o,b,i,day,calls=self.problem(path);store.put(day,b,i,{'accepted':'sha'})
            h.source_id='different'
            with self.assertRaisesRegex(ValueError,'source, day or cutoff'):c.get_batch(h,day,'2000-12-31',o)

if __name__=='__main__':unittest.main()
