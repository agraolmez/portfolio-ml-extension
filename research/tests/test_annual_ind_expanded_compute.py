import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from annual_ind.memory_policy import MemoryPolicy,admit
from annual_ind.numerics_expanded import ordered_months
from annual_ind.parallel_expanded import conditional_sums
from annual_ind.parallel_expanded import ParallelTemporalObjective
from annual_ind.parallel_layers import ParallelTemporalObjective as PreviousObjective
from test_annual_ind_parallel_layers import ParallelLayerTests
from test_annual_ind_retained_inputs import RetainedInputTests
from test_annual_ind_preparation_checks import FakeFiles
from annual_ind.preparation_checks import PreparationFiles
from annual_ind.cache_expanded import ExpandedInputCache,OperatorCache


class CapacityTests(unittest.TestCase):
    def test_expanded_cache_reuse_causality_context_and_mutation(self):
        history,_,day,calls=RetainedInputTests().problem()
        history.files=PreparationFiles(FakeFiles())
        store=SimpleNamespace(context=dict(source_fingerprint='source',reference_fingerprint='reference'),get=lambda *a:None)
        cache=ExpandedInputCache(2**20,lambda:None,store)
        operators=OperatorCache(maximum_bytes=2**20,guard=lambda:None)
        with patch('annual_ind.cache_expanded.admit',return_value=True):
            first=cache.get_batch(history,day,'2000-12-31',operators)
            second=cache.get_batch(history,day,'2000-12-31',operators)
            self.assertIs(first[0].sample,second[0].sample);self.assertEqual(len(calls),1)
            with self.assertRaisesRegex(ValueError,'cutoff'):cache.get_batch(history,day,'1998-12-31',operators)
            history.reference_fingerprint='different'
            with self.assertRaisesRegex(ValueError,'context'):cache.get_batch(history,day,'2000-12-31',operators)
            history.reference_fingerprint='reference'
            first[1][0].values.setflags(write=True);first[1][0].values[0,0]+=1
            with self.assertRaisesRegex(ValueError,'mutated'):cache.get_batch(history,day,'2000-12-31',operators)

    def test_twelve_worker_neutral_and_learned_profiles_are_exact(self):
        args,kwargs=ParallelLayerTests().problem(8)
        old=PreviousObjective(*args,workers=8,**kwargs)
        new=ParallelTemporalObjective(*args,workers=12,**kwargs)
        for gains in (np.ones(13),np.linspace(.7,1.3,13)):
            a,b=old(gains),new(gains)
            for key in ('score','risk','cost','beta','gradient','value'):
                np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(a['training_source_fingerprint'],b['training_source_fingerprint'])
            self.assertEqual(a['coverage'],b['coverage'])
        new.workers=8
        with self.assertRaisesRegex(ValueError,'Execution settings changed'):new(np.ones(13))

    def test_warning_and_recovery_do_not_abort(self):
        p=MemoryPolicy()
        for t in range(30):p.check(t,2,8.)
        p.check(30,4,8.);p.check(39,4,8.);p.check(40,1,8.)
        p.check(100,4,8.)

    def test_sustained_critical_and_low_headroom_still_stop(self):
        p=MemoryPolicy();p.check(0,4,8.)
        with self.assertRaisesRegex(MemoryError,'Critical'):p.check(10,4,8.)
        p=MemoryPolicy();p.check(0,1,.9)
        with self.assertRaisesRegex(MemoryError,'one GiB'):p.check(10,1,.9)

    def test_unknown_pressure_is_not_ignored(self):
        with self.assertRaises(MemoryError):MemoryPolicy().check(0,3,10.)

    def test_admission_respects_both_worker_headroom_and_capacity(self):
        with patch('annual_ind.memory_policy.psutil.virtual_memory',return_value=SimpleNamespace(available=5*2**30)),patch('annual_ind.memory_policy.psutil.Process') as p:
            p.return_value.memory_info.return_value.rss=20*2**30
            self.assertTrue(admit(2**30));self.assertFalse(admit(3*2**30))
            p.return_value.memory_info.return_value.rss=24*2**30
            self.assertFalse(admit(1))

    def test_twelve_workers_preserve_order_and_guard_errors(self):
        self.assertEqual(list(ordered_months(lambda x:x*x,range(30),workers=12,guard=lambda:None)),[x*x for x in range(30)])
        def fail():raise RuntimeError('stop')
        with self.assertRaisesRegex(RuntimeError,'stop'):
            list(ordered_months(lambda x:x,range(30),workers=12,guard=fail))

    def test_conditional_reduction_preserves_cancellation_order(self):
        items=[1e16,1.,-1e16,7.,2.]
        def moments(x):return dict(score=np.array([x,x/3]),risk=np.eye(2)*abs(x),cost=np.eye(2))
        a=conditional_sums(items,moments,columns=2,workers=1,guard=lambda:None)
        b=conditional_sums(items,moments,columns=2,workers=12,guard=lambda:None)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
