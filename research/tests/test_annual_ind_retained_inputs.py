from types import SimpleNamespace
from unittest.mock import patch
import unittest
from annual_ind.retained_inputs import InputCache
import test_annual_ind_parallel_layers as fixtures


class RetainedInputTests(unittest.TestCase):
    def problem(self):
        args,_=fixtures.ParallelLayerTests().problem(8)
        basis,innovation=next(iter(args[0]()));calls=[]
        def sample(day,cutoff):calls.append(day);return basis.sample
        history=SimpleNamespace(source_fingerprint='source',reference_fingerprint='reference',
            training_sample=sample,innovations=lambda market:innovation,validate_market=lambda market,bound:bound)
        cache=InputCache(2**20,lambda:None)
        return history,cache,basis.sample.market.decision_date,calls

    def test_reuses_same_verified_sample_and_reference_without_rebuilding(self):
        history,cache,day,calls=self.problem()
        with patch('annual_ind.retained_inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=32*2**30)):
            first=cache.get(history,day,'2000-12-31');second=cache.get(history,day,'2001-12-31')
            self.assertIs(first[0],second[0]);self.assertIs(first[1],second[1]);self.assertEqual(len(calls),1)
            self.assertLessEqual(cache.bytes,cache.maximum_bytes);self.assertEqual(cache.hits,1)

    def test_future_cutoff_and_changed_source_rejected_on_hits(self):
        history,cache,day,_=self.problem()
        with patch('annual_ind.retained_inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=32*2**30)):
            cache.get(history,day,'2000-12-31')
            with self.assertRaisesRegex(ValueError,'cutoff'):cache.get(history,day,'1998-12-31')
            history.reference_fingerprint='different'
            with self.assertRaisesRegex(ValueError,'source context'):cache.get(history,day,'2000-12-31')

    def test_mutated_reference_rejected_on_hit(self):
        history,cache,day,_=self.problem()
        with patch('annual_ind.retained_inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=32*2**30)):
            _,innovation=cache.get(history,day,'2000-12-31')
            innovation[0].values.setflags(write=True);innovation[0].values[0,0]+=1
            with self.assertRaisesRegex(ValueError,'mutated'):cache.get(history,day,'2000-12-31')

    def test_low_memory_and_oversized_months_are_not_retained(self):
        for available,limit in ((5*2**30,2**20),(32*2**30,1)):
            history,_,day,calls=self.problem();cache=InputCache(limit,lambda:None)
            with patch('annual_ind.retained_inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=available)):
                cache.get(history,day,'2000-12-31');cache.get(history,day,'2000-12-31')
                self.assertEqual(len(calls),2);self.assertEqual(cache.bytes,0)

if __name__=='__main__':unittest.main()
