"""Exact disk-reuse prototype checks; no historical inputs or experiments."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import test_temporal_cache as fixtures
from test_temporal_extension import temporal_fixture
from temporal_numerics.spill import OverflowStore, payload_fingerprint
from temporal_extension.streaming import StreamingObjective
from temporal_extension.reference import Innovation


class OverflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def store(self, folder, **changes):
        def verify(path,digest):
            if (path,digest) != ('source','a'*64):raise ValueError('Changed accepted input')
        return OverflowStore(folder, **dict(dict(source_fingerprint='source',reference_fingerprint='reference',
            study_lock_sha256='lock',maximum_bytes=10**8,verify_input=verify),**changes))

    def test_roundtrip_reuses_exact_operators_profile_and_gradient(self):
        history,parent,bases,histories=fixtures.CacheTests().fixture()
        _,_,_,_,ps,_,fmap,_=temporal_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(tmp)
            for b,h in zip(bases,histories):store.put(b.sample.market.decision_date,b,h,{'source':'a'*64})
            restored=[store.get(b.sample.market.decision_date,parent.settings.cutoff) for b in bases]
            for (a,h),(b,j) in zip(zip(bases,histories),restored):
                self.assertEqual(payload_fingerprint(a,h),payload_fingerprint(b,j))
                signal=[fmap.signals(z,v,x,np.ones(fmap.gain_count)) for z,v,x in zip(a.sample.market.characteristics,a.sample.market.volatility,h)]
                for original,loaded in zip(a.operators(signal),b.operators(signal)):np.testing.assert_array_equal(original,loaded)
            gains=np.linspace(.4,1.6,fmap.gain_count)
            original=StreamingObjective(lambda:zip(bases,histories),fmap,ps,rho=.01,objective_scale=.7)(gains)
            loaded=StreamingObjective(lambda:iter(restored),fmap,ps,rho=.01,objective_scale=.7)(gains)
            for key in ('value','gradient','beta','score','risk','cost'):np.testing.assert_array_equal(original[key],loaded[key])

    def test_wrong_source_future_cutoff_and_changed_input_rejected(self):
        _,parent,bases,histories=fixtures.CacheTests().fixture();day=bases[0].sample.market.decision_date
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(tmp);store.put(day,bases[0],histories[0],{'source':'a'*64})
            with self.assertRaisesRegex(ValueError,'maturity'):store.get(day,day)
            with self.assertRaises(ValueError):self.store(tmp,source_fingerprint='other').get(day,parent.settings.cutoff)
            def reject(*args):raise ValueError('Changed accepted input')
            with self.assertRaisesRegex(ValueError,'accepted input'):self.store(tmp,verify_input=reject).get(day,parent.settings.cutoff)

    def test_changed_partial_or_duplicate_artifact_never_silently_reused(self):
        _,parent,bases,histories=fixtures.CacheTests().fixture();day=bases[0].sample.market.decision_date
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(tmp);store.put(day,bases[0],histories[0],{'source':'a'*64})
            with self.assertRaises(FileExistsError):store.put(day,bases[0],histories[0],{'source':'a'*64})
            (Path(tmp)/(day+'.npz')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'payload changed'):store.get(day,parent.settings.cutoff)
            (Path(tmp)/(day+'.json')).unlink()
            with self.assertRaisesRegex(ValueError,'Partial'):store.get(day,parent.settings.cutoff)

    def test_disk_cap_and_missing_cache_do_not_create_financial_results(self):
        _,parent,bases,histories=fixtures.CacheTests().fixture();day=bases[0].sample.market.decision_date
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(tmp,maximum_bytes=1)
            self.assertIsNone(store.get(day,parent.settings.cutoff))
            with self.assertRaisesRegex(OSError,'disk budget'):store.put(day,bases[0],histories[0],{'source':'a'*64})
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_reason_dictionary_order_does_not_change_payload_identity(self):
        _,parent,bases,histories=fixtures.CacheTests().fixture()
        history=tuple(Innovation(x.values,x.valid,dict(cells=x.values.size,
            adjusted_eligible=int(x.valid.sum()),bypassed=int((~x.valid).sum()),
            current_unobserved=0,prior_unobserved_or_absent=0,prior_security_month_absent=0))
            for x in histories[0])
        reordered=tuple(Innovation(x.values,x.valid,dict(reversed(list(x.reasons.items())))) for x in history)
        self.assertEqual(payload_fingerprint(bases[0],history),payload_fingerprint(bases[0],reordered))
        day=bases[0].sample.market.decision_date
        with tempfile.TemporaryDirectory() as tmp:
            store=self.store(tmp);store.put(day,bases[0],history,{'source':'a'*64})
            basis,restored=store.get(day,parent.settings.cutoff)
            self.assertEqual(payload_fingerprint(bases[0],history),payload_fingerprint(basis,restored))


if __name__ == '__main__':unittest.main()
