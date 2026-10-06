"""Overflow reuse keeps chronological samples, source authority and exact profiles."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import test_temporal_cache as fixtures
from test_temporal_extension import temporal_fixture
from temporal_storage.cache import StorageBatchCache
from temporal_study.cache import batch_bytes
from temporal_extension.streaming import StreamingObjective
from geometry_study.kernel import LeanBasis


class StorageCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def test_overflow_reused_across_passes_and_restart_without_rebuilding(self):
        history,parent,bases,histories=fixtures.CacheTests().fixture()
        seen=[]
        history.files=SimpleNamespace(consumed={'source':'a'*64},unchanged=lambda:None,
                                      path=lambda relative:seen.append(relative))
        size=max(batch_bytes(LeanBasis(b.sample),h) for b,h in zip(bases,histories))
        with tempfile.TemporaryDirectory() as tmp:
            kwargs=dict(maximum_bytes=size,directory=Path(tmp)/'spill',maximum_disk_bytes=10**8,study_lock_sha256='lock')
            cache=StorageBatchCache(history,**kwargs)
            first=list(cache.factory(parent)());second=list(cache.factory(parent)())
            self.assertEqual((cache.hits,cache.misses,cache.disk_hits,cache.disk_writes),(1,2,1,1))
            self.assertLessEqual(cache.peak_retained_bytes,size)
            restarted=StorageBatchCache(history,**kwargs)
            third=list(restarted.factory(parent)())
            self.assertEqual((restarted.misses,restarted.disk_hits),(1,1))
            _,_,_,_,ps,_,fmap,_=temporal_fixture();gains=np.linspace(.4,1.6,fmap.gain_count)
            original=StreamingObjective(lambda:iter(first),fmap,ps,rho=.01,objective_scale=.7)(gains)
            reloaded=StreamingObjective(lambda:iter(third),fmap,ps,rho=.01,objective_scale=.7)(gains)
            for key in ('value','gradient','beta','risk','cost','score'):np.testing.assert_array_equal(original[key],reloaded[key])
            self.assertIn('source',seen)
            history.files.consumed['source']='b'*64
            with self.assertRaisesRegex(ValueError,'accepted source inventory'):
                list(cache.factory(parent)())

    def test_wrong_study_or_changed_reference_cannot_reuse_spill(self):
        history,parent,bases,histories=fixtures.CacheTests().fixture()
        history.files=SimpleNamespace(consumed={'source':'a'*64},unchanged=lambda:None,path=lambda p:None)
        size=max(batch_bytes(LeanBasis(b.sample),h) for b,h in zip(bases,histories))
        with tempfile.TemporaryDirectory() as tmp:
            kwargs=dict(maximum_bytes=size,directory=Path(tmp)/'spill',maximum_disk_bytes=10**8)
            cache=StorageBatchCache(history,study_lock_sha256='original',**kwargs)
            list(cache.factory(parent)())
            other=StorageBatchCache(history,study_lock_sha256='other',**kwargs)
            with self.assertRaisesRegex(ValueError,'identity'):list(other.factory(parent)())
            factory=cache.factory(parent);history.reference_fingerprint='changed'
            with self.assertRaisesRegex(ValueError,'reference changed'):list(factory())


if __name__=='__main__':unittest.main()
