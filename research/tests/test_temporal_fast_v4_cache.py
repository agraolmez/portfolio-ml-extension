"""Checked spill and mixed RAM/deferred replay retain identities and cutoffs."""
from pathlib import Path
import hashlib,tempfile,unittest
import numpy as np
import pandas as pd
import test_temporal_cache as fixtures
from test_temporal_extension import temporal_fixture
from temporal_extension.history import _AcceptedFiles
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.cache import StorageBatchCache
from temporal_fast_v4.deferred import DeferredMonth,OrderedObjective
from temporal_fast_v4.settings import VARIANTS
from temporal_fast_v3.streaming import OrderedObjective as Previous
from temporal_fast_v3.settings import VARIANTS as PREVIOUS_VARIANTS
from temporal_study.cache import batch_bytes


class DeferredCacheTests(unittest.TestCase):
    def test_source_panels_match_original_month_filter_and_bound_memory(self):
        from temporal_fast_v4.panels import SourcePanels
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'panel.parquet'
            frame=pd.DataFrame(dict(id=[2,1,1,2],eom=pd.to_datetime(['1999-02-28','1999-01-31','1999-02-28','1999-01-31']),
                x=[np.nan,.5,.2,1.],valid=[True,False,True,True]))
            frame.to_parquet(path,index=False)
            for budget in (0,10**6):
                files=_AcceptedFiles(root,{'panel.parquet':hashlib.sha256(path.read_bytes()).hexdigest()})
                cache=SourcePanels(files,{'panel.parquet':frame.columns},budget)
                for month in ('1999-01-31','1999-02-28','1999-03-31'):
                    for columns in (list(frame.columns),['id','x']):
                        expected=pd.read_parquet(path,columns=columns,filters=[('eom','==',pd.Timestamp(month))])
                        pd.testing.assert_frame_equal(cache.read('panel.parquet',columns,month),expected)
                self.assertLessEqual(cache.retained_bytes,budget)
                self.assertEqual(cache.reads,1 if budget else 0)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):cache.read('panel.parquet',['id'],'1999-01-31')

    def test_mixed_prefix_and_deferred_restart_preserve_profile_and_source_guards(self):
        with single_thread_blas(),tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source').write_bytes(b'accepted')
            history,parent,bases,histories=fixtures.CacheTests().fixture()
            history.files=_AcceptedFiles(root,{'source':hashlib.sha256(b'accepted').hexdigest()});history.files.path('source')
            size=max(batch_bytes(b,h) for b,h in zip(bases,histories))
            options=dict(maximum_bytes=size,directory=root/'spill',maximum_disk_bytes=10**8,study_lock_sha256='lock')
            cache=StorageBatchCache(history,**options);factory=cache.factory(parent)
            first=list(factory());second=list(factory())
            self.assertEqual(type(second[1]),DeferredMonth);self.assertEqual(cache.misses,2)
            _,_,_,_,settings,_,fmap,_=temporal_fixture()
            common=dict(rho=.01,objective_scale=.7,workers=4,trig_backend='accelerate')
            expected=Previous(lambda:iter(first),fmap,settings,**common,**dict(PREVIOUS_VARIANTS))(np.full(13,.7))
            actual=OrderedObjective(factory,fmap,settings,**common,**dict(VARIANTS))(np.full(13,.7))
            for key in ('value','gradient','beta','risk','cost','score'):
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-11,rtol=1e-10)
            self.assertEqual(actual['training_source_fingerprint'],expected['training_source_fingerprint'])
            restarted=StorageBatchCache(history,**options);list(restarted.factory(parent)())
            self.assertEqual(restarted.disk_hits,1);self.assertEqual(restarted.misses,1)
            (root/'source').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):second[1].load()


if __name__=='__main__':unittest.main()
