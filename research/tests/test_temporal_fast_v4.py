"""Limits audit: numerical equivalence, memory bounds and retained integrity."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,os,unittest
import numpy as np
import pandas as pd
from geometry_study.kernel import LeanBasis
from scipy.linalg import lu_factor
from test_temporal_extension import temporal_fixture
from temporal_fast.backend import single_thread_blas,worker_init
from temporal_fast_v3.settings import VARIANTS
from temporal_fast_v3.streaming import OrderedObjective as Previous
from temporal_fast_v3.kernels import feature_parts
from temporal_fast_v4 import native
from temporal_fast_v4.kernels import fingerprint,parts_with_mean
from temporal_fast_v4.deferred import OrderedObjective,DeferredMonth
from temporal_fast_v4.risk import BoundFactors
from temporal_fast_v4.source_checks import AcceptedBatchVerifier
from temporal_extension.history import _AcceptedFiles
from temporal_extension.reference import Innovation
from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory
from geometry_study.kernel import lu_solve


class LimitTests(unittest.TestCase):
    def problem(self,current=False,months=4):
        rng,_,_,bases,settings,_,fmap,history=temporal_fixture(months=months)
        bases=[LeanBasis(replace(b.sample, market=replace(b.sample.market,
            decision_date=(pd.Timestamp(b.sample.market.decision_date)+pd.offsets.MonthEnd(0)).date().isoformat()))) for b in bases]
        factory=lambda:zip(bases,history)
        if current:
            fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
            factory=current_only_batch_factory(lambda:iter(bases))
        common=dict(rho=.01,objective_scale=.7,workers=4,trig_backend='accelerate',**dict(VARIANTS))
        return rng,bases,settings,fmap,history,factory,common

    def test_pivoted_direct_solves_concurrently_match_original(self):
        rng=np.random.default_rng(321)
        with single_thread_blas():
            for n,p in ((31,1),(128,13),(251,513)):
                a=rng.normal(size=(n,n));factors=lu_factor(a);b=rng.normal(size=(n,p))
                for trans in (0,1):
                    expected=lu_solve(factors,b,trans=trans,check_finite=False)
                    def one(_):return native.solve_lu(factors,b,trans)
                    with ThreadPoolExecutor(max_workers=4,initializer=worker_init) as pool:
                        for actual in pool.map(one,range(16)):
                            np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-10)
                            np.testing.assert_allclose((a.T if trans else a)@actual,b,atol=1e-10,rtol=1e-10)
        with self.assertRaises(ValueError):native.solve_lu((np.eye(2),np.array([1,0])),np.ones(2))

    def test_native_hash_preserves_every_byte_stream(self):
        from feature_geometry.features import fingerprint as old
        values=[np.array(1.),np.empty((0,3)),np.arange(35).reshape(5,7)[:,::2],
                np.asfortranarray(np.eye(19)),np.array(['abc','def']),np.array([1,2],dtype='>i8'),None,{'a':1}]
        self.assertEqual(fingerprint(*values),old(*values))
        self.assertEqual(native.sha_parts([]),hashlib.sha256().hexdigest())
        self.assertEqual(native.sha_parts([b'a\0b',b'',b'c']),hashlib.sha256(b'a\0bc').hexdigest())

    def test_compact_statistics_reconstruct_original_normalization_exactly(self):
        with single_thread_blas():
            _,bases,_,fmap,history,_,_=self.problem()
            gains=np.linspace(.3,1.7,13)
            for z,vol,innovation in zip(bases[0].sample.market.characteristics,bases[0].sample.market.volatility,history[0]):
                parts,mean=parts_with_mean(fmap,z,vol,innovation,gains)
                old=feature_parts(fmap,z,vol,innovation,gains)
                for x,y in zip(parts,old):np.testing.assert_array_equal(x,y)
                np.testing.assert_array_equal(native.normalized_from_stats(parts[3],parts[4],mean,parts[6]),parts[5])
            # Nearly constant features must retain exact centering, not an inverse
            # reconstruction of the trigonometric values or a variance shortcut.
            c=np.array([[1.,1.],[1.+1e-12,1.-1e-12],[1.-1e-12,1.+1e-12]])
            from temporal_fast_v3 import native as old_native
            out=np.empty((3,4));norm=np.empty(4);mean=np.empty(4)
            self.assertEqual(old_native.library().normalize(3,2,c,c,out,norm,mean),0)
            np.testing.assert_array_equal(native.normalized_from_stats(c,c,mean,norm),out)

    def test_both_models_all_candidates_preserve_profiles_and_neutrality(self):
        options=[dict(fast_hash=True),dict(direct_solve=True),dict(fused_horner=True),dict(vector_projection=True),
                 dict(symmetric_gram=True),dict(reuse_bytes=2**30),dict(reuse_bytes=2**30,reuse_mode='trig'),
                 dict(direct_solve=True,fused_horner=True,vector_projection=True,symmetric_gram=True,reuse_bytes=2**30)]
        with single_thread_blas():
            for current in (False,True):
                _,_,settings,fmap,_,factory,kw=self.problem(current)
                expected=Previous(factory,fmap,settings,**kw)
                for gains in (np.ones(13),np.linspace(0.,2.,13)):
                    a=expected(gains)
                    for options_ in options:
                        objective=OrderedObjective(factory,fmap,settings,**kw,**options_)
                        b=objective(gains)
                        for key in ('value','parent_value','gradient','beta','score','risk','cost'):
                            if np.all(gains==1.):np.testing.assert_array_equal(a[key],b[key])
                            else:np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
                        for key in ('coverage','training_months','training_source_fingerprint','frozen_model_fingerprint'):
                            self.assertEqual(a[key],b[key])
                        self.assertLessEqual(b['solve_residual'],1e-9)

    def test_bounded_reuse_restarts_for_every_gain_vector_and_matches_finite_differences(self):
        with single_thread_blas():
            for current in (False,True):
                _,_,settings,fmap,_,factory,kw=self.problem(current)
                for mode in ('full','trig'):
                    objective=OrderedObjective(factory,fmap,settings,**kw,reuse_bytes=2**30,reuse_mode=mode)
                    gains=np.linspace(.3,1.7,13);a=objective(gains)
                    self.assertEqual(a['computation_diagnostics']['reused_months'],4)
                    for k in (0,6,12):
                        up=gains.copy();down=gains.copy();up[k]+=1e-5;down[k]-=1e-5
                        numeric=(objective(up,False)['value']-objective(down,False)['value'])/2e-5
                        np.testing.assert_allclose(numeric,a['gradient'][k],atol=2e-7,rtol=2e-5)
                    for budget in (0,1,a['computation_diagnostics']['peak_reuse_bytes']//2):
                        other=OrderedObjective(factory,fmap,settings,**kw,reuse_bytes=int(budget),reuse_mode=mode)(gains)
                        self.assertLessEqual(other['computation_diagnostics']['peak_reuse_bytes'],budget)
                        np.testing.assert_allclose(other['gradient'],a['gradient'],atol=1e-11,rtol=1e-10)
                    np.testing.assert_array_equal(objective(gains)['beta'],a['beta'])

    def test_reuse_has_no_admission_when_memory_headroom_is_exhausted(self):
        from types import SimpleNamespace
        _,_,settings,fmap,_,factory,kw=self.problem()
        with patch('temporal_fast_v4.streaming.psutil.Process') as process:
            process.return_value.memory_info.return_value=SimpleNamespace(rss=21*2**30)
            actual=OrderedObjective(factory,fmap,settings,**kw,reuse_bytes=8*2**30)(np.full(13,.8))
        self.assertEqual(actual['computation_diagnostics']['reused_months'],0)

    def test_reuse_preserves_mutation_and_cross_pass_guards(self):
        gains=np.full(13,.7)
        for target in ('returns','operator','innovation'):
            _,bases,settings,fmap,history,factory,kw=self.problem()
            objective=OrderedObjective(factory,fmap,settings,**kw,reuse_bytes=2**30,reuse_mode='trig')
            objective(gains,False)
            a={'returns':bases[0].sample.returns,'operator':bases[0].current_lu[0],'innovation':history[0][0].values}[target]
            a.setflags(write=True);a.flat[0]+=.1;a.setflags(write=False)
            with self.assertRaisesRegex(ValueError,'changed'):objective(gains)
        _,bases,settings,fmap,history,_,kw=self.problem();calls=[0]
        other=[list(h) for h in history];x=other[0][0];other[0][0]=Innovation(x.values+.1,x.valid,x.reasons)
        def factory():
            calls[0]+=1;return zip(bases,history if calls[0]==1 else other)
        with self.assertRaisesRegex(ValueError,'between objective passes'):
            OrderedObjective(factory,fmap,settings,**kw,reuse_bytes=2**30)(gains)

    def test_factor_representation_bound_to_original_covariance(self):
        _,bases,_,_,_,_,_=self.problem();m=bases[0].sample.market;n=len(m.ids)
        specific=np.full(n,np.linalg.eigvalsh(m.covariance)[0]*.5)
        x=np.linalg.cholesky(m.covariance-np.diag(specific))
        terms=BoundFactors(m,m.ids,x,np.eye(n),specific)
        self.assertEqual(terms.get(m),terms.arrays)
        with self.assertRaises(AssertionError):BoundFactors(m,m.ids,x*1.1,np.eye(n),specific)
        terms.arrays[0].setflags(write=True);terms.arrays[0][0,0]+=.01
        with self.assertRaisesRegex(ValueError,'changed'):terms.get(m)

    def test_deferred_both_models_match_loaded_profiles_and_guards(self):
        with single_thread_blas():
            for current in (False,True):
                _,bases,settings,fmap,history,factory,kw=self.problem(current)
                def deferred():
                    for basis,innovations in zip(bases,history):
                        yield DeferredMonth(basis.sample.market.decision_date,settings.cutoff,basis.sample.market.source_id,
                            lambda b=basis,h=innovations:(b,h),current_only=current)
                for gains in (np.ones(13),np.linspace(.3,1.7,13)):
                    a=Previous(factory,fmap,settings,**kw)(gains)
                    b=OrderedObjective(deferred,fmap,settings,**kw,deferred_loads=True,reuse_bytes=2**30,reuse_mode='trig')(gains)
                    for key in ('value','gradient','beta','score','risk','cost'):
                        np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
                    self.assertEqual(a['training_source_fingerprint'],b['training_source_fingerprint'])
                days=list(deferred())
                for items in (days[::-1],days[:1]*2):
                    with self.assertRaisesRegex(ValueError,'unique and ordered'):
                        OrderedObjective(lambda:iter(items),fmap,settings,**kw,deferred_loads=True)(np.ones(13))
                wrong=replace(days[0],source_id='other')
                with self.assertRaisesRegex(ValueError,'source, decision or cutoff'):
                    OrderedObjective(lambda:iter([wrong]),fmap,settings,**kw,deferred_loads=True)(np.ones(13))
                wrong=replace(days[0],cutoff='2099-12-31',loader=lambda:(_ for _ in ()).throw(AssertionError('must not load')))
                with self.assertRaisesRegex(ValueError,'cutoff'):
                    OrderedObjective(lambda:iter([wrong]),fmap,settings,**kw,deferred_loads=True)(np.ones(13))

    def test_batch_source_checks_hash_first_use_and_reject_all_changes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'nested').mkdir();p=root/'nested/input';p.write_bytes(b'abc')
            expected=hashlib.sha256(b'abc').hexdigest();files=_AcceptedFiles(root,{'nested/input':expected})
            verify=AcceptedBatchVerifier(files);verify({'nested/input':expected});self.assertIn('nested/input',files.consumed)
            with self.assertRaises(ValueError):verify({'nested/input':'wrong'})
            with self.assertRaises(ValueError):verify({'absent':'wrong'})
            p.write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError,'changed'):verify({'nested/input':expected})
            p.unlink()
            with self.assertRaisesRegex(ValueError,'changed'):verify({'nested/input':expected})

    def test_batch_source_checks_reject_symlink_escape_even_to_same_inode(self):
        with TemporaryDirectory() as tmp,TemporaryDirectory() as outside:
            root=Path(tmp).resolve();p=root/'input';p.write_bytes(b'abc')
            expected=hashlib.sha256(b'abc').hexdigest();files=_AcceptedFiles(root,{'input':expected})
            verify=AcceptedBatchVerifier(files);verify({'input':expected})
            duplicate=Path(outside)/'same_inode';os.link(p,duplicate);p.unlink();p.symlink_to(duplicate)
            with self.assertRaisesRegex(ValueError,'changed'):verify({'input':expected})

    def test_array_reference_preserves_calendar_gaps_masks_order_and_bindings(self):
        from temporal_fast_v4.reference import ArrayBoundReference
        from temporal_extension.history import BoundReference
        from temporal_extension.reference import CalendarReference
        rng=np.random.default_rng(783);blocks=[];dates=[];ids=[];values=[];masks=[]
        for i,day in enumerate(pd.date_range('1998-01-31','1999-07-31',freq='ME')):
            if i==5:continue
            names=np.array([3,1] if i%3==0 else [3,1,2]);z=rng.normal(size=(len(names),5));mask=rng.random(z.shape)>.2
            blocks.append((day,names,z,mask));dates.extend([day]*len(names));ids.extend(names);values.extend(z);masks.extend(mask)
        for weights in (np.full(6,1/6),np.array([.5,0.,.25,0.,.25,0.])):
            old=BoundReference(CalendarReference(dates,ids,np.array(values),np.array(masks),weights),source_fingerprint='accepted')
            new=ArrayBoundReference(blocks,weights,dimensions=5,source_fingerprint='accepted')
            for day,names,z,mask in blocks:
                names=np.concatenate([names[::-1],[99]]);z=np.vstack([z[::-1],np.zeros(5)])
                a=old.at(day,names,z);b=new.at(day,names,z)
                np.testing.assert_array_equal(a.innovation.values,b.innovation.values)
                np.testing.assert_array_equal(a.innovation.valid,b.innovation.valid)
                self.assertEqual(a.innovation.reasons,b.innovation.reasons);self.assertEqual(a.fingerprint,b.fingerprint)
            day,names,z,mask=blocks[-1]
            with self.assertRaisesRegex(ValueError,'archived rank'):new.at(day,names,z+.01)
            target=next(iter(new.blocks.values()))[1];target.setflags(write=True);target[0,0]+=.1
            with self.assertRaisesRegex(ValueError,'mutated'):new.at(day,names,z)


if __name__=='__main__':unittest.main()
