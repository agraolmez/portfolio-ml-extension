"""Independent reference, derivative, chronology and concurrency checks."""
from dataclasses import replace
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_temporal_extension import temporal_fixture
from temporal_extension.streaming import StreamingObjective
from temporal_extension.reference import Innovation
from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory
from temporal_fast.streaming import OrderedObjective,feature_parts,signal_from_parts,gradient_from_parts
from temporal_fast.backend import accelerate,single_thread_blas,sincos,worker_init


class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def make(self, current=False):
        rng,_,_,bases,settings,_,fmap,history=temporal_fixture(months=5)
        factory=lambda:zip(bases,history)
        if current:
            fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
            factory=current_only_batch_factory(lambda:iter(bases))
        return rng,bases,settings,fmap,history,factory

    def test_serial_and_parallel_preserve_both_complete_profiles(self):
        for current in (False,True):
            _,_,settings,fmap,_,factory=self.make(current)
            original=StreamingObjective(factory,fmap,settings,rho=.01,objective_scale=.7)
            for workers in (1,2,4):
                fast=OrderedObjective(factory,fmap,settings,rho=.01,objective_scale=.7,workers=workers)
                for gains in (np.ones(13),np.linspace(0,2,13)):
                    expected=original(gains);actual=fast(gains)
                    for key in ('value','parent_value','beta','gradient','score','risk','cost'):
                        np.testing.assert_allclose(actual[key],expected[key],atol=1e-11,rtol=1e-10)
                    for key in ('coverage','training_months','training_source_fingerprint','frozen_model_fingerprint'):
                        self.assertEqual(actual[key],expected[key])
                    self.assertLess(actual['solve_residual'],1e-9)
                    no_gradient=fast(gains,with_gradient=False)
                    self.assertIsNone(no_gradient['gradient'])
                    np.testing.assert_array_equal(no_gradient['beta'],actual['beta'])

    def test_reused_parts_have_exact_signals_and_matching_adjoint(self):
        for current in (False,True):
            rng,bases,_,fmap,history,_=self.make(current)
            for z,vol,innovation in zip(bases[0].sample.market.characteristics,bases[0].sample.market.volatility,history[0]):
                for gains in (np.ones(13),np.linspace(.2,1.8,13)):
                    parts=feature_parts(fmap,z,vol,innovation,gains)
                    np.testing.assert_array_equal(signal_from_parts(parts),fmap.signals(z,vol,innovation,gains))
                    upstream=rng.normal(size=(len(z),fmap.parent.columns))
                    np.testing.assert_array_equal(gradient_from_parts(fmap,z,innovation,parts,upstream),
                                                  fmap.vjp(z,vol,innovation,gains,upstream))

    def test_parallel_derivative_matches_finite_difference(self):
        for current in (False,True):
            _,_,settings,fmap,_,factory=self.make(current)
            fast=OrderedObjective(factory,fmap,settings,rho=.03,objective_scale=.7,workers=4)
            gains=np.linspace(.4,1.6,13);analytic=fast(gains)['gradient'];epsilon=1e-5
            for index in (0,6,12):
                up=gains.copy();down=gains.copy();up[index]+=epsilon;down[index]-=epsilon
                numeric=(fast(up,with_gradient=False)['value']-fast(down,with_gradient=False)['value'])/(2*epsilon)
                np.testing.assert_allclose(analytic[index],numeric,atol=2e-7,rtol=2e-5)

    def test_invalid_workers_unknown_maps_and_unguarded_basis_rejected(self):
        _,bases,settings,fmap,history,factory=self.make()
        for workers in (0,5,True,1.5):
            with self.assertRaises(ValueError):OrderedObjective(factory,fmap,settings,rho=0,objective_scale=1,workers=workers)
        from feature_geometry.moments import MomentBasis
        dense=[MomentBasis(b.sample) for b in bases]
        fast=OrderedObjective(lambda:zip(dense,history),fmap,settings,rho=0,objective_scale=1,workers=2)
        with self.assertRaisesRegex(ValueError,'locked-solve'):fast(np.ones(13))

    def test_mutations_and_cross_pass_changes_still_fail(self):
        for target in ('returns','operator','innovation'):
            _,bases,settings,fmap,history,factory=self.make()
            fast=OrderedObjective(factory,fmap,settings,rho=.01,objective_scale=1,workers=4)
            fast(np.ones(13),with_gradient=False)
            array={'returns':bases[0].sample.returns,'operator':bases[0].current_lu[0],
                   'innovation':history[0][0].values}[target]
            array.setflags(write=True);array.flat[0]+=.1;array.setflags(write=False)
            with self.assertRaisesRegex(ValueError,'changed'):fast(np.ones(13))
        _,bases,settings,fmap,history,_=self.make();calls=[0]
        changed=[list(h) for h in history];old=changed[0][0]
        changed[0][0]=Innovation(old.values+.1,old.valid,old.reasons)
        def factory():
            calls[0]+=1
            return zip(bases,history if calls[0]==1 else changed)
        with self.assertRaisesRegex(ValueError,'between objective passes'):
            OrderedObjective(factory,fmap,settings,rho=0,objective_scale=1,workers=4)(np.ones(13))

    def test_failed_worker_does_not_leave_objective_locked(self):
        _,_,settings,fmap,history,factory=self.make()
        fast=OrderedObjective(factory,fmap,settings,rho=0,objective_scale=1,workers=4)
        history[2][0].reasons['current_unobserved']=-1
        with self.assertRaisesRegex(ValueError,'reason count'):fast(np.ones(13))
        del history[2][0].reasons['current_unobserved']
        self.assertTrue(np.isfinite(fast(np.ones(13))['value']))

    def test_vector_trigonometry_preserves_neutral_and_complete_profiles(self):
        for current in (False,True):
            _,bases,settings,fmap,history,factory=self.make(current)
            expected=StreamingObjective(factory,fmap,settings,rho=.01,objective_scale=.7)
            fast=OrderedObjective(factory,fmap,settings,rho=.01,objective_scale=.7,workers=4,trig_backend='accelerate')
            for gains in (np.ones(13),np.linspace(.05,1.95,13)):
                a,b=expected(gains),fast(gains)
                for key in ('value','gradient','beta','score','risk','cost'):
                    np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
            z=bases[0].sample.market.characteristics[0];vol=bases[0].sample.market.volatility[0]
            neutral=signal_from_parts(feature_parts(fmap,z,vol,history[0][0],np.ones(13),'accelerate'))
            np.testing.assert_array_equal(neutral,fmap.signals(z,vol,history[0][0],np.ones(13)))

    def test_native_thread_setting_is_restored_and_applied_to_workers(self):
        from concurrent.futures import ThreadPoolExecutor
        lib=accelerate();before=lib.BLASGetThreading()
        with self.assertRaisesRegex(ValueError,'fixture'):
            with single_thread_blas():
                self.assertEqual(lib.BLASGetThreading(),1)
                with ThreadPoolExecutor(max_workers=4,initializer=worker_init) as pool:
                    self.assertEqual(list(pool.map(lambda _:lib.BLASGetThreading(),range(8))),[1]*8)
                raise ValueError('fixture')
        self.assertEqual(lib.BLASGetThreading(),before)

    def test_vector_trig_accuracy_and_bad_inputs(self):
        values=np.concatenate([np.linspace(-100,100,10000),[-1e12,1e12,0.]])
        cosine,sine=sincos(values)
        np.testing.assert_allclose(cosine,np.cos(values),atol=1e-15,rtol=1e-15)
        np.testing.assert_allclose(sine,np.sin(values),atol=1e-15,rtol=1e-15)
        with self.assertRaises(ValueError):sincos(np.array([np.nan]))
        _,_,settings,fmap,_,factory=self.make()
        fast=OrderedObjective(factory,fmap,settings,rho=0,objective_scale=1,workers=4)
        fast.workers=8
        with self.assertRaisesRegex(ValueError,'Execution settings'):fast(np.ones(13))


if __name__=='__main__':unittest.main()
