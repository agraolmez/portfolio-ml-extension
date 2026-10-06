"""Native computation retains original numerical and data-integrity gates."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from test_temporal_extension import temporal_fixture
from temporal_fast.backend import single_thread_blas
from temporal_fast.streaming import feature_parts, signal_from_parts
from temporal_fast_v2.kernels import rank_one_gradient
from temporal_fast_v2.streaming import OrderedObjective as Previous
from temporal_fast_v3.streaming import OrderedObjective
from temporal_fast_v3 import native, kernels
from temporal_extension.reference import Innovation
from temporal_study.controls import CurrentOnlyFeatureMap, current_only_batch_factory

VARIANTS=dict(parallel_checks=True,rank_one=True,packed=False,shared=True)


class NativeTests(unittest.TestCase):
    def problem(self,current=False):
        rng,_,_,bases,settings,_,fmap,history=temporal_fixture(months=4)
        factory=lambda:zip(bases,history)
        if current:
            fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
            factory=current_only_batch_factory(lambda:iter(bases))
        return rng,bases,settings,fmap,history,factory

    def objective(self,settings,fmap,factory,parts=True,gradient=True,workers=4):
        return OrderedObjective(factory,fmap,settings,rho=.01,objective_scale=.7,
            workers=workers,trig_backend='accelerate',**VARIANTS,
            fused_parts=parts,fused_gradient=gradient)

    def test_normalization_and_signal_match_numpy_and_preserve_inputs(self):
        rng=np.random.default_rng(913)
        for n,k in ((3,2),(806,256),(703,256)):
            angles=rng.normal(size=(n,k));c,s=np.cos(angles),np.sin(angles)
            before=(c.copy(),s.copy())
            raw=np.column_stack((c,s));center=raw-raw.mean(axis=0)
            norm=np.sqrt((center*center).sum(axis=0));expected=center/norm
            actual,actual_norm=native.normalize(c,s)
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13)
            np.testing.assert_allclose(actual_norm,norm,atol=1e-14,rtol=1e-13)
            vol=rng.uniform(.1,2,n)
            np.testing.assert_allclose(native.signals(actual,vol),
                np.column_stack((np.full(n,1/np.sqrt(n)),expected))/vol[:,None],atol=1e-14,rtol=1e-13)
            np.testing.assert_array_equal(c,before[0]);np.testing.assert_array_equal(s,before[1])
            # Noncontiguous inputs are safely copied before entering C.
            a,b=native.normalize(c[:,::2],s[:,::2])
            np.testing.assert_allclose(a,np.column_stack((expected[:,:k:2],expected[:,k::2])),atol=1e-14,rtol=1e-13)

    def test_near_constant_centering_and_invalid_inputs(self):
        x=np.array([[1.,1.],[1.+1e-12,1.-1e-12],[1.-1e-12,1.+1e-12]])
        actual,norm=native.normalize(x,x)
        self.assertTrue(np.isfinite(actual).all());self.assertTrue((norm>0).all())
        for bad in (np.ones((3,2)),np.full((3,2),np.nan),np.full((3,2),np.inf)):
            with self.assertRaises(ValueError):native.normalize(bad,bad)
        with self.assertRaises(ValueError):native.normalize(np.ones(2),np.ones(2))
        with self.assertRaises(ValueError):native.normalize(x,x[:2])
        with self.assertRaises(ValueError):native.signals(x,np.array([1.,0.,1.]))
        with self.assertRaises(ValueError):native.characteristic_adjoint(x,np.ones(3),x)

    def test_feature_adjoint_and_independent_full_finite_difference(self):
        with single_thread_blas():
            for current in (False,True):
                rng,bases,settings,fmap,history,factory=self.problem(current)
                if current:history=list(factory())[0][1];innov=history[0]
                else:innov=history[0][0]
                market=bases[0].sample.market;z=market.characteristics[0];vol=market.volatility[0]
                gains=np.linspace(.3,1.7,13)
                old=feature_parts(fmap,z,vol,innov,gains,'accelerate')
                new=kernels.feature_parts(fmap,z,vol,innov,gains)
                for a,b in zip(old,new):np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-10)
                adj=rng.normal(size=len(z));beta=rng.normal(size=fmap.parent.columns)
                np.testing.assert_allclose(kernels.rank_one_gradient(fmap,z,innov,new,adj,beta),
                    rank_one_gradient(fmap,z,innov,old,adj,beta),atol=1e-11,rtol=1e-10)
                objective=self.objective(settings,fmap,factory)
                analytic=objective(gains)['gradient'];eps=1e-5
                for k in (0,6,12):
                    up=gains.copy();down=gains.copy();up[k]+=eps;down[k]-=eps
                    numeric=(objective(up,False)['value']-objective(down,False)['value'])/(2*eps)
                    np.testing.assert_allclose(numeric,analytic[k],atol=2e-7,rtol=2e-5)

    def test_profiles_all_variants_neutral_exact_and_sources_unchanged(self):
        with single_thread_blas():
            for current in (False,True):
                _,_,settings,fmap,_,factory=self.problem(current)
                reference=Previous(factory,fmap,settings,rho=.01,objective_scale=.7,
                    workers=4,trig_backend='accelerate',**VARIANTS)
                for gains in (np.ones(13),np.linspace(0.,2.,13)):
                    expected=reference(gains)
                    for parts,gradient in ((True,False),(False,True),(True,True)):
                        for workers in (1,4):
                            objective=self.objective(settings,fmap,factory,parts,gradient,workers)
                            actual=objective(gains)
                            for key in ('value','parent_value','gradient','beta','score','risk','cost'):
                                if np.all(gains==1.):np.testing.assert_array_equal(actual[key],expected[key])
                                else:np.testing.assert_allclose(actual[key],expected[key],atol=1e-11,rtol=1e-10)
                            for key in ('coverage','training_months','training_source_fingerprint','frozen_model_fingerprint'):
                                self.assertEqual(actual[key],expected[key])
                            self.assertLessEqual(actual['solve_residual'],1e-9)
                            no_gradient=objective(gains,False)
                            np.testing.assert_array_equal(no_gradient['beta'],actual['beta'])

    def test_native_concurrent_calls_are_deterministic_and_build_is_strict(self):
        rng=np.random.default_rng(43);a=rng.normal(size=(703,256))
        expected=native.normalize(np.cos(a),np.sin(a))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for actual in pool.map(lambda _:native.normalize(np.cos(a),np.sin(a)),range(12)):
                for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
        identity=native.identity();flags=identity['recipe']['flags']
        self.assertIn('-fno-fast-math',flags);self.assertIn('-ffp-contract=off',flags)
        # Fake a changed on-disk hash without touching the running library.
        with patch.object(native,'digest',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'artifact changed'):native.identity()

    def test_integrity_chronology_maturity_and_configuration_guards(self):
        gains=np.linspace(.7,1.3,13)
        for target in ('returns','operator','innovation'):
            _,bases,settings,fmap,history,factory=self.problem()
            objective=self.objective(settings,fmap,factory);objective(gains,False)
            a={'returns':bases[0].sample.returns,'operator':bases[0].current_lu[0],'innovation':history[0][0].values}[target]
            a.setflags(write=True);a.flat[0]+=.1;a.setflags(write=False)
            with self.assertRaisesRegex(ValueError,'changed'):objective(gains)
        _,bases,settings,fmap,history,_=self.problem();calls=[0]
        changed=[list(h) for h in history];old=changed[0][0]
        changed[0][0]=Innovation(old.values+.1,old.valid,old.reasons)
        def factory():
            calls[0]+=1;return zip(bases,history if calls[0]==1 else changed)
        with self.assertRaisesRegex(ValueError,'between objective passes'):
            self.objective(settings,fmap,factory)(gains)
        for factory in (lambda:zip(bases[::-1],history[::-1]),lambda:zip(bases[:1]*2,history[:1]*2)):
            with self.assertRaisesRegex(ValueError,'unique and ordered'):self.objective(settings,fmap,factory)(gains)
        with self.assertRaisesRegex(ValueError,'Unmatured'):
            self.objective(replace(settings,cutoff='1960-12-31'),fmap,lambda:zip(bases,history))(gains)
        objective=self.objective(settings,fmap,lambda:zip(bases,history))
        objective.fused_parts=False
        with self.assertRaisesRegex(ValueError,'variants changed'):objective(gains)


if __name__=='__main__':unittest.main()
