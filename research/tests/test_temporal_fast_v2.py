"""Equivalence and unchanged guard coverage for further computational variants."""
from dataclasses import replace
import unittest
import numpy as np
from test_temporal_extension import temporal_fixture
from temporal_fast.streaming import OrderedObjective as Previous,feature_parts,gradient_from_parts
from temporal_fast.backend import single_thread_blas
from temporal_fast_v2.streaming import OrderedObjective
from temporal_fast_v2.kernels import fingerprint,rank_one_gradient,packed_operators,shared_horner_operators
from feature_geometry.features import fingerprint as original_fingerprint
from temporal_extension.reference import Innovation
from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory


class FurtherFastTests(unittest.TestCase):
    def problem(self,current=False):
        rng,_,_,bases,settings,_,fmap,history=temporal_fixture(months=5)
        factory=lambda:zip(bases,history)
        if current:
            fmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
            factory=current_only_batch_factory(lambda:iter(bases))
        return rng,bases,settings,fmap,history,factory

    def objective(self,settings,fmap,factory,**kw):
        return OrderedObjective(factory,fmap,settings,rho=.01,objective_scale=.7,
            workers=4,trig_backend='accelerate',**kw)

    def test_hash_stream_is_identical_for_shapes_types_and_noncontiguous_arrays(self):
        arrays=[np.array(1.),np.empty((0,3)),np.arange(20).reshape(4,5),
                np.arange(20).reshape(4,5)[:,::2],np.array([True,False]),
                np.array(['abc','def']),np.asfortranarray(np.eye(4)),np.array([1,2],dtype='>i8')]
        self.assertEqual(fingerprint('before',*arrays,None,{'a':1}),original_fingerprint('before',*arrays,None,{'a':1}))

    def test_variants_preserve_profiles_neutrality_and_source_identity(self):
        with single_thread_blas():
            for current in (False,True):
                _,_,settings,fmap,_,factory=self.problem(current)
                expected=Previous(factory,fmap,settings,rho=.01,objective_scale=.7,workers=4,trig_backend='accelerate')
                for checks,rank,packed in ((True,False,False),(False,True,False),(False,False,True),(True,True,True)):
                    actual=self.objective(settings,fmap,factory,parallel_checks=checks,rank_one=rank,packed=packed)
                    for gains in (np.ones(13),np.linspace(0.,2.,13)):
                        a,b=expected(gains),actual(gains)
                        for key in ('value','parent_value','gradient','beta','score','risk','cost'):
                            np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
                        for key in ('coverage','training_months','training_source_fingerprint','frozen_model_fingerprint'):
                            self.assertEqual(a[key],b[key])
                        self.assertLessEqual(b['solve_residual'],1e-9)
                        no_gradient=actual(gains,with_gradient=False)
                        np.testing.assert_array_equal(no_gradient['beta'],b['beta'])
                        self.assertIsNone(no_gradient['gradient'])

    def test_rank_one_derivative_matches_general_adjoint_and_finite_difference(self):
        with single_thread_blas():
            for current in (False,True):
                rng,bases,settings,fmap,history,factory=self.problem(current)
                gains=np.linspace(.3,1.7,13);market=bases[0].sample.market
                for z,vol,innovation in zip(market.characteristics,market.volatility,history[0]):
                    parts=feature_parts(fmap,z,vol,innovation,gains,'accelerate')
                    adj=rng.normal(size=len(z));beta=rng.normal(size=fmap.parent.columns)
                    np.testing.assert_allclose(rank_one_gradient(fmap,z,innovation,parts,adj,beta),
                        gradient_from_parts(fmap,z,innovation,parts,np.outer(adj,beta)),atol=1e-11,rtol=1e-10)
                objective=self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=True)
                gradient=objective(gains)['gradient'];epsilon=1e-5
                for k in (0,6,12):
                    up=gains.copy();down=gains.copy();up[k]+=epsilon;down[k]-=epsilon
                    numeric=(objective(up,False)['value']-objective(down,False)['value'])/(2*epsilon)
                    np.testing.assert_allclose(numeric,gradient[k],atol=2e-7,rtol=2e-5)

    def test_packed_noncommuting_horner_preserves_both_operators_and_inputs(self):
        with single_thread_blas():
            rng,bases,_,_,_,_=self.problem()
            for columns in (1,17,513):
                signals=[rng.normal(size=(len(bases[0].sample.market.ids),columns)) for _ in range(13)]
                before=fingerprint(*signals)
                a=bases[0].operators(signals);b=packed_operators(bases[0],signals)
                for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-10)
                self.assertEqual(before,fingerprint(*signals))

    def test_parallel_audit_preserves_mutation_cross_pass_and_sequence_guards(self):
        for target in ('returns','operator','innovation'):
            _,bases,settings,fmap,history,factory=self.problem()
            objective=self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=True)
            objective(np.ones(13),False)
            a={'returns':bases[0].sample.returns,'operator':bases[0].current_lu[0],'innovation':history[0][0].values}[target]
            a.setflags(write=True);a.flat[0]+=.1;a.setflags(write=False)
            with self.assertRaisesRegex(ValueError,'changed'):objective(np.ones(13))
        _,bases,settings,fmap,history,_=self.problem();calls=[0]
        changed=[list(h) for h in history];old=changed[0][0]
        changed[0][0]=Innovation(old.values+.1,old.valid,old.reasons)
        def factory():
            calls[0]+=1;return zip(bases,history if calls[0]==1 else changed)
        with self.assertRaisesRegex(ValueError,'between objective passes'):
            self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=True)(np.ones(13))
        for factory in (lambda:zip(bases[::-1],history[::-1]),lambda:zip(bases[:1]*2,history[:1]*2)):
            with self.assertRaisesRegex(ValueError,'unique and ordered'):
                self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=True)(np.ones(13))
        with self.assertRaisesRegex(ValueError,'Unmatured'):
            self.objective(replace(settings,cutoff='1960-12-31'),fmap,lambda:zip(bases,history),
                parallel_checks=True,rank_one=True,packed=True)(np.ones(13))

    def test_worker_failure_recovers_and_variant_mutation_fails(self):
        _,_,settings,fmap,history,factory=self.problem()
        objective=self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=True)
        history[2][0].reasons['current_unobserved']=-1
        with self.assertRaisesRegex(ValueError,'reason count'):objective(np.ones(13))
        del history[2][0].reasons['current_unobserved']
        self.assertTrue(np.isfinite(objective(np.ones(13))['value']))
        objective.rank_one=False
        with self.assertRaisesRegex(ValueError,'variants changed'):objective(np.ones(13))

    def test_shared_horner_noncommuting_recursion_and_pivoted_LU(self):
        from scipy.linalg import lu_factor
        from types import SimpleNamespace
        with single_thread_blas():
            rng,bases,_,_,_,_=self.problem()
            for columns in (1,17,513):
                signals=[rng.normal(size=(len(bases[0].sample.market.ids),columns)) for _ in range(13)]
                before=fingerprint(*signals)
                for x,y in zip(bases[0].operators(signals),shared_horner_operators(bases[0],signals)):
                    np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-10)
                self.assertEqual(before,fingerprint(*signals))
            # Independent dense reference with genuine row pivoting, unequal growth
            # and noncommuting products; not just diagonal or near-identity fixtures.
            for _ in range(20):
                n=8;m=rng.normal(size=(n,n))*.4
                growth=tuple(rng.uniform(.6,1.4,n) for _ in range(13))
                cur=[np.eye(n)];lag=[np.eye(n)]
                for i in range(11):
                    cur.append(cur[-1]@(m*growth[i][None,:]))
                    lag.append(lag[-1]@(m*growth[i+1][None,:]))
                A,B=sum(cur),sum(lag)
                if max(np.linalg.cond(A),np.linalg.cond(B))>1e5:continue
                signals=[rng.normal(size=(n,5)) for _ in range(13)]
                basis=SimpleNamespace(sample=SimpleNamespace(market=SimpleNamespace(m=m,growth=growth)),
                    current_lu=lu_factor(A),prior_lu=lu_factor(B))
                expected=np.linalg.solve(A,sum(c@s for c,s in zip(cur,signals[:12])))
                previous=np.linalg.solve(B,sum(c@s for c,s in zip(lag,signals[1:])))
                actual,change=shared_horner_operators(basis,signals)
                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(change,expected-growth[0][:,None]*previous,atol=1e-10,rtol=1e-10)

    def test_shared_profile_preserves_both_models_and_gradient(self):
        with single_thread_blas():
            for current in (False,True):
                _,_,settings,fmap,_,factory=self.problem(current)
                baseline=Previous(factory,fmap,settings,rho=.01,objective_scale=.7,workers=4,trig_backend='accelerate')
                fast=self.objective(settings,fmap,factory,parallel_checks=True,rank_one=True,packed=False,shared=True)
                gains=np.linspace(.3,1.7,13)
                for g in (gains,np.ones(13)):
                    a,b=baseline(g),fast(g)
                    for key in ('value','gradient','beta','score','risk','cost'):
                        np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
                    if np.all(g==1.):
                        for key in ('value','beta','score','risk','cost'):
                            np.testing.assert_array_equal(a[key],b[key])
                analytical=fast(gains)['gradient'];epsilon=1e-5
                for k in (0,6,12):
                    up=gains.copy();down=gains.copy();up[k]+=epsilon;down[k]-=epsilon
                    numeric=(fast(up,False)['value']-fast(down,False)['value'])/(2*epsilon)
                    np.testing.assert_allclose(numeric,analytical[k],atol=2e-7,rtol=2e-5)


if __name__=='__main__':unittest.main()
