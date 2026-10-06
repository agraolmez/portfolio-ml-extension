"""Exact recurrence, normalized optimization and chronology tests for the study."""
from dataclasses import replace
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from test_feature_geometry import fixture
from feature_geometry.training import profile,fit_geometry,joint_value_gradient
from geometry_study.kernel import LeanBasis,evaluate,gradient,fit,matrix_fit
from geometry_study.joint import state
from geometry_study.features import CachedFeatureMap
from concurrent.futures import ThreadPoolExecutor

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def test_noncommuting_horner_and_adjoint_recover_original(self):
        rng,fmap,old,settings,th=fixture()
        for a in old:
            b=LeanBasis(a.sample);signals=a.signals(fmap,th)
            self.assertLess(b.nbytes,a.nbytes/10)
            for x,y in zip(a.operators(signals),b.operators(signals)):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-11)
            u,v=rng.normal(size=len(a.sample.returns)),rng.normal(size=len(a.sample.returns))
            for x,y in zip(a.pullback(u,v),b.pullback(u,v)):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-11)
            vector=[s@rng.normal(size=fmap.columns) for s in signals]
            for x,y in zip((v[:,0] for v in a.operators([v[:,None] for v in vector])),b.operators(vector)):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-11)

    def test_streaming_objectives_gradients_and_fits_recover_prototype(self):
        _,fmap,old,settings,th=fixture();bases=[LeanBasis(b.sample) for b in old]
        for objective in ['full','risk_only','return_prediction']:
            a=profile(old,fmap,th,settings,objective);b=evaluate(bases,fmap,th,settings,objective)
            np.testing.assert_allclose(a['value'],b['value'],atol=1e-12,rtol=1e-11)
            np.testing.assert_allclose(a['beta'],b['beta'],atol=1e-12,rtol=1e-11)
            np.testing.assert_allclose(a['gradient'],gradient(bases,fmap,th,b['beta'],settings,objective),atol=1e-12,rtol=1e-10)
            x=fit_geometry(old,fmap,settings,objective);y=fit(bases,fmap,settings,objective)
            for key in ['theta','beta']:np.testing.assert_allclose(x[key],y[key],atol=1e-10,rtol=1e-9)

    def test_normalized_joint_steps_and_separate_coefficient_refit(self):
        _,fmap,old,settings,_=fixture();bases=[LeanBasis(b.sample) for b in old]
        normalizer=evaluate(bases,fmap,np.zeros(fmap.dimensions),settings)['value']
        for joint in [False,True]:
            result=fit(bases,fmap,settings,joint=joint,objective_scale=normalizer)
            self.assertTrue(np.all(np.diff([r['value'] for r in result['trace']])>=0))
            expected=evaluate(bases,fmap,result['theta'],settings)
            np.testing.assert_allclose(result['beta'],expected['beta'],atol=1e-11)
        result=fit(bases,fmap,settings,objective='return_prediction',coefficient_ridge=7.)
        expected=evaluate(bases,fmap,result['theta'],replace(settings,ridge=7.))
        np.testing.assert_allclose(result['beta'],expected['beta'],atol=1e-11)

    def test_concurrent_solves_and_cache_recover_serial_results(self):
        _,fmap,old,settings,th=fixture(dimensions=8,frequencies=12,months=2)
        bases=[LeanBasis(b.sample) for b in old]
        cached=CachedFeatureMap(fmap.frequencies,fmap.source)
        ref=profile(old,fmap,th,settings)
        for _ in range(4):
            actual=evaluate(bases,cached,th,settings)
            np.testing.assert_allclose(actual['beta'],ref['beta'],atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(gradient(bases,cached,th,actual['beta'],settings,'full'),ref['gradient'],atol=1e-11,rtol=1e-10)
        signals=bases[0].signals(fmap,th)
        expected=bases[0].operators(signals)
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda _:bases[0].operators(signals),range(20)))
        for value in results:
            for x,y in zip(value,expected):np.testing.assert_array_equal(x,y)
        # Changing theta must invalidate cached feature parts.
        np.testing.assert_array_equal(cached.signals(bases[0].sample.market.characteristics[0],bases[0].sample.market.volatility[0],th+.01),fmap.signals(bases[0].sample.market.characteristics[0],bases[0].sample.market.volatility[0],th+.01))

    def test_vector_joint_value_gradients_and_optimizer_match_matrix_reference(self):
        rng,fmap,old,settings,theta=fixture(dimensions=8,frequencies=12)
        bases=[LeanBasis(b.sample) for b in old];beta=rng.normal(size=fmap.columns)
        reference=joint_value_gradient(old,fmap,theta,beta,settings)
        result=state(bases,fmap,theta,beta,settings)
        for key in ['value','theta_gradient','beta_gradient']:
            np.testing.assert_allclose(result[key],reference[key],atol=1e-10,rtol=1e-10)
        normalizer=evaluate(bases,fmap,np.zeros(fmap.dimensions),settings)['value']
        reference=matrix_fit(bases,fmap,settings,joint=True,objective_scale=normalizer)
        result=fit(bases,fmap,settings,joint=True,objective_scale=normalizer)
        for key in ['theta','beta','geometry_beta']:
            np.testing.assert_allclose(result[key],reference[key],atol=1e-10,rtol=1e-9)
        self.assertEqual([r.get('backtracks') for r in result['trace']],[r.get('backtracks') for r in reference['trace']])

    def test_maturity_and_duplicate_month_fail_closed(self):
        _,fmap,old,settings,_=fixture();bases=[LeanBasis(b.sample) for b in old]
        for bad,setting in [(bases,replace(settings,cutoff='1999-02-01')),([bases[0],bases[0]],settings),(list(reversed(bases)),settings)]:
            with self.assertRaises(ValueError):fit(bad,fmap,setting)

if __name__=='__main__':unittest.main()
