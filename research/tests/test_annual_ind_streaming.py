from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from geometry_study import kernel
from geometry_study.features import CachedFeatureMap
from annual_ind.inputs import OperatorCache, YearInputs
from annual_ind.streaming_geometry import GeometryObjective
from annual_ind.parallel_layers import ParallelTemporalObjective
from temporal_numerics.optimizer import optimize, LBFGSBSettings
from test_feature_geometry import fixture
import test_annual_ind_parallel_layers as layer_fixtures


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.threads=threadpool_limits(limits=1)
        _,fmap,bases,self.settings,self.theta=fixture()
        self.bases=[kernel.LeanBasis(b.sample) for b in bases]
        self.fmap=CachedFeatureMap(fmap.frequencies,fmap.source)

    def tearDown(self):self.threads.restore_original_limits()

    def objective(self,workers=8):
        return GeometryObjective(lambda:iter(self.bases),len(self.bases),self.fmap,
                                 self.settings,workers=workers,guard=lambda:None)

    def test_complete_geometry_trace_and_every_value_match(self):
        for workers in (1,8):
            for ridge in (0.,.3,10000.):
                self.settings=replace(self.settings,ridge=ridge)
                obj=self.objective(workers)
                for theta in (np.zeros(self.fmap.dimensions),self.theta):
                    original=kernel.evaluate(self.bases,self.fmap,theta,self.settings)
                    actual=obj.evaluate(theta)
                    for k in original:np.testing.assert_array_equal(actual[k],original[k])
                    np.testing.assert_array_equal(obj.gradient(theta,actual['beta']),
                        kernel.gradient(self.bases,self.fmap,theta,actual['beta'],self.settings,'full'))
                norm=obj.evaluate(np.zeros(self.fmap.dimensions))['value']
                expected=kernel.fit(self.bases,self.fmap,self.settings,objective_scale=norm)
                actual=obj.fit(norm,lambda _:None)
                for k in set(expected)-{'seconds'}:
                    if k=='trace':self.assertEqual(actual[k],expected[k])
                    else:np.testing.assert_array_equal(actual[k],expected[k])

    def test_cached_neutral_revalidates_source_count_and_context(self):
        for mode in ('count','sample','frequency','cutoff'):
            obj=self.objective();obj.evaluate(np.zeros(self.fmap.dimensions))
            if mode=='count':obj.factory=lambda:iter(self.bases[:-1])
            if mode=='sample':obj.factory=lambda:iter([kernel.LeanBasis(replace(
                self.bases[0].sample,returns=self.bases[0].sample.returns+1)),self.bases[1]])
            if mode=='frequency':obj.fmap=CachedFeatureMap(self.fmap.frequencies*2,'changed')
            if mode=='cutoff':obj.settings=replace(self.settings,cutoff='2001-01-01')
            with self.assertRaises(ValueError):obj.evaluate(np.zeros(self.fmap.dimensions))
        for value in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):self.objective().fit(value,lambda _:None)

    def test_future_duplicate_and_mutated_arrays_rejected(self):
        obj=self.objective();obj.settings=replace(self.settings,cutoff='1998-12-31')
        with self.assertRaises(ValueError):obj.evaluate(self.theta)
        obj=self.objective();obj.factory=lambda:iter([self.bases[0]]*2)
        with self.assertRaises(ValueError):obj.evaluate(self.theta)
        obj=self.objective();obj.evaluate(np.zeros(self.fmap.dimensions))
        self.bases[0].sample.returns.setflags(write=True);self.bases[0].sample.returns[0]+=1
        with self.assertRaisesRegex(ValueError,'arrays changed'):obj.evaluate(np.zeros(self.fmap.dimensions))

    def test_cached_lu_identical_bounded_and_corruption_rejected(self):
        with patch('annual_ind.inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=32*2**30)):
            cache=OperatorCache(maximum_bytes=2**20,guard=lambda:None)
            first=cache.basis(self.bases[0].sample);second=cache.basis(self.bases[0].sample)
            for k,v in first.matrices(self.fmap,self.theta).items():
                np.testing.assert_array_equal(v,second.matrices(self.fmap,self.theta)[k])
            self.assertEqual(cache.hits,1);self.assertLessEqual(cache.bytes,cache.maximum_bytes)
            first.current_lu[0].setflags(write=True);first.current_lu[0][0,0]+=1
            with self.assertRaises(ValueError):cache.basis(self.bases[0].sample)
        with patch('annual_ind.inputs.psutil.virtual_memory',return_value=SimpleNamespace(available=5*2**30)):
            cache=OperatorCache(maximum_bytes=2**20,guard=lambda:None)
            cache.basis(self.bases[0].sample);self.assertEqual(cache.bytes,0)

    def test_year_schedule_contains_only_matured_months(self):
        for year,count in ((1971,61),(2020,649)):
            inputs=YearInputs(None,None,year,lambda:None,lambda *a,**k:None)
            self.assertEqual(len(inputs),count)
            self.assertEqual(inputs.dates[0],'1965-11-30')
            self.assertEqual(inputs.dates[-1],f'{year-1}-11-30')

    def test_temporal_overflow_preserves_complete_fit_tolerances(self):
        args,kwargs=layer_fixtures.ParallelLayerTests().problem(8)
        settings=LBFGSBSettings(lower=0.,upper=2.,tolerance=1e-5,max_steps=100,
            max_evaluations=250,max_line_searches=16,history_size=10,relative_ftol=0.,wall_seconds=60.)
        for rho in (.1,.01,.001):
            old=optimize(ParallelTemporalObjective(*args,workers=8,**dict(kwargs,rho=rho)),13,settings)
            for limit in (0,128*2**20):
                new=optimize(ParallelTemporalObjective(*args,workers=8,**dict(kwargs,rho=rho,reuse_bytes=limit)),13,settings)
                self.assertTrue(old['converged']);self.assertEqual(old['status'],new['status'])
                for k in ('gains','beta','gradient','objective_trace','projected_gradient_trace'):
                    np.testing.assert_allclose(new[k],old[k],atol=1e-11,rtol=1e-10)

if __name__=='__main__':unittest.main()
