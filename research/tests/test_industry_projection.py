"""Dense independent oracle and synthetic contracts; no historical outcomes."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('industry_projection',ROOT/'scripts/industry_projection_kernels_20260928.py')
k=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(k)


class IndustryProjectionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2918);n=45
        f=rng.normal(size=(25,25));self.factor=.001*(f@f.T+np.eye(25))
        self.loads=rng.normal(size=(n,25));self.specific=rng.uniform(.001,.03,n)
        self.impact=rng.uniform(.01,.08,n);self.wealth=4.
        self.base=rng.normal(0,.1,n);self.c4=rng.normal(0,.1,n)
        self.q=10*(self.loads@self.factor@self.loads.T+np.diag(self.specific))+self.wealth*np.diag(self.impact)
    def call(self,**changes):
        args=dict(baseline_target=self.base,c4_target=self.c4,loadings=self.loads,factor_cov=self.factor,
                  specific_variance=self.specific,impact=self.impact,wealth=self.wealth,factor_names=k.FACTORS)
        args.update(changes);return k.project_industry_targets(**args)
    def test_dense_oracle_kkt_orthogonality_and_norm_control(self):
        result=self.call();b=self.loads[:,:12];d=self.c4-self.base
        v=np.linalg.solve(self.q,b);mult=np.linalg.solve(b.T@v,b.T@d);z=d-v@mult
        np.testing.assert_allclose(result['projected_departure'],z,atol=2e-13,rtol=1e-11)
        np.testing.assert_allclose(b.T@z,0,atol=2e-13)
        np.testing.assert_allclose(self.q@(z-d)+b@mult,0,atol=2e-13)
        self.assertAlmostEqual(z@self.q@(d-z),0,places=12)
        self.assertAlmostEqual(d@self.q@d,z@self.q@z+(d-z)@self.q@(d-z),places=12)
        amp=result['AMPL']-self.base
        self.assertAlmostEqual(amp@self.q@amp,z@self.q@z,places=12)
        self.assertLessEqual(result['amplitude'],1+1e-12)
        self.assertLess(result['kkt_max_residual'],1e-12)
    def test_projection_idempotence(self):
        first=self.call();second=self.call(c4_target=first['IND'])
        np.testing.assert_allclose(second['IND'],first['IND'],atol=2e-13,rtol=1e-11)
        self.assertAlmostEqual(second['amplitude'],1,places=12)
    def test_neutral_exact_target_recovery(self):
        result=self.call(force_neutral=True)
        np.testing.assert_array_equal(result['IND'],self.c4)
        np.testing.assert_array_equal(result['AMPL'],self.c4)
        self.assertEqual(result['amplitude'],1.)
    def test_zero_departure_and_input_preservation(self):
        base=self.base.copy();loads=self.loads.copy();result=self.call(c4_target=self.base)
        np.testing.assert_array_equal(result['IND'],self.base)
        np.testing.assert_array_equal(result['AMPL'],self.base)
        self.assertEqual(result['qnorm_before'],0.)
        self.assertEqual(result['amplitude'],1.)
        np.testing.assert_array_equal(self.base,base);np.testing.assert_array_equal(self.loads,loads)
    def test_factor_metadata_conflict_and_rank_failure(self):
        with self.assertRaises(ValueError):self.call(factor_names=tuple(reversed(k.FACTORS)))
        loads=self.loads.copy();loads[:,1]=loads[:,0]
        with self.assertRaises(ValueError):self.call(loadings=loads)
    def test_invalid_inputs_are_not_repaired(self):
        for change in (dict(impact=-self.impact),dict(specific_variance=-self.specific),
                       dict(wealth=0),dict(c4_target=np.full(45,np.nan)),
                       dict(factor_cov=-self.factor),dict(loadings=self.loads[:,:12])):
            with self.assertRaises(ValueError):self.call(**change)
        f=self.factor.copy();f[0,1]+=.1
        with self.assertRaises(ValueError):self.call(factor_cov=f)
    def test_no_outcome_or_penalty_tuning_api(self):
        with self.assertRaises(TypeError):self.call(realized_returns=np.zeros(45))
        with self.assertRaises(TypeError):self.call(ridge=.01)


class IndustryReplayTests(unittest.TestCase):
    setUp=IndustryProjectionTests.setUp
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('industry_screen',ROOT/'scripts/industry_projection_screen_20260928.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def test_accounting_controls_and_target_vs_actual_exposure(self):
        s=self.runner;rng=np.random.default_rng(174);ids=np.arange(45)
        common=rng.normal(0,.1,45);parent_h=np.repeat(common[:,None],9,axis=1)
        own_h=np.repeat(common[:,None],5,axis=1)
        risk=dict(ids=ids,loadings=self.loads,factor_cov=self.factor,specific_variance=self.specific)
        for month in range(3):
            parent,_=s.shared.replay_packet(dict(ids=ids,next_ids=ids,
                m=np.diag(np.linspace(.1,.8,45)),targets=s.shared.combine_targets(rng.normal(0,.1,(45,6))),
                pretrade=parent_h,impact=self.impact,wealth=np.array(self.wealth),
                total_returns=rng.normal(0,.02,45),market_growth=np.array(.01),
                excess_returns=rng.normal(0,.02,45),methods=np.array(s.shared.METHODS)))
            actual,rows=s.replay_month(parent,risk,own_h,k.FACTORS)
            np.testing.assert_array_equal(actual['pretrade'],own_h)
            trade=actual['weights']-own_h
            expected=.5*self.wealth*np.sum(self.impact[:,None]*trade*trade,axis=0)
            np.testing.assert_allclose([r['impact_cost'] for r in rows],expected,atol=1e-14)
            np.testing.assert_allclose(actual['industry_exposure_after'],0,atol=1e-12)
            # The unchanged nonuniform m can create executed exposure even
            # when the incremental target itself is industry-neutral.
            self.assertGreater(np.linalg.norm(actual['actual_industry_increment_vs_bens'][:,3]),1e-6)
            own_h=actual['next_pretrade'];parent_h=parent['next_pretrade']

    def test_gate_cannot_ignore_amplitude_control_or_robustness(self):
        s=self.runner
        metrics=[dict(period=p,method=m,source_utility=.2 if m=='IND' else .1)
                 for p in ('primary','robustness') for m in s.METHODS]
        intervals=[dict(period='primary',candidate=a,control=b,block_months=l,lower=.01)
                   for a,b in s.PAIRS for l in (12,24)]
        self.assertTrue(s.gate(metrics,intervals)['exploratory_gate_passed'])
        intervals[-1]['lower']=0.
        self.assertFalse(s.gate(metrics,intervals)['exploratory_gate_passed'])
        intervals[-1]['lower']=.01
        for row in metrics:
            if row['period']=='robustness' and row['method']=='AMPL':row['source_utility']=.2
        self.assertFalse(s.gate(metrics,intervals)['exploratory_gate_passed'])
        with self.assertRaises(ValueError):s.gate(metrics,intervals[:-1])


if __name__=='__main__':unittest.main()
