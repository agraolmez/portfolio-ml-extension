"""Fitting and persistence preserve the scientific and convergence gates."""
from dataclasses import asdict,replace
import pandas as pd
from geometry_study.kernel import LeanBasis
from temporal_fast_v4.deferred import DeferredMonth
from pathlib import Path
import json,tempfile,unittest
import numpy as np
from test_temporal_extension import temporal_fixture
from temporal_extension.streaming import StreamingObjective
from temporal_numerics.optimizer import LBFGSBSettings
from temporal_numerics import training as original
from temporal_fast_v4 import training as fast
from temporal_fast_v4.settings import EXECUTION
from temporal_study.controls import CurrentOnlyFeatureMap,current_only_batch_factory
from temporal_extension.training import save_result
from temporal_extension.model import TemporalModel


class FastTrainingTests(unittest.TestCase):
    def problem(self):
        _,_,_,bases,settings,_,fmap,history=temporal_fixture(dimensions=13,frequencies=7,months=3)
        bases=[LeanBasis(replace(b.sample,market=replace(b.sample.market,decision_date=(pd.Timestamp(b.sample.market.decision_date)+pd.offsets.MonthEnd(0)).date().isoformat()))) for b in bases]
        factory=lambda:zip(bases,history)
        beta=StreamingObjective(factory,fmap,settings,rho=.03,objective_scale=.7)(np.ones(13))['beta']
        limits=LBFGSBSettings(lower=0.,upper=2.,tolerance=1e-8,max_steps=4,max_evaluations=50,
                             max_line_searches=16,history_size=10,relative_ftol=0.,wall_seconds=30.)
        common=dict(expected_neutral_beta=beta,expected_parent_feature_fingerprint=fmap.fingerprint,
            parent_fingerprint='parent',source_fingerprint='source',reference_fingerprint='reference',
            rho=.03,objective_scale=.7,optimizer_settings=limits)
        return bases,settings,fmap,factory,common

    def test_temporal_fit_preserves_optimizer_trajectory_and_diagnostic_persistence(self):
        _,settings,fmap,factory,common=self.problem()
        baseline,expected=original.train_profile(factory,fmap,settings,**common)
        def deferred():
            for basis,history in factory():
                yield DeferredMonth(basis.sample.market.decision_date,settings.cutoff,basis.sample.market.source_id,
                    lambda b=basis,h=history:(b,h))
        model,actual=fast.train_profile(deferred,fmap,settings,**common)
        self.assertEqual(actual['status'],expected['status'])
        self.assertEqual(actual['accepted_steps'],expected['accepted_steps'])
        np.testing.assert_allclose(actual['objective_trace'],expected['objective_trace'],atol=1e-11,rtol=1e-10)
        np.testing.assert_allclose(model.gains,baseline.gains,atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(model.beta,baseline.beta,atol=1e-11,rtol=1e-10)
        self.assertEqual(actual['optimizer_settings'],asdict(common['optimizer_settings']))
        self.assertEqual(actual['execution_settings'],dict(EXECUTION))
        with tempfile.TemporaryDirectory() as tmp:
            body=save_result(Path(tmp)/'fit',model,actual)
            loaded=TemporalModel.load(Path(tmp)/'fit'/body['model_file'],expected_parent_fingerprint='parent')
            np.testing.assert_array_equal(loaded.beta,model.beta)

    def test_current_only_fit_uses_fast_profile_and_original_control_identity(self):
        bases,settings,parent,_,common=self.problem();common.pop('reference_fingerprint')
        fmap=CurrentOnlyFeatureMap(parent.parent,parent.theta,parent.groups)
        baseline,expected=original.train_current_only_profile(lambda:iter(bases),fmap,settings,**common)
        def deferred():
            for basis in bases:
                yield DeferredMonth(basis.sample.market.decision_date,settings.cutoff,basis.sample.market.source_id,
                    lambda b=basis:(b,None))
        model,actual=fast.train_current_only_profile(deferred,fmap,settings,**common)
        self.assertEqual(actual['status'],expected['status'])
        self.assertEqual(actual['execution_settings'],dict(EXECUTION))
        self.assertEqual(actual['reference_fingerprint'],expected['reference_fingerprint'])
        np.testing.assert_allclose(model.gains,baseline.gains,atol=1e-9,rtol=1e-8)

    def test_fixed_control_keeps_neutral_gate_and_exact_conditional_solution(self):
        from temporal_study.controls import fit_fixed_profile
        _,settings,fmap,factory,common=self.problem();common.pop('optimizer_settings')
        common.update(gains=np.full(13,.5),lower=0.,upper=2.)
        baseline,expected=fit_fixed_profile(factory,fmap,settings,**common)
        model,actual=fast.fit_fixed_profile(factory,fmap,settings,**common)
        np.testing.assert_allclose(model.beta,baseline.beta,atol=1e-11,rtol=1e-10)
        self.assertEqual(actual['status'],'fixed_configuration_solved')
        self.assertTrue(actual['neutral_gate']['passed'])
        self.assertEqual(actual['execution_settings'],dict(EXECUTION))
        common['expected_neutral_beta']=common['expected_neutral_beta']+.1
        with self.assertRaises(AssertionError):fast.fit_fixed_profile(factory,fmap,settings,**common)

    def test_recorded_execution_matches_configuration(self):
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'config/extension3_speed_v4.json').read_text())
        self.assertEqual(config['execution'],dict(EXECUTION))
        self.assertFalse(config['authorization']['full_historical_optimization'])


if __name__=='__main__':unittest.main()
