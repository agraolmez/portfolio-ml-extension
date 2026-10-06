"""Numerical revision: hard budgets, accepted-state integrity and exact reuse."""
from dataclasses import replace
import copy
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

from temporal_numerics.optimizer import LBFGSBSettings, optimize
from temporal_numerics.cache import RetainedBatchCache
from temporal_numerics.training import train_profile, train_current_only_profile
from temporal_numerics.pilot import validate_config, run_pilot
from temporal_numerics.logging import DurableLog
from temporal_study.cache import AnnualBatchCache, batch_bytes
from temporal_study.controls import CurrentOnlyFeatureMap, CurrentOnlyModel, save_current_only_result
from temporal_extension.training import save_result
from temporal_extension.streaming import StreamingObjective
from geometry_study.kernel import LeanBasis, evaluate as parent_evaluate
import test_temporal_cache as cache_fixtures
from test_temporal_extension import temporal_fixture

ROOT = Path(__file__).resolve().parents[1]


def settings(**changes):
    result = LBFGSBSettings(0., 2., 1e-7, 100, 250, 16, 10, 0., 30.)
    return replace(result, **changes)


def quadratic(target, weights=None):
    target = np.asarray(target, dtype=float)
    weights = np.ones_like(target) if weights is None else np.asarray(weights)
    def evaluate(point, with_gradient=True):
        delta = point-target
        return dict(value=-.5*float(delta@(weights*delta)), gradient=-weights*delta,
                    beta=np.r_[point, point.sum()], solve_residual=1e-16)
    return evaluate


class NumericalOptimizerTests(unittest.TestCase):
    def test_interior_coupled_curvature_and_deterministic_trace(self):
        target = np.array([.2, 1.7, .8, 1.25])
        matrix = np.array([[8,2,1,0],[2,3,.1,0],[1,.1,1,0],[0,0,0,.2]])
        def evaluate(point, with_gradient=True):
            d = point-target
            return dict(value=-.5*float(d@matrix@d), gradient=-matrix@d,
                        beta=point.copy(), solve_residual=1e-16)
        one = optimize(evaluate, 4, settings())
        two = optimize(evaluate, 4, settings())
        self.assertTrue(one['converged'])
        np.testing.assert_allclose(one['gains'], target, atol=5e-7)
        self.assertEqual(one['evaluations'], two['evaluations'])
        self.assertLessEqual(one['projected_gradient_norm'], 1e-7)
        self.assertTrue(np.all(np.diff(one['objective_trace']) >= -1e-14))

    def test_boundary_optimum_uses_constrained_stationarity(self):
        result = optimize(quadratic([-1., 3.]), 2, settings())
        self.assertTrue(result['converged'])
        np.testing.assert_array_equal(result['gains'], [0., 2.])
        self.assertGreater(np.max(abs(result['gradient'])), .5)
        self.assertEqual(result['projected_gradient_norm'], 0.)

    def test_hard_evaluation_cap_includes_neutral_and_unaccepted_trials(self):
        def solver(fun, point, **kwargs):
            fun(point)
            fun(np.array([1.1]))
            fun(np.array([1.2]))
            self.fail('Hard evaluator cap failed')
        with patch('temporal_numerics.optimizer.minimize', solver):
            result = optimize(quadratic([1.8]), 1, settings(max_evaluations=2))
        self.assertEqual(result['status'], 'evaluation_limit')
        self.assertEqual(result['evaluation_count'], 2)
        self.assertEqual(result['accepted_steps'], 0)
        np.testing.assert_array_equal(result['gains'], [1.])
        np.testing.assert_array_equal(result['beta'], [1., 1.])
        self.assertFalse(result['evaluations'][-1]['accepted'])

    def test_interruption_keeps_matching_last_accepted_point_not_last_trial(self):
        seen = []
        base = quadratic([2.])
        def evaluate(point, with_gradient=True):
            seen.append(point.copy())
            if len(seen) == 4:
                raise TimeoutError('approved budget')
            return base(point)
        def solver(fun, point, callback, **kwargs):
            fun(point)
            fun(np.array([1.2])); callback(np.array([1.2]))
            fun(np.array([1.6]))
            fun(np.array([1.8]))
        with patch('temporal_numerics.optimizer.minimize', solver):
            result = optimize(evaluate, 1, settings())
        self.assertEqual(result['status'], 'wall_time_limit')
        self.assertEqual(result['accepted_steps'], 1)
        np.testing.assert_array_equal(result['gains'], [1.2])
        np.testing.assert_array_equal(result['beta'], [1.2,1.2])
        np.testing.assert_allclose(result['gradient'], [.8])
        self.assertFalse(result['converged'])

    def test_library_success_never_overrides_independent_accuracy_gate(self):
        def solver(fun, point, **kwargs):
            fun(point)
            return SimpleNamespace(x=point, success=True, status=0, nit=0, nfev=1,
                                   message='relative objective stagnation')
        with patch('temporal_numerics.optimizer.minimize', solver):
            result = optimize(quadratic([1.9]), 1, settings())
        self.assertEqual(result['status'], 'library_stopped_before_stationarity')
        self.assertFalse(result['converged'])

    def test_step_cap_and_final_allowed_optimum(self):
        def solver(fun, point, callback, **kwargs):
            fun(point)
            fun(np.array([1.2])); callback(np.array([1.2]))
            self.fail('Step cap failed')
        with patch('temporal_numerics.optimizer.minimize', solver):
            limited = optimize(quadratic([1.9]), 1, settings(max_steps=1))
            final = optimize(quadratic([1.2]), 1, settings(max_steps=1))
        self.assertEqual(limited['status'], 'step_limit')
        self.assertTrue(final['converged'])
        self.assertEqual(final['accepted_steps'], 1)

    def test_zero_budget_invalid_settings_and_callback_mutation(self):
        calls = []
        result = optimize(lambda *a, **k: calls.append(1), 1, settings(max_evaluations=0))
        self.assertEqual(result['status'], 'evaluation_limit')
        self.assertEqual(calls, [])
        for change in [dict(tolerance=0),dict(max_steps=True),dict(history_size=0),
                       dict(max_line_searches=0),dict(relative_ftol=-1),dict(wall_seconds=np.inf)]:
            with self.subTest(change=change), self.assertRaises(ValueError): settings(**change)
        with self.assertRaises(ValueError): optimize(quadratic([1.]),1,settings(lower=1.1))
        def mutates(point, with_gradient=True):
            point[0]=0
        result = optimize(mutates,1,settings())
        self.assertEqual(result['status'],'evaluation_failed')
        self.assertIsNone(result['beta'])

    def test_inaccurate_solve_and_over_budget_completed_eval_are_ineligible(self):
        result = optimize(lambda *a,**k: dict(value=0.,gradient=np.zeros(1),beta=np.ones(2),solve_residual=1e-3),1,settings())
        self.assertEqual(result['status'],'evaluation_failed')
        now=[0.]
        def slow(point,with_gradient=True):
            now[0]=40.
            return quadratic([1.])(point)
        with patch('temporal_numerics.optimizer.monotonic',lambda:now[0]):
            result=optimize(slow,1,settings())
        self.assertEqual(result['status'],'wall_time_limit')
        self.assertIsNone(result['beta'])
        self.assertEqual(result['evaluations'][0]['status'],'completed_after_time_limit')


class DurableLoggingTests(unittest.TestCase):
    def test_broken_chat_connection_does_not_interrupt_persistent_log(self):
        class Broken:
            calls=0
            def write(self,line):
                self.calls+=1
                raise BrokenPipeError('chat disconnected')
        destination=io.StringIO(); echo=Broken();log=DurableLog(destination,echo=echo)
        log.emit({'event':'first'});log.emit({'event':'second'})
        self.assertEqual(echo.calls,1)
        self.assertEqual([json.loads(s)['event'] for s in destination.getvalue().splitlines()],['first','second'])

    def test_closed_echo_is_optional_but_persistent_log_failures_are_not(self):
        destination=io.StringIO();echo=io.StringIO();echo.close()
        log=DurableLog(destination,echo=echo);log.emit({'event':'survives'})
        self.assertIsNone(log.echo)
        destination.close()
        with self.assertRaises(ValueError):log.emit({'event':'must_fail'})

    def test_invalid_payload_cannot_be_silently_written(self):
        destination=io.StringIO();log=DurableLog(destination,echo=io.StringIO())
        with self.assertRaises(ValueError):log.emit({'invalid':float('nan')})
        self.assertEqual(destination.getvalue(),'')


class NumericalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls): cls.threads.restore_original_limits()

    def test_cache_overflow_retains_hits_and_matches_full_cache_profile(self):
        history,parent,bases,histories=cache_fixtures.CacheTests().fixture()
        size=max(batch_bytes(LeanBasis(b.sample),h) for b,h in zip(bases,histories))
        cache=RetainedBatchCache(history,maximum_bytes=size)
        first=list(cache.factory(parent)()); second=list(cache.factory(parent)())
        self.assertEqual((cache.hits,cache.misses),(1,3))
        self.assertIs(first[0][0],second[0][0])
        self.assertIsNot(first[1][0],second[1][0])
        self.assertLessEqual(cache.peak_retained_bytes,size)
        self.assertTrue(cache.admission_closed)
        _,_,_,_,ps,_,fmap,_=temporal_fixture()
        gains=np.linspace(.4,1.6,fmap.gain_count)
        big=AnnualBatchCache(history,maximum_bytes=10**8)
        one=StreamingObjective(cache.factory(parent),fmap,ps,rho=.01,objective_scale=.7)(gains)
        two=StreamingObjective(big.factory(parent),fmap,ps,rho=.01,objective_scale=.7)(gains)
        for key in ('value','gradient','beta','score','risk','cost'):
            np.testing.assert_array_equal(one[key],two[key])
        with self.assertRaises(MemoryError): list(RetainedBatchCache(history,maximum_bytes=1).factory(parent)())

    def test_cache_still_rejects_changed_reference_and_unmatured_cutoff(self):
        history,parent,_,_=cache_fixtures.CacheTests().fixture()
        cache=RetainedBatchCache(history,maximum_bytes=10**8)
        factory=cache.factory(parent);history.reference_fingerprint='changed'
        with self.assertRaises(ValueError):list(factory())
        parent.settings=SimpleNamespace(cutoff=parent.metadata['training_last'])
        with self.assertRaises(ValueError):cache.factory(parent)

    def problem(self, fixed=False):
        _,parent,_,bases,ps,theta,fmap,histories=temporal_fixture(13,7)
        beta=parent_evaluate(bases,parent,theta,ps)['beta']
        opts=settings(lower=1. if fixed else 0.,upper=1. if fixed else 2.,max_steps=3)
        kwargs=dict(expected_neutral_beta=beta,expected_parent_feature_fingerprint=fmap.fingerprint,
            parent_fingerprint='parent',source_fingerprint='source',reference_fingerprint='reference',
            rho=.03,objective_scale=.7,optimizer_settings=opts)
        return bases,ps,fmap,histories,kwargs

    def test_temporal_neutral_and_nonconverged_serialization(self):
        for fixed in (True,False):
            bases,ps,fmap,histories,kwargs=self.problem(fixed)
            model,receipt=train_profile(lambda:zip(bases,histories),fmap,ps,**kwargs)
            self.assertTrue(receipt['neutral_gate']['passed'])
            self.assertEqual(receipt['eligible_for_portfolio_evaluation'],fixed)
            self.assertEqual(receipt['optimizer_method'],'L-BFGS-B')
            with tempfile.TemporaryDirectory() as tmp:
                saved=save_result(Path(tmp)/'fit',model,receipt)
                self.assertEqual(saved['model_file'],'model.npz' if fixed else 'diagnostic_model.npz')
            if not fixed:
                expected=StreamingObjective(lambda:zip(bases,histories),fmap,ps,rho=.03,objective_scale=.7)(model.gains)
                np.testing.assert_array_equal(model.beta,expected['beta'])

    def test_bad_neutral_coefficient_stops_before_any_gain_update(self):
        bases,ps,fmap,histories,kwargs=self.problem()
        kwargs['expected_neutral_beta']+=1.
        model,receipt=train_profile(lambda:zip(bases,histories),fmap,ps,**kwargs)
        self.assertIsNone(model)
        self.assertFalse(receipt['eligible_for_portfolio_evaluation'])
        self.assertEqual(receipt['evaluation_count'],1)

    def test_current_only_uses_same_new_solver_and_distinct_persistence(self):
        bases,ps,fmap,histories,kwargs=self.problem(True)
        kwargs.pop('reference_fingerprint')
        cmap=CurrentOnlyFeatureMap(fmap.parent,fmap.theta,fmap.groups)
        model,receipt=train_current_only_profile(lambda:iter(bases),cmap,ps,**kwargs)
        self.assertIsInstance(model,CurrentOnlyModel)
        self.assertTrue(receipt['converged'])
        self.assertEqual(receipt['optimizer_method'],'L-BFGS-B')
        with tempfile.TemporaryDirectory() as tmp:
            saved=save_current_only_result(Path(tmp)/'control',model,receipt)
            actual=CurrentOnlyModel.load(Path(tmp)/'control'/saved['model_file'],expected_parent_fingerprint='parent')
            np.testing.assert_array_equal(actual.beta,model.beta)

    def test_revision_config_rejects_changed_science_or_unapproved_budget(self):
        base=json.loads((ROOT/'config/extension3_study_v1.json').read_text())
        revised=json.loads((ROOT/'config/extension3_study_v2.json').read_text())
        validate_config(revised,base)
        for path,value in [('reference_weights',[.5,.5]),('cache_policy','lru'),('optimizer_method','other'),
                           ('resources',dict(revised['resources'],basis_cache_gib=24)),
                           ('pilot',dict(revised['pilot'],combined_wall_seconds=7200)),
                           ('optimizer',dict(revised['optimizer'],tolerance=.001))]:
            changed=copy.deepcopy(revised);changed[path]=value
            with self.subTest(path=path),self.assertRaises(ValueError):validate_config(changed,base)

    def test_previous_attempt_cannot_be_omitted_to_reset_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/'output').mkdir()
            for name in ('extension3_study_v1.json','extension3_study_v2.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            pilot=root/'output/extension3_numerical_pilot_20260919'
            prior=pilot/'run01';prior.mkdir(parents=True)
            (prior/'pilot_lock.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'previous attempts'):
                run_pilot(root,pilot/'run02',config_path=root/'config/extension3_study_v2.json')
            self.assertFalse((pilot/'run02').exists())

    def test_exhausted_additional_hour_rejected_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/'output').mkdir()
            for name in ('extension3_study_v1.json','extension3_study_v2.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            config=json.loads((root/'config/extension3_study_v2.json').read_text())
            pilot=root/'output/extension3_numerical_pilot_20260919'
            prior=pilot/'run01';prior.mkdir(parents=True)
            (prior/'pilot_lock.json').write_text(json.dumps({'config':config}))
            (prior/'pilot_result.json').write_text(json.dumps({'seconds':3600.}))
            with patch('temporal_numerics.pilot.preflight') as preflight:
                with self.assertRaisesRegex(ValueError,'budget is exhausted'):
                    run_pilot(root,pilot/'run02',config_path=root/'config/extension3_study_v2.json',prior_attempts=[prior])
                preflight.assert_not_called()
            self.assertFalse((pilot/'run02').exists())

    def test_interrupted_attempt_without_lock_cannot_disappear_from_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/'output').mkdir()
            for name in ('extension3_study_v1.json','extension3_study_v2.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            pilot=root/'output/extension3_numerical_pilot_20260919'
            (pilot/'run01').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'previous attempts'):
                run_pilot(root,pilot/'run02',config_path=root/'config/extension3_study_v2.json')
            self.assertFalse((pilot/'run02').exists())


if __name__ == '__main__': unittest.main()
