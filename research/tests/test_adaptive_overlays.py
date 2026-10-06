"""Synthetic agreement-policy checks; no historical outcomes are consumed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('adaptive_overlay_kernels',
    ROOT/'scripts/adaptive_overlay_kernels_20260928.py')
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)
agreement_targets = kernel.agreement_targets
risk_scale = kernel.risk_scale


class AgreementOverlayTests(unittest.TestCase):
    def test_known_sign_disagreement_and_sample_variance(self):
        baseline = np.zeros((2, 3))
        c4 = np.array([[.02, -.01, .005], [.02, .018, .022]])
        actual = agreement_targets(baseline, c4)
        np.testing.assert_allclose(actual['dbar'], [.005, .02], atol=1e-16)
        np.testing.assert_allclose(actual['v'], [.000225, .000004], atol=1e-16)
        np.testing.assert_allclose(actual['g'], [.1, 100/101], atol=1e-15)
        self.assertAlmostEqual(actual['AGREE'][0], .0005)
        self.assertGreater(actual['AGREE'][1], .019)

    def test_identical_departures_are_exact_neutral(self):
        baseline = np.array([[1., 2., 4.], [-2., 3., 1.]])
        c4 = baseline + np.array([[.5], [-.25]])
        actual = agreement_targets(baseline, c4)
        np.testing.assert_array_equal(actual['g'], np.ones(2))
        np.testing.assert_array_equal(actual['AGREE'], c4.mean(axis=1))
        np.testing.assert_array_equal(actual['ASIZE'], c4.mean(axis=1))
        self.assertEqual(actual['h'], 1.)

    def test_forced_neutral_recovers_c4_ensemble_exactly(self):
        rng = np.random.default_rng(918)
        baseline, c4 = rng.normal(size=(2, 20, 3))
        actual = agreement_targets(baseline, c4, force_neutral=True)
        np.testing.assert_array_equal(actual['AGREE'], c4.mean(axis=1))
        np.testing.assert_array_equal(actual['ASIZE'], c4.mean(axis=1))
        np.testing.assert_array_equal(actual['g'], np.ones(20))
        self.assertEqual(actual['h'], 1.)

    def test_amplitude_control_matches_l1_increment_not_stock_direction(self):
        baseline = np.array([[.2, .1, .15], [-.1, -.2, -.1]])
        delta = np.array([[.02, -.01, .005], [.02, .018, .022]])
        actual = agreement_targets(baseline, baseline+delta)
        selective = actual['AGREE']-actual['BENS']
        uniform = actual['ASIZE']-actual['BENS']
        self.assertAlmostEqual(float(np.abs(selective).sum()), float(np.abs(uniform).sum()))
        np.testing.assert_allclose(uniform, actual['h']*actual['dbar'], atol=1e-15)
        self.assertFalse(np.allclose(selective, uniform))

    def test_seed_permutation_invariance(self):
        rng = np.random.default_rng(39)
        baseline, c4 = rng.normal(size=(2, 30, 3))
        reference = agreement_targets(baseline, c4)
        for permutation in ([1, 2, 0], [2, 0, 1], [2, 1, 0]):
            actual = agreement_targets(baseline[:, permutation], c4[:, permutation])
            for key in reference:
                np.testing.assert_allclose(actual[key], reference[key], atol=1e-14, rtol=1e-13)

    def test_common_scale_invariance_and_input_preservation(self):
        rng = np.random.default_rng(15)
        baseline, c4 = rng.normal(size=(2, 30, 3))
        before_b, before_c = baseline.copy(), c4.copy()
        reference = agreement_targets(baseline, c4)
        for scale in (.001, -2., 1000.):
            actual = agreement_targets(scale*baseline, scale*c4)
            np.testing.assert_allclose(actual['g'], reference['g'], atol=1e-14)
            self.assertAlmostEqual(actual['h'], reference['h'])
            for key in ('AGREE', 'ASIZE', 'BENS', 'CENS', 'dbar'):
                np.testing.assert_allclose(actual[key], scale*reference[key], atol=1e-11, rtol=1e-13)
            np.testing.assert_allclose(actual['v'], scale**2*reference['v'], atol=1e-8, rtol=1e-13)
        np.testing.assert_array_equal(baseline, before_b)
        np.testing.assert_array_equal(c4, before_c)

    def test_zero_departure_and_cancellation(self):
        zero = np.zeros((3, 3))
        actual = agreement_targets(zero, zero)
        np.testing.assert_array_equal(actual['g'], np.ones(3))
        np.testing.assert_array_equal(actual['AGREE'], np.zeros(3))
        self.assertEqual(actual['h'], 1.)
        actual = agreement_targets(zero, np.tile([-1., 0., 1.], (3, 1)))
        np.testing.assert_array_equal(actual['g'], np.zeros(3))
        np.testing.assert_array_equal(actual['AGREE'], np.zeros(3))
        np.testing.assert_array_equal(actual['ASIZE'], np.zeros(3))
        self.assertEqual(actual['h'], 1.)

    def test_invalid_inputs_and_outcomes_not_an_api_input(self):
        valid = np.ones((2, 3))
        for bad in (np.ones((2, 2)), np.ones(3), np.empty((0, 3)),
                    np.full((2, 3), np.nan), np.full((2, 3), np.inf)):
            with self.assertRaises(ValueError):
                agreement_targets(bad, valid)
        with self.assertRaises(ValueError):
            agreement_targets(valid, valid, force_neutral=.5)
        # The kernel has no way to consume realized returns or tune the gate.
        with self.assertRaises(TypeError):
            agreement_targets(valid, valid, realized_returns=np.ones(2))
        with self.assertRaises(TypeError):
            agreement_targets(valid, valid, variance_multiplier=.1)


class RiskCalibrationTests(unittest.TestCase):
    def setUp(self):
        months=np.arange(np.datetime64('2011-01','M'),np.datetime64('2016-03','M'))
        self.dates=(months+np.timedelta64(1,'M')).astype('datetime64[D]')-np.timedelta64(1,'D')
        self.gross=np.linspace(-.04,.04,len(months))
        self.predicted=np.full(len(months),.0001)

    def test_exact_sixty_matured_months_and_sample_variance(self):
        answer=risk_scale('2015-12-31',self.dates,self.gross,self.predicted)
        np.testing.assert_array_equal(answer['indices'],np.arange(60))
        self.assertEqual(answer['window_first'],'2011-01-31')
        self.assertEqual(answer['window_last'],'2015-12-31')
        expected=np.sum((self.gross[:60]-self.gross[:60].mean())**2)/59
        self.assertAlmostEqual(answer['realized_variance'],expected)
        self.assertAlmostEqual(answer['k'],np.sqrt(.0001/expected))

    def test_future_and_old_outcomes_cannot_change_current_scale(self):
        expected=risk_scale('2015-12-31',self.dates,self.gross,self.predicted)
        g=self.gross.copy();v=self.predicted.copy();g[60:]=np.nan;v[60:]=-np.inf
        actual=risk_scale('2015-12-31',self.dates,g,v)
        self.assertEqual(expected['k'],actual['k'])
        next_scale=risk_scale('2016-01-31',self.dates,self.gross,self.predicted)
        g=self.gross.copy();v=self.predicted.copy();g[0]=9999;v[0]=9999
        self.assertEqual(next_scale['k'],risk_scale('2016-01-31',self.dates,g,v)['k'])

    def test_cap_and_zero_variance(self):
        self.assertEqual(risk_scale('2015-12-31',self.dates,self.gross,np.ones(62))['k'],1.)
        self.assertEqual(risk_scale('2015-12-31',self.dates,np.zeros(62),np.zeros(62))['k'],1.)
        self.assertEqual(risk_scale('2015-12-31',self.dates,self.gross,np.zeros(62))['k'],0.)

    def test_missing_history_bad_dates_and_negative_variance_fail(self):
        with self.assertRaises(ValueError):risk_scale('2015-11-30',self.dates,self.gross,self.predicted)
        with self.assertRaises(ValueError):risk_scale('2015-12-30',self.dates,self.gross,self.predicted)
        with self.assertRaises(ValueError):risk_scale('2015-12-31',self.dates,self.gross,-self.predicted)
        dates=self.dates.copy();dates[1]=dates[0]
        with self.assertRaises(ValueError):risk_scale('2015-12-31',dates,self.gross,self.predicted)


class AdaptiveReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('adaptive_screen',ROOT/'scripts/adaptive_policy_screen_20260928.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def test_seven_path_replay_common_start_switch_cost_and_continuity(self):
        rng=np.random.default_rng(914);s=self.runner
        ids=np.array([1,2,3]);common=np.array([.1,-.2,.3])
        parent_h=np.repeat(common[:,None],9,axis=1);own_h=np.repeat(common[:,None],7,axis=1)
        for month in range(4):
            next_ids=np.array([2,3,4]) if month==0 else ids.copy()
            components=rng.normal(0,.2,(3,6));targets=s.shared.combine_targets(components)
            parent,unused=s.shared.replay_packet(dict(ids=ids,next_ids=next_ids,
                m=np.diag([.2,.4,.6]),targets=targets,pretrade=parent_h,
                impact=np.array([.1,.2,.3]),wealth=np.array(10.),
                total_returns=rng.normal(0,.02,3),market_growth=np.array(.01),
                excess_returns=rng.normal(0,.02,3),methods=np.array(s.shared.METHODS)))
            agreement=agreement_targets(components[:,:3],components[:,3:])
            scale=dict(k=.7,indices=np.arange(60),predicted_variance_mean=.00049,realized_variance=.001)
            actual,rows=s.replay_month(parent,own_h,agreement,scale)
            # All seven portfolios retain their own state and pay actual trades.
            np.testing.assert_array_equal(actual['pretrade'],own_h)
            q=actual['weights']-own_h
            expected=.5*10*np.sum(np.array([.1,.2,.3])[:,None]*q*q,axis=0)
            np.testing.assert_allclose([r['impact_cost'] for r in rows],expected,atol=1e-15)
            self.assertGreater(expected.sum(),0.)
            np.testing.assert_array_equal(actual['targets'][:,5],.7*parent['targets'][:,7])
            np.testing.assert_array_equal(actual['targets'][:,6],.7*parent['targets'][:,6])
            if month==0:np.testing.assert_array_equal(actual['next_pretrade'][2],np.zeros(7))
            ids=next_ids;parent_h=parent['next_pretrade'];own_h=actual['next_pretrade']

    def test_combined_family_gate_requires_every_control_and_both_periods(self):
        s=self.runner
        metrics=[dict(period=p,method=m,source_utility=.2 if m in ('AGREE','RISK') else .1)
                 for p in ('primary','robustness') for m in s.METHODS]
        intervals=[dict(period='primary',candidate=a,control=b,block_months=l,lower=.01)
                   for a,b in s.PAIRS for l in (12,24)]
        self.assertEqual(s.gate(metrics,intervals)['candidate_for_larger_study_consideration'],'AGREE')
        intervals[0]['lower']=0.
        self.assertEqual(s.gate(metrics,intervals)['candidate_for_larger_study_consideration'],'RISK')
        for row in metrics:
            if row['method']=='RBASE' and row['period']=='robustness':row['source_utility']=.2
        self.assertFalse(s.gate(metrics,intervals)['exploratory_gate_passed'])
        with self.assertRaises(ValueError):s.gate(metrics,intervals[:-1])


if __name__ == '__main__':
    unittest.main()
