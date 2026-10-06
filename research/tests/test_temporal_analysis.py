"""Synthetic complete-path statistics, selection links and paired inference."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from inference.core import basic_interval
from temporal_study.analysis import analyze_paths, compound_drawdown, path_metrics
from temporal_study.selection import Candidate, CandidatePath, SelectionPlan, select_candidates


def digest(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def analysis_fixture(*, identical=False):
    plan = dict(id='synthetic-temporal-analysis', frequency_seeds=[1, 2, 3],
        first_return_year=2000, last_return_year=2003,
        penalties_in_tie_order=[.1, .01, .001],
        selection=dict(first_return='2001-01-31', last_return='2001-12-31', risk_aversion=10.),
        periods=dict(development=[2000, 2000], selection=[2001, 2001], evaluation=[2002, 2002],
                     exposed_robustness=[2003, 2003]),
        primary_contrasts=[['selected_temporal', method] for method in
                           ('parent', 'fixed_temporal', 'selected_current_only', 'fixed1024')],
        inference=dict(replicas=64, block_months=[3, 6], master_seed=20260918,
                       alpha=.05, family=4, bootstrap_refits=False))
    dates = pd.date_range('2000-01-31', '2003-12-31', freq='ME')
    methods = ['selected_temporal', 'parent', 'fixed_temporal', 'selected_current_only', 'fixed1024']
    rows = []
    initial = digest('common initial holdings')
    for seed in (1, 2, 3):
        for method_index, method in enumerate(methods):
            for i, date in enumerate(dates):
                phase = seed*np.pi/2
                value = .008+.003*seed*np.sin(i*.8+phase)
                if not identical:
                    value += (4-method_index)*.0003+(method_index*.0005)*np.cos(i*.61)
                impact = .0001*(method_index+1)
                rows.append(dict(seed=seed, method=method, decision_eom=date-pd.offsets.MonthEnd(1),
                    return_eom=date, gross_excess=value+impact, impact_cost=impact, net_excess=value,
                    turnover=.1+.01*seed, gross_exposure=1.2+.1*seed,
                    net_exposure=.1, largest_absolute_weight=.1, gross_weight_concentration=.05,
                    parent_cutoff=pd.Timestamp(f'{date.year-1}-12-31'),
                    training_last_decision=pd.Timestamp(f'{date.year-1}-11-30'),
                    training_outcomes_available_at=pd.Timestamp(f'{date.year-1}-12-31'),
                    pretrade_sha256=initial if i == 0 else digest((seed, method, i)),
                    next_pretrade_sha256=digest((seed, method, i+1)), holdings_reset=False))
    frame = pd.DataFrame(rows)
    manifest = dict(status='COMPLETE', months=48, paths=15, monthly_records=len(frame),
                    first_trade_charged=True, annual_or_period_holdings_reset=False)
    receipts = {}
    for family, method in [('temporal', 'selected_temporal'), ('current_only', 'selected_current_only')]:
        candidates = tuple(Candidate(f'rho_{rho}', {'rho': rho}) for rho in plan['penalties_in_tie_order'])
        selection_plan = SelectionPlan(plan['id'], family, (1, 2, 3), candidates,
            '1999-12-31', '2001-01-31', '2001-12-31', '2001-12-31', '2001-12-31',
            '2001-11-30', '2001-12-31', initial, 10., 'Synthetic exposed retrospective selection')
        paths = []
        for seed in (1, 2, 3):
            base = frame[(frame.seed == seed) & frame.method.eq(method)
                         & (frame.return_eom <= pd.Timestamp('2001-12-31'))].copy()
            for j, candidate in enumerate(candidates):
                candidate_frame = base.copy()
                candidate_frame['method'] = candidate.name
                candidate_frame['net_excess'] -= .001*j
                candidate_frame['gross_excess'] -= .001*j
                paths.append(CandidatePath(seed, candidate.name, candidate_frame,
                                            {'synthetic-path': digest((family, seed, candidate.name))}, True))
        receipts[family] = select_candidates(plan=selection_plan, paths=paths)
    exposure = dict(families={family: receipt['candidate_exposure'] for family, receipt in receipts.items()},
                    additional_candidate_exposure=[])
    kwargs = dict(path_manifest=manifest, selection_metadata=receipts,
                  candidate_exposure_metadata=exposure, artifact_sha256={'synthetic-input': digest('inputs')})
    return frame, plan, kwargs


class AnalysisTests(unittest.TestCase):
    def test_complete_metrics_independent_scalar_utility_and_seed_average_not_ensemble(self):
        frame, plan, kwargs = analysis_fixture()
        result = analyze_paths(frame, plan, **kwargs)
        self.assertEqual(len(result['strategy_summary']), 60)
        self.assertEqual(len(result['mean_seed_metrics']), 20)
        self.assertLess(result['result']['independent_scalar_utility_max_error'], 1e-12)
        group = frame.query("seed == 2 and method == 'selected_temporal' and return_eom >= '2002-01-01' and return_eom <= '2002-12-31'")
        numbers = group.net_excess.tolist()
        mean = math.fsum(numbers)/len(numbers)
        variance = math.fsum((value-mean)**2 for value in numbers)/len(numbers)
        row = result['strategy_summary'].query("period == 'evaluation' and seed == 2 and method == 'selected_temporal'").iloc[0]
        self.assertAlmostEqual(row.annual_net_utility, 12*mean-60*variance, places=14)
        averaged = result['mean_seed_metrics'].query("period == 'evaluation' and method == 'selected_temporal'").iloc[0]
        separate = result['strategy_summary'].query("period == 'evaluation' and method == 'selected_temporal'")
        self.assertAlmostEqual(averaged.annual_net_utility, separate.annual_net_utility.mean(), places=15)
        wide = frame.query("method == 'selected_temporal' and return_eom >= '2002-01-01' and return_eom <= '2002-12-31'").pivot(index='return_eom', columns='seed', values='net_excess')
        ensemble = wide.mean(axis=1).to_numpy()
        self.assertGreater(abs(averaged.annual_net_utility-12*(ensemble.mean()-5*ensemble.var())), 1e-5)
        self.assertTrue(result['strategy_summary'].query("period == 'selection' and method == 'selected_temporal'").selected_earlier_history_is_postselected.all())

    def test_paired_identical_paths_have_exact_zero_contrasts_and_intervals(self):
        frame, plan, kwargs = analysis_fixture(identical=True)
        result = analyze_paths(frame, plan, **kwargs)
        intervals = result['primary_intervals']
        for field in ('estimate', 'adjusted_lower', 'adjusted_upper', 'ordinary_lower', 'ordinary_upper'):
            np.testing.assert_array_equal(intervals[field], np.zeros(len(intervals)))
        self.assertFalse(result['result']['all_conditions_met'])
        self.assertFalse(result['result']['primary_family_improvement_established'])
        self.assertEqual(result['result']['separate_model_seeds'], [1, 2, 3])

    def test_four_comparison_correction_recomputed_from_paired_draws(self):
        frame, plan, kwargs = analysis_fixture()
        result = analyze_paths(frame, plan, **kwargs)
        saved = result['bootstrap']['evaluation_L3']
        columns = list(zip(saved['seeds'], saved['methods']))
        left = [columns.index((seed, 'selected_temporal')) for seed in (1, 2, 3)]
        right = [columns.index((seed, 'parent')) for seed in (1, 2, 3)]
        draws = (saved['utility_draws'][:, left]-saved['utility_draws'][:, right]).mean(axis=1)
        row = result['primary_intervals'].query("period == 'evaluation' and block_months == 3 and aggregation == 'mean_seed' and right == 'parent'").iloc[0]
        low, high = basic_interval(row.estimate, draws, alpha=.05, family=4)
        self.assertEqual((row.adjusted_lower, row.adjusted_upper), (low, high))
        self.assertLessEqual(row.adjusted_lower, row.ordinary_lower)
        self.assertGreaterEqual(row.adjusted_upper, row.ordinary_upper)
        self.assertEqual(row.family, 4)
        self.assertEqual(len(result['primary_intervals']), 64)

    def test_drawdown_initial_peak_annual_continuity_and_bankrupt_index_unavailable(self):
        result = compound_drawdown([-.1, 0., -.1, .1])
        self.assertAlmostEqual(result['maximum_drawdown'], .19, places=14)
        self.assertAlmostEqual(result['drawdowns'][0], .1, places=14)
        self.assertGreater(result['maximum_drawdown'], max(compound_drawdown([-.1, 0.])['maximum_drawdown'],
                                                         compound_drawdown([-.1, .1])['maximum_drawdown']))
        broken = compound_drawdown([.1, -1., .1])
        self.assertFalse(broken['available'])
        self.assertIsNone(broken['maximum_drawdown'])
        self.assertIn('-100%', broken['reason'])
        frame = pd.DataFrame(dict(gross_excess=[-.099, .001, -.099, .101],
                                  net_excess=[-.1, 0., -.1, .1], impact_cost=[.001]*4,
                                  turnover=[.1]*4, gross_exposure=[1.]*4))
        metrics = path_metrics(frame, gamma=10., continuous_drawdowns=result['drawdowns'],
                              drawdown_basis='compounded_net_excess_index_not_total_wealth')
        self.assertEqual(metrics['drawdown_basis'], 'compounded_net_excess_index_not_total_wealth')
        self.assertAlmostEqual(metrics['period_maximum_drawdown'], .19, places=14)
        self.assertIsNone(metrics['mean_gross_weight_concentration'])
        self.assertIn('mean_gross_weight_concentration', metrics['unavailable_position_metrics'])

    def test_primary_success_also_requires_robustness_mean_positive(self):
        frame, plan, kwargs = analysis_fixture()
        bad = frame.copy()
        mask = bad.method.eq('selected_temporal') & bad.return_eom.dt.year.eq(2003)
        bad.loc[mask, 'net_excess'] -= .1
        bad.loc[mask, 'gross_excess'] -= .1
        result = analyze_paths(bad, plan, **kwargs)
        self.assertFalse(result['result']['all_conditions_met'])
        self.assertFalse(result['result']['approved_success_conditions']['exposed_robustness_mean_seed_positive_for_all_four'])
        self.assertTrue(all(not row['exposed_robustness_mean_seed_delta_positive']
                            for row in result['result']['primary_contrast_checks']))

    def test_incomplete_duplicate_misaligned_wrong_gamma_or_family_rejected(self):
        frame, plan, kwargs = analysis_fixture()
        bad_inputs = [frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), frame[frame.seed != 3]]
        shifted = frame.copy(); shifted.loc[0, 'decision_eom'] += pd.offsets.MonthEnd(1)
        bad_inputs.append(shifted)
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                analyze_paths(bad, plan, **kwargs)
        for update in ({'selection': dict(plan['selection'], risk_aversion=5.)},
                       {'inference': dict(plan['inference'], family=6)}):
            with self.assertRaises(ValueError):
                analyze_paths(frame, dict(plan, **update), **kwargs)

    def test_selection_tamper_backfill_change_holdings_reset_or_missing_exposure_rejected(self):
        frame, plan, kwargs = analysis_fixture()
        tampered = deepcopy(kwargs)
        tampered['selection_metadata']['temporal']['selected_by_seed'][0]['parameters']['rho'] = .9
        with self.assertRaisesRegex(ValueError, 'receipt content changed'):
            analyze_paths(frame, plan, **tampered)
        changed = frame.copy()
        mask = changed.method.eq('selected_temporal') & changed.seed.eq(1) & changed.return_eom.dt.year.eq(2001)
        changed.loc[mask, 'net_excess'] += .001
        changed.loc[mask, 'gross_excess'] += .001
        with self.assertRaisesRegex(ValueError, 'original scored candidate'):
            analyze_paths(changed, plan, **kwargs)
        changed = frame.copy()
        changed.loc[changed.method.eq('selected_temporal') & changed.return_eom.eq(pd.Timestamp('2002-01-31')),
                    'pretrade_sha256'] = digest('reset')
        with self.assertRaisesRegex(ValueError, 'holdings reset'):
            analyze_paths(changed, plan, **kwargs)
        with self.assertRaisesRegex(ValueError, 'exposure'):
            analyze_paths(frame, plan, **dict(kwargs, candidate_exposure_metadata={}))

    def test_new_only_outputs_preserve_full_selection_exposure_and_no_fresh_holdout_claim(self):
        frame, plan, kwargs = analysis_fixture()
        with tempfile.TemporaryDirectory(prefix='temporal-analysis-check-') as temporary:
            folder = Path(temporary)/'analysis'
            result = analyze_paths(frame, plan, **kwargs, destination=folder)
            self.assertTrue((folder/'analysis_manifest.json').exists())
            saved = json.loads((folder/'analysis_result.json').read_text())
            self.assertEqual(len(saved['candidate_exposure_metadata']['families']['temporal']), 9)
            self.assertFalse(saved['fresh_holdout'])
            self.assertFalse(saved['originality_established'])
            self.assertEqual(result['bootstrap'], {})
            with self.assertRaises(FileExistsError):
                analyze_paths(frame, plan, **kwargs, destination=folder)


if __name__ == '__main__':
    unittest.main()
