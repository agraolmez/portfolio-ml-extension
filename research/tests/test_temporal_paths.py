"""Synthetic annual continuity, eligibility and original-accounting checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline.core import transition
from feature_geometry.features import FeatureMap, fingerprint
from feature_geometry.moments import MarketMoment
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.history import VerifiedParent, BoundInnovation
from temporal_extension.model import TemporalModel
from temporal_extension.reference import Innovation
from temporal_study.paths import AnnualPolicy, InitialState, RealizedMonth, run_paths


def parent_fixture(year, seed=1):
    frequency = FeatureMap(np.array([[.8, -.3], [.2, 1.1]])*seed, 'synthetic-fixed-frequency')
    theta = np.array([.1, -.1])
    fmap = TemporalFeatureMap(frequency, theta, np.arange(2))
    cutoff = f'{year-1}-12-31'
    beta = np.array([.04, .01, -.02, .03, .02]) * (year-1998)
    model = GeometryModel(frequency, theta, beta, cutoff, np.log(2.), 'parent-inputs')
    settings = FitSettings(cutoff)
    metadata = dict(training_first='1965-11-30', training_last=f'{year-1}-11-30',
                    status='ITERATION_BUDGET_REACHED')
    return VerifiedParent(year, seed, model, fmap, settings, .7, metadata,
                          {'synthetic_model.npz': fingerprint(year, seed, beta)})


class SyntheticSource:
    source_fingerprint = 'synthetic-source'
    reference_fingerprint = 'synthetic-reference'
    consumed_inputs = {'synthetic': 'not-historical-data'}

    def __init__(self):
        self.dates = ('2000-11-30', '2000-12-31', '2001-01-31')
        self.events = []
        self.markets = {}
        self.archives = {}
        for i, day in enumerate(self.dates):
            ids = np.array([1, 2, 3]) if i == 0 else np.array([2, 3, 4])
            z = np.array([[.2, .8], [.7, .4], [.9, .1]]) + i*.02
            self.markets[day] = MarketMoment(day, ids, tuple(z.copy() for _ in range(13)),
                tuple(np.ones(3) for _ in range(13)), tuple(np.ones(3) for _ in range(13)),
                np.array([[.6, .01, 0.], [0., .5, .02], [.01, 0., .4]]),
                np.diag([.04, .03, .02]), np.array([.1, .2, .3]), 2.)

    def market(self, day):
        self.events.append(('market', day))
        return self.markets[day]

    def innovation(self, market):
        z = market.characteristics[0]
        return BoundInnovation(Innovation(z-.3, np.ones_like(z, dtype=bool), {}),
                               market.decision_date, market.ids, fingerprint(z),
                               self.source_fingerprint, self.reference_fingerprint)

    def realized(self, market):
        self.events.append(('realized', market.decision_date))
        next_ids = np.array([2, 3, 4])
        return RealizedMonth(market.decision_date, market.ids,
                             np.array([.03, -.02, .04]), np.array([.02, -.03, .03]),
                             .01, next_ids)

    def unchanged(self):
        self.events.append(('unchanged',))

    def archived_positions(self, day, seed):
        return self.archives[day, seed]


def fitted_policy(parent, kind='learned_temporal', gains=(1.2, .7)):
    gains = np.array(gains)
    model = TemporalModel(parent.feature_map, gains, parent.model.beta*.9,
                          parent.model.cutoff, parent.model.cutoff, parent.fingerprint,
                          SyntheticSource.source_fingerprint, SyntheticSource.reference_fingerprint,
                          np.zeros(2), np.full(2, 2.))
    receipt = dict(model_kind=kind, year=parent.year, seed=parent.seed,
                   parent_fingerprint=parent.fingerprint,
                   parent_feature_fingerprint=parent.feature_map.fingerprint,
                   source_fingerprint=model.source_fingerprint,
                   reference_fingerprint=model.reference_fingerprint,
                   training_first=parent.metadata['training_first'],
                   training_last=parent.metadata['training_last'], gains=gains, beta=model.beta,
                   eligible_for_portfolio_evaluation=True, neutral_gate={'passed': True},
                   solve_residual=1e-15, status='converged', converged=True,
                   projected_gradient_norm=1e-10,
                   optimizer_settings=dict(lower=0., upper=2., tolerance=1e-8))
    if kind.startswith('fixed_'):
        receipt.update(status='fixed_configuration_solved', prescribed_gains=gains,
                       lower=model.lower, upper=model.upper)
        receipt.pop('converged')
        receipt.pop('projected_gradient_norm')
        receipt.pop('optimizer_settings')
    return model, receipt


class PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def problem(self):
        source = SyntheticSource()
        initial = InitialState(source.dates[0], [1, 2, 3], [.02, -.01, .015], 'synthetic-common-R1')
        policies = {}
        for year in (2000, 2001):
            for seed in (1, 2):
                parent = parent_fixture(year, seed)
                policies[year, seed, 'parent'] = AnnualPolicy(parent, kind='archived_parent')
                policies[year, seed, 'neutral'] = AnnualPolicy(parent, kind='neutral_replay')
                model, receipt = fitted_policy(parent)
                policies[year, seed, 'learned'] = AnnualPolicy(parent, kind='learned_temporal',
                                                               model=model, receipt=receipt)
        options = dict(destination=None, save_positions=True, compare_archived=False,
                       accounting_atol=1e-12, accounting_rtol=1e-11,
                       neutral_atol=1e-12, neutral_rtol=1e-11)
        return source, initial, policies, options

    def test_neutral_replay_and_continuous_own_holdings_across_annual_and_universe_change(self):
        source, initial, policies, options = self.problem()
        result = run_paths(source, source.dates, policies, initial, **options)
        rows, positions = result['monthly'], result['positions']
        self.assertEqual(len(rows), 18)
        self.assertEqual(result['manifest']['neutral_maximum_position_error'], 0.)
        specs = list(zip(positions[0]['seeds'], positions[0]['methods']))
        for i in range(2):
            np.testing.assert_array_equal(positions[i]['next_pretrade'], positions[i+1]['pretrade'])
            for seed, method in specs:
                path = rows[(rows.seed == seed) & (rows.method == method)]
                self.assertEqual(path.iloc[i].next_pretrade_sha256, path.iloc[i+1].pretrade_sha256)
        learned, neutral, parent = [specs.index((1, method)) for method in ('learned', 'neutral', 'parent')]
        self.assertGreater(np.max(abs(positions[1]['pretrade'][:, learned]-positions[1]['pretrade'][:, parent])), 0.)
        np.testing.assert_array_equal(positions[1]['pretrade'][:, neutral], positions[1]['pretrade'][:, parent])
        self.assertEqual(positions[1]['pretrade'][2, learned], 0.)  # entrant receives no inherited position
        self.assertTrue((rows.impact_cost > 0).all())  # includes first trade
        self.assertFalse(rows.holdings_reset.any())
        self.assertTrue((rows.parent_cutoff <= rows.decision_eom).all())
        self.assertEqual(len(result['exit_ledger']), 6)
        self.assertFalse(result['exit_ledger'].liquidation_cost_added.any())

    def test_accounting_matches_independent_original_transition_and_own_returns(self):
        source, initial, policies, options = self.problem()
        result = run_paths(source, source.dates, policies, initial, **options)
        first = result['positions'][0]
        for j, (seed, method) in enumerate(zip(first['seeds'], first['methods'])):
            market = source.markets[source.dates[0]]
            outcomes = source.realized(market)
            expected = transition(market.ids, initial.holdings, first['target'][:, j], market.m,
                                  outcomes.total_returns, outcomes.market_growth, market.impact,
                                  market.wealth, outcomes.excess_returns, outcomes.next_ids)
            row = result['monthly'].query('seed == @seed and method == @method').iloc[0]
            self.assertAlmostEqual(row.gross_excess, expected['gross_excess'], places=15)
            self.assertAlmostEqual(row.impact_cost, expected['impact_cost'], places=15)
            self.assertAlmostEqual(row.net_excess, expected['gross_excess']-expected['impact_cost'], places=15)
            np.testing.assert_array_equal(first['weight'][:, j], expected['weights'])

    def test_nonconverged_ineligible_changed_parent_and_wrong_year_receipts_fail(self):
        parent = parent_fixture(2000)
        model, receipt = fitted_policy(parent)
        bad = [dict(status='step_limit'), dict(converged=False),
               dict(eligible_for_portfolio_evaluation=False), dict(projected_gradient_norm=.01),
               dict(seed=2), dict(year=2001), dict(neutral_gate={'passed': False}),
               dict(training_first='1980-01-31'), dict(solve_residual=1e-3),
               dict(model_kind='learned_current_only')]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                AnnualPolicy(parent, kind='learned_temporal', model=model, receipt=dict(receipt, **change))
        changed = replace(model, cutoff='2000-01-31')
        with self.assertRaisesRegex(ValueError, 'frozen annual parent'):
            AnnualPolicy(parent, kind='learned_temporal', model=changed, receipt=receipt)

    def test_fixed_control_exact_solve_needs_no_optimizer_and_must_match_prescription(self):
        parent = parent_fixture(2000)
        model, receipt = fitted_policy(parent, kind='fixed_temporal')
        policy = AnnualPolicy(parent, kind='fixed_temporal', model=model, receipt=receipt,
                              prescribed_gains=[1.2, .7])
        source = SyntheticSource()
        actual = policy.aim(source.markets[source.dates[0]], source.innovation(source.markets[source.dates[0]]))
        self.assertEqual(actual.shape, (3,))
        with self.assertRaisesRegex(ValueError, 'prescribed'):
            AnnualPolicy(parent, kind='fixed_temporal', model=model, receipt=receipt,
                         prescribed_gains=[.5, .5])

    def test_future_expired_seed_missing_year_and_nonconsecutive_paths_fail_closed(self):
        source, initial, policies, options = self.problem()
        early = replace(source.markets[source.dates[0]], decision_date='1999-11-30')
        expired = replace(source.markets[source.dates[0]], decision_date='2000-12-31')
        for market in (early, expired):
            with self.assertRaisesRegex(ValueError, 'annual model'):
                policies[2000, 1, 'parent'].aim(market, None)
        wrong = dict(policies); wrong[2000, 2, 'parent'] = policies[2000, 1, 'parent']
        with self.assertRaisesRegex(ValueError, 'parent/seed'):
            run_paths(source, source.dates, wrong, initial, **options)
        missing = dict(policies); missing.pop((2001, 1, 'parent'))
        with self.assertRaisesRegex(ValueError, 'every evaluation year'):
            run_paths(source, source.dates, missing, initial, **options)
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            run_paths(source, source.dates[::2], policies, initial, **options)

    def test_mutated_receipt_or_model_and_wrong_reference_cannot_enter_paths(self):
        source, initial, policies, options = self.problem()
        policy = policies[2000, 1, 'learned']
        policy.receipt['neutral_gate']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_paths(source, source.dates, policies, initial, **options)
        source, initial, policies, options = self.problem()
        source.reference_fingerprint = 'changed-reference'
        with self.assertRaisesRegex(ValueError, 'reference'):
            run_paths(source, source.dates, policies, initial, **options)

    def test_current_only_explicit_model_kind_runs_without_history_and_rejects_temporal_model(self):
        from temporal_study.controls import (CurrentOnlyFeatureMap, CurrentOnlyModel,
                                              current_only_reference_fingerprint)
        source, initial, _, options = self.problem()
        def forbidden_history(market):
            raise AssertionError('Current-only paths must not access historical innovations')
        source.innovation = forbidden_history
        policies = {}
        for year in (2000, 2001):
            parent = parent_fixture(year)
            temporal_model, receipt = fitted_policy(parent)
            fmap = CurrentOnlyFeatureMap(parent.feature_map.parent, parent.feature_map.theta,
                                         parent.feature_map.groups)
            model = CurrentOnlyModel(fmap, temporal_model.gains, temporal_model.beta,
                parent.model.cutoff, parent.model.cutoff, parent.fingerprint,
                source.source_fingerprint, current_only_reference_fingerprint(source.source_fingerprint),
                temporal_model.lower, temporal_model.upper)
            receipt.update(model_kind='learned_current_only',
                           reference_fingerprint=model.reference_fingerprint,
                           control_feature_fingerprint=fmap.fingerprint)
            policies[year, 1, 'current_only'] = AnnualPolicy(parent, kind='learned_current_only',
                                                             model=model, receipt=receipt)
            with self.assertRaisesRegex(ValueError, 'model implementation'):
                AnnualPolicy(parent, kind='learned_current_only', model=temporal_model, receipt=receipt)
        result = run_paths(source, source.dates, policies, initial, **options)
        self.assertEqual(len(result['monthly']), 3)
        self.assertFalse(result['manifest']['archived_replay_checked'])
        self.assertEqual(result['manifest']['neutral_monthly_pairs_checked'], 0)
        self.assertEqual(result['manifest']['initial_pretrade_sha256'],
                         result['monthly'].iloc[0].pretrade_sha256)

    def test_archive_replay_checks_all_positions_and_new_only_output(self):
        source, initial, policies, options = self.problem()
        baseline = run_paths(source, source.dates, policies, initial, **options)
        for day, positions in zip(source.dates, baseline['positions']):
            for seed in (1, 2):
                column = list(zip(positions['seeds'], positions['methods'])).index((seed, 'parent'))
                source.archives[day, seed] = {
                    key: positions[key].copy() if key in ('ids', 'next_ids') else positions[key][:, column].copy()
                    for key in ('ids', 'next_ids', 'pretrade', 'target', 'weight', 'trade',
                                'drifted_weight', 'next_pretrade')}
        with tempfile.TemporaryDirectory(prefix='temporal-path-check-') as temporary:
            folder = Path(temporary)/'new'
            configured = dict(options, destination=folder, compare_archived=True)
            result = run_paths(source, source.dates, policies, initial, **configured)
            self.assertEqual(result['manifest']['archived_replay_maximum_position_error'], 0.)
            self.assertTrue((folder/'paths_manifest.json').exists())
            self.assertEqual(len(pd.read_parquet(folder/'monthly.parquet')), 18)
            with self.assertRaises(FileExistsError):
                run_paths(source, source.dates, policies, initial, **configured)
            source.archives[source.dates[1], 1]['trade'][0] += .01
            failed = Path(temporary)/'failed'
            with self.assertRaises(AssertionError):
                run_paths(source, source.dates, policies, initial,
                          **dict(configured, destination=failed))
            self.assertTrue((failed/'failed_run.json').exists())
            self.assertFalse((failed/'paths_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
