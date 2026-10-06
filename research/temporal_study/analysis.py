"""Configured complete-path statistics and paired stationary-bootstrap inference.

This module opens no historical data and starts no experiment. Its caller must
provide complete verified inputs, selection provenance and candidate-exposure
metadata. Random seeds are model replicates: their statistics are averaged;
their returns are never combined into an ensemble portfolio.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from feature_geometry.history import sha
from inference.core import stationary, utility, bootstrap_utility, basic_interval


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        raise TypeError(type(item).__name__)
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day != day.normalize() or day != day + pd.offsets.MonthEnd(0):
        raise ValueError('Explicit calendar month-end dates required')
    return day


def _inventory(value, name):
    if (not isinstance(value, dict) or not value
            or any(not isinstance(key, str) or not key
                   or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                   for key, digest in value.items())):
        raise ValueError(f'{name} requires a nonempty path-to-SHA256 inventory')
    return dict(value)


def compound_drawdown(returns):
    """Drawdowns include the initial peak of one and never reset each year.

    An observation at or below -100% invalidates ordinary compound wealth.
    Such inputs return an explicit unavailable reason, never clipped returns.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or not len(returns) or not np.isfinite(returns).all():
        raise ValueError('Finite nonempty one-dimensional returns required')
    if np.any(returns <= -1.):
        return dict(available=False, reason='Return at or below -100%; compound index undefined',
                    drawdowns=None, wealth_index=None, maximum_drawdown=None)
    # Log space avoids needless underflow/overflow in the drawdown ratio.
    log_wealth = np.cumsum(np.log1p(returns))
    log_peak = np.maximum.accumulate(np.r_[0., log_wealth])[1:]
    drawdowns = -np.expm1(log_wealth-log_peak)
    wealth = np.exp(log_wealth) if np.all(np.abs(log_wealth) < 700.) else None
    return dict(available=True, reason=None, drawdowns=drawdowns, wealth_index=wealth,
                maximum_drawdown=float(drawdowns.max()))


def path_metrics(frame, *, gamma, continuous_drawdowns=None,
                 drawdown_basis, drawdown_unavailable_reason=None):
    """Statistics for one seed/method/reporting period; population variances."""
    if gamma != 10.:
        raise ValueError('The reused inference utility requires exactly risk aversion 10')
    gross = frame.gross_excess.to_numpy(dtype=float)
    net = frame.net_excess.to_numpy(dtype=float)
    cost = frame.impact_cost.to_numpy(dtype=float)
    if not len(net) or not np.isfinite(np.column_stack((gross, net, cost))).all():
        raise ValueError('Finite nonempty aligned gross/net/cost observations required')
    variance, gross_variance = float(np.var(net, ddof=0)), float(np.var(gross, ddof=0))
    vol, gross_vol = math.sqrt(12*variance), math.sqrt(12*gross_variance)
    drawdown_returns = frame['net_total_return'].to_numpy(dtype=float) if drawdown_basis == 'net_total_return' else net
    within = compound_drawdown(drawdown_returns)
    result = dict(months=len(net), annual_gross_excess_mean=float(12*gross.mean()),
        annual_net_excess_mean=float(12*net.mean()), annual_gross_volatility=gross_vol,
        annual_net_volatility=vol, gross_sharpe=float(12*gross.mean()/gross_vol) if gross_vol else None,
        net_sharpe=float(12*net.mean()/vol) if vol else None,
        annual_net_utility=float(utility(net)),
        annual_gross_utility=float(12*(gross.mean()-gamma/2*gross_variance)),
        annual_impact_cost=float(12*cost.mean()), annual_impact_bps=float(120000*cost.mean()),
        mean_monthly_absolute_turnover=float(frame.turnover.mean()),
        mean_gross_exposure=float(frame.gross_exposure.mean()),
        maximum_gross_exposure=float(frame.gross_exposure.max()),
        drawdown_basis=drawdown_basis,
        period_maximum_drawdown=within['maximum_drawdown'],
        period_drawdown_available=within['available'],
        period_drawdown_unavailable_reason=within['reason'],
        continuous_path_maximum_drawdown_in_period=(float(np.max(continuous_drawdowns))
                                                    if continuous_drawdowns is not None else None),
        continuous_drawdown_available=continuous_drawdowns is not None,
        continuous_drawdown_unavailable_reason=drawdown_unavailable_reason)
    optional = dict(mean_net_exposure=('net_exposure', 'mean'),
                    mean_largest_absolute_weight=('largest_absolute_weight', 'mean'),
                    maximum_absolute_weight=('largest_absolute_weight', 'max'),
                    mean_gross_weight_concentration=('gross_weight_concentration', 'mean'),
                    maximum_gross_weight_concentration=('gross_weight_concentration', 'max'))
    unavailable = []
    for output, (source, operation) in optional.items():
        if source not in frame or frame[source].isna().any():
            result[output] = None
            unavailable.append(output)
        else:
            values = frame[source].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite optional exposure/concentration metric')
            result[output] = float(np.mean(values) if operation == 'mean' else np.max(values))
    result['unavailable_position_metrics'] = unavailable
    # Independent scalar implementation catches accidental sample-variance or
    # ensemble-return substitutions without reusing the vectorized helper.
    numbers = [float(x) for x in net]
    mean = math.fsum(numbers)/len(numbers)
    scalar_variance = math.fsum((x-mean)**2 for x in numbers)/len(numbers)
    result['independent_utility_error'] = abs(result['annual_net_utility']-12*(mean-5*scalar_variance))
    if result['independent_utility_error'] > 1e-12:
        raise ValueError('Independent scalar net utility differs')
    return result


def validate_paths(monthly, plan, path_manifest):
    """Require every configured month and all three independently held seeds."""
    plan = _json(plan)
    if plan['selection']['risk_aversion'] != 10.:
        raise ValueError('The reused inference utility requires exactly risk aversion 10')
    seeds = plan['frequency_seeds']
    if seeds != [1, 2, 3]:
        raise ValueError('This study requires exactly separately held seeds 1, 2 and 3')
    pairs = plan['primary_contrasts']
    if (len(pairs) != 4 or len({tuple(pair) for pair in pairs}) != 4
            or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs)
            or plan['inference']['family'] != len(pairs)):
        raise ValueError('The declared four distinct comparisons require a four-comparison family')
    inf = plan['inference']
    if (not isinstance(inf['replicas'], int) or isinstance(inf['replicas'], bool) or inf['replicas'] < 1
            or not 0 < inf['alpha'] < 1 or inf['bootstrap_refits'] is not False
            or not inf['block_months'] or len(set(inf['block_months'])) != len(inf['block_months'])
            or any(not isinstance(length, int) or length < 1 for length in inf['block_months'])):
        raise ValueError('Explicit valid stationary bootstrap settings required')
    required = {'seed', 'method', 'decision_eom', 'return_eom', 'gross_excess', 'impact_cost',
                'net_excess', 'turnover', 'gross_exposure'}
    if not required.issubset(monthly.columns):
        raise ValueError('Missing required complete-path accounting columns: '+str(sorted(required-set(monthly.columns))))
    frame = monthly.copy(deep=True)
    for column in ('decision_eom', 'return_eom'):
        frame[column] = pd.to_datetime(frame[column], errors='raise')
        if frame[column].isna().any() or not frame[column].map(lambda day: _date(day) == day).all():
            raise ValueError('Invalid monthly path date')
    if (frame.duplicated(['seed', 'method', 'return_eom']).any()
            or frame[['seed', 'method']].isna().any().any()
            or set(frame.seed) != set(seeds)
            or not all(isinstance(value, str) and value for value in frame.method)):
        raise ValueError('Duplicate path keys or invalid seed/method identities')
    methods = sorted(frame.method.unique())
    if not {method for pair in pairs for method in pair}.issubset(methods):
        raise ValueError('Every declared comparison method must have a complete path')
    dates = pd.date_range(f"{plan['first_return_year']}-01-31",
                          f"{plan['last_return_year']}-12-31", freq='ME')
    for (seed, method), group in frame.groupby(['seed', 'method']):
        if not group.sort_values('return_eom').return_eom.reset_index(drop=True).equals(pd.Series(dates)):
            raise ValueError('A seed/method path has missing, extra or misaligned calendar months')
    if len(frame) != len(dates)*len(seeds)*len(methods):
        raise ValueError('Every method requires all three complete seed paths')
    if not (frame.decision_eom+pd.offsets.MonthEnd(1)).eq(frame.return_eom).all():
        raise ValueError('Decision and realized-return dates are not one calendar month apart')
    financial = frame[['gross_excess', 'impact_cost', 'net_excess', 'turnover', 'gross_exposure']].to_numpy(dtype=float)
    if (not np.isfinite(financial).all() or np.any(frame.impact_cost < 0)
            or np.any(frame.turnover < 0) or np.any(frame.gross_exposure < 0)
            or not np.allclose(frame.gross_excess-frame.impact_cost, frame.net_excess,
                               atol=1e-14, rtol=0.)):
        raise ValueError('Nonfinite, negative-cost or inconsistent actual-order accounting')
    if (path_manifest.get('status') != 'COMPLETE'
            or path_manifest.get('months') != len(dates)
            or path_manifest.get('paths') != len(seeds)*len(methods)
            or path_manifest.get('monthly_records') != len(frame)
            or path_manifest.get('first_trade_charged') is not True
            or path_manifest.get('annual_or_period_holdings_reset') is not False):
        raise ValueError('Complete continuous-path manifest with charged initial trade required')
    for name, (first, last) in plan['periods'].items():
        if first > last or first < plan['first_return_year'] or last > plan['last_return_year']:
            raise ValueError('Reporting period lies outside the complete path')
    if not {'evaluation', 'exposed_robustness'}.issubset(plan['periods']):
        raise ValueError('Primary and exposed-robustness reporting periods required')
    reported_years = [year for first, last in plan['periods'].values() for year in range(first, last+1)]
    if sorted(reported_years) != list(range(plan['first_return_year'], plan['last_return_year']+1)):
        raise ValueError('Reporting periods must partition the complete history without gaps or overlaps')
    return frame.sort_values(['seed', 'method', 'return_eom']).reset_index(drop=True), methods, dates


def _mean_seed_metrics(summary):
    """Average already computed statistics; never compute an ensemble's utility."""
    nonnumeric = {'period', 'seed', 'method', 'selected_earlier_history_is_postselected',
                  'drawdown_basis', 'period_drawdown_available', 'period_drawdown_unavailable_reason',
                  'continuous_drawdown_available', 'continuous_drawdown_unavailable_reason',
                  'unavailable_position_metrics'}
    numeric = [column for column in summary if column not in nonnumeric]
    records = []
    for (period, method), group in summary.groupby(['period', 'method'], sort=False):
        row = dict(period=period, method=method, aggregation='arithmetic mean of three seed statistics')
        for key in numeric:
            # Do not silently average two seeds when a statistic is unavailable
            # for the third (e.g. undefined Sharpe or invalid compound wealth).
            row[key] = float(group[key].mean()) if group[key].notna().all() else None
        bases = list(group.drawdown_basis.unique())
        row['drawdown_basis'] = bases[0] if len(bases) == 1 else 'mixed: '+', '.join(bases)
        row['position_metrics_complete_for_all_seeds'] = all(not missing for missing in group.unavailable_position_metrics)
        records.append(row)
    return pd.DataFrame(records)


def _selection_provenance(frame, plan, selection_metadata, candidate_exposure_metadata):
    from .selection import Candidate, CandidatePath, SelectionPlan, _checked_path, _digest
    families = {'temporal': 'selected_temporal', 'current_only': 'selected_current_only'}
    if not isinstance(selection_metadata, dict) or set(selection_metadata) != set(families):
        raise ValueError('Both full frozen per-seed selection receipts are required')
    exposure = _json(candidate_exposure_metadata)
    if (set(exposure) != {'families', 'additional_candidate_exposure'}
            or set(exposure['families']) != set(families)
            or not isinstance(exposure['additional_candidate_exposure'], list)):
        raise ValueError('Explicit complete candidate exposure and additional-exposure list required')
    records = []
    for family, method in families.items():
        receipt = _json(selection_metadata[family])
        expected_hash = receipt.pop('receipt_sha256', None)
        if expected_hash != _digest(receipt):
            raise ValueError('Frozen selection receipt content changed')
        if (receipt.get('status') != 'SELECTION_FROZEN'
                or receipt.get('selection_pools_seeds') is not False
                or receipt.get('development_and_selection_are_retrospective') is not True
                or receipt.get('selected_configuration_known_at_initial_decision') is not False
                or receipt.get('evaluation_is_previously_exposed') is not True):
            raise ValueError('Selection receipt lacks required per-seed/exposure declarations')
        definition = dict(receipt['plan'])
        definition['candidates'] = tuple(Candidate(**candidate) for candidate in definition['candidates'])
        selection_plan = SelectionPlan(**definition)
        expected_initial = f"{plan['first_return_year']-1}-12-31"
        evaluation_decision = f"{plan['periods']['evaluation'][0]-1}-12-31"
        if (receipt['plan_sha256'] != selection_plan.fingerprint
                or selection_plan.protocol_id != plan['id'] or selection_plan.family_id != family
                or list(selection_plan.seeds) != plan['frequency_seeds'] or selection_plan.gamma != 10.
                or selection_plan.initial_decision != expected_initial
                or selection_plan.validation_first_return != plan['selection']['first_return']
                or selection_plan.validation_last_return != plan['selection']['last_return']
                or selection_plan.selection_available_at != plan['selection']['last_return']
                or selection_plan.first_evaluation_decision != evaluation_decision
                or [dict(candidate.parameters) for candidate in selection_plan.candidates]
                   != [dict(rho=rho) for rho in plan['penalties_in_tie_order']]):
            raise ValueError('Frozen selection differs from approved dates, seeds or ordered penalty grid')
        if receipt['candidate_exposure'] != exposure['families'][family]:
            raise ValueError('Candidate exposure inventory differs from frozen selection')
        expected_keys = {(seed, candidate.name) for seed in selection_plan.seeds
                         for candidate in selection_plan.candidates}
        candidate_rows = receipt['candidate_exposure']
        keys = [(row['seed'], row['candidate']) for row in candidate_rows]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError('Every inspected candidate and seed must remain in exposure metadata')
        inventory = {(row['seed'], row['candidate']): row for row in candidate_rows}
        choices = receipt['selected_by_seed']
        if sorted(choice['seed'] for choice in choices) != plan['frequency_seeds']:
            raise ValueError('Exactly one frozen choice per seed required')
        for choice in choices:
            seed, candidate = choice['seed'], choice['candidate']
            selected = frame[(frame.seed == seed) & frame.method.eq(method)].sort_values('return_eom')
            bounded = selected[selected.return_eom <= pd.Timestamp(selection_plan.validation_last_return)].copy()
            # The report-facing method is an alias for this frozen candidate;
            # restore its original label before invoking selector verification.
            bounded['method'] = candidate
            original = inventory.get((seed, candidate))
            if original is None:
                raise ValueError('Selected configuration has no recorded candidate exposure')
            _inventory(original['artifact_sha256'], 'Selected candidate provenance')
            checked = _checked_path(CandidatePath(seed, candidate, bounded,
                                                  original['artifact_sha256'], True), selection_plan)
            if (checked['input_table_sha256'] != original['input_table_sha256']
                    or checked['terminal_pretrade_sha256'] != choice['terminal_pretrade_sha256']
                    or checked['annual_net_utility'] != choice['annual_net_utility']):
                raise ValueError('Selected historical path is not the original scored candidate path')
            if (selected.holdings_reset.any()
                    or tuple(selected.next_pretrade_sha256.iloc[:-1]) != tuple(selected.pretrade_sha256.iloc[1:])):
                raise ValueError('Selected path holdings reset or changed when selection became available')
            evaluation = selected[selected.decision_eom == pd.Timestamp(evaluation_decision)]
            if len(evaluation) != 1 or evaluation.pretrade_sha256.iloc[0] != choice['terminal_pretrade_sha256']:
                raise ValueError('Evaluation did not inherit the chosen candidate own holdings')
            # Verify every later annual parent too; no later model may enter an
            # earlier decision just because the penalty was selected safely.
            for row in selected.itertuples(index=False):
                year = row.return_eom.year
                cutoff, last_training = pd.Timestamp(f'{year-1}-12-31'), pd.Timestamp(f'{year-1}-11-30')
                if (_date(row.parent_cutoff) != cutoff or _date(row.training_last_decision) != last_training
                        or not cutoff <= row.decision_eom
                        or not last_training+pd.offsets.MonthEnd(1)
                               <= _date(row.training_outcomes_available_at) <= cutoff):
                    raise ValueError('Selected path annual training/parent chronology differs')
            records.append(dict(family=family, seed=seed, method=method, candidate=candidate,
                                parameters=choice['parameters'], selection_receipt_sha256=expected_hash,
                                selection_input_table_sha256=checked['input_table_sha256'],
                                terminal_selection_pretrade_sha256=choice['terminal_pretrade_sha256']))
    return records


def analyze_paths(monthly, plan, *, path_manifest, selection_metadata,
                  candidate_exposure_metadata, artifact_sha256, destination=None, progress=None):
    """Compute configured metrics and four-comparison paired inference.

    ``selection_metadata`` maps temporal/current_only to full selector receipts.
    Exposure metadata repeats those complete family inventories and explicitly
    lists any additional inspected candidate paths. The supplied SHA inventory
    must have been verified by the caller; this pure data interface cannot
    attest that files it never opens still have those bytes.
    """
    plan = _json(plan)
    inventory = _inventory(artifact_sha256, 'Analysis inputs')
    frame, methods, complete_dates = validate_paths(monthly, plan, path_manifest)
    selection_links = _selection_provenance(frame, plan, selection_metadata, candidate_exposure_metadata)
    folder = None if destination is None else Path(destination)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=False)
    rows, comparisons, decompositions, intervals, sampling = [], [], [], [], []
    bootstrap_arrays = {}
    seeds = plan['frequency_seeds']
    specs = [(seed, method) for seed in seeds for method in methods]
    inf = plan['inference']
    continuous = {}
    drawdown_basis = {}
    for key, group in frame.groupby(['seed', 'method'], sort=False):
        has_total = 'net_total_return' in group and group.net_total_return.notna().all()
        if has_total and not np.isfinite(group.net_total_return.to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite supplied total returns')
        basis = 'net_total_return' if has_total else 'compounded_net_excess_index_not_total_wealth'
        drawdown_basis[key] = basis
        values = group.net_total_return.to_numpy(dtype=float) if has_total else group.net_excess.to_numpy(dtype=float)
        continuous[key] = (group.return_eom.to_numpy(), values)
    try:
        for period, (first, last) in plan['periods'].items():
            sub = frame[frame.return_eom.dt.year.between(first, last)]
            for (seed, method), group in sub.groupby(['seed', 'method'], sort=False):
                dates, values = continuous[seed, method]
                active = (dates >= np.datetime64(f'{first}-01-01')) & (dates <= np.datetime64(f'{last}-12-31'))
                through_period = dates <= np.datetime64(f'{last}-12-31')
                curve = compound_drawdown(values[through_period])
                result = path_metrics(group, gamma=10., drawdown_basis=drawdown_basis[seed, method],
                    continuous_drawdowns=curve['drawdowns'][active[through_period]] if curve['available'] else None,
                    drawdown_unavailable_reason=curve['reason'])
                rows.append(dict(period=period, seed=int(seed), method=method,
                    selected_earlier_history_is_postselected=(method.startswith('selected_')
                                                              and period in ('development', 'selection')),
                    **result))
            wide = sub.pivot(index='return_eom', columns=['seed', 'method'], values='net_excess')
            wide = wide.reindex(columns=pd.MultiIndex.from_tuples(specs)).sort_index()
            matrix = wide.to_numpy(dtype=float)
            point = utility(matrix)
            for left, right in plan['primary_contrasts']:
                deltas = []
                for seed in seeds:
                    a = sub[(sub.seed == seed) & sub.method.eq(left)].sort_values('return_eom')
                    b = sub[(sub.seed == seed) & sub.method.eq(right)].sort_values('return_eom')
                    mean_part = float(12*(a.gross_excess.mean()-b.gross_excess.mean()))
                    impact_part = float(-12*(a.impact_cost.mean()-b.impact_cost.mean()))
                    variance_part = float(-60*(a.net_excess.var(ddof=0)-b.net_excess.var(ddof=0)))
                    delta = float(point[specs.index((seed, left))]-point[specs.index((seed, right))])
                    if abs(delta-mean_part-impact_part-variance_part) > 1e-12:
                        raise ValueError('Utility contrast accounting decomposition differs')
                    deltas.append(delta)
                    decompositions.append(dict(period=period, seed=seed, left=left, right=right,
                        delta_annual_net_utility=delta, gross_mean_contribution=mean_part,
                        impact_cost_contribution=impact_part, net_variance_contribution=variance_part))
                comparisons.append(dict(period=period, left=left, right=right,
                    mean_seed_delta=float(np.mean(deltas)), all_seed_points_positive=bool(min(deltas) > 0),
                    **{f'seed_{seed}_delta': delta for seed, delta in zip(seeds, deltas)}))
            if period not in ('evaluation', 'exposed_robustness'):
                continue
            for length in inf['block_months']:
                indices, starts, restart, digests = stationary(plan['id'], inf['master_seed'], length,
                    f'{first}-01-31', len(wide), replicas=inf['replicas'])
                draws = bootstrap_utility(matrix, indices)
                tag = f'{period}_L{length}'
                arrays = dict(indices=indices, starts=starts, restart=restart,
                              utility_draws=draws, seeds=np.array([seed for seed, _ in specs]),
                              methods=np.array([method for _, method in specs]), seed_sha256=np.array(digests))
                if folder is None:
                    bootstrap_arrays[tag] = arrays
                else:
                    with (folder/f'{tag}_bootstrap.npz').open('xb') as handle:
                        np.savez_compressed(handle, **arrays)
                for left, right in plan['primary_contrasts']:
                    ia = [specs.index((seed, left)) for seed in seeds]
                    ib = [specs.index((seed, right)) for seed in seeds]
                    per_seed = point[ia]-point[ib]
                    per_seed_draws = draws[:, ia]-draws[:, ib]
                    estimates = [('mean_seed', None, float(np.mean(per_seed)), np.mean(per_seed_draws, axis=1))]
                    estimates.extend(('individual_seed', seed, float(per_seed[i]), per_seed_draws[:, i])
                                     for i, seed in enumerate(seeds))
                    for aggregation, seed, estimate, samples in estimates:
                        lo, hi = basic_interval(estimate, samples, alpha=inf['alpha'], family=inf['family'])
                        ordinary_lo, ordinary_hi = basic_interval(estimate, samples, alpha=inf['alpha'], family=1)
                        intervals.append(dict(period=period, block_months=length, left=left, right=right,
                            aggregation=aggregation, seed=seed, estimate=estimate,
                            adjusted_lower=lo, adjusted_upper=hi, ordinary_lower=ordinary_lo,
                            ordinary_upper=ordinary_hi, family=inf['family'], replicas=inf['replicas'],
                            confidence=1-inf['alpha'],
                            correction_scope='four contrasts within this period, block length and reported seed aggregation',
                            interpretation=('primary seed-average contrast' if aggregation == 'mean_seed'
                                            else 'individual-seed diagnostic; no simultaneous claim across all seeds')))
                sampling.append(dict(period=period, block_months=length, months=len(wide),
                    replicas=inf['replicas'], paired_across_all_methods_and_seeds=True,
                    restart_fraction_after_first=float(restart[:, 1:].mean()),
                    bootstrap_refits=False, model_seeds_are_not_resampled=True))
                if progress is not None:
                    progress(dict(stage='inference', period=period, block_months=length,
                                  replicas=inf['replicas']))
        summary = pd.DataFrame(rows)
        tables = dict(strategy_summary=summary, mean_seed_metrics=_mean_seed_metrics(summary),
                      seed_contrasts=pd.DataFrame(comparisons), utility_decomposition=pd.DataFrame(decompositions),
                      primary_intervals=pd.DataFrame(intervals), sampling_diagnostics=pd.DataFrame(sampling),
                      selection_links=pd.DataFrame(selection_links))
        primary = tables['primary_intervals'].query("period == 'evaluation' and aggregation == 'mean_seed'")
        decision_checks = []
        for left, right in plan['primary_contrasts']:
            checked = primary[primary.left.eq(left) & primary.right.eq(right)]
            observed = tables['seed_contrasts'].query("period == 'evaluation'")
            observed = observed[observed.left.eq(left) & observed.right.eq(right)].iloc[0]
            robust = tables['seed_contrasts'].query("period == 'exposed_robustness'")
            robust = robust[robust.left.eq(left) & robust.right.eq(right)].iloc[0]
            decision_checks.append(dict(left=left, right=right,
                mean_seed_delta=float(observed.mean_seed_delta),
                all_seed_point_estimates_positive=bool(observed.all_seed_points_positive),
                adjusted_lower_positive_at_every_declared_block_length=bool((checked.adjusted_lower > 0).all()),
                exposed_robustness_mean_seed_delta=float(robust.mean_seed_delta),
                exposed_robustness_mean_seed_delta_positive=bool(robust.mean_seed_delta > 0)))
        conditions = dict(
            primary_mean_seed_positive_for_all_four=all(row['mean_seed_delta'] > 0 for row in decision_checks),
            primary_all_three_seed_points_positive_for_all_four=all(row['all_seed_point_estimates_positive'] for row in decision_checks),
            primary_family_adjusted_lower_positive_at_all_block_lengths_for_all_four=all(
                row['adjusted_lower_positive_at_every_declared_block_length'] for row in decision_checks),
            exposed_robustness_mean_seed_positive_for_all_four=all(
                row['exposed_robustness_mean_seed_delta_positive'] for row in decision_checks))
        result = dict(status='COMPLETE_ANALYSIS', protocol_id=plan['id'],
            approved_study_plan=plan,
            complete_path_months=len(complete_dates), separate_model_seeds=seeds,
            metrics_aggregation='arithmetic mean of separately computed seed statistics; not an ensemble',
            primary_comparison_family=plan['primary_contrasts'],
            inference_scope='conditional on fitted and selected models; paired market-month resampling without refitting',
            correction_scope='four contrasts per period and block length; no adjustment across periods or block lengths',
            evaluation_previously_exposed=True, fresh_holdout=False, originality_established=False,
            source_publication_timestamps_certified=False,
            earlier_selected_histories='development and selection are postselected descriptive histories of the chosen continuous candidate',
            drawdown_interpretation='period curve starts at one; continuous curve keeps peaks from the common initial path; no annual reset',
            net_excess_drawdown_warning='compounded net excess return is a diagnostic index, not investor total-return wealth',
            independent_scalar_utility_max_error=float(summary.independent_utility_error.max()),
            primary_contrast_checks=decision_checks,
            approved_success_conditions=conditions,
            all_conditions_met=all(conditions.values()),
            primary_family_improvement_established=all(conditions.values()),
            formal_originality_or_fresh_holdout_claim=False,
            candidate_exposure_metadata=_json(candidate_exposure_metadata),
            selection_metadata=_json(selection_metadata), supplied_inputs_sha256=inventory,
            filesystem_input_hashes_verified_by='caller; this function operates only on supplied tables',
            unavailable_position_metrics=[dict(period=row.period, seed=int(row.seed), method=row.method,
                                                metrics=row.unavailable_position_metrics)
                                          for row in summary.itertuples() if row.unavailable_position_metrics])
        if folder is not None:
            for name, table in tables.items():
                table.to_csv(folder/f'{name}.csv', index=False, float_format='%.17g')
            with (folder/'analysis_result.json').open('x') as handle:
                json.dump(_json(result), handle, indent=2, allow_nan=False)
                handle.write('\n')
            manifest = dict(status='COMPLETE', artifact_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
            with (folder/'analysis_manifest.json').open('x') as handle:
                json.dump(manifest, handle, indent=2)
                handle.write('\n')
        return dict(**tables, result=result, bootstrap=bootstrap_arrays)
    except Exception as error:
        if folder is not None:
            with (folder/'failed_analysis.json').open('x') as handle:
                json.dump(dict(status='FAILED_NOT_ACCEPTED', error_type=type(error).__name__, error=str(error)), handle, indent=2)
                handle.write('\n')
        raise
