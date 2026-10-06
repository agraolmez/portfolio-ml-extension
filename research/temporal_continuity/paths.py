"""Preserved portfolio rules with canonical ID hashes and checked risk diagnostics.

run_paths is versioned from temporal_study.paths.run_paths. Only the three
holdings fingerprints and the equivalent volatility diagnostic differ.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline.core import transition
from feature_geometry.history import sha
from temporal_extension.reference import month_key
from temporal_study.paths import AnnualPolicy, InitialState, _day, _json
from .identity import holdings_fingerprint, annual_volatility

def run_paths(source, dates, policies, initial, *, destination, save_positions,
              accounting_atol, accounting_rtol, neutral_atol, neutral_rtol,
              compare_archived, progress=None):
    """Evaluate explicit ``(year, seed, method) -> AnnualPolicy`` entries.

    Dates must form one consecutive sequence. All seed/method combinations
    must exist for every required return year. ``destination`` is new-only;
    leaving it None supports small synthetic checks without files. The caller
    must authorize historical evaluation and choose its protocol separately.
    """
    dates = tuple(_day(day) for day in dates)
    if (not dates or not isinstance(initial, InitialState)
            or dates[0] != initial.decision_date
            or any(month_key(b) != month_key(a) + 1 for a, b in zip(dates, dates[1:]))):
        raise ValueError('A consecutive monthly sequence from the explicit initial state is required')
    tolerances = (accounting_atol, accounting_rtol, neutral_atol, neutral_rtol)
    if not np.isfinite(tolerances).all() or min(tolerances) < 0:
        raise ValueError('Explicit finite nonnegative comparison tolerances required')
    if not isinstance(save_positions, bool) or not isinstance(compare_archived, bool):
        raise ValueError('Explicit position-storage and archive-comparison decisions required')
    years = sorted({(pd.Timestamp(day) + pd.offsets.MonthEnd(1)).year for day in dates})
    if any(not isinstance(key, tuple) or len(key) != 3 for key in policies):
        raise ValueError('Policies require explicit (year, seed, method) keys')
    specs = sorted({(key[1], key[2]) for key in policies})
    required = {(year, seed, method) for year in years for seed, method in specs}
    if not specs or set(policies) != required:
        raise ValueError('Every seed/method requires exactly one policy for every evaluation year')
    for key, policy in policies.items():
        year, seed, method = key
        if (not isinstance(policy, AnnualPolicy) or policy.parent.year != year
                or policy.parent.seed != seed or not isinstance(method, str) or not method):
            raise ValueError('Policy key differs from verified annual parent/seed')
        policy.validate()
        if policy.model is not None:
            if policy.model.source_fingerprint != source.source_fingerprint:
                raise ValueError('Fitted model and path source differ')
            if (not policy.kind.endswith('current_only')
                    and policy.model.reference_fingerprint != source.reference_fingerprint):
                raise ValueError('Fitted temporal reference and path reference differ')
    for seed, method in specs:
        if len({policies[year, seed, method].kind for year in years}) != 1:
            raise ValueError('A portfolio method cannot change its model kind between years')
    folder = None if destination is None else Path(destination)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=False)
    holding_ids = initial.ids.copy()
    h = np.repeat(initial.holdings[:, None], len(specs), axis=1)
    monthly, exits, position_records = [], [], []
    max_neutral, max_archive = 0., 0.
    neutral_pairs_checked, archived_paths_checked = 0, 0
    policy_identities = {str(key): policy.fingerprint for key, policy in policies.items()}
    try:
        for day in dates:
            year = (pd.Timestamp(day) + pd.offsets.MonthEnd(1)).year
            market = source.market(day)
            if market.decision_date != day or not np.array_equal(holding_ids, market.ids):
                raise ValueError('Inherited own-holdings universe does not match the next market')
            active = [policies[year, seed, method] for seed, method in specs]
            needs_history = any(p.kind not in ('archived_parent', 'learned_current_only',
                                               'fixed_current_only') for p in active)
            bound = source.innovation(market) if needs_history else None
            if bound is not None:
                bound.validate(day, market.ids, market.characteristics[0],
                               source.source_fingerprint, source.reference_fingerprint)
            # Portfolio aims are computed before obtaining any realized outcome.
            targets = np.column_stack([policy.aim(market, bound) for policy in active])
            realized = source.realized(market)
            if realized.decision_date != day or not np.array_equal(realized.ids, market.ids):
                raise ValueError('Realized outcomes differ from the decision universe/date')
            arrays = dict(ids=market.ids.copy(), next_ids=realized.next_ids.copy(), pretrade=h.copy(),
                          target=targets, weight=np.empty_like(h), trade=np.empty_like(h),
                          drifted_weight=np.empty_like(h),
                          next_pretrade=np.empty((len(realized.next_ids), len(specs))))
            for j, ((seed, method), policy) in enumerate(zip(specs, active)):
                result = transition(market.ids, h[:, j], targets[:, j], market.m,
                                    realized.total_returns, realized.market_growth,
                                    market.impact, market.wealth, realized.excess_returns,
                                    realized.next_ids)
                for key, result_key in [('weight', 'weights'), ('trade', 'trades'),
                                        ('drifted_weight', 'drifted_weights'),
                                        ('next_pretrade', 'next_pretrade')]:
                    arrays[key][:, j] = result[result_key]
                gross = float(sum(float(w) * float(r) for w, r in
                                  zip(result['weights'], realized.excess_returns)))
                cost = float(.5 * market.wealth * sum(float(l) * float(q)**2
                             for l, q in zip(market.impact, result['trades'])))
                np.testing.assert_allclose([gross, cost],
                    [result['gross_excess'], result['impact_cost']],
                    atol=accounting_atol, rtol=accounting_rtol,
                    err_msg='Independent actual-position return/cost accounting differs')
                weights = result['weights']; absolute = np.abs(weights)
                exposure = float(absolute.sum())
                monthly.append(dict(seed=seed, method=method, model_kind=policy.kind,
                    decision_eom=pd.Timestamp(day), return_eom=pd.Timestamp(day)+pd.offsets.MonthEnd(1),
                    parent_cutoff=pd.Timestamp(policy.parent.model.cutoff),
                    training_last_decision=pd.Timestamp(policy.parent.metadata['training_last']),
                    training_outcomes_available_at=pd.Timestamp(policy.parent.model.cutoff),
                    pretrade_sha256=holdings_fingerprint(market.ids, h[:, j]),
                    next_pretrade_sha256=holdings_fingerprint(realized.next_ids, result['next_pretrade']),
                    holdings_reset=False,
                    gross_excess=gross, impact_cost=cost, net_excess=gross-cost,
                    turnover=result['turnover'], gross_exposure=exposure,
                    net_exposure=float(weights.sum()), largest_absolute_weight=float(absolute.max()),
                    gross_weight_concentration=float(np.sum((absolute/exposure)**2)) if exposure else 0.,
                    modeled_annual_volatility=annual_volatility(weights, market.covariance)))
                for sid, value in zip(result['dropped_ids'], result['dropped_drifted_weights']):
                    exits.append(dict(seed=seed, method=method, decision_eom=pd.Timestamp(day),
                                      id=int(sid), omitted_drifted_weight=float(value),
                                      liquidation_cost_added=False))
                if compare_archived and policy.kind in ('archived_parent', 'neutral_replay'):
                    archived_paths_checked += 1
                    archive = source.archived_positions(day, seed)
                    for key in ('ids', 'next_ids'):
                        np.testing.assert_array_equal(arrays[key], archive[key])
                    for key in ('pretrade', 'target', 'weight', 'trade', 'drifted_weight', 'next_pretrade'):
                        actual = arrays[key][:, j]
                        np.testing.assert_allclose(actual, archive[key], atol=neutral_atol,
                                                   rtol=neutral_rtol, err_msg='Archived parent replay differs')
                        max_archive = max(max_archive, float(np.max(abs(actual-archive[key]), initial=0.)))
            # Neutral controls must inherit their own positions and still agree
            # with their paired parent; equality is checked after actual trading.
            for j, policy in enumerate(active):
                if policy.kind != 'neutral_replay':
                    continue
                for k, other in enumerate(active):
                    if other.kind == 'archived_parent' and other.parent.seed == policy.parent.seed:
                        neutral_pairs_checked += 1
                        for key in ('pretrade', 'target', 'weight', 'trade', 'drifted_weight', 'next_pretrade'):
                            np.testing.assert_allclose(arrays[key][:, j], arrays[key][:, k],
                                atol=neutral_atol, rtol=neutral_rtol, err_msg='Neutral replay differs from paired parent')
                            max_neutral = max(max_neutral, float(np.max(abs(arrays[key][:, j]-arrays[key][:, k]), initial=0.)))
            arrays.update(seeds=np.array([s for s, _ in specs]), methods=np.array([m for _, m in specs]))
            if save_positions:
                if folder is None:
                    position_records.append(arrays)
                else:
                    with (folder/f'{day}.npz').open('xb') as handle:
                        np.savez_compressed(handle, **arrays)
            h, holding_ids = arrays['next_pretrade'], realized.next_ids.copy()
            if progress is not None:
                progress(dict(decision_date=day, completed_months=len(monthly)//len(specs),
                              total_months=len(dates)))
        source.unchanged()
        if policy_identities != {str(key): policy.fingerprint for key, policy in policies.items()}:
            raise ValueError('Annual policy changed during path evaluation')
        frame = pd.DataFrame(monthly)
        exit_frame = pd.DataFrame(exits, columns=['seed', 'method', 'decision_eom', 'id',
                                               'omitted_drifted_weight', 'liquidation_cost_added'])
        manifest = dict(status='COMPLETE', decision_first=dates[0], decision_last=dates[-1],
            months=len(dates), paths=len(specs), monthly_records=len(frame),
            first_trade_charged=True, annual_or_period_holdings_reset=False,
            initial_source_fingerprint=initial.source_fingerprint,
            initial_pretrade_sha256=holdings_fingerprint(initial.ids, initial.holdings),
            source_fingerprint=source.source_fingerprint,
            reference_fingerprint=source.reference_fingerprint,
            independent_position_accounting_all_months=True,
            neutral_maximum_position_error=max_neutral,
            neutral_monthly_pairs_checked=neutral_pairs_checked,
            archived_replay_requested=compare_archived,
            archived_replay_checked=archived_paths_checked > 0,
            archived_monthly_paths_checked=archived_paths_checked,
            archived_replay_maximum_position_error=max_archive,
            comparison_tolerances=dict(accounting_atol=accounting_atol, accounting_rtol=accounting_rtol,
                                       neutral_atol=neutral_atol, neutral_rtol=neutral_rtol),
            policy_fingerprints=policy_identities, consumed_inputs_sha256=source.consumed_inputs,
            saved_positions=save_positions)
        if folder is not None:
            frame.to_parquet(folder/'monthly.parquet', index=False)
            exit_frame.to_parquet(folder/'exit_ledger.parquet', index=False)
            manifest['artifact_sha256'] = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
            with (folder/'paths_manifest.json').open('x') as handle:
                json.dump(_json(manifest), handle, indent=2, allow_nan=False)
                handle.write('\n')
        return dict(monthly=frame, exit_ledger=exit_frame, positions=position_records,
                    manifest=manifest, final_ids=holding_ids, final_pretrade=h)
    except Exception as error:
        if folder is not None:
            with (folder/'failed_run.json').open('x') as handle:
                json.dump(dict(status='FAILED_NOT_ACCEPTED', error_type=type(error).__name__,
                               error=str(error), completed_months=len(monthly)//len(specs)),
                          handle, indent=2)
                handle.write('\n')
        raise
