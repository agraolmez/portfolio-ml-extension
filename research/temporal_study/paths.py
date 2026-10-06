"""Annual model binding and continuous original-rule portfolio evaluation.

There are no study dates, scientific control values, tolerance choices, or run
authorization defaults here. Production inputs come from AcceptedPathSource;
the small source interface also permits synthetic accounting tests. Every path
inherits its own holdings through annual boundaries and pays its first trade.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from baseline.core import transition
from feature_geometry.features import array, fingerprint
from feature_geometry.history import sha
from temporal_extension.history import AcceptedTemporalHistory, VerifiedParent, PARENT
from temporal_extension.model import TemporalModel
from temporal_extension.reference import month_key


def _day(value):
    month_key(value)
    return pd.Timestamp(value).date().isoformat()


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


@dataclass(frozen=True)
class InitialState:
    """One explicit, common initial portfolio, before each path's first trade."""
    decision_date: str
    ids: np.ndarray
    holdings: np.ndarray
    source_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, 'decision_date', _day(self.decision_date))
        ids = np.array(self.ids, copy=True)
        holdings = array(self.holdings, 1)
        if (ids.ndim != 1 or pd.isna(ids).any() or len(np.unique(ids)) != len(ids)
                or holdings.shape != ids.shape or not self.source_fingerprint):
            raise ValueError('Aligned unique initial IDs, holdings and provenance required')
        ids.setflags(write=False)
        object.__setattr__(self, 'ids', ids)
        object.__setattr__(self, 'holdings', holdings)


@dataclass(frozen=True)
class RealizedMonth:
    """Outcomes supplied only after all dated portfolio aims have been formed."""
    decision_date: str
    ids: np.ndarray
    total_returns: np.ndarray
    excess_returns: np.ndarray
    market_growth: float
    next_ids: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'decision_date', _day(self.decision_date))
        for name in ('ids', 'next_ids'):
            value = np.array(getattr(self, name), copy=True)
            if (value.ndim != 1 or pd.isna(value).any()
                    or len(np.unique(value)) != len(value)):
                raise ValueError('Unique ordered outcome IDs required')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ('total_returns', 'excess_returns'):
            value = array(getattr(self, name), 1)
            if value.shape != self.ids.shape:
                raise ValueError('Realized outcomes must match current IDs')
            object.__setattr__(self, name, value)
        if not np.isfinite(self.market_growth) or self.market_growth <= -1:
            raise ValueError('Finite positive realized market gross growth required')


class AnnualPolicy:
    """A strategy-year artifact pinned to its exact Extension-2 year and seed.

    ``neutral_replay`` uses the archived beta with all gains one. A separate
    annual coefficient-neutrality gate is required in fitted receipts; replay
    by itself does not claim that a historical conditional solve was checked.
    """
    KINDS = ('archived_parent', 'neutral_replay', 'learned_temporal',
             'fixed_temporal', 'learned_current_only', 'fixed_current_only')

    def __init__(self, parent, *, kind, model=None, receipt=None, prescribed_gains=None):
        if not isinstance(parent, VerifiedParent) or kind not in self.KINDS:
            raise ValueError('Verified annual parent and explicit supported policy kind required')
        self.parent = parent
        self.kind = kind
        self.model = model
        self.receipt = MappingProxyType(_json(receipt or {}))
        self.prescribed_gains = (None if prescribed_gains is None
                                 else array(prescribed_gains, 1))
        self._identity = self._snapshot()
        self.validate()

    def _snapshot(self):
        p = self.parent
        return fingerprint(p.fingerprint, p.model.feature_map.fingerprint,
                           p.model.theta, p.model.beta,
                           self.kind, dict(self.receipt), self.prescribed_gains,
                           None if self.model is None else self.model._content_hash())

    @property
    def fingerprint(self):
        self.validate()
        return self._identity

    def validate(self):
        p, model, receipt = self.parent, self.model, self.receipt
        if (not np.array_equal(p.model.theta, p.feature_map.theta)
                or p.model.feature_map.fingerprint != p.feature_map.parent.fingerprint
                or self._snapshot() != self._identity):
            raise ValueError('Annual parent, policy or fitting receipt changed')
        if self.kind in ('archived_parent', 'neutral_replay'):
            if model is not None or receipt or self.prescribed_gains is not None:
                raise ValueError('Archived and neutral policies use only the verified parent')
            return
        current_only = self.kind.endswith('current_only')
        if current_only:
            from .controls import CurrentOnlyModel
            correct_type = type(model) is CurrentOnlyModel
        else:
            correct_type = type(model) is TemporalModel
        if not correct_type:
            raise ValueError('Explicit model kind and saved model implementation differ')
        model._check_unchanged()
        if (not np.array_equal(model.feature_map.theta, p.feature_map.theta)
                or not np.array_equal(model.feature_map.groups, p.feature_map.groups)
                or model.feature_map.parent.fingerprint != p.feature_map.parent.fingerprint
                or model.parent_fingerprint != p.fingerprint
                or model.cutoff != p.model.cutoff or model.parent_cutoff != p.model.cutoff):
            raise ValueError('Fitted policy changed the frozen annual parent')
        common = dict(model_kind=self.kind, year=p.year, seed=p.seed,
                      parent_fingerprint=p.fingerprint,
                      parent_feature_fingerprint=p.feature_map.fingerprint,
                      source_fingerprint=model.source_fingerprint,
                      reference_fingerprint=model.reference_fingerprint,
                      training_first=p.metadata['training_first'],
                      training_last=p.metadata['training_last'])
        if any(receipt.get(key) != value for key, value in common.items()):
            raise ValueError('Fitting receipt year/seed, parent, history or input binding differs')
        if current_only:
            from .controls import current_only_reference_fingerprint
            if (receipt.get('control_feature_fingerprint') != model.feature_map.fingerprint
                    or model.reference_fingerprint != current_only_reference_fingerprint(model.source_fingerprint)):
                raise ValueError('Current-only control fingerprint or input definition differs')
        if (receipt.get('eligible_for_portfolio_evaluation') is not True
                or receipt.get('neutral_gate', {}).get('passed') is not True
                or not np.array_equal(receipt.get('gains'), model.gains)
                or not np.array_equal(receipt.get('beta'), model.beta)):
            raise ValueError('Eligible neutral-checked fitting receipt and matching parameters required')
        residual = receipt.get('solve_residual')
        if (not isinstance(residual, (float, int)) or isinstance(residual, bool)
                or not np.isfinite(residual) or not 0 <= residual <= 1e-9):
            raise ValueError('Accurate exact conditional coefficient solve required')
        if self.kind.startswith('learned_'):
            settings = receipt.get('optimizer_settings', {})
            norm, tolerance = receipt.get('projected_gradient_norm'), settings.get('tolerance')
            if (receipt.get('status') != 'converged' or receipt.get('converged') is not True
                    or norm is None or tolerance is None
                    or not np.isfinite([norm, tolerance]).all() or not 0 <= norm <= tolerance
                    or tolerance <= 0 or self.prescribed_gains is not None):
                raise ValueError('Nonconverged or diagnostic model cannot enter a portfolio path')
            try:
                lower = np.broadcast_to(np.asarray(settings['lower'], float), model.gains.shape)
                upper = np.broadcast_to(np.asarray(settings['upper'], float), model.gains.shape)
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError('Explicit fitted gain bounds required') from error
        else:
            if (receipt.get('status') != 'fixed_configuration_solved'
                    or self.prescribed_gains is None
                    or not np.array_equal(model.gains, self.prescribed_gains)
                    or not np.array_equal(receipt.get('prescribed_gains'), self.prescribed_gains)):
                raise ValueError('Fixed control differs from its prescribed gain configuration')
            lower = np.asarray(receipt.get('lower'), float)
            upper = np.asarray(receipt.get('upper'), float)
        if not np.array_equal(lower, model.lower) or not np.array_equal(upper, model.upper):
            raise ValueError('Fitting receipt and model bounds differ')

    def aim(self, market, bound_innovation):
        self.validate()
        self.parent.assert_decision(market.decision_date)
        z, vol = market.characteristics[0], market.volatility[0]
        if self.kind == 'archived_parent':
            return self.parent.model.aim(z, vol, market.decision_date)
        if self.kind == 'neutral_replay':
            # Check the current binding even though exact all-one identity does
            # not numerically depend on the history values.
            if bound_innovation is None:
                raise ValueError('Dated temporal innovation required for neutral replay')
            innovation = bound_innovation.validate(
                market.decision_date, market.ids, z,
                bound_innovation.source_fingerprint, bound_innovation.reference_fingerprint)
            return self.parent.feature_map.signals(
                z, vol, innovation, np.ones(self.parent.feature_map.gain_count)) @ self.parent.model.beta
        return self.model.aim(market.decision_date, market.ids, z, vol,
                              None if self.kind.endswith('current_only') else bound_innovation)


class AcceptedPathSource:
    """Realized accounting joined to unchanged, verified original market inputs."""
    def __init__(self, history):
        if not isinstance(history, AcceptedTemporalHistory):
            raise ValueError('Accepted temporal history adapter required')
        self.history = history
        self.source_fingerprint = history.source_fingerprint
        self.reference_fingerprint = history.reference_fingerprint
        self._cached_archive = None

    def market(self, date):
        return self.history.market(date)

    def innovation(self, market):
        return self.history.innovations(market)[0]

    def _archive(self, date):
        day = _day(date)
        path = self.history.files.path(f'{PARENT}/paths/{day}.npz')
        if self._cached_archive is not None and self._cached_archive[0] == day:
            return self._cached_archive[1]
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        self._cached_archive = day, arrays
        return arrays

    def archived_positions(self, date, seed):
        data = self._archive(date)
        indexes = np.flatnonzero((data['seeds'] == seed) & (data['methods'] == 'full'))
        if len(indexes) != 1:
            raise ValueError('Unique archived full learned-scale seed path required')
        column = indexes[0]
        return {key: data[key].copy() if key in ('ids', 'next_ids') else data[key][:, column].copy()
                for key in ('ids', 'next_ids', 'pretrade', 'target', 'weight', 'trade',
                            'drifted_weight', 'next_pretrade')}

    def common_initial(self):
        relative = 'data/prepared/step9_v1/continuous_R1/PolicyDecision.parquet'
        path = self.history.files.path(relative)
        frame = pd.read_parquet(path, columns=['id', 'eom', 'pretrade'],
                                filters=[('eom', '==', pd.Timestamp('1980-12-31'))]).sort_values('id')
        return InitialState('1980-12-31', frame.id.to_numpy(), frame.pretrade.to_numpy(),
                            fingerprint(relative, sha(path), '1980-12-31'))

    def realized(self, market):
        frame, _ = self.history._month(market.decision_date)
        rows = frame.loc[market.ids]
        archive = self._archive(market.decision_date)
        if not np.array_equal(archive['ids'], market.ids) or rows.mu_ld1.nunique() != 1:
            raise ValueError('Archived path universe or realized market growth differs')
        # Use the parent path's terminal next-ID convention as well as its
        # intermediate universes; never invent a last-month liquidation.
        return RealizedMonth(market.decision_date, market.ids, rows.tr_ld1.to_numpy(),
                             rows.ret_ld1.to_numpy(), float(rows.mu_ld1.iloc[0]),
                             archive['next_ids'])

    def unchanged(self):
        self.history.files.unchanged()

    @property
    def consumed_inputs(self):
        return dict(self.history.files.consumed)


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
                    pretrade_sha256=fingerprint(market.ids, h[:, j]),
                    next_pretrade_sha256=fingerprint(realized.next_ids, result['next_pretrade']),
                    holdings_reset=False,
                    gross_excess=gross, impact_cost=cost, net_excess=gross-cost,
                    turnover=result['turnover'], gross_exposure=exposure,
                    net_exposure=float(weights.sum()), largest_absolute_weight=float(absolute.max()),
                    gross_weight_concentration=float(np.sum((absolute/exposure)**2)) if exposure else 0.,
                    modeled_annual_volatility=float(np.sqrt(12*weights@market.covariance@weights))))
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
            initial_pretrade_sha256=fingerprint(initial.ids, initial.holdings),
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
