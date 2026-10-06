"""One frozen forty-coefficient direct-utility development probe.

No experiment runs on import. Prepare binds accepted inputs; timing measures
three fixed months before any fit or replay; run completes all three fixed
annual episodes before reporting their gate. Completed parents are read-only.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from baseline.core import moment, report_returns, transition
from feature_geometry.history import CanonicalHistory
from extension4.runtime import Runtime, now
from theme_policy_kernels_20260928 import (
    ACCEPTED_DICTIONARY_SHA256, ThemeDictionary, signal_names, theme_signals,
)

CONFIG = ROOT / 'config/theme_policy_probe_20260929.json'
BASE = 'data/prepared/step9_v1/baseline'
REFERENCE = 'output/feature_geometry_study_20260914'
RAW = 'data/prepared/step3_author_v3/monthly_inputs.parquet'
YEARS = (1981, 1990, 2000)
METHODS = ('B1', 'THEME40')
FIRST = '1965-11-30'
TIMING_DATES = ('1965-11-30', '1980-11-30', '1999-11-30')
REVIEW_RESERVE_SECONDS = 120.
FIT_REPLAY_RESERVE_SECONDS = 120.
ATOL, RTOL = 1e-10, 1e-8
PROTECTED = ('GOAL.md', 'docs/STEP_1_THEORY.md', 'docs/RECONSTRUCTION_AMENDMENT.md',
             'config/protocol_v1.json', 'config/reconstruction_amendment.json')
MANIFESTS = (
    'output/step3_author_conventions_20260911/bundle_checksums.json',
    'output/step4_20260911/acceptance_manifest.json',
    'output/step6_20260911/preparation_manifest.json',
    'output/step9_20260913/acceptance_manifest.json',
    REFERENCE + '/acceptance_manifest.json',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path, guard=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        if hasattr(fcntl, 'F_NOCACHE'):
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
        for block in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(block)
            guard()
    return digest.hexdigest()


def save_json(path, value):
    """An existing artifact is never replaced; incomplete temporary files remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.partial_' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def native_pressure():
    level = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(level))
    code = ctypes.CDLL(None, use_errno=True).sysctlbyname(
        b'kern.memorystatus_vm_pressure_level', ctypes.byref(level), ctypes.byref(size), None, 0)
    if code or level.value != 1:
        raise MemoryError(f'Native memory pressure is not normal: level={level.value}, error={code}')


class GuardedRuntime(Runtime):
    def sample(self):
        row = super().sample()
        native_pressure()
        return row


def training_calendar(year):
    """The only dates whose moments may enter this predetermined year's fit."""
    if type(year) is not int or year not in YEARS:
        raise ValueError('Only the three prespecified development years are allowed')
    return tuple(pd.date_range(FIRST, f'{year - 1}-11-30', freq='ME'))


def aggregate_training(records, year):
    """Access the causal prefix only; never inspect later records or outcomes."""
    score = np.zeros(40); risk = np.zeros((40, 40)); cost = np.zeros((40, 40))
    dates = training_calendar(year)
    for day in dates:
        record = records[str(day.date())]
        for key, shape in (('r_tilde', (40,)), ('risk', (40, 40)), ('tc', (40, 40))):
            if np.shape(record[key]) != shape or not np.isfinite(record[key]).all():
                raise ValueError('Invalid new-representation training moment')
        score += record['r_tilde']; risk += record['risk']; cost += record['tc']
    return dict(score=score / len(dates), risk=risk / len(dates), cost=cost / len(dates), count=len(dates))


def validate_config(config):
    expected = dict(years=list(YEARS), training_first=FIRST, training_last='1999-11-30',
                    timing_dates=list(TIMING_DATES), methods=list(METHODS), gamma=10,
                    ridge_rule='accepted_baseline_prior_december_numeric_lambda',
                    output_directory='output/theme_policy_probe_20260929')
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError('Frozen specification differs: ' + key)
    basis = config['basis']
    for key, value in dict(dictionary='config/feature_dictionary_v1.csv',
                           dictionary_sha256=ACCEPTED_DICTIONARY_SHA256, groups='sorted',
                           degrees=[1, 2, 3], ordering='degree_major', constant=True, columns=40,
                           learned_scales=False, temporal_gains=False, random_mixtures=False,
                           linear_14_fit=False).items():
        if basis[key] != value:
            raise ValueError('Frozen representation differs: ' + key)
    for key in ('all_three_years_positive_both_utilities', 'all_technical_checks',
                'all_fixed_years_required_regardless_of_interim_outcomes'):
        if config['gate'][key] is not True:
            raise ValueError('Frozen development gate differs')
    if config['gate']['automatic_full_study'] is not False or config['gate']['significance_test'] is not False:
        raise ValueError('This probe does not authorize an exhaustive study or significance search')
    budget = config['compute_budget']
    for key, value in dict(total_active_seconds=1800, pilot_active_seconds=600,
                           max_physical_footprint_gib=16, min_system_headroom_gib=4,
                           min_free_disk_gib=50, max_additional_disk_gib=2, maximum_workers=1,
                           blas_threads=1, native_memory_pressure='normal only', swap_growth_bytes=0,
                           one_heavy_job=True, local_only=True, paid=False).items():
        if budget[key] != value:
            raise ValueError('Frozen resource limit differs: ' + key)


def verify_inventory(inventory, guard=lambda: None):
    verified = {}
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Dependency escapes the authorized project')
        if path not in verified:
            verified[path] = sha(path, guard)
        if verified[path] != expected:
            raise ValueError('Changed dependency: ' + relative)


def _inventories(guard):
    accepted = {}; receipts = {}
    for relative in MANIFESTS:
        value = read(ROOT / relative)
        if 'gates' in value and not all(x['passed'] for x in value['gates']):
            raise ValueError('Failed parent acceptance: ' + relative)
        receipts[relative] = sha(ROOT / relative, guard)
        for name, digest in value.get('artifact_sha256', value).items():
            if isinstance(digest, str) and len(digest) == 64:
                accepted.setdefault(name, set()).add(digest)
    return accepted, receipts


def prepare(config, out, guard=lambda: None):
    validate_config(config)
    freeze = read(out / 'protocol_freeze.json')
    if (freeze['config_sha256'] != sha(CONFIG, guard)
            or freeze['new_financial_performance_calculated'] is not False
            or freeze['new_models_fitted'] != 0):
        raise ValueError('Pre-outcome protocol freeze differs')
    preflight = read(out / 'financial_preflight.json')
    if preflight != read(ROOT / REFERENCE / 'financial_preflight.json'):
        raise ValueError('Fresh financial preflight differs from the accepted economic bundle')
    if (out / 'lock.json').exists():
        lock = read(out / 'lock.json')
        if lock['config'] != config:
            raise ValueError('Configuration differs from immutable lock')
        verify_inventory(lock['artifact_sha256'], guard)
        return lock
    accepted, consumed = _inventories(guard)

    def bind(relative):
        canonical = str((ROOT / relative).resolve().relative_to(ROOT))
        choices = accepted.get(relative, accepted.get(canonical))
        if choices is None or len(choices) != 1:
            raise ValueError('Missing/conflicting accepted input: ' + relative)
        consumed[relative] = next(iter(choices))

    for relative in (RAW, 'data/prepared/step4_v1/risk/scales.parquet',
                     'output/step2_20260910/source_settings.json', BASE + '/annual_selection.parquet',
                     REFERENCE + '/paths/monthly.parquet',
                     'baseline/core.py', 'baseline/io.py'):
        bind(relative)
    # CanonicalHistory uses accepted Step6 economic matrices. Bind Step9 aliases
    # too and require identical content to the actual baseline replay inputs.
    for date in pd.date_range(FIRST, '2000-11-30', freq='ME'):
        key = str(date.date())
        for kind in ('matrices', 'risk'):
            early = f'data/prepared/step6_v1/{kind}/{key}.npz'
            baseline = f'{BASE}/{kind}/{key}.npz'
            bind(early); bind(baseline)
            if consumed[early] != consumed[baseline]:
                raise ValueError('Canonical economic input differs from accepted baseline: ' + key)
    for year in YEARS:
        for date in pd.date_range(f'{year - 1}-12-31', f'{year}-11-30', freq='ME'):
            bind(f'{REFERENCE}/paths/{date.date()}.npz')
        bind(f'{REFERENCE}/fits/{year}/seed1/fixed512/fit.json')
    # Read only tuning metadata here, never candidate performance.
    verify_inventory(consumed, guard)
    selection = pd.read_parquet(ROOT / BASE / 'annual_selection.parquet')
    annual = {}
    for year in YEARS:
        date = pd.Timestamp(f'{year - 1}-12-31')
        rows = selection[selection.return_eom.eq(date)]
        if len(rows) != 1:
            raise ValueError('Missing unique prior-December baseline selection')
        row = rows.iloc[0]
        gi, width, li = int(row.g_index), int(row.p), int(row.lambda_index)
        if width != 512 or int(row.fit_year) != year - 1:
            raise ValueError('Unexpected chronological baseline reference')
        coefficient = f'{BASE}/coefficients/g{gi}/{year}_p512.npz'
        bind(coefficient)
        verify_inventory({coefficient: consumed[coefficient]}, guard)
        with np.load(ROOT / coefficient, allow_pickle=False) as data:
            penalty = float(data['lambdas'][li])
            if (penalty != float(row['lambda']) or penalty < 0 or not np.isfinite(penalty)
                    or str(data['training_cutoff'].item()) != f'{year - 1}-11-30'
                    or int(data['training_months']) != len(training_calendar(year))):
                raise ValueError('Original coefficient metadata or exact numerical ridge differs')
        parent = read(ROOT / REFERENCE / f'fits/{year}/seed1/fixed512/fit.json')
        expected = dict(year=year, seed=1, method='fixed512', bandwidth_index=gi,
                        coefficient_ridge=penalty, cutoff=str(date.date()), training_first=FIRST,
                        training_last=f'{year - 1}-11-30', training_months=len(training_calendar(year)),
                        latest_available_at=str(date.date()))
        if any(parent[key] != value for key, value in expected.items()):
            raise ValueError('Archived baseline chronology disagrees with source selection')
        annual[str(year)] = dict(**expected, original_coefficients=coefficient,
                                 lambda_index=li, selection_available_at=str(date.date()),
                                 candidate_columns=40, effective_regularization_equal=False)
    dictionary = ROOT / config['basis']['dictionary']
    if sha(dictionary, guard) != ACCEPTED_DICTIONARY_SHA256:
        raise ValueError('Accepted characteristic dictionary changed')
    code = ['scripts/theme_policy_probe_20260929.py', 'scripts/theme_policy_kernels_20260928.py',
            'tests/test_theme_policy.py', 'baseline/__init__.py', 'baseline/core.py', 'baseline/io.py',
            'feature_geometry/__init__.py', 'feature_geometry/history.py', 'feature_geometry/moments.py',
            'feature_geometry/features.py', 'feature_geometry/training.py', 'extension4/__init__.py',
            'extension4/runtime.py', 'extension4/storage.py', 'temporal_fast_v4/__init__.py',
            'temporal_fast_v4/memory.py']
    for relative in (*PROTECTED, config['basis']['dictionary'], *code,
                     REFERENCE + '/financial_preflight.json',
                     str(CONFIG.relative_to(ROOT)), str((out / 'protocol_freeze.json').relative_to(ROOT)),
                     str((out / 'financial_preflight.json').relative_to(ROOT))):
        actual = sha(ROOT / relative, guard)
        if relative in consumed and actual != consumed[relative]:
            raise ValueError('Accepted source code changed: ' + relative)
        consumed[relative] = actual
    verify_inventory(consumed, guard)
    lock = dict(created_at_utc=now(), config=config, artifact_sha256=consumed, annual_settings=annual,
                inherited_learned_models=False, baseline_snapshots_used_only_as_reference=True,
                freshly_bound_comparison_metadata=[REFERENCE + '/financial_preflight.json'],
                source_closure=code, review_reserve_seconds=REVIEW_RESERVE_SECONDS,
                fit_replay_reserve_seconds=FIT_REPLAY_RESERVE_SECONDS)
    save_json(out / 'lock.json', lock)
    return lock


def stage_verify(folder, context, guard=lambda: None):
    commit = read(folder / 'commit.json')
    if commit['context'] != context or commit['status'] != 'COMPLETE':
        raise ValueError('Incomplete/incompatible immutable stage')
    inventory = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'commit.json'}
    if inventory != set(commit['files_sha256']):
        raise ValueError('Stage inventory differs')
    for name, digest in commit['files_sha256'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path, guard) != digest:
            raise ValueError('Changed committed stage: ' + str(folder))
    return commit


def stage_write(folder, context, writer, guard=lambda: None):
    if folder.exists():
        return stage_verify(folder, context, guard)
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = folder.with_name('.' + folder.name + '.partial_' + uuid.uuid4().hex)
    temporary.mkdir()
    writer(temporary)
    files = {str(p.relative_to(temporary)): sha(p, guard) for p in temporary.rglob('*') if p.is_file()}
    save_json(temporary / 'commit.json', dict(status='COMPLETE', context=context, files_sha256=files))
    guard(); native_pressure()
    if folder.exists():
        raise FileExistsError('Concurrent stage creation; incomplete attempt retained')
    temporary.rename(folder)
    return stage_verify(folder, context, guard)


def context(out, **details):
    return dict(lock_sha256=sha(out / 'lock.json'), **details)


def load_history(out, last):
    started = time.monotonic()
    source = CanonicalHistory(ROOT, FIRST, last, read(out / 'financial_preflight.json'))
    rows = pd.read_csv(ROOT / 'config/feature_dictionary_v1.csv').to_dict('records')
    dictionary = ThemeDictionary.from_rows(rows, source.features)
    del source.data
    gc.collect()
    return source, dictionary, time.monotonic() - started


def build_moment(date, source, dictionary, out, lock, runtime):
    key = str(pd.Timestamp(date).date())
    folder = out / 'moments' / key
    ctx = context(out, stage='monthly_moment', decision_eom=key, columns=40)
    if folder.exists():
        stage_verify(folder, ctx, runtime.guard)
        return read(folder / 'metadata.json')
    started = time.monotonic()
    runtime.guard(); native_pressure()
    sample = source.training_sample(key, '1999-12-31')
    market = sample.market
    signals = [theme_signals(z, vol, dictionary) for z, vol in zip(market.characteristics, market.volatility)]
    if any(x.shape != (len(market.ids), 40) for x in signals):
        raise ValueError('Fixed polynomial signal dimension changed')
    measured = key in TIMING_DATES
    result = moment(signals, market.growth, market.m, market.covariance, market.impact,
                    market.wealth, sample.returns, gamma=market.gamma, return_operators=measured)
    runtime.guard(); native_pressure()
    for value in result.values():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite exact monthly moment')
    for name in ('risk', 'tc', 'denom'):
        np.testing.assert_allclose(result[name], result[name].T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['denom'], result['risk'] + result['tc'], atol=1e-10, rtol=1e-10)
    metadata = dict(decision_eom=key, available_at=sample.available_at,
                    market_fingerprint=market.fingerprint, sample_fingerprint=sample.fingerprint,
                    stocks=len(market.ids), columns=40, signal_names=list(signal_names(dictionary)),
                    accepted_inputs_sha256={name: lock['artifact_sha256'][name] for name in (
                        RAW, 'data/prepared/step4_v1/risk/scales.parquet',
                        f'data/prepared/step6_v1/matrices/{key}.npz', f'data/prepared/step6_v1/risk/{key}.npz',
                        'config/feature_dictionary_v1.csv')}, representative_review_packet=measured)

    def write(folder):
        save_npz(folder / 'moments.npz', r_tilde=result['r_tilde'], risk=result['risk'], tc=result['tc'])
        if measured:
            packet = dict(ids=market.ids, m=market.m, covariance=market.covariance, impact=market.impact,
                          wealth=np.array(market.wealth), gamma=np.array(market.gamma), returns=sample.returns,
                          omega=result['omega'], omega_change=result['omega_change'])
            for i in range(13):
                packet.update({f'z{i}': market.characteristics[i], f'vol{i}': market.volatility[i],
                               f'growth{i}': market.growth[i], f'signal{i}': signals[i]})
            save_npz(folder / 'review_packet.npz', **packet)
        metadata['measured_seconds'] = time.monotonic() - started
        save_json(folder / 'metadata.json', metadata)

    stage_write(folder, ctx, write, runtime.guard)
    runtime.event('training_moment_committed', decision_eom=key, stocks=len(market.ids),
                  seconds=metadata['measured_seconds'], coefficients_fitted=0)
    return metadata


def timing(config, out, lock, runtime):
    folder = out / 'timing'
    ctx = context(out, stage='fixed_timing_gate', dates=list(TIMING_DATES))
    if folder.exists():
        stage_verify(folder, ctx, runtime.guard)
        return read(folder / 'result.json')
    source, dictionary, loading = load_history(out, '1999-11-30')
    measurements = [build_moment(date, source, dictionary, out, lock, runtime) for date in TIMING_DATES]
    consumed = source.consumed.copy()
    del source
    gc.collect()
    slowest = max(x['measured_seconds'] for x in measurements)
    # All409 months at the slowest observed cost is intentionally conservative,
    # even though three representative moments have already been committed.
    already = runtime.previous + time.monotonic() - runtime.start
    projection = already + loading + 409 * slowest + REVIEW_RESERVE_SECONDS + FIT_REPLAY_RESERVE_SECONDS
    passed = projection <= config['compute_budget']['total_active_seconds']
    result = dict(status='TIMING_PASSED' if passed else 'INCONCLUSIVE_PREPARATION_EXCEEDS_BUDGET',
                  passed=bool(passed), initialization_seconds=loading, measurements=measurements,
                  maximum_measured_seconds_per_month=slowest, projected_months=409,
                  cumulative_seconds_at_projection=already, review_reserve_seconds=REVIEW_RESERVE_SECONDS,
                  fit_replay_reserve_seconds=FIT_REPLAY_RESERVE_SECONDS,
                  projected_total_seconds=projection, total_budget_seconds=1800,
                  coefficients_fitted=0, candidate_performance_computed=False,
                  consumed_input_sha256=consumed)
    stage_write(folder, ctx, lambda p: save_json(p / 'result.json', result), runtime.guard)
    runtime.event('timing_gate', passed=bool(passed), projected_total_seconds=projection,
                  next_action='fixed_three_year_probe' if passed else 'stop_inconclusive')
    return result


def fit_year(year, out, lock, runtime):
    annual = lock['annual_settings'][str(year)]
    directory = out / 'years' / str(year) / 'fit'
    commits = {str(day.date()): sha(out / 'moments' / str(day.date()) / 'commit.json', runtime.guard)
               for day in training_calendar(year)}
    ctx = context(out, stage='exact_fit', year=year, training_moment_commit_sha256=commits,
                  coefficient_ridge=annual['coefficient_ridge'], columns=40)
    if directory.exists():
        stage_verify(directory, ctx, runtime.guard)
        with np.load(directory / 'coefficients.npz', allow_pickle=False) as z:
            return z['beta'].copy()

    class Records:
        def __getitem__(self, key):
            folder = out / 'moments' / key
            stage_verify(folder, context(out, stage='monthly_moment', decision_eom=key, columns=40), runtime.guard)
            metadata = read(folder / 'metadata.json')
            if (metadata['available_at'] > annual['cutoff']
                    or metadata['decision_eom'] != key
                    or pd.Timestamp(metadata['available_at']) != pd.Timestamp(key) + pd.offsets.MonthEnd(1)):
                raise ValueError('Immature/misdated training outcome')
            with np.load(folder / 'moments.npz', allow_pickle=False) as z:
                return {name: z[name].copy() for name in ('r_tilde', 'risk', 'tc')}

    stats = aggregate_training(Records(), year)
    if stats['count'] != annual['training_months']:
        raise ValueError('Wrong causal training prefix')
    penalty = annual['coefficient_ridge']
    hessian = (stats['risk'] + stats['cost'])
    system = (hessian + hessian.T) / 2 + penalty * np.eye(40)
    beta = cho_solve(cho_factor(system, lower=True, check_finite=True), stats['score'], check_finite=True)
    residual = float(np.linalg.norm(system @ beta - stats['score']) / max(np.linalg.norm(stats['score']), 1e-30))
    value = float(stats['score'] @ beta - .5 * beta @ system @ beta)
    if not np.isfinite(beta).all() or not np.isfinite(value) or residual > 1e-9:
        raise ValueError('Failed exact coefficient solve; no substitute ridge or jitter allowed')

    def write(folder):
        save_npz(folder / 'coefficients.npz', beta=beta, score=stats['score'], risk=stats['risk'], cost=stats['cost'])
        save_json(folder / 'metadata.json', dict(year=year, cutoff=annual['cutoff'], training_first=FIRST,
            training_last=annual['training_last'], training_months=stats['count'], coefficient_ridge=penalty,
            constant_penalized=True, coefficients=40, solve_residual=residual, training_value=value,
            independent_hyperparameter_search=False, later_moments_consumed=False))

    stage_write(directory, ctx, write, runtime.guard)
    runtime.event('annual_fit_committed', year=year, training_months=stats['count'],
                  solve_residual=residual, candidate_performance_computed=False)
    return beta


def replay_year(year, beta, source, dictionary, out, lock, runtime):
    directory = out / 'years' / str(year) / 'paths'
    directory.mkdir(parents=True, exist_ok=True)
    fit_hash = sha(out / 'years' / str(year) / 'fit' / 'commit.json')
    start, end = f'{year - 1}-12-31', f'{year}-11-30'
    raw = pd.read_parquet(ROOT / RAW, columns=['id', 'eom', 'tr_ld1', 'mu_ld1'],
                          filters=[('eom', '>=', pd.Timestamp(start)), ('eom', '<=', pd.Timestamp(end))])
    outcomes = {day: rows.sort_values('id').set_index('id') for day, rows in raw.groupby('eom')}
    baseline = pd.read_parquet(ROOT / REFERENCE / 'paths/monthly.parquet')
    baseline = baseline[(baseline.method == 'fixed512') & (baseline.seed == 1)].set_index('decision_eom')
    previous = None; holdings = None; next_ids = None; records = []
    for date in pd.date_range(start, end, freq='ME'):
        runtime.guard(); native_pressure()
        key = str(date.date())
        relative = f'{REFERENCE}/paths/{key}.npz'
        ctx = context(out, stage='episode_month', year=year, decision_eom=key,
                      previous_commit_sha256=previous, fit_commit_sha256=fit_hash,
                      baseline_snapshot_sha256=lock['artifact_sha256'][relative])
        folder = directory / key
        with np.load(ROOT / relative, allow_pickle=False) as z:
            columns = np.flatnonzero((z['methods'] == 'fixed512') & (z['seeds'] == 1))
            if len(columns) != 1:
                raise ValueError('Missing unique accepted baseline snapshot')
            c = int(columns[0])
            archive = {name: z[name].copy() for name in ('ids', 'next_ids')}
            archive.update({name: z[name][:, c].copy() for name in
                            ('target', 'pretrade', 'weight', 'trade', 'drifted_weight', 'next_pretrade')})
        if holdings is None:
            holdings = np.repeat(archive['pretrade'][:, None], 2, axis=1)
            next_ids = archive['ids']
        np.testing.assert_array_equal(next_ids, archive['ids'])
        if folder.exists():
            stage_verify(folder, ctx, runtime.guard)
            with np.load(folder / 'packet.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['pretrade'], holdings)
                np.testing.assert_array_equal(z['ids'], archive['ids'])
                holdings = z['next_pretrade'].copy(); next_ids = z['next_ids'].copy()
            records.extend(read(folder / 'monthly.json'))
            previous = sha(folder / 'commit.json')
            continue
        market = source.market(date)
        np.testing.assert_array_equal(market.ids, archive['ids'])
        current_signal = theme_signals(market.characteristics[0], market.volatility[0], dictionary)
        targets = np.column_stack((archive['target'], current_signal @ beta))
        # Target construction never accesses this month's realized return
        # fields; the outcome table is already loaded for subsequent accounting.
        row = outcomes[date].loc[market.ids]
        excess = source.months[date].loc[market.ids, 'ret_ld1'].to_numpy()
        if row.mu_ld1.nunique() != 1:
            raise ValueError('Inconsistent market denominator within a month')
        market_total = float(row.mu_ld1.iloc[0])
        packet = dict(ids=market.ids, next_ids=archive['next_ids'], m=market.m,
                      z=market.characteristics[0], volatility=market.volatility[0], signals=current_signal,
                      impact=market.impact, wealth=np.array(market.wealth), total_returns=row.tr_ld1.to_numpy(),
                      market_growth=np.array(market_total), excess_returns=excess, methods=np.array(METHODS),
                      targets=targets, pretrade=holdings.copy(), weights=np.empty_like(holdings),
                      trades=np.empty_like(holdings), drifted_weights=np.empty_like(holdings),
                      next_pretrade=np.empty((len(archive['next_ids']), 2)),
                      archived_baseline_pretrade=archive['pretrade'], archived_baseline_target=archive['target'])
        monthly = []; errors = dict(accounting=0., trade_rule=0., drift=0., baseline=0.)
        for column, method in enumerate(METHODS):
            result = transition(market.ids, holdings[:, column], targets[:, column], market.m,
                                row.tr_ld1.to_numpy(), market_total, market.impact, market.wealth,
                                excess, archive['next_ids'])
            direct = holdings[:, column] + (np.eye(len(market.ids)) - market.m) @ (targets[:, column] - holdings[:, column])
            gross = math.fsum(float(w) * float(r) for w, r in zip(direct, excess))
            cost = .5 * market.wealth * math.fsum(float(l) * float(q) ** 2 for l, q in
                        zip(market.impact, direct - holdings[:, column]))
            drift = direct * (1 + row.tr_ld1.to_numpy()) / (1 + market_total)
            lookup = dict(zip(market.ids, drift))
            expected_next = np.array([lookup.get(i, 0.) for i in archive['next_ids']])
            for actual, expected, name in ((result['weights'], direct, 'trade_rule'),
                                          (np.array([result['gross_excess'], result['impact_cost']]), np.array([gross, cost]), 'accounting'),
                                          (result['next_pretrade'], expected_next, 'drift')):
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
                errors[name] = max(errors[name], float(np.max(np.abs(actual - expected))))
            for name in ('weights', 'trades', 'drifted_weights', 'next_pretrade'):
                packet[name][:, column] = result[name]
            record = dict(year=year, method=method, decision_eom=key,
                          return_eom=str((date + pd.offsets.MonthEnd(1)).date()),
                          gross_excess=gross, impact_cost=cost, net_excess=gross - cost,
                          turnover=result['turnover'], gross_exposure=result['gross_exposure'])
            if column == 0:
                for name, saved in (('weights', 'weight'), ('trades', 'trade'),
                                    ('drifted_weights', 'drifted_weight'), ('next_pretrade', 'next_pretrade')):
                    np.testing.assert_allclose(result[name], archive[saved], atol=ATOL, rtol=RTOL)
                    errors['baseline'] = max(errors['baseline'], float(np.max(abs(result[name] - archive[saved]))))
                np.testing.assert_allclose(holdings[:, 0], archive['pretrade'], atol=ATOL, rtol=RTOL)
                old = baseline.loc[date]
                if pd.Timestamp(old.return_eom) != date + pd.offsets.MonthEnd(1):
                    raise ValueError('Archived baseline return maturity differs')
                for name in ('gross_excess', 'impact_cost', 'net_excess', 'turnover', 'gross_exposure'):
                    np.testing.assert_allclose(record[name], old[name], atol=ATOL, rtol=RTOL)
            monthly.append(record)

        def write(destination):
            save_npz(destination / 'packet.npz', **packet)
            save_json(destination / 'monthly.json', monthly)
            save_json(destination / 'checks.json', dict(passed=True, max_errors=errors,
                initial_holdings_common=key == start, first_switch_charged=True,
                inherited_source_extra_exit_cost=0., coefficient_fit_cutoff=f'{year - 1}-12-31'))

        stage_write(folder, ctx, write, runtime.guard)
        holdings = packet['next_pretrade']; next_ids = packet['next_ids']
        records.extend(monthly); previous = sha(folder / 'commit.json')
        runtime.event('episode_month_committed', year=year, decision_eom=key,
                      completed_months=len(records) // 2, interim_gate_evaluated=False)
    return records


def summarize(frame):
    if len(frame) != 72 or frame.duplicated(['year', 'method', 'decision_eom']).any():
        raise ValueError('All36months and both paths are required before any development gate')
    metrics = []; comparisons = []
    names = ('source_gross_sample_variance_utility', 'net_population_utility')
    for year in YEARS:
        by_method = {}
        expected = list(pd.date_range(f'{year}-01-31', f'{year}-12-31', freq='ME').strftime('%Y-%m-%d'))
        for method in METHODS:
            x = frame[(frame.year == year) & (frame.method == method)].sort_values('return_eom')
            if x.return_eom.tolist() != expected:
                raise ValueError('Missing or substituted prespecified development month')
            value = report_returns(x.gross_excess.to_numpy(), x.impact_cost.to_numpy(), gamma=10.)
            by_method[method] = value
            metrics.append(dict(year=year, method=method, **value))
        comparisons.append(dict(year=year, **{name: by_method['THEME40'][name] - by_method['B1'][name] for name in names}))
    passed = all(row[name] > 0 for row in comparisons for name in names)
    result = dict(status='COMPLETED_BOUNDED_DEVELOPMENT_PROBE', diagnostic_complete=True,
                  independent_review_complete=False, technical_execution_checks_passed=True,
                  scientific_study_complete=False, years=list(YEARS), paths=list(METHODS), monthly_rows=72,
                  coefficients_per_candidate_fit=40, candidate_fits=3,
                  metrics=metrics, candidate_minus_baseline=comparisons,
                  development_gate_passed=bool(passed), automatic_full_study_authorized=False,
                  statistically_significant_outperformance_established=False,
                  fresh_holdout=False, original_paper_outperformance_established=False,
                  selection_rule='All3years strictly positive in both utilities; no rounding or subset selection',
                  interpretation='Previously exposed, separate12month development episodes; exact original accounting')
    return metrics, result


def run_probe(config, out, lock, runtime):
    timing_folder = out / 'timing'
    stage_verify(timing_folder, context(out, stage='fixed_timing_gate', dates=list(TIMING_DATES)), runtime.guard)
    if read(timing_folder / 'result.json')['passed'] is not True:
        raise RuntimeError('Timing gate failed; no fitting or performance calculation is authorized')
    source, dictionary, loading = load_history(out, '2000-11-30')
    runtime.event('history_ready', loading_seconds=loading)
    for date in training_calendar(2000):
        build_moment(date, source, dictionary, out, lock, runtime)
    # Every causal prefix is ready before any fit; all fits are ready before any
    # episode outcomes are compared. No early winning/losing-year termination.
    fits = {year: fit_year(year, out, lock, runtime) for year in YEARS}
    rows = []
    for year in YEARS:
        rows.extend(replay_year(year, fits[year], source, dictionary, out, lock, runtime))
    frame = pd.DataFrame(rows)
    directory = out / 'analysis'
    ctx = context(out, stage='complete_development_summary', years=list(YEARS))

    def write(destination):
        metrics, result = summarize(frame)
        frame.to_csv(destination / 'monthly.csv', index=False)
        pd.DataFrame(metrics).to_csv(destination / 'summary.csv', index=False)
        save_json(destination / 'result.json', result)
        save_json(destination / 'consumed_input_sha256.json', source.consumed)

    stage_write(directory, ctx, write, runtime.guard)
    del source
    gc.collect()
    verify_inventory(lock['artifact_sha256'], runtime.guard)
    runtime.guard(); native_pressure()
    complete = out / 'execution_complete.json'
    if not complete.exists():
        artifacts = [p for name in ('moments', 'timing', 'years', 'analysis')
                     for p in (out / name).rglob('*') if p.is_file()
                     and not any(x.startswith('.') for x in p.relative_to(out).parts)]
        artifacts += [out / 'lock.json', out / 'protocol_freeze.json', out / 'financial_preflight.json']
        save_json(complete, dict(completed_at_utc=now(), status='COMPLETE_PENDING_INDEPENDENT_REVIEW',
            diagnostic_execution_complete=True, independent_review_complete=False, scientific_study_complete=False,
            models_fitted=3, monthly_rows=72, accepted_inputs_unchanged=True,
            artifact_sha256={str(p.relative_to(ROOT)): sha(p, runtime.guard) for p in artifacts}))
    else:
        verify_inventory(read(complete)['artifact_sha256'], runtime.guard)
    runtime.event('bounded_execution_complete', independent_review_pending=True,
                  full_study_authorized=False)


def _timeout(signum, frame):
    raise TimeoutError('Approved phase/cumulative awake-time limit reached; partial stages retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', action='store_true')
    choice.add_argument('--timing', action='store_true')
    choice.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = read(CONFIG); validate_config(config)
    out = ROOT / config['output_directory']
    if not out.is_dir():
        raise FileNotFoundError('Root must first create the frozen protocol and fresh preflight')
    with (out / '.execution.lock').open('a') as exclusive:
        fcntl.flock(exclusive.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            lock = prepare(config, out)
            print(json.dumps(dict(prepared=True, dependencies=len(lock['artifact_sha256']),
                                  coefficients_fitted=0, candidate_performance_computed=False)))
            return
        native_pressure()
        runtime = GuardedRuntime(out, config, 'pilot' if args.timing else 'development_probe')
        old_handler = signal.signal(signal.SIGALRM, _timeout)
        remaining = config['compute_budget']['total_active_seconds'] - runtime.previous
        if args.timing:
            remaining = min(remaining, config['compute_budget']['pilot_active_seconds'] - runtime.previous_phases.get('pilot', 0.))
        if remaining <= 0:
            runtime.close('STOPPED_RESOURCE', 'No approved time remains')
            raise TimeoutError('No approved time remains')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            lock = prepare(config, out, runtime.guard)
            runtime.event('lock_verified', candidate='THEME40')
            if args.timing:
                result = timing(config, out, lock, runtime)
                status = 'COMPLETE' if result['passed'] else 'STOPPED_TIMING_INCONCLUSIVE'
            else:
                run_probe(config, out, lock, runtime)
                status = 'COMPLETE'
            runtime.close(status)
        except BaseException as error:
            runtime.close('STOPPED_REVIEW_REQUIRED', error)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
