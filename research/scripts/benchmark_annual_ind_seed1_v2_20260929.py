"""Seed-1 exact-reuse benchmark; no historical selection or portfolio results.

Geometry checks the zero, middle and final original ridge. Temporal fits use all
three existing restraints with paired reference/cache/cache/reference ordering.
The late fixture covers September-November 2019 for matrix-size measurement,
not a complete 2020 expanding-window fit. All outcomes are before its cutoff.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from geometry_study import kernel
from geometry_study.features import CachedFeatureMap
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_fast_v4.settings import VARIANTS
from temporal_extension.model import TemporalModel
from temporal_numerics.optimizer import optimize, LBFGSBSettings
from extension4.storage import verify_stage, write_json
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime
from annual_ind.numerics import ordered_months
from annual_ind.parallel_layers import ParallelTemporalObjective
from annual_ind.geometry_reuse import NeutralMoments, fit_full
from annual_ind.temporal_reuse import SharedTemporalProfiles
from annual_ind.pipeline_probe import plain
from baseline.full import preflight

OUT = ROOT/'output/annual_ind_seed1_speed_v2_20260929'


def same_fit(first, second, kind):
    fields = (('theta', 'beta', 'geometry_beta', 'trace', 'profile_evaluations',
               'solve_residual', 'final_training_value', 'status') if kind == 'geometry'
              else ('gains', 'beta', 'gradient', 'objective_trace', 'projected_gradient_trace',
                    'accepted_steps', 'evaluation_count', 'status', 'converged'))
    for key in fields:
        if key == 'trace':
            if first[key] != second[key]:
                raise ValueError('Geometry trace changed')
        else:
            np.testing.assert_array_equal(first[key], second[key], err_msg=kind+' '+key)


def main():
    lockfile = (ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    ledger_path = ROOT/'output/annual_ind_parallel_review_20260929/resource_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    from extension4.runtime import prior_seconds
    prior_attempt_seconds = prior_seconds(ROOT/'output/annual_ind_seed1_speed_20260929')[0]
    seconds = min(1500., ledger['remaining_engineering_seconds'] - prior_attempt_seconds - 120.)
    if seconds <= 0:
        raise TimeoutError('Engineering time allowance exhausted')
    limits = dict(total_active_seconds=seconds, pilot_active_seconds=seconds,
                  max_physical_footprint_gib=16., min_system_headroom_gib=4.,
                  max_additional_disk_gib=2., min_free_disk_gib=50.)
    reference = ROOT/'output/annual_ind_fresh_probe_pipeline_r50_20260929'
    old = json.loads((reference/'lock.json').read_text())
    for label in ('code_sha256', 'protected_sha256'):
        for rel, expected in old[label].items():
            if sha(ROOT/rel) != expected:
                raise ValueError('Previously verified source changed: '+rel)
    paths = [Path(__file__), ROOT/'annual_ind/geometry_reuse.py',
             ROOT/'annual_ind/temporal_reuse.py', ROOT/'config/annual_ind_seed1_20260929.json']
    write_json(OUT/'lock.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose=__doc__, seeds=[1], geometry_ridge_indices=[0, 50, 100],
        temporal_restraints=[.1, .01, .001], temporal_pair_order=['reference','reuse','reuse','reference'],
        late_fixture_decisions=['2019-09-30','2019-10-31','2019-11-30'],
        late_fixture_is_full_history=False, budget=limits,
        prior_resource_ledger=str(ledger_path.relative_to(ROOT)),
        prior_resource_ledger_sha256=sha(ledger_path), prior_charged_seconds=ledger['charged_seconds'],
        additional_prior_attempt_seconds=prior_attempt_seconds,
        code_sha256={**old['code_sha256'], **{str(p.relative_to(ROOT)):sha(p) for p in paths}},
        protected_sha256=old['protected_sha256'], full_study_started=False, AI_calls=False))
    runtime = None
    original_ordered = kernel.ordered
    try:
        runtime = ProbeRuntime(OUT, {'compute_budget':limits}, 'seed1_exact_reuse_benchmark')
        runtime.event('financial_preflight_started')
        receipt, _ = preflight()
        write_json(OUT/'financial_preflight.json', receipt)
        runtime.event('financial_preflight_complete')
        history = StudyHistory(ROOT, receipt, weights=[1/6]*6)
        preparation = ROOT/'output/step6_20260911/preparation_manifest.json'
        assert sha(preparation) == receipt['preparation_manifest_sha256']
        history.files.inventory.update(json.loads(preparation.read_text())['artifact_sha256'])
        kernel.ordered = lambda f, items: ordered_months(f, items, workers=8, guard=runtime.guard)
        grid = json.loads((ROOT/'output/step2_20260910/source_settings.json').read_text())['settings']['pf_ml']
        for stage in ('fresh_parent', 'fresh_temporal'):
            verify_stage(reference/stage)
        parent = GeometryModel.load(reference/'fresh_parent/model.npz')
        parent_record = json.loads((reference/'fresh_parent/fit.json').read_text())
        temporal_record = json.loads((reference/'temporal_attempt.json').read_text())
        temporal = TemporalModel.load(reference/'fresh_temporal/model.npz',
            expected_parent_fingerprint=temporal_record['parent_fingerprint'])
        settings = FitSettings(**parent_record['settings'])
        fmap = CachedFeatureMap(parent.feature_map.frequencies, parent.feature_map.source)
        optimizer = LBFGSBSettings(**temporal_record['optimizer_settings'])
        bases, batches = [], []
        prep_start = time.monotonic()
        for date in pd.date_range('1965-11-30', '1970-11-30', freq='ME'):
            runtime.guard()
            basis = kernel.LeanBasis(history.training_sample(str(date.date()), settings.cutoff))
            bound = history.innovations(basis.sample.market)
            batches.append((basis, history.validate_market(basis.sample.market, bound)))
            bases.append(basis)
        history.panels.release(); gc.collect()
        runtime.event('early_inputs_ready', months=len(bases), seconds=time.monotonic()-prep_start)
        geometry = []
        with threadpool_limits(limits=1), single_thread_blas():
            started = time.monotonic()
            neutral = NeutralMoments.build(bases, fmap, settings.cutoff)
            neutral_build_seconds = time.monotonic()-started
            write_json(OUT/'neutral_build.json', dict(seconds=neutral_build_seconds,
                identity=neutral.identity, bytes=neutral.score.nbytes+neutral.risk.nbytes+neutral.cost.nbytes))
            for ridge_index in ():
                ridge = float(grid['l_vec'][ridge_index])
                for reuse in (False, True):
                    runtime.event('geometry_started', ridge_index=ridge_index, reuse=reuse)
                    start = time.monotonic()
                    base_settings = replace(settings, ridge=ridge, scale_penalty=0.)
                    zero = (neutral.evaluate(bases, fmap, base_settings) if reuse else
                            kernel.evaluate(bases, fmap, np.zeros(115), base_settings))
                    fit_settings = replace(base_settings, scale_penalty=.01*zero['value'])
                    result = (fit_full(bases, fmap, fit_settings, neutral=neutral,
                                       objective_scale=zero['value']) if reuse else
                              kernel.fit(bases, fmap, fit_settings, objective_scale=zero['value']))
                    elapsed = time.monotonic()-start
                    if not reuse:
                        expected = result
                    else:
                        same_fit(expected, result, 'geometry')
                    write_json(OUT/f'geometry_r{ridge_index}_{int(reuse)}.json', plain(result))
                    row = dict(ridge_index=ridge_index, reuse=reuse, seconds=elapsed,
                               status=result['status'], includes_normalizer_solve=True,
                               identical_to_reference=bool(reuse))
                    geometry.append(row); runtime.event('geometry_complete', **row)
            temporal_trials = []
            for ordinal, reuse in enumerate((False, True, True, False)):
                def factory():
                    for batch in batches:
                        runtime.guard()
                        yield batch
                kwargs = dict(objective_scale=parent_record['objective_normalizer'],
                              workers=8, trig_backend='accelerate', **dict(VARIANTS))
                bank = (SharedTemporalProfiles(factory, temporal.feature_map, settings,
                         maximum_bytes=128*2**20, guard=runtime.guard, **kwargs) if reuse else None)
                for rho in (.1, .01, .001):
                    runtime.event('temporal_started', ordinal=ordinal, rho=rho, reuse=reuse)
                    objective = (bank.for_restraint(rho) if reuse else
                        ParallelTemporalObjective(factory, temporal.feature_map, settings, rho=rho, **kwargs))
                    start = time.monotonic()
                    result = optimize(objective, 13, optimizer)
                    write_json(OUT/f'raw_temporal_{ordinal}_{rho}.json', plain(result))
                    elapsed = time.monotonic()-start
                    name = str(rho).replace('.', 'p')
                    if ordinal == 0:
                        write_json(OUT/f'temporal_reference_{name}.json', plain(result))
                    else:
                        expected = json.loads((OUT/f'temporal_reference_{name}.json').read_text())
                        same_fit(expected, result, 'temporal')
                    if not result['converged']:
                        write_json(OUT/f'temporal_failed_{ordinal}_{name}.json', plain(result))
                        raise ValueError('Candidate failed convergence; no historical launch or candidate filtering')
                    if rho == .1:
                        for key in ('gains', 'beta', 'objective_trace', 'projected_gradient_trace'):
                            np.testing.assert_array_equal(result[key], temporal_record[key])
                    write_json(OUT/f'temporal_{ordinal}_{name}.json', plain(result))
                    row = dict(ordinal=ordinal, rho=rho, reuse=reuse, seconds=elapsed,
                               identical_to_reference=ordinal != 0,
                               evaluations=result['evaluation_count'], updates=result['accepted_steps'],
                               cache=None if bank is None else bank.diagnostics())
                    temporal_trials.append(row); runtime.event('temporal_complete', **row)
                del bank
                gc.collect()
            write_json(OUT/'early_results.json', dict(geometry=geometry, temporal=temporal_trials,
                                                      neutral_build_seconds=neutral_build_seconds))
            # Size/throughput fixture only; never label this as a full 2020 fit.
            del bases, batches, neutral
            history.panels.release(); gc.collect()
            late = []
            for date in pd.date_range('2019-09-30', '2019-11-30', freq='ME'):
                runtime.guard()
                start = time.monotonic()
                basis = kernel.LeanBasis(history.training_sample(str(date.date()), '2019-12-31'))
                late.append(basis)
                runtime.event('late_month_prepared', date=str(date.date()), n=len(basis.sample.market.ids),
                              seconds=time.monotonic()-start)
            history.panels.release(); gc.collect()
            late_settings = replace(settings, cutoff='2019-12-31', scale_penalty=0.)
            start = time.monotonic()
            late_neutral = NeutralMoments.build(late, fmap, late_settings.cutoff)
            late_build = time.monotonic()-start
            late_rows = []
            for reuse in (False, True):
                start = time.monotonic()
                zero = (late_neutral.evaluate(late, fmap, late_settings) if reuse else
                        kernel.evaluate(late, fmap, np.zeros(115), late_settings))
                ss = replace(late_settings, scale_penalty=.01*zero['value'])
                result = (fit_full(late, fmap, ss, neutral=late_neutral, objective_scale=zero['value'])
                          if reuse else kernel.fit(late, fmap, ss, objective_scale=zero['value']))
                elapsed = time.monotonic()-start
                if reuse:
                    same_fit(late_original, result, 'geometry')
                else:
                    late_original = result
                write_json(OUT/f'late_geometry_{int(reuse)}.json', plain(result))
                late_rows.append(dict(reuse=reuse, seconds=elapsed, identical_to_reference=reuse))
            history.files.unchanged()
            for rel, expected in old['protected_sha256'].items():
                assert sha(ROOT/rel) == expected, rel
            summary = dict(status='PASS_SINGLE_SEED_EXACT_REUSE_BENCHMARK_ONLY',
                geometry=geometry, temporal=temporal_trials, neutral_build_seconds=neutral_build_seconds,
                late_geometry=late_rows, late_neutral_build_seconds=late_build,
                early_training_months=61, late_fixture_months=3,
                late_fixture_is_full_history=False, late_security_counts=[len(b.sample.market.ids) for b in late],
                new_portfolio_paths=0, full_study_started=False,
                complete_historical_runtime_established=False,
                active_seconds=time.monotonic()-runtime.start)
            write_json(OUT/'result.json', summary)
            runtime.event('benchmark_complete', status=summary['status'])
        runtime.close('COMPLETE')
    except BaseException as error:
        if runtime is not None:
            runtime.close('STOPPED', error)
        write_json(OUT/'stop.json', dict(status='STOPPED', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        kernel.ordered = original_ordered


if __name__ == '__main__':
    main()
