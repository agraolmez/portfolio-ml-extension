"""One numerical concurrency/width benchmark; no portfolio run.

The pilot coordinate is mechanical: first bandwidth, maximum released width,
middle ridge index, first validation year and seed 1. No historical performance
selects this coordinate. Accepted old coefficients are a neutral oracle only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from geometry_study.features import CachedFeatureMap
from geometry_study import kernel
from temporal_fast_v4.history import StudyHistory
from extension4.storage import commit_stage, write_json
from extension4.runtime import Runtime, prior_seconds
from extension4_execution.memory_v2 import pressure_level
from temporal_fast.backend import single_thread_blas


class ProbeRuntime(Runtime):
    def sample(self):
        if pressure_level() != 1:
            raise MemoryError('Native memory pressure is not NORMAL')
        return super().sample()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=[64,128,256,512], required=True)
    parser.add_argument('--workers', type=int, choices=[1,2], required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():
        raise ValueError('Fresh isolated project output required')
    process_lock = (ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit_dir = ROOT/'output/annual_ind_audit_v3_20260929'
    audit = json.loads((audit_dir/'audit.json').read_text())
    if audit['status'] != 'AUDIT_AND_SELECTOR_REGRESSION_PASS_NOT_A_HISTORICAL_RUN':
        raise ValueError('Passing financial/source audit required')
    for rel, expected in audit['input_and_code_sha256'].items():
        if sha(ROOT/rel) != expected: raise ValueError('Audited dependency changed: '+rel)
    receipt = json.loads((audit_dir/'financial_preflight.json').read_text())
    settings_doc = json.loads((ROOT/'output/step2_20260910/source_settings.json').read_text())
    grid = settings_doc['settings']['pf_ml']
    width = args.width; ridge_index = len(grid['l_vec'])//2
    ridge = float(grid['l_vec'][ridge_index]); year, seed, gi = 1971, 1, 0
    cutoff = '1970-12-31'
    charged_attempts = {str(p.relative_to(ROOT)):prior_seconds(p)[0]
        for p in (ROOT/'output').glob('annual_ind_fresh_probe*') if (p/'lock.json').exists()}
    remaining_seconds = 7200.-sum(charged_attempts.values())
    if remaining_seconds<=0: raise TimeoutError('Two-hour cumulative numerical probe allowance exhausted')
    resource_limits = dict(total_active_seconds=remaining_seconds, pilot_active_seconds=remaining_seconds,
        max_physical_footprint_gib=16., min_system_headroom_gib=4.,
        max_additional_disk_gib=2., min_free_disk_gib=50.)
    out.mkdir()
    protected = ['GOAL.md','docs/STEP_1_THEORY.md','docs/RECONSTRUCTION_AMENDMENT.md',
                 'config/protocol_v1.json','config/reconstruction_amendment.json']
    source_paths = [Path(__file__), ROOT/'annual_ind/selection.py']
    source_paths += [p for folder in ['feature_geometry','geometry_study','temporal_fast_v4',
        'temporal_extension','temporal_study','extension4','extension4_execution']
        for p in (ROOT/folder).glob('*.py')]
    lock = dict(status='LOCKED_NUMERICAL_FEASIBILITY_PROBE_NOT_FULL_STUDY',
        at_utc=datetime.now(timezone.utc).isoformat(), year=year, seed=seed,
        bandwidth_index=gi, feature_count=width, ridge_index=ridge_index, ridge=ridge,
        coordinate_rule='first bandwidth, enumerated original width, middle ridge index, first year/seed',
        user_authorization=['Let\'s run our best model from scratch as they did and see what happens',
                            'Do exactly what they did in the paper'],
        no_proposed_1966_1970_startup=True, old_coefficients_oracle_only=True,
        numerical_threads=1, geometry_month_workers=args.workers, budget=resource_limits,
        charged_prior_probe_attempt_seconds=charged_attempts,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        audit_sha256=sha(audit_dir/'audit.json'),
        protected_sha256={p:sha(ROOT/p) for p in protected},
        source_accounting='accepted corrected wealth; timing probe, not paper-accounting evaluation')
    write_json(out/'lock.json', lock)
    runtime = None
    try:
        runtime = ProbeRuntime(out, {'compute_budget':resource_limits}, 'pilot')
        history = StudyHistory(ROOT, receipt, weights=[1/6]*6)
        preparation_path = ROOT/'output/step6_20260911/preparation_manifest.json'
        if sha(preparation_path)!=receipt['preparation_manifest_sha256']:
            raise ValueError('Early baseline oracle preparation differs from financial preflight')
        # Early coefficient files are retained under step6 symlink targets;
        # their original acceptance is required rather than treating them as new.
        history.files.inventory.update(json.loads(preparation_path.read_text())['artifact_sha256'])
        bases = []
        for i,date in enumerate(pd.date_range('1965-11-30','1970-11-30',freq='ME'),1):
            runtime.guard()
            bases.append(kernel.LeanBasis(history.training_sample(str(date.date()), cutoff)))
            if i==1 or i%12==0: runtime.event('fresh_training_inputs', months=i, date=str(date.date()))
        history.panels.release(); gc.collect()
        original_ordered = kernel.ordered
        def guarded_ordered(function, items):
            def one(item):
                single_thread_blas()
                runtime.guard()
                result = function(item)
                runtime.guard()
                return result
            if args.workers == 1:
                for item in items:
                    yield one(item)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    iterator = iter(items)
                    pending = []
                    for _ in range(args.workers):
                        item = next(iterator, None)
                        if item is not None: pending.append(pool.submit(one,item))
                    while pending:
                        future = pending.pop(0)
                        result = future.result()
                        item = next(iterator, None)
                        if item is not None: pending.append(pool.submit(one,item))
                        yield result
        # Preserve chronological aggregation and every numerical equation. This
        # isolated process merely admits one or two bounded monthly calculations at a time, in original order.
        kernel.ordered = guarded_ordered
        directions_rel = f'output/feature_geometry_study_20260914/directions/seed{seed}_g{gi}.bin'
        directions = history.files.path(directions_rel)
        frequencies = np.fromfile(directions,dtype='<f8').reshape(115,512)[:,:width//2]
        fmap = CachedFeatureMap(frequencies,'fresh_probe_original_draw_'+sha(directions))
        geometry_plan_path = ROOT/'config/feature_geometry_study_v1.json'
        gp = json.loads(geometry_plan_path.read_text())['geometry_settings']
        settings = FitSettings(cutoff, ridge=ridge, scale_penalty=0.,
            **{k:v for k,v in gp.items() if k!='relative_scale_penalty'})
        started = time.monotonic()
        with threadpool_limits(limits=1):
            neutral_started = time.monotonic()
            zero = kernel.evaluate(bases,fmap,np.zeros(115),settings)
            neutral_seconds = time.monotonic()-neutral_started
            coefficient_rel = f'data/prepared/step9_v1/baseline/coefficients/g0/1971_p{width}.npz'
            with np.load(history.files.path(coefficient_rel),allow_pickle=False) as oracle:
                expected = oracle['coefficients'][:,ridge_index]
            np.testing.assert_allclose(zero['beta'],expected,atol=1e-9,rtol=1e-8)
            normalizer = float(zero['value'])
            if not np.isfinite(normalizer) or normalizer<=0: raise ValueError('Nonpositive normalizer')
            settings = replace(settings,scale_penalty=gp['relative_scale_penalty']*normalizer)
            runtime.event('neutral_fresh_solve_passed',seconds=neutral_seconds,
                          beta_max_error=float(np.max(abs(zero['beta']-expected))))
            fit_started = time.monotonic()
            result = kernel.fit(bases,fmap,settings,objective_scale=normalizer,
                progress=lambda row: runtime.event('fresh_scale_update',**row))
            fit_seconds = time.monotonic()-fit_started
        kernel.ordered = original_ordered
        source_key = fingerprint(*(b.sample.fingerprint for b in bases))
        model = GeometryModel(fmap,result['theta'],result['beta'],cutoff,settings.bound,source_key)
        context = dict(lock_sha256=sha(out/'lock.json'),year=year,seed=seed,gi=gi,
                       p=width,ridge_index=ridge_index)
        def writer(folder):
            model.save(folder/'model.npz')
            loaded = GeometryModel.load(folder/'model.npz')
            np.testing.assert_array_equal(loaded.theta,model.theta)
            np.testing.assert_array_equal(loaded.beta,model.beta)
            write_json(folder/'fit.json',clean(dict(result,settings=asdict(settings),
                objective_normalizer=normalizer,training_months=len(bases),
                training_first='1965-11-30',training_last='1970-11-30',
                source_fingerprint=source_key,inputs_sha256=dict(history.files.consumed))))
        commit_stage(out/'fresh_parent',context,writer)
        history.files.unchanged()
        for rel,expected_hash in lock['protected_sha256'].items():
            if sha(ROOT/rel)!=expected_hash:raise ValueError('Protected source changed')
        result_summary = dict(status='FRESH_E2_FIT_PROBE_COMPLETE_NOT_FULL_PIPELINE_PILOT',
            numerical_only=True,new_fits=1,new_portfolio_paths=0,validation_returns_scored=0,
            year=year,seed=seed,p=width,workers=args.workers,ridge_index=ridge_index,training_months=len(bases),
            neutral_seconds=neutral_seconds,geometry_fit_seconds=fit_seconds,
            fitting_elapsed_seconds=time.monotonic()-started,
            geometry_status=result['status'],profile_evaluations=result['profile_evaluations'],
            optimizer_updates=len(result['trace'])-1,solve_residual=result['solve_residual'],
            this_width_full_grid_parent_fits=50*3*2*len(grid['l_vec']),
            this_width_parent_hours_at_probe_rate=fit_seconds*(50*3*2*len(grid['l_vec']))/3600,
            total_all_width_parent_fits=50*3*2*len(grid['l_vec'])*len(grid['p_vec']),
            projection_limitation='Scenario at one early-window throughput for this width, not a reliable full-run ETA or lower bound; excludes other widths, temporal fits and conditional layers',
            total_active_seconds=time.monotonic()-runtime.start,
            fresh_model_sha256=sha(out/'fresh_parent/model.npz'))
        write_json(out/'probe_result.json',result_summary)
        runtime.event('probe_complete',**result_summary)
        runtime.close('COMPLETE')
    except BaseException as exc:
        if runtime is not None: runtime.close('STOPPED',exc)
        write_json(out/'stop.json',dict(status='STOPPED',error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
