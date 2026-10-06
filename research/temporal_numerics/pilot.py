"""Approved, bounded local pilot; never dispatches a full historical experiment."""
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from baseline.full import preflight
from feature_geometry.history import sha
from temporal_extension.history import load_parent
from .optimizer import LBFGSBSettings as OptimizerSettings
from temporal_extension.training import save_result, _json_value
from .training import train_profile
from .cache import RetainedBatchCache as AnnualBatchCache
from temporal_study.history import StudyHistory
from .logging import DurableLog


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, default=_json_value, allow_nan=False)
        stream.write('\n')


def validate_config(config, base):
    """Only the approved numerical/cache revision may differ from v1 science."""
    revised = {'id', 'status', 'approval', 'optimizer', 'optimizer_method', 'cache_policy', 'pilot'}
    if ({k:v for k,v in config.items() if k not in revised}
            != {k:v for k,v in base.items() if k not in revised}):
        raise ValueError('Numerical revision cannot change the scientific comparison')
    expected = dict(lower=0., upper=2., tolerance=1e-5, max_steps=100, max_evaluations=250,
                    max_line_searches=16, history_size=10, relative_ftol=0., wall_seconds=3600.)
    if (config['optimizer'] != expected or config.get('optimizer_method') != 'L-BFGS-B'
            or config.get('cache_policy') != 'retained_prefix_stream_overflow'
            or config['approval'].get('numerical_revision_approved') is not True
            or config['approval'].get('full_historical_experiment') is not False
            or not config['approval'].get('user_confirmation', '').strip()):
        raise ValueError('Explicit approval for the exact numerical revision is required')
    if config['pilot'] != dict(base['pilot'], output_root='output/extension3_numerical_pilot_20260919'):
        raise ValueError('Only one additional approved first-year local hour is supported')


def run_pilot(root, destination, *, config_path, prior_attempts=()):
    root, destination, config_path = map(lambda p: Path(p).resolve(), (root, destination, config_path))
    if destination == root/'output' or not destination.is_relative_to(root/'output'):
        raise ValueError('Pilot outputs must use a new project output subdirectory')
    config = json.loads(config_path.read_text())
    if (config['approval']['historical_pilot'] is not True
            or config['approval']['paid_resources'] is not False
            or config['approval']['overwrite_existing_results'] is not False):
        raise ValueError('Recorded local pilot authorization is required')
    validate_config(config, json.loads((root/'config/extension3_study_v1.json').read_text()))
    proposal = config['pilot']
    if (proposal['year'], proposal['seed'], proposal['rho']) != (1981, 1, .01):
        raise ValueError('This entry point is restricted to the approved 1981 seed-1 pilot')
    if proposal['combined_wall_seconds'] > 3600:
        raise ValueError('Pilot exceeds the approved combined one-hour budget')
    resources = config['resources']
    pilot_root = (root/proposal['output_root']).resolve()
    if destination.parent != pilot_root or not destination.name.startswith('run'):
        raise ValueError('Use a new run directory within the approved revision pilot folder')
    required_prior = {p.resolve() for p in pilot_root.glob('run*') if p.is_dir()}
    if set(map(lambda p: Path(p).resolve(), prior_attempts)) != required_prior:
        raise ValueError('All previous attempts under this approval must be counted')
    prior_seconds = 0.
    prior_receipts = {}
    for prior in prior_attempts:
        prior = Path(prior).resolve()
        if not prior.is_relative_to(root/'output') or prior == destination or str(prior) in prior_receipts:
            raise ValueError('Distinct existing project pilot attempts required')
        previous_lock = json.loads((prior/'pilot_lock.json').read_text())
        if previous_lock['config'] != config:
            raise ValueError('Pilot retry cannot change approved scientific settings')
        result_path = prior/'pilot_result.json'
        if not result_path.exists():
            result_path = prior/'failed_attempt.json'
        prior_result = json.loads(result_path.read_text())
        spent = float(prior_result['seconds'])
        if not np.isfinite(spent) or spent < 0:
            raise ValueError('Invalid previous pilot time receipt')
        prior_seconds += spent
        prior_receipts[str(prior)] = dict(result_sha256=sha(result_path), seconds=spent)
    remaining_seconds = proposal['combined_wall_seconds'] - prior_seconds
    if remaining_seconds <= 0:
        raise ValueError('Approved pilot budget is exhausted across retained attempts')
    lock_file = (root/'output/.extension3_heavy_job.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except Exception:
        lock_file.close()
        raise
    started = time.monotonic()
    log = (destination/'progress.log').open('x', buffering=1)
    log_writer = DurableLog(log)
    process = psutil.Process()
    peak_rss = 0
    events = []
    next_storage_check = 0.

    def event(message, **values):
        nonlocal peak_rss
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        payload = dict(at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic()-started, event=message,
                       rss_gib=rss/2**30, **values)
        log_writer.emit(payload)
        events.append(payload)

    def guard():
        nonlocal next_storage_check
        if process.memory_info().rss > resources['process_memory_gib']*2**30:
            raise MemoryError('Pilot process exceeded its declared memory limit')
        now = time.monotonic()
        if now >= next_storage_check:
            next_storage_check = now + 15.
            if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) > resources['maximum_new_storage_gib']*2**30:
                raise OSError('Pilot outputs exceed the declared storage budget')
            if shutil.disk_usage(root).free < resources['minimum_free_disk_gib']*2**30:
                raise OSError('Free disk is below the declared reserve')
        if now-started >= remaining_seconds:
            raise TimeoutError('Combined local pilot time budget reached')

    try:
        guard()
        files = [config_path, root/'config/extension3_study_v1.json', root/'scripts/extension3_numerical_pilot.py']
        for package in ('temporal_extension', 'temporal_study', 'temporal_numerics', 'feature_geometry', 'geometry_study', 'baseline', 'inference'):
            files.extend(sorted((root/package).glob('*.py')))
        inventory = {str(p.relative_to(root)): sha(p) for p in files}
        snapshot = destination/'source_snapshot'
        for relative in inventory:
            target = snapshot/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/relative, target)
        write_new(destination/'pilot_lock.json', dict(config=config, code_sha256=inventory,
            prior_attempts=prior_receipts, prior_seconds=prior_seconds, remaining_seconds=remaining_seconds,
            created_at_utc=datetime.now(timezone.utc).isoformat(), full_run_authorized=False))
        event('financial_preflight_started')
        receipt, _ = preflight()
        write_new(destination/'financial_preflight.json', receipt)
        guard()
        event('financial_preflight_passed')
        parent = load_parent(root, proposal['year'], proposal['seed'], expected_cutoff='1980-12-31')
        history = StudyHistory(root, receipt, weights=config['reference_weights'])
        cache = AnnualBatchCache(history, maximum_bytes=int(resources['basis_cache_gib']*2**30))
        factory = cache.factory(parent)
        before = parent.fingerprint
        pass_number = 0

        def batches():
            nonlocal pass_number
            pass_number += 1
            this_pass = pass_number
            guard()
            event('training_pass_started', pass_number=this_pass)
            count = 0
            for count, batch in enumerate(factory(), 1):
                guard()
                if count == 1 or count % 25 == 0:
                    event('training_pass_progress', pass_number=this_pass, months=count,
                          last_decision=batch[0].sample.market.decision_date,
                          cache_gib=cache.retained_bytes/2**30)
                yield batch
            event('training_pass_finished', pass_number=this_pass, months=count)

        settings = OptimizerSettings(**config['optimizer'])
        settings = replace(settings, wall_seconds=max(.001, remaining_seconds-(time.monotonic()-started)))
        results = {}
        active_kind = 'temporal'
        def numerical_progress(value):
            event('optimizer_'+value['event'], model_kind=active_kind, optimizer=value)

        with threadpool_limits(limits=resources['threads']):
            event('temporal_fit_started', training_first=parent.metadata['training_first'],
                  training_last=parent.metadata['training_last'], training_months=parent.metadata['training_months'])
            model, result = train_profile(batches, parent.feature_map, parent.settings,
                expected_neutral_beta=parent.model.beta,
                expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
                parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
                reference_fingerprint=history.reference_fingerprint, rho=proposal['rho'],
                objective_scale=parent.objective_scale, optimizer_settings=settings, progress=numerical_progress)
            result.update(model_kind='learned_temporal', year=parent.year, seed=parent.seed,
                training_first=parent.metadata['training_first'], training_last=parent.metadata['training_last'],
                accepted_inputs_sha256=dict(history.files.consumed), parent_artifacts_sha256=dict(parent.hashes),
                parent_scale_optimizer_status=parent.metadata['status'],
                authorization_config_sha256=sha(config_path))
            results['temporal'] = save_result(destination/'temporal', model, result)
            event('temporal_fit_finished', status=result['status'], converged=result['converged'],
                  projected_gradient_norm=result['projected_gradient_norm'], evaluations=result['evaluation_count'])
            # A second historical control fit consumes only the remaining shared
            # pilot budget and never runs after a failed temporal convergence gate.
            if (result['converged'] and proposal['current_only_after_temporal_converges_if_budget_remains']
                    and remaining_seconds-(time.monotonic()-started) > 1):
                from temporal_study.controls import CurrentOnlyFeatureMap, save_current_only_result
                from .training import train_current_only_profile
                active_kind = 'current_only'
                settings = replace(settings, wall_seconds=remaining_seconds-(time.monotonic()-started))
                fmap = CurrentOnlyFeatureMap(parent.feature_map.parent, parent.feature_map.theta, parent.feature_map.groups)
                event('current_only_fit_started')
                def bases():
                    for basis, _ in batches():
                        yield basis
                current_model, current_result = train_current_only_profile(bases, fmap, parent.settings,
                    expected_neutral_beta=parent.model.beta,
                    expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
                    parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
                    rho=proposal['rho'], objective_scale=parent.objective_scale, optimizer_settings=settings, progress=numerical_progress)
                current_result.update(year=parent.year, seed=parent.seed,
                    training_first=parent.metadata['training_first'], training_last=parent.metadata['training_last'],
                    accepted_inputs_sha256=dict(history.files.consumed), parent_artifacts_sha256=dict(parent.hashes),
                    parent_scale_optimizer_status=parent.metadata['status'], authorization_config_sha256=sha(config_path))
                results['current_only'] = save_current_only_result(destination/'current_only', current_model, current_result)
                event('current_only_fit_finished', status=current_result['status'], converged=current_result['converged'])
        history.files.unchanged()
        if parent.fingerprint != before:
            raise ValueError('Frozen parent changed during pilot')
        for relative, digest in inventory.items():
            if sha(root/relative) != digest:
                raise ValueError('Pilot code/configuration changed during execution: '+relative)
        for relative, digest in history.files.consumed.items():
            if sha(root/relative) != digest:
                raise ValueError('Accepted pilot input changed during execution: '+relative)
        summary = dict(status='PILOT_COMPLETE', scientific_study_complete=False,
            full_historical_experiment_run=False, historical_performance_measured=False,
            seconds=time.monotonic()-started, prior_attempt_seconds=prior_seconds,
            combined_attempt_seconds=prior_seconds+time.monotonic()-started,
            peak_sampled_process_rss_gib=peak_rss/2**30,
            cache=cache.diagnostics(), all_attempted_fits_converged=all(r['converged'] for r in results.values()),
            numerical_results={name:{key:r.get(key) for key in (
                'status','converged','neutral_gate','projected_gradient_norm','accepted_steps',
                'evaluation_count','elapsed_seconds','solve_residual','gains','coverage')}
                for name,r in results.items()}, next_step='Review timing and convergence before full-run authorization')
        write_new(destination/'pilot_result.json', summary)
        event('pilot_completed', all_attempted_fits_converged=summary['all_attempted_fits_converged'])
        log.close()
        write_new(destination/'pilot_manifest.json', dict(status='COMPLETE_PILOT_ONLY',
            artifact_sha256={str(p.relative_to(root)):sha(p) for p in destination.rglob('*') if p.is_file()},
            parent_fingerprint=before, scientific_study_complete=False))
        return summary
    except Exception as exc:
        event('pilot_failed', error_type=type(exc).__name__, error=str(exc))
        write_new(destination/'failed_attempt.json', dict(status='FAILED_PRESERVED',
            error_type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started,
            historical_fit_may_have_started=True, full_historical_experiment_run=False))
        raise
    finally:
        if not log.closed:
            log.close()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
