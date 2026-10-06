"""Version-2 numerical wiring for the preserved complete temporal study.

Versioned separately so the first workflow and pilot source inventories stay intact.

Importing this module performs no work.  A full run needs a separate approval
receipt binding the exact configuration, a successful local pilot and a total
wall-clock budget.  The pilot configuration alone never authorizes this entry
point. Completed artifacts may be verified and reused. A separately approved
budget extension can resume a stop between complete stages. It does not retry
partial or diagnostic fits: a budget interruption during gain evaluation can
be retained as an ineligible evaluation-failure receipt, which still stops the
complete comparison family and requires a separately reviewed retry design.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from baseline.full import preflight
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from temporal_extension.history import load_parent, PARENT
from temporal_extension.model import TemporalModel
from .optimizer import LBFGSBSettings as OptimizerSettings
from .pilot import validate_config
import scipy
from temporal_extension.training import save_result, _json_value
from .training import train_profile, train_current_only_profile
from .cache import RetainedBatchCache as AnnualBatchCache
from temporal_study.controls import (CurrentOnlyFeatureMap, CurrentOnlyModel,
                       fit_fixed_profile, save_current_only_result, save_fixed_result)
from temporal_study.paths import AnnualPolicy, AcceptedPathSource, run_paths
from temporal_study.selection import Candidate, CandidatePath, SelectionPlan, select_candidates


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, default=_json_value, allow_nan=False)
        stream.write('\n')


def _json(path):
    return json.loads(Path(path).read_text())


def _inside(root, value, *, output=False):
    root = Path(root).resolve()
    path = Path(value)
    path = (root/path).resolve() if not path.is_absolute() else path.resolve()
    base = root/'output' if output else root
    if path == base or not path.is_relative_to(base):
        raise ValueError('A file or child directory inside the authorized project is required')
    return path


def _verify(root, inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('A nonempty immutable artifact inventory is required')
    for relative, digest in inventory.items():
        path = _inside(root, relative)
        if sha(path) != digest:
            raise ValueError('Changed locked dependency: '+relative)


def _merge_inventories(*inventories):
    """Preserve dependencies from reused stages as well as current reads."""
    result = {}
    for inventory in inventories:
        for path, digest in inventory.items():
            if path in result and result[path] != digest:
                raise ValueError('Accepted input hash differs between stage receipts: '+path)
            result[path] = digest
    return result


def _config(config):
    """Require exactly the approved numerical revision and preserved science."""
    base = _json(Path(__file__).resolve().parents[1]/'config/extension3_study_v1.json')
    validate_config(config, base)
    return OptimizerSettings(**config['optimizer'])


def verify_approval(root, config_path, approval_path):
    """Validate full-run permission before opening any historical fit inputs.

    The approval is written only after the user approves the concrete study and
    pilot review. A pilot receipt with no temporal fit, failed neutrality, or
    any attempted nonconverged fit cannot authorize a full historical run.
    """
    root = Path(root).resolve()
    config_path, approval_path = (_inside(root, p) for p in (config_path, approval_path))
    config, approval = _json(config_path), _json(approval_path)
    _config(config)
    if (approval.get('full_historical_experiment') is not True
            or approval.get('paid_resources') is not False
            or approval.get('overwrite_existing_results') is not False
            or approval.get('config_sha256') != sha(config_path)
            or not isinstance(approval.get('user_confirmation'), str)
            or not approval['user_confirmation'].strip()):
        raise ValueError('Explicit full-run approval for this exact configuration is required')
    seconds = approval.get('maximum_total_wall_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('An explicitly approved positive total wall-clock budget is required')
    pilot_path = _inside(root, approval['pilot_result_path'], output=True)
    manifest_path = _inside(root, approval['pilot_manifest_path'], output=True)
    if (sha(pilot_path) != approval['pilot_result_sha256']
            or sha(manifest_path) != approval['pilot_manifest_sha256']):
        raise ValueError('Approved pilot result or manifest changed')
    pilot, manifest = _json(pilot_path), _json(manifest_path)
    if (pilot.get('status') != 'PILOT_COMPLETE'
            or pilot.get('all_attempted_fits_converged') is not True
            or manifest.get('status') != 'COMPLETE_PILOT_ONLY'):
        raise ValueError('A successfully completed and converged pilot is required')
    results = pilot.get('numerical_results', {})
    if 'temporal' not in results:
        raise ValueError('The approved pilot must include the temporal model')
    for result in results.values():
        residual = result.get('solve_residual')
        if (result.get('converged') is not True or result.get('status') != 'converged'
                or result.get('neutral_gate', {}).get('passed') is not True
                or not isinstance(residual, (int, float)) or not math.isfinite(residual)
                or not 0 <= residual <= 1e-9):
            raise ValueError('Pilot convergence, neutrality or conditional solve gate failed')
    for kind in results:
        fit_path = pilot_path.parent/kind/'fit.json'
        fit = _json(fit_path)
        if (fit.get('optimizer_method') != 'L-BFGS-B'
                or fit.get('scipy_version') != scipy.__version__):
            raise ValueError('Pilot fitter or SciPy version differs from the requested numerical revision')
        settings = fit.get('optimizer_settings', {})
        if ({k:v for k,v in settings.items() if k != 'wall_seconds'}
                != {k:v for k,v in config['optimizer'].items() if k != 'wall_seconds'}):
            raise ValueError('Pilot numerical settings differ from the requested revision')
        if manifest['artifact_sha256'].get(str(fit_path.relative_to(root))) != sha(fit_path):
            raise ValueError('Pilot fit is absent from its completion inventory')
    _verify(root, manifest['artifact_sha256'])
    if manifest['artifact_sha256'].get(str(pilot_path.relative_to(root))) != sha(pilot_path):
        raise ValueError('Pilot result is absent from its completion inventory')
    pilot_lock_path = pilot_path.parent/'pilot_lock.json'
    lock = _json(pilot_lock_path)
    if lock.get('config') != config:
        raise ValueError('Pilot and requested full study use different scientific settings')
    _verify(root, lock['code_sha256'])
    return config, approval


def candidate_name(rho):
    return 'rho_'+format(float(rho), '.12g').replace('.', 'p')


def fit_specs(config, selections=None):
    """Return the declared jobs, without opening data or launching a fit."""
    split = pd.Timestamp(config['selection']['last_return']).year
    rows = []
    for year in range(config['first_return_year'], config['last_return_year']+1):
        for seed in config['frequency_seeds']:
            for kind in ('temporal', 'current_only'):
                if year <= split:
                    penalties = config['penalties_in_tie_order']
                elif selections is None:
                    continue
                else:
                    choices = [r for r in selections[kind]['selected_by_seed'] if r['seed'] == seed]
                    if len(choices) != 1 or choices[0]['parameters']['rho'] not in config['penalties_in_tie_order']:
                        raise ValueError('A uniquely selected declared penalty is required for each seed')
                    penalties = [choices[0]['parameters']['rho']]
                for rho in penalties:
                    rows.append(dict(year=year, seed=seed, kind=kind, rho=float(rho), candidate=candidate_name(rho)))
            if year <= split or selections is not None:
                rows.append(dict(year=year, seed=seed, kind='fixed_temporal', rho=0., candidate='fixed_temporal'))
    return rows


def _source_files(root, config_path):
    files = [config_path, root/'scripts/extension3_numerical_pilot.py', root/'scripts/extension3_numerical_study.py',
             root/'scripts/verify_extension3_inference.py', root/'scripts/extension3_inference_reference.R']
    for package in ('temporal_extension', 'temporal_study', 'temporal_numerics', 'feature_geometry',
                    'geometry_study', 'baseline', 'inference'):
        files.extend((root/package).glob('*.py'))
    files.extend([root/'config/extension3_temporal_audit_v1.json', root/'config/protocol_v1.json',
                  root/'config/extension3_study_v1.json'])
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(files))}


def _attempt_seconds(destination):
    """Conservatively include an interrupted attempt's entire elapsed time."""
    spent = 0.
    for start_path in sorted((destination/'attempts').glob('*_start.json')):
        start = _json(start_path)
        end_path = start_path.with_name(start_path.name.replace('_start.json', '_end.json'))
        if end_path.exists():
            end = _json(end_path)
            if end.get('start_sha256') != sha(start_path):
                raise ValueError('Changed prior attempt budget receipt')
            duration = end['seconds']
        else:
            then = datetime.fromisoformat(start['started_at_utc'])
            duration = (datetime.now(timezone.utc)-then).total_seconds()
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid prior full-study elapsed-time receipt')
        spent += duration
    return spent


def _budget_authorizations(root, destination, config_path, approval_path, approval,
                           extension_paths=(), *, allow_new=True):
    """Verify and append explicit budget-only approvals without altering a lock.

    Locked extensions are inherited on resume even when no CLI flag is given.
    Supplying an already locked receipt again is harmless; changing its bytes
    or omitting a link in a newly supplied chain fails closed. Existing attempt
    durations remain untouched. The ceiling is cumulative, never a fresh budget
    for each attempt. This function itself grants no experiment permission and
    cannot make a partial or nonconverged fit eligible, or authorize its retry.
    """
    root = Path(root).resolve()
    destination = _inside(root, destination, output=True)
    config_path, approval_path = (_inside(root, p) for p in (config_path, approval_path))
    approval_hash, config_hash = sha(approval_path), sha(config_path)
    if (_json(approval_path) != approval
            or approval.get('full_historical_experiment') is not True
            or approval.get('paid_resources') is not False
            or approval.get('overwrite_existing_results') is not False
            or approval.get('config_sha256') != config_hash
            or not isinstance(approval.get('user_confirmation'), str)
            or not approval['user_confirmation'].strip()):
        raise ValueError('An unchanged original full-run approval is required before extending its budget')
    ceiling = approval.get('maximum_total_wall_seconds')
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('The original authorized total wall-clock ceiling is invalid')
    lock_path = destination/'study_lock.json'
    lock = _json(lock_path)
    if lock.get('approval_sha256') != approval_hash or lock.get('config') != _json(config_path):
        raise ValueError('Budget extension must preserve the original study lock and scientific configuration')
    lock_hash = sha(lock_path)
    previous_hash = approval_hash
    journal = destination/'budget_authorizations'
    record_paths = sorted(journal.glob('*.json')) if journal.exists() else []
    if [p.name for p in record_paths] != [f'{i:04d}.json' for i in range(1, len(record_paths)+1)]:
        raise ValueError('Budget authorization journal has a missing or reordered record')
    required_fields = {'extend_full_run_budget', 'original_approval_sha256', 'config_sha256',
        'previous_authorization_sha256', 'previous_maximum_total_wall_seconds',
        'maximum_total_wall_seconds', 'user_confirmation', 'paid_resources',
        'overwrite_existing_results', 'scientific_settings_unchanged'}
    records, known, new = [], {}, []

    def consume(path, body, digest):
        nonlocal ceiling, previous_hash
        if (not isinstance(body, dict) or set(body) != required_fields
                or body.get('extend_full_run_budget') is not True
                or body.get('original_approval_sha256') != approval_hash
                or body.get('config_sha256') != config_hash
                or body.get('previous_authorization_sha256') != previous_hash
                or body.get('previous_maximum_total_wall_seconds') != ceiling
                or body.get('paid_resources') is not False
                or body.get('overwrite_existing_results') is not False
                or body.get('scientific_settings_unchanged') is not True
                or not isinstance(body.get('user_confirmation'), str)
                or not body['user_confirmation'].strip()):
            raise ValueError('Explicit chained budget-only approval with unchanged science is required')
        next_ceiling = body['maximum_total_wall_seconds']
        old_ceiling = body['previous_maximum_total_wall_seconds']
        if (isinstance(old_ceiling, bool) or not isinstance(old_ceiling, (int, float))
                or not math.isfinite(old_ceiling)
                or isinstance(next_ceiling, bool) or not isinstance(next_ceiling, (int, float))
                or not math.isfinite(next_ceiling) or next_ceiling <= ceiling):
            raise ValueError('An extension must explicitly authorize a strictly larger cumulative ceiling')
        record = dict(study_lock_sha256=lock_hash, original_approval_sha256=approval_hash,
                      config_sha256=config_hash, receipt_path=str(path.relative_to(root)),
                      receipt_sha256=digest, receipt=body)
        records.append(record)
        known[path] = record
        ceiling, previous_hash = next_ceiling, digest
        return record

    for saved in record_paths:
        record = _json(saved)
        path = _inside(root, record['receipt_path'])
        if path in known:
            raise ValueError('Duplicate locked budget-authorization receipt')
        digest = sha(path)
        if digest != record['receipt_sha256'] or _json(path) != record['receipt']:
            raise ValueError('A locked budget-extension receipt changed')
        expected = consume(path, record['receipt'], digest)
        if expected != record:
            raise ValueError('Budget authorization belongs to another study lock or chain')
    requested = [_inside(root, p) for p in extension_paths]
    if len(set(requested)) != len(requested):
        raise ValueError('Duplicate requested budget-extension paths')
    for path in requested:
        digest = sha(path)
        if path in known:
            if digest != known[path]['receipt_sha256']:
                raise ValueError('A locked budget-extension receipt changed')
            continue
        if not allow_new:
            raise ValueError('A numerically completed study does not accept additional execution budget')
        new.append(consume(path, _json(path), digest))
    # Validate the whole requested chain before persisting any newly approved
    # link. Existing authority files, receipts and completed fits are untouched.
    if new:
        journal.mkdir(exist_ok=True)
        for number, record in enumerate(new, len(record_paths)+1):
            write_new(journal/f'{number:04d}.json', record)
    return dict(original_approval_sha256=approval_hash,
                original_maximum_total_wall_seconds=approval['maximum_total_wall_seconds'],
                maximum_total_wall_seconds=ceiling, latest_authorization_sha256=previous_hash,
                extension_receipts=[dict(receipt_path=r['receipt_path'], receipt_sha256=r['receipt_sha256'])
                                    for r in records],
                authorization_receipts_sha256={r['receipt_path']:r['receipt_sha256'] for r in records})


class Runtime:
    def __init__(self, root, destination, config, approval, log, prior_seconds):
        self.root, self.destination, self.config, self.approval = root, destination, config, approval
        self.log, self.prior_seconds = log, prior_seconds
        self.start = time.monotonic()
        self.process = psutil.Process()
        self.peak_rss = 0
        self.next_storage_check = 0.

    def event(self, event, **values):
        value = dict(at_utc=datetime.now(timezone.utc).isoformat(), event=event,
                     attempt_seconds=time.monotonic()-self.start,
                     cumulative_seconds=self.prior_seconds+time.monotonic()-self.start, **values)
        self.log.write(json.dumps(value, default=_json_value, allow_nan=False)+'\n')
        self.log.flush()
        print(json.dumps(value, default=_json_value, allow_nan=False), flush=True)

    def guard(self, force=False):
        now = time.monotonic()
        if self.prior_seconds + now-self.start >= self.approval['maximum_total_wall_seconds']:
            raise TimeoutError('Approved total full-study wall-clock budget exhausted across attempts')
        rss = self.process.memory_info().rss
        self.peak_rss = max(rss, self.peak_rss)
        resources = self.config['resources']
        if rss > resources['process_memory_gib']*2**30:
            raise MemoryError('Full-study process exceeded its declared memory limit')
        if force or now >= self.next_storage_check:
            self.next_storage_check = now+15.
            if shutil.disk_usage(self.root).free < resources['minimum_free_disk_gib']*2**30:
                raise OSError('Free disk is below the approved reserve')
            size = sum(p.stat().st_size for p in self.destination.rglob('*') if p.is_file())
            if size > resources['maximum_new_storage_gib']*2**30:
                raise OSError('Full-study artifacts exceeded their approved storage budget')


def _fit_folder(destination, spec):
    return destination/'fits'/str(spec['year'])/f"seed{spec['seed']}"/spec['kind']/spec['candidate']


def _load_fit(root, folder, spec, parent, lock_sha256, config):
    if not folder.exists():
        return None
    commit_path = folder/'fit_commit.json'
    if not commit_path.exists():
        raise ValueError('Preserved partial fit requires a separately reviewed retry: '+str(folder))
    commit = _json(commit_path)
    if commit.get('study_lock_sha256') != lock_sha256 or commit.get('fit_spec') != spec:
        raise ValueError('Saved fit belongs to a different locked experiment')
    _verify(root, commit['artifact_sha256'])
    receipt = _json(folder/'fit.json')
    if (receipt.get('objective_scale') != parent.objective_scale
            or receipt.get('parent_settings') != asdict(parent.settings)
            or receipt.get('parent_artifacts_sha256') != dict(parent.hashes)
            or receipt.get('parent_scale_optimizer_status') != parent.metadata['status']):
        raise ValueError('Saved fit objective normalization or frozen parent authority differs')
    if spec['kind'] != 'fixed_temporal' and receipt.get('optimizer_settings') != config['optimizer']:
        raise ValueError('Saved fit optimizer settings differ from the approved configuration')
    if spec['kind'] != 'fixed_temporal' and (receipt.get('optimizer_method') != 'L-BFGS-B'
            or receipt.get('scipy_version') != scipy.__version__):
        raise ValueError('Saved fitting method or SciPy version differs from the numerical revision')
    if spec['kind'] == 'fixed_temporal' and not np.array_equal(
            receipt.get('prescribed_gains'), np.full(parent.feature_map.gain_count, config['fixed_temporal_gain'])):
        raise ValueError('Saved fixed temporal gains differ from the approved configuration')
    if spec['kind'] == 'fixed_temporal' and (
            not np.array_equal(receipt.get('lower'), np.full(parent.feature_map.gain_count, config['optimizer']['lower']))
            or not np.array_equal(receipt.get('upper'), np.full(parent.feature_map.gain_count, config['optimizer']['upper']))):
        raise ValueError('Saved fixed temporal bounds differ from the approved configuration')
    if receipt.get('eligible_for_portfolio_evaluation') is not True:
        raise ValueError('A nonconverged candidate stops the complete comparison family: '+str(folder))
    model_path = folder/receipt['model_file']
    if model_path.name != 'model.npz' or sha(model_path) != receipt['model_sha256']:
        raise ValueError('Completed fit model integrity failed')
    model_type = CurrentOnlyModel if spec['kind'] == 'current_only' else TemporalModel
    model = model_type.load(model_path, expected_parent_fingerprint=parent.fingerprint)
    kind = 'fixed_temporal' if spec['kind'] == 'fixed_temporal' else 'learned_'+spec['kind']
    AnnualPolicy(parent, kind=kind, model=model, receipt=receipt,
                 prescribed_gains=model.gains if kind == 'fixed_temporal' else None)
    if receipt.get('rho') != spec['rho'] or receipt.get('study_lock_sha256') != lock_sha256:
        raise ValueError('Saved candidate penalty or study identity differs')
    if receipt.get('fit_spec') != spec:
        raise ValueError('Fitting receipt candidate identity differs from its commit')
    return model, receipt


def _fit(root, destination, spec, parent, history, cache, runtime, lock_sha256):
    folder = _fit_folder(destination, spec)
    cached = _load_fit(root, folder, spec, parent, lock_sha256, runtime.config)
    if cached is not None:
        runtime.event('fit_reused', **spec)
        return cached
    config = runtime.config
    runtime.guard(force=True)
    before = parent.fingerprint
    factory = cache.factory(parent)
    passes = 0

    def batches():
        nonlocal passes
        passes += 1
        this_pass = passes
        runtime.event('fit_pass_started', **spec, pass_number=this_pass)
        for count, value in enumerate(factory(), 1):
            runtime.guard()
            if count == 1 or count % 60 == 0:
                runtime.event('fit_pass_progress', **spec, pass_number=this_pass, months=count,
                              decision=value[0].sample.market.decision_date,
                              cache_gib=cache.retained_bytes/2**30)
            yield value

    common = dict(expected_neutral_beta=parent.model.beta,
                  expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
                  parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
                  rho=spec['rho'], objective_scale=parent.objective_scale)
    def progress(value):
        runtime.event('optimizer_'+value['event'], **spec, optimizer=value)

    runtime.event('fit_started', **spec, training_first=parent.metadata['training_first'],
                  training_last=parent.metadata['training_last'])
    if spec['kind'] == 'fixed_temporal':
        model, receipt = fit_fixed_profile(batches, parent.feature_map, parent.settings,
            gains=np.full(parent.feature_map.gain_count, config['fixed_temporal_gain']),
            lower=config['optimizer']['lower'], upper=config['optimizer']['upper'],
            reference_fingerprint=history.reference_fingerprint, **common)
        saver = save_fixed_result
    elif spec['kind'] == 'current_only':
        fmap = CurrentOnlyFeatureMap(parent.feature_map.parent, parent.feature_map.theta, parent.feature_map.groups)
        def bases():
            for basis, _ in batches():
                yield basis
        model, receipt = train_current_only_profile(bases, fmap, parent.settings,
            optimizer_settings=OptimizerSettings(**config['optimizer']), progress=progress, **common)
        saver = save_current_only_result
    else:
        model, receipt = train_profile(batches, parent.feature_map, parent.settings,
            reference_fingerprint=history.reference_fingerprint,
            optimizer_settings=OptimizerSettings(**config['optimizer']), progress=progress, **common)
        receipt['model_kind'] = 'learned_temporal'
        saver = save_result
    history.files.unchanged()
    if parent.fingerprint != before:
        raise ValueError('Frozen parent changed during a fit')
    receipt.update(year=parent.year, seed=parent.seed,
        training_first=parent.metadata['training_first'], training_last=parent.metadata['training_last'],
        accepted_inputs_sha256=dict(history.files.consumed), parent_artifacts_sha256=dict(parent.hashes),
        parent_scale_optimizer_status=parent.metadata['status'], study_lock_sha256=lock_sha256,
        fit_spec=spec, cache_diagnostics=cache.diagnostics())
    body = saver(folder, model, receipt)
    write_new(folder/'fit_commit.json', dict(status='COMPLETE_ARTIFACT', study_lock_sha256=lock_sha256,
        fit_spec=spec, artifact_sha256={str(p.relative_to(root)): sha(p)
                                      for p in folder.iterdir() if p.is_file()}))
    runtime.event('fit_finished', **spec, status=body['status'],
                  eligible=body['eligible_for_portfolio_evaluation'])
    runtime.guard(force=True)
    if body['eligible_for_portfolio_evaluation'] is not True:
        raise RuntimeError('Candidate did not converge; diagnostics preserved and study stopped without filtering candidates')
    return _load_fit(root, folder, spec, parent, lock_sha256, config)


def _policies(root, destination, config, parents, specs, lock_sha256, *, selected):
    policies = {}
    years = sorted({spec['year'] for spec in specs})
    for year in years:
        for seed in config['frequency_seeds']:
            parent = parents[year, seed]
            policies[year, seed, 'parent'] = AnnualPolicy(parent, kind='archived_parent')
            policies[year, seed, 'neutral'] = AnnualPolicy(parent, kind='neutral_replay')
    for spec in specs:
        parent = parents[spec['year'], spec['seed']]
        model, receipt = _load_fit(root, _fit_folder(destination, spec), spec, parent, lock_sha256, config)
        if spec['kind'] == 'fixed_temporal':
            name, kind = 'fixed_temporal', 'fixed_temporal'
        else:
            name = ('selected_'+spec['kind']) if selected else spec['kind']+'__'+spec['candidate']
            kind = 'learned_'+spec['kind']
        key = spec['year'], spec['seed'], name
        if key in policies:
            raise ValueError('Multiple fits would occupy the same annual policy slot')
        policies[key] = AnnualPolicy(parent, kind=kind, model=model, receipt=receipt,
            prescribed_gains=np.full(parent.feature_map.gain_count, config['fixed_temporal_gain'])
            if kind == 'fixed_temporal' else None)
    return policies


def _paths(root, folder, source, dates, policies, initial, runtime, lock_sha256):
    manifest_path = folder/'paths_manifest.json'
    commit_path = folder/'stage_commit.json'
    if folder.exists():
        if not commit_path.exists():
            raise ValueError('Preserved incomplete portfolio stage requires review before retry')
        commit = _json(commit_path)
        if commit.get('study_lock_sha256') != lock_sha256:
            raise ValueError('Portfolio stage study lock differs')
        _verify(root, commit['artifact_sha256'])
        manifest = _json(manifest_path)
        if manifest['policy_fingerprints'] != {str(k): p.fingerprint for k, p in policies.items()}:
            raise ValueError('Portfolio stage annual policy inventory differs')
        return dict(monthly=pd.read_parquet(folder/'monthly.parquet'), manifest=manifest)
    def progress(item):
        runtime.guard()
        if item['completed_months'] == 1 or item['completed_months'] % 12 == 0:
            runtime.event('portfolio_progress', stage=folder.name, **item)
    runtime.event('portfolio_stage_started', stage=folder.name, months=len(dates))
    result = run_paths(source, dates, policies, initial, destination=folder,
        save_positions=True, accounting_atol=1e-11, accounting_rtol=1e-10,
        neutral_atol=1e-11, neutral_rtol=1e-10, compare_archived=True, progress=progress)
    write_new(commit_path, dict(status='COMPLETE', study_lock_sha256=lock_sha256,
        artifact_sha256={str(p.relative_to(root)): sha(p) for p in folder.iterdir() if p.is_file()}))
    runtime.guard(force=True)
    return result


def _select(root, destination, config, first_paths, initial, lock_sha256):
    folder = destination/'selection'
    commit_path = folder/'selection_commit.json'
    if folder.exists():
        if not commit_path.exists():
            raise ValueError('Preserved incomplete selection requires review')
        commit = _json(commit_path)
        if commit.get('study_lock_sha256') != lock_sha256:
            raise ValueError('Selection study lock differs')
        _verify(root, commit['artifact_sha256'])
        return {kind: _json(folder/f'{kind}.json') for kind in ('temporal', 'current_only')}
    folder.mkdir()
    manifest = first_paths['manifest']
    path_folder = destination/'candidate_paths'
    inventory = {str((path_folder/p).relative_to(root)): h for p, h in manifest['artifact_sha256'].items()}
    inventory[str((path_folder/'stage_commit.json').relative_to(root))] = sha(path_folder/'stage_commit.json')
    selections = {}
    for kind in ('temporal', 'current_only'):
        candidates = tuple(Candidate(kind+'__'+candidate_name(rho), dict(rho=rho)) for rho in config['penalties_in_tie_order'])
        last_return = config['selection']['last_return']
        last_decision = (pd.Timestamp(last_return)-pd.offsets.MonthEnd(1)).date().isoformat()
        plan = SelectionPlan(protocol_id=config['id'], family_id=kind,
            seeds=tuple(config['frequency_seeds']), candidates=candidates,
            initial_decision=initial.decision_date,
            validation_first_return=config['selection']['first_return'], validation_last_return=last_return,
            selection_available_at=last_return, first_evaluation_decision=last_return,
            max_available_decision=last_decision, max_available_outcome=last_return,
            initial_pretrade_sha256=fingerprint(initial.ids, initial.holdings),
            gamma=config['selection']['risk_aversion'],
            exposure_label='Previously exposed development and selection; configurations fixed before 2011 evaluation decisions')
        paths = []
        for seed in config['frequency_seeds']:
            for candidate in candidates:
                frame = first_paths['monthly']
                frame = frame[(frame.seed == seed) & (frame.method == candidate.name)]
                paths.append(CandidatePath(seed, candidate.name, frame, inventory,
                                           manifest['first_trade_charged']))
        selections[kind] = select_candidates(plan=plan, paths=paths)
        write_new(folder/f'{kind}.json', selections[kind])
    write_new(commit_path, dict(status='COMPLETE', study_lock_sha256=lock_sha256,
        artifact_sha256={str(p.relative_to(root)): sha(p) for p in folder.iterdir() if p.is_file()}))
    return selections


def _external_benchmark(root, history, source, config, initial, runtime):
    """Read the accepted fixed-1024 path, retaining its original own holdings."""
    relative = f'{PARENT}/paths/monthly.parquet'
    frame = pd.read_parquet(history.files.path(relative))
    frame = frame[frame.method.eq('fixed1024') & frame.seed.isin(config['frequency_seeds'])].copy()
    dates = pd.date_range(f"{config['first_return_year']-1}-12-31", f"{config['last_return_year']}-11-30", freq='ME')
    if len(frame) != len(dates)*len(config['frequency_seeds']) or frame.duplicated(['seed', 'decision_eom']).any():
        raise ValueError('Accepted external benchmark path is incomplete')
    records = []
    previous = {}
    for number, date in enumerate(dates, 1):
        runtime.guard()
        arrays = source._archive(date)
        for seed in config['frequency_seeds']:
            match = np.flatnonzero((arrays['seeds'] == seed) & (arrays['methods'] == 'fixed1024'))
            if len(match) != 1:
                raise ValueError('Unique external fixed-1024 path is required')
            j = match[0]
            current = fingerprint(arrays['ids'], arrays['pretrade'][:, j])
            expected = fingerprint(initial.ids, initial.holdings) if seed not in previous else previous[seed]
            if current != expected:
                raise ValueError('External benchmark holdings are not continuous from the common initial state')
            previous[seed] = fingerprint(arrays['next_ids'], arrays['next_pretrade'][:, j])
            rows = frame[frame.seed.eq(seed) & frame.decision_eom.eq(date)]
            if len(rows) != 1 or rows.return_eom.iloc[0] != date+pd.offsets.MonthEnd(1):
                raise ValueError('External benchmark calendar differs')
            row = rows.iloc[0].to_dict()
            weights = arrays['weight'][:, j]
            absolute = abs(weights)
            exposure = float(absolute.sum())
            np.testing.assert_allclose(exposure, row['gross_exposure'], atol=1e-11, rtol=1e-10)
            np.testing.assert_allclose(row['gross_excess']-row['impact_cost'], row['net_excess'], atol=1e-12, rtol=1e-11)
            row.update(model_kind='accepted_external_fixed1024', net_exposure=float(weights.sum()),
                       largest_absolute_weight=float(absolute.max()),
                       gross_weight_concentration=float(np.sum((absolute/exposure)**2)) if exposure else 0.,
                       pretrade_sha256=current, next_pretrade_sha256=previous[seed], holdings_reset=False)
            records.append(row)
        if number % 120 == 0:
            runtime.event('external_benchmark_verified', months=number)
    return pd.DataFrame(records)


def _completion_result(destination, config, analysis_result):
    result = dict(status='HISTORICAL_NUMERICS_COMPLETE_REPORT_REVIEW_PENDING', configuration_id=config['id'],
                  scientific_study_complete=False, numerical_historical_study_complete=True,
                  full_historical_experiment_run=True, originality_established=False,
                  fresh_holdout=False, learned_parent_scales_reoptimized=False,
                  annual_fits=765, continuous_months=540, frequency_seeds=config['frequency_seeds'],
                  analysis_result=analysis_result)
    path = destination/'completion_result.json'
    if path.exists():
        if _json(path) != result:
            raise ValueError('Saved numerical completion result differs on resume')
    else:
        write_new(path, result)
    return result


def _report_stage(destination, analysis_output, diagnostics, gains):
    """Reuse only a complete evidence-matching report; never overwrite it."""
    from temporal_study.report import write_report, verify_report_inputs
    if not destination.exists():
        return write_report(analysis_output, destination,
                            fit_diagnostics=diagnostics, theme_gains=gains)
    manifest_path = destination/'report_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Preserved partial report requires review before retry')
    manifest = _json(manifest_path)
    if manifest.get('status') != 'REPORT_RENDERED_PENDING_VISUAL_REVIEW':
        raise ValueError('Unrecognized saved report stage status')
    manifest = verify_report_inputs(destination, analysis_output,
        fit_diagnostics=diagnostics, theme_gains=gains)
    return dict(html=destination/'completion_report.html', markdown=destination/'completion_report.md',
                source=destination/'report_source.json', manifest=manifest)


def _reference_stage(destination, monthly, config, analysis_folder, runtime):
    from scripts.verify_extension3_inference import export_inputs, run_reference
    manifest_path = destination/'verification_manifest.json'
    if destination.exists():
        if not manifest_path.exists():
            raise ValueError('Preserved partial independent verification requires review before retry')
        result = _json(manifest_path)
        if (result.get('status') != 'PASS'
                or result.get('r_script_sha256') != sha(runtime.root/'scripts/extension3_inference_reference.R')
                or result.get('exporter_sha256') != sha(runtime.root/'scripts/verify_extension3_inference.py')):
            raise ValueError('Saved independent verification code or status differs')
        for relative, digest in result['artifact_sha256'].items():
            path = (destination/relative).resolve()
            if not path.is_relative_to(destination.resolve()) or sha(path) != digest:
                raise ValueError('Changed saved independent verification artifact')
        inputs = _json(destination/'input_manifest.json')
        if inputs['plan'] != config or inputs['analysis_manifest_sha256'] != sha(analysis_folder/'analysis_manifest.json'):
            raise ValueError('Independent verification belongs to different analysis inputs')
        return result
    runtime.guard(force=True)
    executable = shutil.which('Rscript')
    if executable is None:
        raise RuntimeError('Local R is required to independently verify the historical inference')
    export_inputs(monthly, config, analysis_folder, destination)
    runtime.guard(force=True)
    remaining = runtime.approval['maximum_total_wall_seconds']-runtime.prior_seconds-(time.monotonic()-runtime.start)
    runtime.event('independent_R_verification_started')
    result = run_reference(destination, rscript=executable, timeout_seconds=min(300., remaining))
    runtime.guard(force=True)
    runtime.event('independent_R_verification_passed', receipt=result['receipt'])
    return result


def run_study(root, destination, *, config_path, approval_path, budget_extensions=()):
    """Run/reuse complete approved stages; never infer permission or retry a failed fit.

    Budget extensions alter only the cumulative wall ceiling. They cannot
    resume an optimizer from an interrupted point or remove a diagnostic fit
    that blocks its original candidate's complete comparison family.
    """
    from temporal_study.history import StudyHistory
    root = Path(root).resolve()
    destination = _inside(root, destination, output=True)
    config_path, approval_path = (_inside(root, p) for p in (config_path, approval_path))
    config, approval = verify_approval(root, config_path, approval_path)
    # Shared with the pilot; acquiring this lock never permits two heavy jobs.
    heavy = (root/'output/.extension3_heavy_job.lock').open('a+')
    try:
        fcntl.flock(heavy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        heavy.close()
        raise
    runtime = log = start_path = None
    try:
        lock_path = destination/'study_lock.json'
        if destination.exists():
            if not lock_path.exists():
                raise ValueError('Output destination exists without a completed study lock')
            lock = _json(lock_path)
            if lock.get('config') != config or lock.get('approval_sha256') != sha(approval_path):
                raise ValueError('Resume approval/configuration differs from the locked study')
            if lock.get('runtime_dependencies') != dict(scipy=scipy.__version__, numpy=np.__version__, pandas=pd.__version__):
                raise ValueError('Numerical runtime dependency versions changed on resume')
            _verify(root, lock['code_sha256'])
            _verify(root, lock['authority_sha256'])
            completed = destination/'completion_manifest.json'
            if completed.exists():
                completion = _json(completed)
                budget = _budget_authorizations(root, destination, config_path, approval_path, approval,
                                                budget_extensions, allow_new=False)
                if completion.get('budget_authorization') != budget:
                    raise ValueError('Completed study budget authority differs from its saved journal')
                _verify(root, completion['artifact_sha256'])
                _verify(root, completion['consumed_inputs_sha256'])
                _verify(root, completion['parent_artifacts_sha256'])
                return _json(destination/'completion_result.json')
        else:
            destination.mkdir(parents=True, exist_ok=False)
            files = _source_files(root, config_path)
            for relative in files:
                target = destination/'source_snapshot'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/relative, target)
            authorities = [approval_path, root/PARENT/'acceptance_manifest.json',
                           root/PARENT/'fits_manifest.json', root/PARENT/'paths_manifest.json',
                           root/PARENT/'plan_lock.json']
            lock = dict(status='FULL_STUDY_APPROVED_AND_LOCKED', config=config,
                runtime_dependencies=dict(scipy=scipy.__version__, numpy=np.__version__, pandas=pd.__version__),
                approval_sha256=sha(approval_path), code_sha256=files,
                authority_sha256={str(p.relative_to(root)): sha(p) for p in authorities},
                created_at_utc=datetime.now(timezone.utc).isoformat(), paid_resources=False,
                parent_overwrite=False, expected_annual_fits=765)
            write_new(lock_path, lock)
        budget = _budget_authorizations(root, destination, config_path, approval_path, approval,
                                        budget_extensions)
        effective_approval = dict(approval, maximum_total_wall_seconds=budget['maximum_total_wall_seconds'])
        prior_seconds = _attempt_seconds(destination)
        if prior_seconds >= budget['maximum_total_wall_seconds']:
            raise TimeoutError('Approved full-study budget already exhausted')
        attempts = destination/'attempts'
        attempts.mkdir(exist_ok=True)
        number = len(list(attempts.glob('*_start.json')))+1
        start_path = attempts/f'{number:04d}_start.json'
        write_new(start_path, dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            approval_sha256=sha(approval_path), previous_attempt_seconds=prior_seconds,
            budget_authorization_sha256=budget['latest_authorization_sha256'],
            maximum_total_wall_seconds=budget['maximum_total_wall_seconds']))
        log = (destination/'progress.log').open('a', buffering=1)
        runtime = Runtime(root, destination, config, effective_approval, log, prior_seconds)
        runtime.guard(force=True)
        runtime.event('study_attempt_started', attempt=number)
        receipt, _ = preflight()
        preflight_path = destination/'financial_preflight.json'
        if preflight_path.exists():
            if _json(preflight_path) != receipt:
                raise ValueError('Financial preflight changed on resume')
        else:
            write_new(preflight_path, receipt)
        history = StudyHistory(root, receipt, weights=config['reference_weights'])
        if history.gamma != config['selection']['risk_aversion']:
            raise ValueError('Selection risk aversion differs from the economic parent')
        source = AcceptedPathSource(history)
        initial = source.common_initial()
        cache = AnnualBatchCache(history, maximum_bytes=int(config['resources']['basis_cache_gib']*2**30))
        parents = {}
        for year in range(config['first_return_year'], config['last_return_year']+1):
            for seed in config['frequency_seeds']:
                runtime.guard()
                parents[year, seed] = load_parent(root, year, seed, expected_cutoff=f'{year-1}-12-31')
        lock_sha256 = sha(lock_path)
        with threadpool_limits(limits=config['resources']['threads']):
            first_specs = fit_specs(config)
            for spec in first_specs:
                _verify(root, lock['code_sha256'])
                _fit(root, destination, spec, parents[spec['year'], spec['seed']], history, cache, runtime, lock_sha256)
            runtime.event('preselection_fits_complete', fits=len(first_specs))
            policies = _policies(root, destination, config, parents, first_specs, lock_sha256, selected=False)
            selection_end = pd.Timestamp(config['selection']['last_return'])-pd.offsets.MonthEnd(1)
            dates = pd.date_range(initial.decision_date, selection_end, freq='ME')
            first_paths = _paths(root, destination/'candidate_paths', source, dates, policies, initial, runtime, lock_sha256)
            selections = _select(root, destination, config, first_paths, initial, lock_sha256)
            runtime.event('penalty_selection_frozen', selections={k:v['selected_by_seed'] for k,v in selections.items()})
            all_specs = fit_specs(config, selections)
            later_specs = [s for s in all_specs if s['year'] > selection_end.year]
            for spec in later_specs:
                _verify(root, lock['code_sha256'])
                _fit(root, destination, spec, parents[spec['year'], spec['seed']], history, cache, runtime, lock_sha256)
            choices = {(kind, row['seed']): candidate_name(row['parameters']['rho']) for kind, result in selections.items()
                       for row in result['selected_by_seed']}
            selected_specs = [s for s in all_specs if s['kind'] == 'fixed_temporal'
                              or s['candidate'] == choices[s['kind'], s['seed']]]
            policies = _policies(root, destination, config, parents, selected_specs, lock_sha256, selected=True)
            dates = pd.date_range(initial.decision_date, f"{config['last_return_year']}-11-30", freq='ME')
            final_paths = _paths(root, destination/'selected_paths', source, dates, policies, initial, runtime, lock_sha256)
            # Selection must not reset holdings at its boundary; replayed chosen
            # candidate rows are also checked byte/content-wise by analysis.
            benchmark = _external_benchmark(root, history, source, config, initial, runtime)
            monthly = pd.concat([final_paths['monthly'], benchmark], ignore_index=True)
            saved_fits = [(spec, _json(_fit_folder(destination, spec)/'fit.json')) for spec in all_specs]
            consumed_inputs = _merge_inventories(
                *(fit['accepted_inputs_sha256'] for _, fit in saved_fits),
                first_paths['manifest']['consumed_inputs_sha256'],
                final_paths['manifest']['consumed_inputs_sha256'], history.files.consumed)
            combined = dict(final_paths['manifest'], paths=6*len(config['frequency_seeds']), monthly_records=len(monthly),
                includes_accepted_external_benchmark=True, external_benchmark='fixed1024',
                consumed_inputs_sha256=consumed_inputs)
            combined_path = destination/'combined_monthly.parquet'
            if combined_path.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(combined_path), monthly)
            else:
                monthly.to_parquet(combined_path, index=False)
            combined_manifest = destination/'combined_paths_manifest.json'
            if combined_manifest.exists():
                if _json(combined_manifest) != combined:
                    raise ValueError('Combined path manifest changed on resume')
            else:
                write_new(combined_manifest, combined)
        diagnostics, gains = [], []
        names = sorted(pd.read_csv(root/'config/feature_dictionary_v1.csv').cluster.unique())
        for spec, fit in saved_fits:
            diagnostics.append(dict(**spec, status=fit['status'],
                neutral_beta_max_error=fit['neutral_gate']['maximum_beta_error'],
                solve_residual=fit['solve_residual'], projected_gradient_norm=fit.get('projected_gradient_norm'),
                accepted_steps=fit.get('accepted_steps'), evaluations=fit.get('evaluation_count'),
                elapsed_seconds=fit.get('elapsed_seconds'), coverage=fit.get('coverage'),
                boundary_flags=fit.get('boundary_flags'), parent_scale_optimizer_status=fit['parent_scale_optimizer_status']))
            for group, gain in zip(names, fit['gains']):
                gains.append(dict(**spec, theme=group, gain=gain,
                    interpretation=('current-input scaling' if spec['kind'] == 'current_only'
                                    else 'temporal innovation gain: below one smooths; above one emphasizes'),
                    optimizer_convergence_applicable=spec['kind'] != 'fixed_temporal'))
        for name, frame in [('fit_diagnostics.csv', pd.DataFrame(diagnostics)), ('theme_gains.csv', pd.DataFrame(gains))]:
            path = destination/name
            text = frame.to_csv(index=False)
            if path.exists():
                if path.read_text() != text:
                    raise ValueError('Existing fit diagnostic table differs on resume')
            else:
                with path.open('x') as stream:
                    stream.write(text)
        runtime.guard(force=True)
        analysis_folder = destination/'analysis'
        analysis_inputs = {str(combined_path.relative_to(root)):sha(combined_path),
            str(combined_manifest.relative_to(root)):sha(combined_manifest),
            str((destination/'selection/selection_commit.json').relative_to(root)):sha(destination/'selection/selection_commit.json'),
            str(lock_path.relative_to(root)):lock_sha256}
        exposure_metadata = dict(families={kind:r['candidate_exposure'] for kind,r in selections.items()},
                                 additional_candidate_exposure=[])
        if analysis_folder.exists():
            manifest = _json(analysis_folder/'analysis_manifest.json')
            if manifest.get('status') != 'COMPLETE':
                raise ValueError('Preserved incomplete analysis requires review')
            # Analysis manifests use their own folder-relative paths.
            for relative, digest in manifest['artifact_sha256'].items():
                path = analysis_folder/relative
                if not path.resolve().is_relative_to(analysis_folder.resolve()) or sha(path) != digest:
                    raise ValueError('Changed saved analysis artifact')
            analysis_result = _json(analysis_folder/'analysis_result.json')
            if (analysis_result.get('supplied_inputs_sha256') != analysis_inputs
                    or analysis_result.get('protocol_id') != config['id']
                    or analysis_result.get('approved_study_plan') != config
                    or analysis_result.get('selection_metadata') != selections
                    or analysis_result.get('candidate_exposure_metadata') != exposure_metadata):
                raise ValueError('Saved analysis belongs to different paths, selection or study lock')
            analysis_output = {name:pd.read_csv(analysis_folder/f'{name}.csv', float_precision='round_trip') for name in (
                'strategy_summary', 'mean_seed_metrics', 'seed_contrasts', 'utility_decomposition',
                'primary_intervals', 'sampling_diagnostics', 'selection_links')}
            analysis_output.update(result=analysis_result, bootstrap={})
        else:
            from temporal_study.analysis import analyze_paths
            runtime.event('analysis_started')
            analysis_output = analyze_paths(monthly, config, path_manifest=combined,
                selection_metadata=selections,
                candidate_exposure_metadata=exposure_metadata, artifact_sha256=analysis_inputs,
                destination=analysis_folder, progress=lambda item: (runtime.guard(), runtime.event('analysis_progress', **item)))
            analysis_result = analysis_output['result']
        verification = _reference_stage(destination/'reference_verification', monthly, config, analysis_folder, runtime)
        # Preserve the completed analytical receipt; enrich only the downstream
        # report's evidence snapshot with the independent verification record.
        analysis_result = dict(analysis_result,
            analysis_calculation_result_sha256=sha(analysis_folder/'analysis_result.json'),
            independent_R_verification=verification['receipt'],
            supplied_inputs_sha256=dict(analysis_result['supplied_inputs_sha256'], **{
                str((destination/'reference_verification/verification_manifest.json').relative_to(root)):
                    sha(destination/'reference_verification/verification_manifest.json')}))
        analysis_output['result'] = analysis_result
        history.files.unchanged()
        _verify(root, consumed_inputs)
        _verify(root, lock['code_sha256'])
        _verify(root, lock['authority_sha256'])
        if budget['authorization_receipts_sha256']:
            _verify(root, budget['authorization_receipts_sha256'])
        if _budget_authorizations(root, destination, config_path, approval_path, approval,
                                  allow_new=False) != budget:
            raise ValueError('Budget authority journal changed during execution')
        parent_inventory = {p:h for parent in parents.values() for p,h in parent.hashes.items()}
        _verify(root, parent_inventory)
        for spec in all_specs:
            commit = _json(_fit_folder(destination, spec)/'fit_commit.json')
            if commit['study_lock_sha256'] != lock_sha256 or commit['fit_spec'] != spec:
                raise ValueError('Fitted comparison family changed after portfolio evaluation')
            _verify(root, commit['artifact_sha256'])
        for stage in ('candidate_paths', 'selected_paths', 'selection'):
            name = 'selection_commit.json' if stage == 'selection' else 'stage_commit.json'
            _verify(root, _json(destination/stage/name)['artifact_sha256'])
        runtime.guard(force=True)
        if len(all_specs) != 765:
            raise ValueError('The declared annual comparison family is incomplete')
        analysis_output['completion_gates'] = dict(approved_full_historical_run=True,
            complete_annual_fits=True, complete_continuous_paths=True,
            neutral_equivalence_passed=bool(combined['archived_replay_checked']
                                          and combined['neutral_monthly_pairs_checked'] == 540*3),
            frozen_inputs_verified=True)
        _report_stage(destination/'report', analysis_output, pd.DataFrame(diagnostics), pd.DataFrame(gains))
        runtime.guard(force=True)
        result = _completion_result(destination, config, analysis_result)
        runtime.event('historical_numerics_completed_report_review_pending', annual_fits=len(all_specs), months=540, paths=18)
        write_new(start_path.with_name(start_path.name.replace('_start.json', '_end.json')),
                  dict(status='COMPLETE', start_sha256=sha(start_path), seconds=time.monotonic()-runtime.start))
        log.close()
        write_new(destination/'completion_manifest.json', dict(status='HISTORICAL_NUMERICS_COMPLETE_REPORT_REVIEW_PENDING',
            scientific_study_complete=False, numerical_historical_study_complete=True,
            approval_sha256=sha(approval_path), study_lock_sha256=lock_sha256,
            budget_authorization=budget,
            consumed_inputs_sha256=consumed_inputs,
            parent_artifacts_sha256=parent_inventory,
            peak_sampled_rss_gib=runtime.peak_rss/2**30,
            cumulative_seconds=runtime.prior_seconds+time.monotonic()-runtime.start,
            cache=cache.diagnostics(), artifact_sha256={str(p.relative_to(root)):sha(p)
                for p in destination.rglob('*') if p.is_file()}))
        return result
    except Exception as exc:
        if runtime is not None:
            if not log.closed:
                runtime.event('study_stopped', error_type=type(exc).__name__, error=str(exc))
            end_path = start_path.with_name(start_path.name.replace('_start.json', '_end.json'))
            if not end_path.exists():
                write_new(end_path, dict(status='STOPPED_PRESERVED', start_sha256=sha(start_path),
                    seconds=time.monotonic()-runtime.start, error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        if log is not None and not log.closed:
            log.close()
        fcntl.flock(heavy, fcntl.LOCK_UN)
        heavy.close()
