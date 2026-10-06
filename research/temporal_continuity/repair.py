"""Preserved-artifact derivation and explicit supplemental continuation lock."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from temporal_fast_v4.study import _inside, _json, _verify, write_new
from .identity import canonical_ids, holdings_fingerprint


def derive_candidate_metadata(root, original, destination, study_lock_sha256):
    """Create a new path table only after checking every original array/hash.

    No fitting, trading, performance scoring, row reordering or rounding.
    Original position archives remain in place and are referenced by hash.
    """
    root = Path(root).resolve()
    original, destination = (_inside(root, p, output=True) for p in (original, destination))
    if destination.exists():
        raise FileExistsError('A metadata derivative must have a new destination')
    commit = _json(original/'stage_commit.json')
    manifest = _json(original/'paths_manifest.json')
    if commit.get('status') != 'COMPLETE' or commit.get('study_lock_sha256') != study_lock_sha256:
        raise ValueError('Original completed candidate stage and exact study lock required')
    _verify(root, commit['artifact_sha256'])
    for name, expected in manifest['artifact_sha256'].items():
        path = _inside(root, original/name, output=True)
        if sha(path) != expected:
            raise ValueError('Original path manifest hash differs')
    original_table = pd.read_parquet(original/'monthly.parquet')
    frame = original_table.copy(deep=True)
    keys = ['decision_eom', 'seed', 'method']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate monthly path identity')
    row_numbers = {tuple(row): number for number, row in enumerate(frame[keys].itertuples(index=False, name=None))}
    dates = pd.date_range(manifest['decision_first'], manifest['decision_last'], freq='ME')
    if len(dates) != manifest['months'] or len(frame) != manifest['monthly_records']:
        raise ValueError('Incomplete recorded candidate calendar')
    previous = None
    transitions = checked_rows = mismatches = 0
    initial_identity = None
    for date in dates:
        path = original/f'{date.date().isoformat()}.npz'
        if commit['artifact_sha256'].get(str(path.relative_to(root))) != sha(path):
            raise ValueError('Position archive absent from original stage commit')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in
                      ('ids', 'next_ids', 'pretrade', 'next_pretrade', 'seeds', 'methods')}
        canonical_ids(arrays['ids']); canonical_ids(arrays['next_ids'])
        if previous is not None:
            if (not np.array_equal(previous['seeds'], arrays['seeds'])
                    or not np.array_equal(previous['methods'], arrays['methods'])
                    or not np.array_equal(previous['next_ids'], arrays['ids'])
                    or previous['next_pretrade'].dtype != arrays['pretrade'].dtype
                    or previous['next_pretrade'].shape != arrays['pretrade'].shape
                    or previous['next_pretrade'].tobytes() != arrays['pretrade'].tobytes()):
                raise ValueError('Actual holdings, ordered IDs or path identities are discontinuous')
        if len(arrays['seeds']) != manifest['paths'] or arrays['methods'].shape != arrays['seeds'].shape:
            raise ValueError('Complete aligned path inventory required')
        for j, (seed, method) in enumerate(zip(arrays['seeds'], arrays['methods'])):
            key = (date, int(seed), str(method))
            if key not in row_numbers:
                raise ValueError('Position archive has no unique monthly row')
            index = row_numbers[key]
            row = frame.iloc[index]
            before = fingerprint(arrays['ids'], arrays['pretrade'][:, j])
            after = fingerprint(arrays['next_ids'], arrays['next_pretrade'][:, j])
            if row.pretrade_sha256 != before or row.next_pretrade_sha256 != after:
                raise ValueError('Original holdings fingerprint differs from its saved array')
            new_before = holdings_fingerprint(arrays['ids'], arrays['pretrade'][:, j])
            new_after = holdings_fingerprint(arrays['next_ids'], arrays['next_pretrade'][:, j])
            if previous is None:
                if before != manifest['initial_pretrade_sha256']:
                    raise ValueError('Original common initial holdings differ')
                if initial_identity is not None and new_before != initial_identity:
                    raise ValueError('Candidate initial portfolios differ')
                initial_identity = new_before
            else:
                old_after = fingerprint(previous['next_ids'], previous['next_pretrade'][:, j])
                mismatches += int(old_after != before)
                if holdings_fingerprint(previous['next_ids'], previous['next_pretrade'][:, j]) != new_before:
                    raise ValueError('Canonical holdings continuity failed')
                transitions += 1
            frame.at[frame.index[index], 'pretrade_sha256'] = new_before
            frame.at[frame.index[index], 'next_pretrade_sha256'] = new_after
            checked_rows += 1
        previous = arrays
    if checked_rows != len(frame):
        raise ValueError('Monthly table contains unmatched rows')
    hash_columns = ['pretrade_sha256', 'next_pretrade_sha256']
    pd.testing.assert_frame_equal(original_table.drop(columns=hash_columns), frame.drop(columns=hash_columns), check_exact=True)
    # Only write after all validations pass; original artifacts are never edited.
    destination.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(destination/'monthly.parquet', index=False)
    audit = dict(status='PASS', monthly_rows=checked_rows, adjacent_path_links=transitions,
                 raw_fingerprint_mismatches=mismatches, actual_holdings_bitwise_equal=True,
                 ordered_id_values_equal=True, nonidentity_columns_unchanged=True,
                 original_stage_commit_sha256=sha(original/'stage_commit.json'),
                 original_monthly_sha256=sha(original/'monthly.parquet'),
                 canonicalization='lossless int64 ID representation only; exact holdings bytes retained',
                 original_artifacts_changed=False, historical_fit_run=False)
    write_new(destination/'metadata_repair.json', audit)
    inventory = {os.path.relpath(original/name, destination): digest
                 for name, digest in manifest['artifact_sha256'].items() if name != 'monthly.parquet'}
    inventory.update({p.name:sha(p) for p in destination.iterdir() if p.is_file()})
    consumed = dict(manifest['consumed_inputs_sha256'])
    for name in ('stage_commit.json', 'paths_manifest.json', 'monthly.parquet'):
        consumed[str((original/name).relative_to(root))] = sha(original/name)
    repaired = dict(manifest, initial_pretrade_sha256=initial_identity,
                    artifact_sha256=inventory, consumed_inputs_sha256=consumed,
                    metadata_correction=audit)
    write_new(destination/'paths_manifest.json', repaired)
    bound = {str((original/name).relative_to(root)):sha(original/name)
             for name in ('stage_commit.json', 'paths_manifest.json')}
    bound.update(commit['artifact_sha256'])
    bound.update({str(p.relative_to(root)):sha(p) for p in destination.iterdir() if p.is_file()})
    write_new(destination/'stage_commit.json', dict(status='COMPLETE',
              study_lock_sha256=study_lock_sha256, metadata_only_derivative=True, artifact_sha256=bound))
    return dict(monthly=frame, manifest=repaired, audit=audit)


def verify_repair(root, destination, config_path, approval_path, repair_path):
    root = Path(root).resolve()
    destination = _inside(root, destination, output=True)
    repair = _json(repair_path)
    expected = dict(status='VERIFIED_CONTINUITY_REPAIR',
        original_study_output=str(destination.relative_to(root)),
        original_study_lock_sha256=sha(destination/'study_lock.json'),
        original_approval_sha256=sha(approval_path), config_sha256=sha(config_path),
        scientific_settings_changed=False, fitting_engine_changed=False,
        original_results_overwritten=False, resource_limits_changed=False,
        paid_resources=False)
    if any(repair.get(key) != value for key, value in expected.items()):
        raise ValueError('Repair does not bind this exact unchanged authorized study')
    if not repair.get('user_confirmation') or not repair.get('gates') or not all(g.get('passed') is True for g in repair['gates']):
        raise ValueError('Completed repair verification and continuation authority required')
    required = {str(p.relative_to(root)) for p in (root/'temporal_continuity').glob('*.py')}
    required.add('scripts/extension3_continuity_study.py')
    if not required <= repair.get('code_sha256', {}).keys():
        raise ValueError('Supplemental source inventory is incomplete')
    _verify(root, repair['code_sha256'])
    _verify(root, repair['artifact_sha256'])
    return repair
