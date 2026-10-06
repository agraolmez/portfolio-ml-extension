"""Versioned Step 6 orchestration; all accepted Step 3–5 files remain read-only.

The accepted Engine is reused unchanged. Only its module's output/preparation
roots are rebound in this process; source data, formulas and grids are fixed.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import engine as machinery
from .acceptance import verify_baseline
from .core import report_returns
from .io import ROOT, DATA, PREP as PRIOR, OUT as PRIOR_OUT, PROTOCOL, sha, save_json, verify_data
from .wealth import validate

PREP = ROOT / 'data/prepared/step6_v1'
OUT = ROOT / 'output/step6_20260911'
RUN = ROOT / 'output/step6_baseline_1981_2020'
DATES = pd.date_range('1980-12-31', '2020-11-30', freq='ME')
MOMENT_DATES = pd.date_range('1965-11-30', DATES[-1], freq='ME')
PROTECTED = ['GOAL.md', 'docs/STEP_1_THEORY.md', 'docs/RECONSTRUCTION_AMENDMENT.md',
             'config/protocol_v1.json', 'config/reconstruction_amendment.json']


def verify_pilot():
    path = ROOT / 'output/step5_20260911/acceptance_manifest.json'
    manifest = json.loads(path.read_text())
    if (manifest.get('status') != 'ACCEPTED_HISTORICAL_PILOT'
            or manifest.get('step5_complete') is not True
            or not manifest.get('gates') or not all(g['passed'] for g in manifest['gates'])):
        raise ValueError('Accepted Step 5 pilot required')
    for rel, digest in manifest['artifact_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'Changed accepted pilot dependency: {rel}')
    return {'manifest_sha256': sha(path), 'files_verified': len(manifest['artifact_sha256'])}


def dependencies():
    if PROTOCOL['periods']['baseline'] != ['1981-01', '2020-12']:
        raise ValueError('This runner is scoped to the approved 1981–2020 baseline')
    return {'source': verify_data(), 'baseline': verify_baseline(), 'pilot': verify_pilot(),
            'wealth': validate(pd.read_parquet(DATA / 'market_wealth.parquet'))}


def configure():
    if not (PREP / 'risk').is_symlink() or (PREP / 'risk').resolve() != PRIOR / 'risk':
        raise ValueError('Step 6 must use the accepted risk bundle')
    machinery.PREP, machinery.OUT = PREP, OUT
    return machinery.Engine(end=DATES[-1])


def initialize():
    receipt = dependencies()
    if (OUT / 'start.json').exists():
        raise ValueError('Step 6 already initialized; resume the incomplete stage explicitly')
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / 'risk').symlink_to(PRIOR / 'risk', target_is_directory=True)
    copied = {}
    for subfolder in ['matrices', 'moments/g0', 'moments/g1']:
        target = PREP / subfolder
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((PRIOR / subfolder).glob('*.npz')):
            if pd.Timestamp(path.stem) <= DATES[-1]:
                dest = target / path.name
                shutil.copy2(path, dest)
                if sha(path) != sha(dest):
                    raise ValueError('Seed-copy integrity error')
                copied[str(dest.relative_to(ROOT))] = {'source': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    (OUT / 'reference').mkdir(exist_ok=True)
    for gi in [0, 1]:
        shutil.copy2(PRIOR_OUT / f'reference/rff_g{gi}.bin', OUT / f'reference/rff_g{gi}.bin')
    save_json(OUT / 'start.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User: Let us continue with step 6 (September 11, 2026)',
        'dependencies': receipt, 'protected': {p: sha(ROOT / p) for p in PROTECTED},
        'accepted_seed_copies': copied, 'decisions': [str(DATES[0].date()), str(DATES[-1].date())],
        'planned_checks': ['All 40 annual model choices and all source-grid coefficients/validation scores versus actual R',
            'All 480 months of actual R holdings and returns on identical R1 inputs',
            'Actual R input rebuild at every December and June decision, plus final November 2020 (81 months × 2 bandwidths)',
            'All-month independent accounting, entries, exits, selection/training cutoffs and unchanged pilot prefix',
            'Published table reconciliation with metric-definition and input-vintage limitations'],
        'full_input_R_check_is_sampled': True, 'scientific_amendment': False})
    print('Initialized separate Step 6 preparation roots; accepted history preserved.', flush=True)


def prepare():
    if (OUT / 'preparation_manifest.json').exists():
        raise ValueError('Completed preparation is immutable')
    dependencies()
    engine = configure()
    engine.build_moments()
    engine.fit(end_year=2020)
    aims = engine.aims(DATES)
    aims.to_parquet(PREP / 'baseline_aims.parquet', index=False)
    # All preparation that the financial path will consume is locked before it runs.
    files = [p for folder in ['matrices', 'moments', 'coefficients'] for p in (PREP / folder).rglob('*') if p.is_file()]
    files += list(PREP.glob('*.parquet'))
    files += [ROOT / f'baseline/{name}' for name in ['full.py', 'engine.py', 'core.py', 'risk.py', 'io.py', 'wealth.py', 'acceptance.py']]
    files += [OUT / f'reference/rff_g{gi}.bin' for gi in [0, 1]]
    save_json(OUT / 'preparation_manifest.json', {'status': 'PREPARED_PENDING_FULL_REVIEW',
        'moment_months': len(MOMENT_DATES), 'return_months': len(DATES),
        'artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}})
    print(json.dumps(preflight()[0], indent=2), flush=True)


def preflight():
    dep = dependencies()
    manifest_path = OUT / 'preparation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for rel, digest in manifest['artifact_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'Changed prepared artifact: {rel}')
    start = json.loads((OUT / 'start.json').read_text())
    if any(sha(ROOT / p) != h for p, h in start['protected'].items()):
        raise ValueError('Protected research decisions changed')
    aims = pd.read_parquet(PREP / 'baseline_aims.parquet')
    if set(aims.eom) != set(DATES) or aims.duplicated(['id', 'eom']).any() or not np.isfinite(aims.aim).all():
        raise ValueError('Incomplete, duplicate, or nonfinite baseline aims')
    active = pd.read_parquet(DATA / 'monthly_inputs.parquet', columns=['id', 'eom', 'valid'],
        filters=[('eom', '>=', DATES[0]), ('eom', '<=', DATES[-1])]).query('valid')
    if set(zip(aims.id, aims.eom)) != set(zip(active.id, active.eom)):
        raise ValueError('Target membership differs from accepted universe')
    for year, block in aims.groupby('fit_year'):
        if block[['g_index', 'p', 'lambda_index']].drop_duplicates().shape[0] != 1:
            raise ValueError('Within-year model changed')
        if not block.selection_return_year.eq(year - 1).all() or not (block.eom + pd.offsets.MonthEnd(1)).dt.year.eq(year).all():
            raise ValueError('Aim date/selection chronology error')
        row = block.iloc[0]
        with np.load(PREP / f'coefficients/g{row.g_index}/{year}_p{row.p}.npz') as z:
            if str(z['training_cutoff']) != f'{year-1}-11-30':
                raise ValueError('Unmatured coefficient estimation')
    old = pd.read_parquet(PRIOR / 'pilot_aims_preflight.parquet').sort_values(['eom', 'id']).reset_index(drop=True)
    prefix = aims[aims.eom <= old.eom.max()].sort_values(['eom', 'id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, prefix, check_exact=True)
    return {'dependencies': dep, 'preparation_manifest_sha256': sha(manifest_path),
        'prepared_artifacts_verified': len(manifest['artifact_sha256']), 'months': len(DATES),
        'target_stock_months': len(aims), 'return_first': '1981-01-31', 'return_last': '2020-12-31',
        'pilot_targets_identical': True, 'financial_path_run': False}, aims


def run():
    if (RUN / 'ExperimentManifest.json').exists():
        raise ValueError('Existing full baseline results must not be overwritten')
    receipt, aims = preflight()
    engine = configure()
    # The old gate is checked in preflight; the new prepared-file inventory is checked too.
    # Engine.run's internal gate points at its output root, so use the explicit gates above.
    monthly, positions, exits = engine.run(DATES, aims, require_step4=False)
    RUN.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(RUN / 'monthly_performance.parquet', index=False)
    positions.to_parquet(RUN / 'PolicyDecision.parquet', index=False)
    aims.to_parquet(RUN / 'AimArtifact.parquet', index=False)
    exits.to_parquet(RUN / 'source_exit_ledger.parquet', index=False)
    snapshots = []
    for d in DATES:
        frame = engine.months[d].query('valid')
        snapshots.append({'decision_eom': d, 'return_eom': d + pd.offsets.MonthEnd(1),
            'eligible_stocks': len(frame), 'wealth_usd': frame.wealth_usd.iloc[0],
            'market_total_growth': frame.mu_ld1.iloc[0], 'rf_french_for_matrix': engine.rf[d],
            'risk_sha256': sha(PRIOR / f'risk/{d:%Y-%m-%d}.npz')})
    pd.DataFrame(snapshots).to_parquet(RUN / 'MarketSnapshot.parquet', index=False)
    summary = report_returns(monthly.gross_excess, monthly.impact_cost, PROTOCOL['economic']['risk_aversion'])
    save_json(RUN / 'summary.json', summary)
    save_json(RUN / 'ExperimentManifest.json', {'status': 'FULL_BASELINE_COMPUTED_PENDING_STEP6_REVIEW',
        'protocol_id': PROTOCOL['protocol_id'], 'preflight': receipt, 'continuous_holdings': True,
        'initialization': 'Source first-month value weights; first trade charged; no resets',
        'accounting': 'Author conventions; entrants zero; exits omitted without extra liquidation; terminal drift retained',
        'outputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in RUN.iterdir() if p.is_file()},
        'scope': 'Conditional historical R1 reconstruction; not paper-vintage equality or causal execution'})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initialize', 'prepare', 'preflight', 'run'])
    args = parser.parse_args()
    if args.stage == 'preflight':
        print(json.dumps(preflight()[0], indent=2))
    else:
        globals()[args.stage]()
