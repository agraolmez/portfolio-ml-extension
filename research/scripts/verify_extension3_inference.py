"""Export supplied analysis artifacts and independently verify them in local R.

This opens no research database and fits no model. The caller supplies a saved
monthly path table, explicit plan, and completed Extension-3 analysis artifacts.
The output folder must be new. Verification reuses saved resampling indices;
it does not regenerate the random sample or change the comparison family.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_geometry.history import sha


def export_inputs(monthly, plan, analysis_folder, destination):
    analysis_folder, destination = Path(analysis_folder).resolve(), Path(destination).resolve()
    if plan['inference']['family'] != 4 or len(plan['primary_contrasts']) != 4:
        raise ValueError('An explicit four-contrast plan is required')
    manifest_path = analysis_folder/'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError('Completed analysis artifacts are required')
    inventory = manifest['artifact_sha256']
    for relative, expected in inventory.items():
        path = analysis_folder/relative
        if not path.resolve().is_relative_to(analysis_folder) or sha(path) != expected:
            raise ValueError('Changed or invalid supplied analysis artifact: '+relative)
    frame = monthly.copy(deep=True)
    frame['return_eom'] = pd.to_datetime(frame.return_eom)
    if frame.duplicated(['return_eom', 'seed', 'method']).any():
        raise ValueError('Duplicate supplied monthly path identities')
    intervals = pd.read_csv(analysis_folder/'primary_intervals.csv')
    required = {'primary_intervals.csv', *(f'{period}_L{block}_bootstrap.npz'
        for period in ('evaluation', 'exposed_robustness')
        for block in plan['inference']['block_months'])}
    if not required.issubset(inventory):
        raise ValueError('Every consumed analysis artifact must have a verified hash')
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(plan['primary_contrasts'], columns=['left', 'right']).to_csv(
        destination/'contrasts.csv', index=False)
    cases = []
    for period in ('evaluation', 'exposed_robustness'):
        first, last = plan['periods'][period]
        bounded = frame[frame.return_eom.dt.year.between(first, last)]
        for block in plan['inference']['block_months']:
            case = f'{period}_L{block}'
            with np.load(analysis_folder/f'{case}_bootstrap.npz', allow_pickle=False) as data:
                indices, expected = data['indices'].copy(), data['utility_draws'].copy()
                seeds, methods = data['seeds'].copy(), data['methods'].copy()
            if seeds.shape != methods.shape or seeds.ndim != 1:
                raise ValueError('Misaligned saved bootstrap identities')
            columns = list(zip(seeds, methods))
            if len(set(columns)) != len(columns):
                raise ValueError('Duplicate saved bootstrap column identities')
            wide = bounded.pivot(index='return_eom', columns=['seed', 'method'], values='net_excess')
            wide = wide.reindex(columns=pd.MultiIndex.from_tuples(columns)).sort_index()
            dates = pd.date_range(f'{first}-01-31', f'{last}-12-31', freq='ME')
            matrix = wide.to_numpy(dtype=float)
            replicas = plan['inference']['replicas']
            if (not wide.index.equals(dates) or not np.isfinite(matrix).all()
                    or indices.shape != (replicas, len(dates)) or indices.dtype.kind not in 'iu'
                    or np.any(indices < 0) or np.any(indices >= len(dates))
                    or expected.shape != (replicas, len(columns)) or not np.isfinite(expected).all()):
                raise ValueError('Saved indices, utilities or supplied complete return paths differ')
            names = dict(returns_csv=f'{case}_returns.csv', indices_bin=f'{case}_indices.bin',
                         utilities_csv=f'{case}_utilities.csv', columns_csv=f'{case}_columns.csv',
                         intervals_csv=f'{case}_intervals.csv')
            np.savetxt(destination/names['returns_csv'], matrix, delimiter=',', fmt='%.17g')
            np.asarray(indices, dtype='<i4').tofile(destination/names['indices_bin'])
            np.savetxt(destination/names['utilities_csv'], expected, delimiter=',', fmt='%.17g')
            pd.DataFrame(dict(position=np.arange(1, len(columns)+1), seed=seeds, method=methods)).to_csv(
                destination/names['columns_csv'], index=False)
            rows = intervals[intervals.period.eq(period) & intervals.block_months.eq(block)]
            rows.to_csv(destination/names['intervals_csv'], index=False, float_format='%.17g')
            cases.append(dict(case=case, **names, replicas=replicas, months=len(dates),
                              alpha=plan['inference']['alpha'], family=4))
    pd.DataFrame(cases).to_csv(destination/'verification_cases.csv', index=False, float_format='%.17g')
    source = dict(analysis_manifest_sha256=sha(manifest_path),
                  analysis_artifact_sha256=inventory, plan=plan,
                  exported_input_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file()},
                  historical_fit_run=False, historical_database_opened=False)
    with (destination/'input_manifest.json').open('x') as handle:
        json.dump(source, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return source


def run_reference(destination, *, rscript, timeout_seconds=300.):
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('An explicit finite positive subprocess time limit is required')
    destination = Path(destination).resolve()
    manifest = json.loads((destination/'input_manifest.json').read_text())
    for name, expected in manifest['exported_input_sha256'].items():
        if sha(destination/name) != expected:
            raise ValueError('Changed exported verification input: '+name)
    script = ROOT/'scripts/extension3_inference_reference.R'
    command = [str(rscript), str(script), str(destination)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
    receipt = json.loads((destination/'R_verification.json').read_text())
    if receipt.get('status') != 'PASS' or receipt.get('comparison_family') != 4:
        raise ValueError('Independent R verification did not pass')
    for name, expected in manifest['exported_input_sha256'].items():
        if sha(destination/name) != expected:
            raise ValueError('Verification input changed while R was running: '+name)
    final = dict(status='PASS', independent_runtime='R', r_script_sha256=sha(script),
                 exporter_sha256=sha(Path(__file__)), receipt=receipt,
                 stdout=result.stdout, stderr=result.stderr,
                 artifact_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file()},
                 historical_fit_run=False, originality_or_performance_claim=False)
    with (destination/'verification_manifest.json').open('x') as handle:
        json.dump(final, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monthly', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rscript', default=shutil.which('Rscript'))
    args = parser.parse_args()
    if args.rscript is None:
        raise RuntimeError('Local Rscript is required for independent verification')
    frame = pd.read_parquet(args.monthly)
    plan = json.loads(args.plan.read_text())
    export_inputs(frame, plan, args.analysis, args.output)
    result = run_reference(args.output, rscript=args.rscript)
    print(json.dumps(result['receipt'], indent=2))


if __name__ == '__main__':
    main()
