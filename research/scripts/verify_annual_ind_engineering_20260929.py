"""Read-only independent artifact/equivalence/resource review of bounded probes."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from extension4.storage import verify_stage, write_json
from extension4.runtime import prior_seconds
from feature_geometry.history import sha


def main():
    destination = ROOT/'output/annual_ind_engineering_20260929/benchmark_review.json'
    if destination.exists():
        raise FileExistsError('Preserve completed review; use a new version for later work')
    audit = json.loads((ROOT/'output/annual_ind_audit_v3_20260929/audit.json').read_text())
    for rel, expected in audit['input_and_code_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Audited dependency changed: '+rel)
    results = []
    for batch in ('annual_ind_background_benchmarks_20260929', 'annual_ind_thread_benchmarks_20260929'):
        folder = ROOT/'output'/batch
        summary = json.loads((folder/'summary.json').read_text())
        if not summary['worker_equivalence_passed'] or summary['full_study_dispatched']:
            raise ValueError('Incomplete/incorrect benchmark scope')
        for row in summary['results']:
            out = ROOT/row['output']
            verify_stage(out/'fresh_parent')
            if sha(out/'fresh_parent/model.npz') != row['fresh_model_sha256']:
                raise ValueError('Model digest differs')
            lock = json.loads((out/'lock.json').read_text())
            for key in ('code_sha256', 'protected_sha256'):
                for rel, expected in lock[key].items():
                    if sha(ROOT/rel) != expected:
                        raise ValueError('Locked dependency changed: '+rel)
            oracle = ROOT/f"output/annual_ind_fresh_probe_bench_p{row['p']}_w1_20260929/fresh_parent/model.npz"
            differences = {}
            with np.load(oracle, allow_pickle=False) as a, np.load(out/'fresh_parent/model.npz', allow_pickle=False) as b:
                for key in ('theta', 'beta', 'frequencies'):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-11, rtol=1e-10)
                    differences[key] = float(np.max(abs(a[key]-b[key])))
            telemetry = [json.loads(line) for p in (out/'attempts').glob('*_resources.jsonl')
                         for line in p.read_text().splitlines()]
            peak = max(r['physical_footprint_gib'] for r in telemetry)
            reserve = min(r['system_available_gib'] for r in telemetry)
            swap = max(r['swap_growth_bytes'] for r in telemetry)
            if peak > 16 or reserve < 4 or swap > 0:
                raise ValueError('Recorded memory safeguard breached')
            results.append(dict(output=row['output'], width=row['p'], workers=row['workers'],
                geometry_fit_seconds=row['geometry_fit_seconds'], max_absolute_difference=differences,
                peak_process_tree_gib=peak, minimum_system_available_gib=reserve, swap_growth_bytes=swap))
    attempts = {str(p.relative_to(ROOT)): prior_seconds(p)[0]
                for p in (ROOT/'output').glob('annual_ind_fresh_probe*') if (p/'lock.json').exists()}
    if sum(attempts.values()) > 7200:
        raise ValueError('Cumulative two-hour numerical-probe allowance exceeded')
    write_json(destination, dict(status='PASS_ENGINEERING_ONLY', at_utc=datetime.now(timezone.utc).isoformat(),
        results=results, attempt_active_seconds=attempts, total_active_seconds=sum(attempts.values()),
        full_study_started=False, full_pipeline_pilot_complete=False, new_portfolio_paths=0,
        limitation='All timing coordinates are 1971 seed1 first bandwidth middle ridge; no full-history ETA.'))
    print(json.dumps(dict(status='PASS_ENGINEERING_ONLY', models_checked=len(results),
                          total_active_seconds=sum(attempts.values()))))


if __name__ == '__main__':
    main()
