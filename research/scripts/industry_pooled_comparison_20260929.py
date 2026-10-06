"""User-requested pooled IND versus B1 inference; saved paths, no refitting."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.core import stationary, basic_interval

PARENT = ROOT / 'output/industry_projection_screen_20260928'
OUT = ROOT / 'output/industry_pooled_comparison_20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def utility(gross, net, axis=0):
    return 12 * (net.mean(axis=axis) - 5 * gross.var(axis=axis, ddof=1))


def main():
    OUT.mkdir(exist_ok=False)
    parent_manifest = PARENT / 'acceptance_manifest.json'
    accepted = json.loads(parent_manifest.read_text())
    assert accepted['accepted'] and accepted['advancement_gate_passed'] is False
    inventory = {**accepted['source_sha256'], **accepted['artifact_sha256']}
    inputs = [PARENT / 'monthly.csv', PARENT / 'review/result.json',
              ROOT / 'config/industry_projection_screen_20260928.json', ROOT / 'inference/core.py']
    for path in inputs:
        assert sha(path) == inventory[str(path.relative_to(ROOT))], str(path)
    plan = dict(
        status='FROZEN_BEFORE_POOLED_RESULT', at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly requested combined2016–2025 significance; hoping for5% does not change calculation or reporting.',
        comparison='IND minus B1', methods=['B1', 'IND'], period=['2016-01-31', '2025-12-31'],
        months=120, utility='12*(mean(net excess)-5*sample variance(gross excess))',
        no_averaging_period_utilities_or_combining_p_values=True,
        bootstrap='paired stationary circular over the entire continuous120month period',
        expected_block_months=[12, 24], replicas=10000, master_seed=20260924,
        seed_namespace='industry_pooled_2016_2025_20260929', seed_date='2025-12-31',
        interval='basic two-sided, linear quantiles', alpha=.05,
        p_value='centered equal-tail two-sided, plus-one MonteCarlo tail counts',
        families=[1, 33], family33_scope='Prior32 finite-family contrasts plus this one newly requested pooled contrast; both block lengths are sensitivity checks.',
        success_rule_unchanged=True, original_gate_passed=False, pooling_is_post_hoc=True,
        all_outcomes_previously_exposed=True, inference_conditions_on_saved_targets=True,
        new_model_fits=0, new_portfolio_paths=0, original_study_outputs_unchanged=True,
        resources='Local single numerical thread; small saved-return analysis only; no raw-data extraction, paid resources or training.',
        numpy_version=np.__version__, pandas_version=pd.__version__,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs + [parent_manifest, Path(__file__).resolve()]})
    write(OUT / 'plan.json', plan)
    started = time.monotonic()
    all_rows = pd.read_csv(PARENT / 'monthly.csv')
    dates = pd.date_range(*plan['period'], freq='ME').strftime('%Y-%m-%d').tolist()
    gross, costs, frames = [], [], []
    for method in plan['methods']:
        x = all_rows[all_rows.method.eq(method)].sort_values('return_eom')
        assert x.return_eom.tolist() == dates and len(x) == 120
        assert x.decision_eom.tolist() == pd.date_range('2015-12-31', '2025-11-30', freq='ME').strftime('%Y-%m-%d').tolist()
        assert (x.impact_cost >= 0).all()
        np.testing.assert_allclose(x.net_excess, x.gross_excess - x.impact_cost, atol=1e-15, rtol=0)
        gross.append(x.gross_excess.to_numpy()); costs.append(x.impact_cost.to_numpy()); frames.append(x)
    gross = np.column_stack(gross); costs = np.column_stack(costs); net = gross - costs
    assert np.isfinite(gross).all() and np.isfinite(costs).all()
    pd.concat(frames, ignore_index=True).to_csv(OUT / 'monthly_used.csv', index=False)
    values = utility(gross, net); observed = float(values[1] - values[0])
    pd.DataFrame([dict(method=m, months=120, annual_source_utility=float(values[j]),
        annual_net_mean=float(12 * net[:, j].mean()), annual_impact_cost=float(12 * costs[:, j].mean()),
        annual_gross_volatility=float(np.sqrt(12 * gross[:, j].var(ddof=1))))
        for j, m in enumerate(plan['methods'])]).to_csv(OUT / 'metrics.csv', index=False)
    results = []
    for length in plan['expected_block_months']:
        idx, starts, restarts, digests = stationary(plan['seed_namespace'], plan['master_seed'], length,
            plan['seed_date'], 120, replicas=10000)
        assert idx.shape == (10000, 120) and 0 <= idx.min() <= idx.max() < 120
        np.asarray(idx + 1, dtype='<i4').tofile(OUT / f'indices_L{length}.bin')
        np.savez_compressed(OUT / f'primitives_L{length}.npz', starts=starts, restarts=restarts,
                            seed_digests=np.asarray(digests))
        draws = np.empty(10000)
        for first in range(0, 10000, 250):
            ix = idx[first:first + 250]
            u = utility(gross[ix], net[ix], axis=1)
            draws[first:first + 250] = u[:, 1] - u[:, 0]
        pd.DataFrame({'IND_vs_B1': draws}).to_csv(OUT / f'draws_L{length}.csv', index=False)
        upper_count = int(np.count_nonzero(draws - observed >= observed))
        lower_count = int(np.count_nonzero(draws - observed <= observed))
        p = min(1., 2 * min((upper_count + 1) / 10001, (lower_count + 1) / 10001))
        intervals = {}
        for family in plan['families']:
            lo, hi = basic_interval(observed, draws, alpha=.05, family=family)
            intervals[str(family)] = dict(lower=lo, upper=hi, positive=bool(lo > 0))
        results.append(dict(block_months=length, observed=observed, nominal_two_sided_p=p,
            bonferroni33_two_sided_p=min(1., 33 * p), centered_upper_tail_count=upper_count,
            centered_lower_tail_count=lower_count, intervals=intervals))
    write(OUT / 'result.json', dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
        period=plan['period'], months=120, baseline_utility=float(values[0]), candidate_utility=float(values[1]),
        annual_utility_difference=observed, inference=results,
        nominal_5_percent_both_blocks=bool(all(x['nominal_two_sided_p'] < .05 and x['intervals']['1']['positive'] for x in results)),
        adjusted_5_percent_both_blocks=bool(all(x['bonferroni33_two_sided_p'] < .05 and x['intervals']['33']['positive'] for x in results)),
        original_gate_changed=False, new_models_fitted=0, new_paths_run=0, new_pooled_resamples=20000,
        elapsed_seconds=time.monotonic()-started, independent_verification_pending=True,
        limits='New post-hoc pooling of exposed history; no fresh holdout, no model-reestimation uncertainty, stationarity approximation, finite-family adjustment does not cover all adaptive research.'))
    print(json.dumps(json.loads((OUT / 'result.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
