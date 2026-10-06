"""Describe nominal significance from accepted saved draws; no new experiment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'output/industry_projection_screen_20260928'
OUT = ROOT / 'output/industry_significance_levels_20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the existing significance clarification')
    accepted = json.loads((PARENT / 'acceptance_manifest.json').read_text())
    assert accepted['accepted'] and accepted['advancement_gate_passed'] is False
    inventory = {**accepted['source_sha256'], **accepted['artifact_sha256']}
    inputs = [PARENT / name for name in ('intervals.json', 'metrics.csv', 'review/result.json')]
    inputs += [PARENT / f'analysis/bootstrap/{period}_L{length}_draws.csv'
               for period in ('primary', 'robustness') for length in (12, 24)]
    inputs += [ROOT / 'config/industry_projection_screen_20260928.json']
    for path in inputs:
        assert sha(path) == inventory[str(path.relative_to(ROOT))], path
    OUT.mkdir()
    write(OUT / 'source_lock.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'User-requested descriptive significance levels; original study gate unchanged',
        'new_candidates': 0, 'new_fits': 0, 'new_paths': 0, 'new_resamples': 0,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs +
                          [PARENT / 'acceptance_manifest.json', Path(__file__).resolve()]},
    })
    intervals = json.loads((PARENT / 'intervals.json').read_text())
    metrics = pd.read_csv(PARENT / 'metrics.csv').set_index(['period', 'method'])
    rows = []
    for old in intervals:
        period, length, control = old['period'], old['block_months'], old['control']
        draws = pd.read_csv(PARENT / f'analysis/bootstrap/{period}_L{length}_draws.csv')
        x = draws[f'IND_vs_{control}'].to_numpy()
        delta = float(old['observed'])
        assert x.shape == (10000,) and np.isfinite(x).all() and delta > 0
        np.testing.assert_allclose(delta, metrics.loc[(period, 'IND'), 'source_utility'] -
                                   metrics.loc[(period, control), 'source_utility'], atol=1e-14, rtol=0)
        original = 2 * delta - np.quantile(x, [.05 / 64, 1 - .05 / 64], method='linear')[::-1]
        np.testing.assert_allclose(original, [old['lower'], old['upper']], atol=2e-15, rtol=0)
        # Under basic-CI logic, null-centered error is draw - observed; compare
        # it with observed - 0. Hence the right tail is draw >= 2*observed.
        upper_count = int(np.count_nonzero(x >= 2 * delta))
        lower_count = int(np.count_nonzero(x <= 2 * delta))
        upper = (1 + upper_count) / (len(x) + 1)
        lower = (1 + lower_count) / (len(x) + 1)
        p = min(1., 2 * min(upper, lower))
        row = dict(period=period, candidate='IND', control=control, block_months=length,
                   observed=delta, draws=len(x), centered_upper_tail_count=upper_count,
                   centered_lower_tail_count=lower_count, nominal_two_sided_p=p,
                   bonferroni32_two_sided_p=min(1., 32 * p))
        for level in (90, 95, 99):
            alpha = 1 - level / 100
            ci = 2 * delta - np.quantile(x, [alpha / 2, 1 - alpha / 2], method='linear')[::-1]
            row[f'nominal_{level}_lower'], row[f'nominal_{level}_upper'] = map(float, ci)
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == 16 and not frame.duplicated(['period', 'control', 'block_months']).any()
    frame.to_csv(OUT / 'significance_levels.csv', index=False)
    write(OUT / 'result.json', {
        'status': 'COMPUTED_DESCRIPTIVE_CLARIFICATION_PENDING_INDEPENDENT_REVIEW',
        'comparisons': 16, 'draws_regenerated': 0, 'models_fitted': 0, 'portfolio_paths_run': 0,
        'original_family32_intervals_reproduced': True, 'original_gate_changed': False,
        'original_gate_passed': False,
        'method': 'Approximate two-sided equal-tail centered bootstrap p-values with plus-one tail counts',
        'finite_simulation': 'Tail counts use (count+1)/(10000+1); linear-quantile CI inversion is approximate at grid boundaries.',
        'interpretation': 'Nominal levels describe individual comparisons only; Bonferroni32 controls the recorded finite family approximately, not all adaptive research.',
        'limitations': ['All historical outcomes exposed', 'Conditional on saved fitted targets',
                        'No model re-estimation uncertainty', 'No p-value is a probability that a hypothesis is true',
                        'No change to block lengths, metric, dates or success gate'],
    })
    print(frame.loc[frame.control.eq('B1'), ['period', 'block_months',
          'nominal_two_sided_p', 'bonferroni32_two_sided_p']].to_string(index=False))


if __name__ == '__main__':
    main()
