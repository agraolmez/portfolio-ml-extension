"""Apply the authors' released Table 3 inference to accepted IND/B1 paths."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'output/industry_projection_screen_20260928'
OUT = ROOT / 'output/industry_author_test_20260929'
AUTHOR = ROOT / 'references/author_code/6 - Base analysis.R'
PERIODS = [
    ('pooled', '2016-01-31', '2025-12-31', 120),
    ('primary', '2016-01-31', '2020-12-31', 60),
    ('robustness', '2021-01-31', '2025-12-31', 60),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    parent_manifest = PARENT / 'acceptance_manifest.json'
    receipt = json.loads(parent_manifest.read_text())
    assert receipt['accepted'] and not receipt['advancement_gate_passed']
    monthly_path = PARENT / 'monthly.csv'
    assert sha(monthly_path) == receipt['artifact_sha256'][str(monthly_path.relative_to(ROOT))]
    author_text = AUTHOR.read_text()
    for required in ['e_var_adj := (r-mean(r))^2',
                     'utility_t := r-tc-0.5*e_var_adj*pf_set$gamma_rel',
                     '1-pnorm(q = 0, mean=mean(diff), sd = sd(diff)/sqrt(length(diff)))']:
        assert required in author_text, required
    sources = [monthly_path, parent_manifest, AUTHOR,
               ROOT / 'references/author_code/7 - Tables.R',
               ROOT / 'references/Portfolio-ML.pdf', Path(__file__).resolve()]
    plan = dict(
        status='FROZEN_BEFORE_AUTHOR_METHOD_RESULTS', at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested the original paper statistical method applied to our best observed model.',
        comparison='IND minus B1; no new model selection', methods=['B1', 'IND'],
        periods=[dict(label=l, start=s, end=e, months=n) for l, s, e, n in PERIODS],
        controlling_implementation='Released 6 - Base analysis.R lines 25-26 and 101-110',
        monthly_flow='gross - impact_cost - 5*(gross - within-period mean(gross))^2',
        statistic='z = mean(paired flow difference) / (sample sd(paired flow difference)/sqrt(N))',
        outperformance_probability='Phi(z)', conventional_one_sided_p='Phi(-z)',
        normal_distribution=True, multiple_comparison_adjustment=False,
        serial_correlation_adjustment=False, bootstrap=False, model_reestimation=False,
        original_gates_unchanged=True, previously_exposed_history=True,
        annual_table3_flow_uses_population_variance=True,
        annual_table2_summary_uses_sample_variance=True,
        text_code_discrepancies=[
            'Printed p32/Table3 uses centered net returns in risk term; released code uses gross returns.',
            'Printed p32 posterior variance shows sample variance without division by N; released code uses sd/sqrt(N).',
            'Use released executable author method; do not choose between versions after observing outcomes.'
        ],
        resources='Local single-thread saved-return arithmetic only; no model fitting, paths, bootstrap resampling or paid resources.',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    write(OUT / 'plan.json', plan)
    data = pd.read_csv(monthly_path)
    results, flows = [], []
    for label, start, end, n in PERIODS:
        dates = pd.date_range(start, end, freq='ME').strftime('%Y-%m-%d').tolist()
        values = {}
        for method in plan['methods']:
            x = data[data.method.eq(method) & data.return_eom.between(start, end)].sort_values('return_eom')
            assert len(x) == n and x.return_eom.tolist() == dates
            gross = x.gross_excess.to_numpy(); cost = x.impact_cost.to_numpy()
            assert np.isfinite(gross).all() and np.isfinite(cost).all() and (cost >= 0).all()
            np.testing.assert_allclose(gross-cost, x.net_excess, atol=1e-15, rtol=0)
            flow = gross-cost-5*(gross-gross.mean())**2
            values[method] = (flow, float(12*(np.mean(gross-cost)-5*np.var(gross, ddof=1))))
            for j, date in enumerate(dates):
                flows.append(dict(period=label, method=method, return_eom=date,
                    gross_excess=float(gross[j]), impact_cost=float(cost[j]),
                    within_period_gross_mean=float(gross.mean()), utility_flow=float(flow[j])))
        difference = values['IND'][0]-values['B1'][0]
        mean = float(difference.mean()); sd = float(difference.std(ddof=1)); se=sd/math.sqrt(n)
        assert se > 0
        z = mean/se
        p = .5*math.erfc(z/math.sqrt(2)); probability = .5*math.erfc(-z/math.sqrt(2))
        results.append(dict(period=label, start=start, end=end, months=n,
            annual_baseline_mean_flow=float(12*values['B1'][0].mean()),
            annual_ind_mean_flow=float(12*values['IND'][0].mean()),
            annual_flow_difference=12*mean,
            annual_baseline_table2_utility=values['B1'][1],
            annual_ind_table2_utility=values['IND'][1],
            monthly_difference_mean=mean, monthly_difference_sample_sd=sd,
            monthly_mean_standard_error=se, z=z, outperformance_probability=probability,
            unadjusted_one_sided_p=p, nominal_5_percent=bool(p < .05), nominal_1_percent=bool(p < .01)))
    pd.DataFrame(flows).to_csv(OUT / 'monthly_flows.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / 'comparison.csv', index=False)
    result = dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
        method='Authors released-code Table3 normal one-sided pairwise calculation',
        comparison='IND versus reconstructed baseline B1', results=results,
        original_gate_changed=False, original_gate_passed=False,
        new_models_fitted=0, new_portfolio_paths=0, new_bootstrap_draws=0,
        published_paper_superiority_established=False,
        limitations='Post-hoc method translation on previously inspected outcomes; ordinary sd/sqrt(N) assumes away serial dependence; no search/multiplicity or model-selection adjustment.')
    write(OUT / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
