"""Verify report bindings and independently recompute final study metrics."""
import json
import math
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from feature_geometry.history import sha
from geometry_study.run import ROOT, OUT, locked, write


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.current, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.current = []
        elif tag == 'tr' and self.current is not None: self.row = []
        elif tag in ('th', 'td') and self.row is not None: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split())); self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.current.append(self.row); self.row = None
        elif tag == 'table' and self.current is not None:
            self.tables.append(self.current); self.current = None


def verify():
    plan = locked()['plan']
    receipt = json.loads((OUT/'numerical_completion_verification.json').read_text())
    assert receipt['status'] == 'NUMERICAL_STUDY_VERIFIED_PENDING_PRESENTATION'
    for rel, digest in receipt['artifact_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    monthly = pd.read_parquet(OUT/'paths/monthly.parquet')
    metrics = pd.read_csv(OUT/'analysis/strategy_summary.csv')
    parts = pd.read_csv(OUT/'analysis/utility_decomposition.csv')
    intervals = pd.read_csv(OUT/'analysis/primary_intervals.csv')
    seeds = pd.read_csv(OUT/'analysis/seed_contrasts.csv')
    calculated, max_error = {}, 0.0
    for row in metrics.itertuples(index=False):
        first, last = plan['periods'][row.period]
        frame = monthly[(monthly.seed == row.seed) & (monthly.method == row.method)
                        & monthly.return_eom.dt.year.between(first, last)]
        net = frame.net_excess.tolist(); count = len(net)
        assert count == 12*(last-first+1) == row.months
        mean = math.fsum(net)/count
        variance = math.fsum((v-mean)**2 for v in net)/count
        values = dict(annual_net_excess_mean=12*mean,
                      annual_gross_excess_mean=12*math.fsum(frame.gross_excess)/count,
                      annual_net_volatility=math.sqrt(12*variance),
                      annual_net_utility=12*(mean-5*variance),
                      annual_impact_bps=120000*math.fsum(frame.impact_cost)/count,
                      monthly_absolute_turnover=math.fsum(frame.turnover)/count,
                      annual_net_variance=12*variance)
        calculated[row.period, row.seed, row.method] = values
        for key, value in values.items():
            if key == 'annual_net_variance': continue
            error = abs(value-getattr(row, key)); max_error = max(max_error, error)
            assert error < 1e-10, (row.period, row.seed, row.method, key, error)
    for row in parts.itertuples(index=False):
        a, b = row.contrast.split('-', 1)
        full, control = calculated[row.period, row.seed, a], calculated[row.period, row.seed, b]
        expected = dict(delta_utility=full['annual_net_utility']-control['annual_net_utility'],
                        gross_mean_contribution=full['annual_gross_excess_mean']-control['annual_gross_excess_mean'],
                        cost_mean_contribution=-(full['annual_impact_bps']-control['annual_impact_bps'])/10000,
                        net_variance_contribution=-5*(full['annual_net_variance']-control['annual_net_variance']))
        for key, value in expected.items(): assert abs(value-getattr(row, key)) < 1e-12
    labels = {'fixed512':'Fixed 512','full':'Full economic geometry','risk_only':'Risk-only geometry',
              'return_prediction':'Prediction geometry','joint':'Joint gradient training',
              'joint_refit':'Joint geometry, exact refit','fixed1024':'Fixed 1,024',
              'fixed1024_double_ridge':'Fixed 1,024, double ridge'}
    expected_tables = []
    for period in ['evaluation','exposed_robustness']:
        table = [['Method','Net mean (%/yr)','Net volatility (%/yr)','Net utility (%/yr)','Impact (bp/yr)','Monthly turnover']]
        for method, label in labels.items():
            values = [calculated[period,s,method] for s in [1,2,3]]
            avg = lambda key: math.fsum(v[key] for v in values)/3
            table.append([label, f"{100*avg('annual_net_excess_mean'):.2f}",
                          f"{100*avg('annual_net_volatility'):.2f}",f"{100*avg('annual_net_utility'):.2f}",
                          f"{avg('annual_impact_bps'):.2f}",f"{avg('monthly_absolute_turnover'):.3f}"])
        expected_tables.append(table)
    for period in ['evaluation','exposed_robustness']:
        table = [['Full geometry minus','Mean Δ (bp/yr)','Seed 1 / 2 / 3 (bp/yr)','Adjusted interval, L=12','Adjusted interval, L=24']]
        for contrast in plan['primary_contrasts']:
            control = contrast.split('-',1)[1]
            deltas = [calculated[period,s,'full']['annual_net_utility']-calculated[period,s,control]['annual_net_utility'] for s in [1,2,3]]
            ir = intervals[(intervals.period == period)&(intervals.contrast == contrast)].set_index('block_months')
            table.append([labels[control],f'{math.fsum(deltas)/3*10000:+.2f}',
                          ' / '.join(f'{d*10000:+.2f}' for d in deltas),
                          *[f'[{ir.loc[length].adjusted_lower*10000:+.2f}, {ir.loc[length].adjusted_upper*10000:+.2f}]' for length in [12,24]]])
        expected_tables.append(table)
    part = parts[(parts.period=='evaluation')&(parts.contrast=='full-fixed512')]
    expected_tables.append([['Contribution','Annual utility contribution (bp)']]+[
        [label,f'{part[key].mean()*10000:+.2f}'] for label,key in [
            ('Gross mean return','gross_mean_contribution'),('Mean impact cost','cost_mean_contribution'),
            ('Net-return variance','net_variance_contribution'),('Total net utility','delta_utility')]])
    original = (OUT/'report/completion_report.html').read_text()
    reviewed = (OUT/'report_reviewed/completion_report.html').read_text()
    change = json.loads((OUT/'final_review/presentation_revision.json').read_text())['only_change']
    assert original.replace(change['before'],change['after']) == reviewed
    parser = Tables(); parser.feed(reviewed); assert parser.tables == expected_tables
    for rel in ['findings.md','utility_intervals.png','utility_intervals.svg']:
        assert sha(OUT/'report'/rel) == sha(OUT/'report_reviewed'/rel)
    primary = intervals[intervals.period=='evaluation']
    success = bool((primary.adjusted_lower>0).all() and (primary.mean_seed_delta>0).all()
                   and (seeds[seeds.period=='evaluation'][['seed1','seed2','seed3']]>0).all().all()
                   and (seeds[seeds.period=='exposed_robustness'].mean_seed_delta>0).all())
    assert success is False and receipt['primary_family_improvement_established'] is False
    record = dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  metric_groups_recomputed=len(metrics),decomposition_rows_checked=len(parts),
                  max_metric_absolute_error=max_error,html_tables_checked=len(parser.tables),
                  numerical_inventory_files_reverified=len(receipt['artifact_sha256']),
                  primary_success_criterion_met=success,presentation_only_change=True,
                  original_artifacts_unchanged=True,verification_code_sha256=sha(Path(__file__)))
    write(OUT/'final_review/final_checks.json',record)
    print(json.dumps(record))


if __name__ == '__main__': verify()
