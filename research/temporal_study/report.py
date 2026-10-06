"""Self-contained, source-backed HTML/Markdown for completed analysis only.

Rendering never opens historical datasets, fits models, approves a study, or
changes source artifacts. Full-study completion requires explicit caller gates
in addition to completed analytical calculations. Optional diagnostic gaps are
shown rather than invented or silently replaced with a successful status.
"""
import html
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_geometry.history import sha


LABELS = {'parent': 'Extension 2 parent', 'neutral': 'All-one neutral replay',
          'selected_temporal': 'Extension 3 temporal gains',
          'fixed_temporal': 'Fixed temporal control',
          'selected_current_only': 'Current-only gain control',
          'fixed1024': 'Fixed 1,024-feature benchmark'}
GATES = ('approved_full_historical_run', 'complete_annual_fits', 'complete_continuous_paths',
         'neutral_equivalence_passed', 'frozen_inputs_verified')


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        raise TypeError(type(item).__name__)
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _records(table):
    """Preserve structured diagnostic cells through a CSV-based resume."""
    clean = table.astype(object).where(pd.notna(table), None)
    for column in ('unavailable_position_metrics', 'parameters', 'coverage', 'boundary_flags'):
        if column in clean:
            clean[column] = clean[column].map(lambda value: ast.literal_eval(value)
                if isinstance(value, str) and value[:1] in ('[', '{') else value)
    return _json(clean.to_dict('records'))


def verify_report_inputs(destination, analysis_result, *, fit_diagnostics=None, theme_gains=None):
    """Verify immutable report bytes and their complete source snapshot on resume."""
    folder = Path(destination)
    manifest = json.loads((folder/'report_manifest.json').read_text())
    for relative, expected in manifest['artifact_sha256'].items():
        path = (folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError('Changed or escaped report artifact: '+relative)
    source = json.loads((folder/'report_source.json').read_text())
    if (source['analysis_result'] != _json(analysis_result['result'])
            or source['completion_gates'] != dict(analysis_result.get('completion_gates', {}))
            or source['synthetic_fixture'] != (analysis_result.get('synthetic_fixture') is True)):
        raise ValueError('Report no longer matches its analysis or completion gates')
    for name, rows in source['full_analysis_tables'].items():
        if name not in analysis_result or rows != _records(analysis_result[name]):
            raise ValueError('Report source table differs: '+name)
    if isinstance(theme_gains, pd.DataFrame) and 'method' not in theme_gains and 'kind' in theme_gains:
        theme_gains = theme_gains.rename(columns={'kind': 'method'})
    for name, table in [('fit_diagnostics', fit_diagnostics), ('theme_gains', theme_gains)]:
        expected = None if table is None else _records(table)
        if source[name] != expected:
            raise ValueError('Report diagnostic evidence differs: '+name)
    return manifest


def _number(value, *, multiplier=1., suffix='', places=2, signed=False):
    if value is None or pd.isna(value):
        return 'Unavailable'
    return f'{float(value)*multiplier:{"+" if signed else ""}.{places}f}{suffix}'


def _html_table(frame, caption):
    if frame.empty:
        return '<p class="muted">No records were supplied.</p>'
    heads = ''.join(f'<th scope="col">{html.escape(str(column))}</th>' for column in frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        cells = ''.join(f'<td>{html.escape("Unavailable" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))}</td>' for value in row)
        rows.append('<tr>'+cells+'</tr>')
    return ('<div class="table-wrap" tabindex="0"><table><caption>'+html.escape(caption)
            +'</caption><thead><tr>'+heads+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>')


def _md_table(frame):
    if frame.empty:
        return 'No records were supplied.'
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| '+' | '.join(map(cell, frame.columns))+' |',
             '| '+' | '.join('---' for _ in frame.columns)+' |']
    lines.extend('| '+' | '.join(cell(value) for value in row)+' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def _period_label(plan, period):
    first, last = plan['periods'][period]
    return f'{first}–{last}'


def write_report(analysis_result, destination, *, fit_diagnostics=None, theme_gains=None):
    """Render an immutable local report, including honest pending gate status.

    ``analysis_result`` is the mapping returned by ``analyze_paths``. The caller
    may add ``completion_gates`` using the five numerical GATES above, each a real
    Boolean. Full acceptance additionally requires report_visual_review_passed
    and final_study_acceptance_passed; neither is inferred from rendering.
    A synthetic QA mapping must additionally set ``synthetic_fixture``
    to True; that label overrides any study-completion wording throughout.
    Diagnostics and gains are optional DataFrames; missing evidence is visible.
    """
    result = _json(analysis_result['result'])
    if result.get('status') != 'COMPLETE_ANALYSIS':
        raise ValueError('Only completed analysis may be rendered as a results report')
    required_tables = ('strategy_summary', 'mean_seed_metrics', 'seed_contrasts',
                       'primary_intervals', 'selection_links', 'utility_decomposition')
    if any(not isinstance(analysis_result.get(name), pd.DataFrame) for name in required_tables):
        raise ValueError('Complete source tables are required for the report')
    tables = {name: analysis_result[name].copy(deep=True) for name in required_tables}
    gates = dict(analysis_result.get('completion_gates', {}))
    if any(name in gates and type(gates[name]) is not bool for name in GATES):
        raise ValueError('Completion gate values must be explicit Booleans')
    synthetic = analysis_result.get('synthetic_fixture') is True
    numerical_complete = all(gates.get(name) is True for name in GATES) and not synthetic
    study_complete = (numerical_complete and gates.get('report_visual_review_passed') is True
                      and gates.get('final_study_acceptance_passed') is True)
    if result.get('all_conditions_met') != all(result['approved_success_conditions'].values()):
        raise ValueError('Success statement does not match the recorded full decision rule')
    plan = result['approved_study_plan']
    primary_period = _period_label(plan, 'evaluation')
    robustness_period = _period_label(plan, 'exposed_robustness')
    contrasts = tables['seed_contrasts'].query("period == 'evaluation'")
    parent_contrast = contrasts[(contrasts.left == 'selected_temporal') & (contrasts.right == 'parent')]
    if len(parent_contrast) != 1:
        raise ValueError('Unique primary Extension-2 comparison required')
    parent_delta = float(parent_contrast.iloc[0].mean_seed_delta)
    mean_metrics = tables['mean_seed_metrics'].query("period == 'evaluation'")
    method_order = ['parent', 'selected_temporal', 'fixed_temporal', 'selected_current_only',
                    'fixed1024', 'neutral']
    mean_metrics = mean_metrics.assign(_display_order=mean_metrics.method.map(
        {method: i for i, method in enumerate(method_order)}).fillna(len(method_order))).sort_values('_display_order')
    utilities = {}
    for method in ('selected_temporal', 'parent'):
        matched = mean_metrics[mean_metrics.method.eq(method)]
        if len(matched) != 1:
            raise ValueError('Unique seed-average parent and Extension-3 metrics required')
        utilities[method] = float(matched.iloc[0].annual_net_utility)
    if abs(utilities['selected_temporal']-utilities['parent']-parent_delta) > 1e-12:
        raise ValueError('Displayed utility difference does not reconcile with source tables')
    headline = ('Temporal inputs met the planned historical comparison criteria'
                if result['all_conditions_met'] else
                'Temporal inputs did not meet every planned historical comparison criterion')
    if synthetic:
        title = 'Synthetic layout check — not research results'
        status = 'Artificial test data; no historical experiment or result is represented here.'
    else:
        title = 'Extension 3: temporal inputs on the learned-scale model'
        status = ('Historical study completed and accepted.' if study_complete else
                  'Historical fitting, portfolio paths and analysis completed. Report visual review and final study acceptance are pending.'
                  if numerical_complete else
                  'Analysis completed; full-study completion has not been certified by all required gates.')
    headline = 'Illustrative test verdict only' if synthetic else headline
    primary_intervals = tables['primary_intervals'].query("period == 'evaluation' and aggregation == 'mean_seed'")
    comparison_rows = []
    for left, right in plan['primary_contrasts']:
        observed = contrasts[contrasts.left.eq(left) & contrasts.right.eq(right)]
        if len(observed) != 1:
            raise ValueError('Every declared contrast needs a unique observed estimate')
        observed = observed.iloc[0]
        row = {'Comparator': LABELS.get(right, right),
               'Utility change (bp/year)': _number(observed.mean_seed_delta, multiplier=10000, signed=True)}
        for length in plan['inference']['block_months']:
            match = primary_intervals[primary_intervals.left.eq(left) & primary_intervals.right.eq(right)
                                      & primary_intervals.block_months.eq(length)]
            if len(match) != 1:
                raise ValueError('All configured paired uncertainty intervals are required')
            interval = match.iloc[0]
            row[f'{length}-month blocks: adjusted interval (bp)'] = (
                _number(interval.adjusted_lower, multiplier=10000, signed=True)+' to '
                +_number(interval.adjusted_upper, multiplier=10000, signed=True))
        row['Positive point in all 3 seeds'] = 'Yes' if observed.all_seed_points_positive else 'No'
        comparison_rows.append(row)
    comparison_table = pd.DataFrame(comparison_rows)
    metric_rows = []
    for row in mean_metrics.itertuples():
        metric_rows.append({'Model': LABELS.get(row.method, row.method),
            'Annual net utility': _number(row.annual_net_utility, multiplier=100, suffix='%'),
            'Annual net excess mean': _number(row.annual_net_excess_mean, multiplier=100, suffix='%'),
            'Annual net volatility': _number(row.annual_net_volatility, multiplier=100, suffix='%'),
            'Net Sharpe': _number(row.net_sharpe),
            'Annual impact (bp)': _number(row.annual_impact_bps),
            'Monthly turnover': _number(row.mean_monthly_absolute_turnover, multiplier=100, suffix='%'),
            'Gross exposure': _number(row.mean_gross_exposure, suffix='×'),
            'Period index drawdown': _number(row.period_maximum_drawdown, multiplier=100, suffix='%')})
    metric_table = pd.DataFrame(metric_rows)
    seed_rows = []
    selected_seed = tables['strategy_summary'][tables['strategy_summary'].method.isin(
        ['selected_temporal', 'parent', 'selected_current_only'])
        & tables['strategy_summary'].period.isin(['evaluation', 'exposed_robustness'])]
    for row in selected_seed.itertuples():
        seed_rows.append({'Period': _period_label(plan, row.period), 'Seed': row.seed,
                         'Model': LABELS.get(row.method, row.method),
                         'Annual net utility': _number(row.annual_net_utility, multiplier=100, suffix='%'),
                         'Net Sharpe': _number(row.net_sharpe),
                         'Annual impact (bp)': _number(row.annual_impact_bps)})
    seed_table = pd.DataFrame(seed_rows)
    robustness_rows = []
    for row in tables['seed_contrasts'].query("period == 'exposed_robustness'").itertuples():
        robustness_rows.append({'Comparator': LABELS.get(row.right, row.right),
                               'Utility change (bp/year)': _number(row.mean_seed_delta, multiplier=10000, signed=True),
                               'Positive mean across seeds': 'Yes' if row.mean_seed_delta > 0 else 'No'})
    robustness_table = pd.DataFrame(robustness_rows)
    choices = []
    exposure_counts = []
    for family, receipt in result['selection_metadata'].items():
        for choice in receipt['selected_by_seed']:
            choices.append({'Model family': family, 'Seed': choice['seed'],
                            'Chosen penalty': choice['parameters']['rho'],
                            'Selection utility': _number(choice['annual_net_utility'], multiplier=100, suffix='%')})
        exposure_counts.append({'Model family': family,
                                'Inspected candidate/seed paths': len(receipt['candidate_exposure']),
                                'Penalties tried per seed': len(receipt['plan']['candidates']),
                                'Selection returns': receipt['plan']['validation_first_return']+' to '
                                                     +receipt['plan']['validation_last_return']})
    choice_table, exposure_table = pd.DataFrame(choices), pd.DataFrame(exposure_counts)
    if fit_diagnostics is not None and not isinstance(fit_diagnostics, pd.DataFrame):
        raise ValueError('Fit diagnostics must be a supplied evidence table')
    if theme_gains is not None:
        if isinstance(theme_gains, pd.DataFrame) and 'method' not in theme_gains and 'kind' in theme_gains:
            theme_gains = theme_gains.rename(columns={'kind': 'method'}).copy(deep=True)
        if not isinstance(theme_gains, pd.DataFrame) or not {'year', 'seed', 'method', 'theme', 'gain'}.issubset(theme_gains):
            raise ValueError('Theme gains require year, seed, method, theme and gain columns')
        if not np.isfinite(theme_gains.gain.to_numpy(dtype=float)).all():
            raise ValueError('Theme gain evidence must be finite')
    diagnostics_text = ('Annual fit diagnostics were not supplied to this report.' if fit_diagnostics is None
                        else f'{len(fit_diagnostics):,} annual fit records are supplied below; statuses remain as recorded.')
    gains_text = ('Annual theme-gain values were not supplied to this report.' if theme_gains is None
                  else f'{len(theme_gains):,} year/seed/theme gain records are supplied below. '
                       'A temporal gain of one reproduces the parent input; below one smooths its innovation; above one emphasizes it.')
    coverage_text = ('Temporal coverage counts cells with genuine current and required past source observations; '
        'ineligible cells keep the archived input unchanged. The current-only control uses current inputs, including '
        'imputed values. Its all-valid input count is not 100% historical coverage or proof that every source value was observed.')
    summary = (f'For {primary_period}, mean annual net utility across the three separate seeds was '
               f'{_number(utilities["selected_temporal"], multiplier=100, suffix="%")}, compared with '
               f'{_number(utilities["parent"], multiplier=100, suffix="%") } for Extension 2: '
               f'{_number(parent_delta, multiplier=10000, signed=True)} basis points per year.')
    success_text = ('The full rule requires positive primary point estimates in all three seeds, positive '
                    'four-comparison-adjusted lower bounds at every declared block length, and positive mean '
                    f'utility differences for all four comparisons in {robustness_period}. '
                    +('Every condition was met.' if result['all_conditions_met'] else 'At least one condition was not met.'))
    implementation = ('Extension 3 changes the inputs before the existing nonlinear transformation. It compares each '
        'current characteristic with its lagged reference, then learns how strongly each economic theme should react '
        'to that difference. The archived Extension-2 characteristic scales and random feature directions stay fixed. '
        'The original trading rule, risk inputs, trading-cost inputs and annual fitting cutoffs remain unchanged. '
        'Portfolio coefficients are solved again for each proposed gain configuration.')
    evidence = ('All figures average statistics computed separately for seeds 1, 2 and 3. They do not describe a '
        'portfolio made by averaging those three strategies. Annual net utility is 12 × '
        '[mean monthly net excess return − 5 × population variance]. Trading costs come from each strategy’s '
        'actual orders, with continuous own holdings and the first trade charged.')
    uncertainty = (f'The intervals use {plan["inference"]["replicas"]:,} paired stationary-bootstrap samples. '
        'Each resampled month is shared across all methods and seeds; mean block lengths are '
        +', '.join(map(str, plan['inference']['block_months']))+' months. The four contrasts form one '
        'Bonferroni-adjusted family within each period and block length. Models and penalty selection are not refitted '
        'inside the bootstrap. These intervals are conditional on the fitted and selected models.')
    limitations = [
        'Evaluation and robustness outcomes were previously exposed during this research. This is not a fresh holdout.',
        'The selected development and selection histories are descriptive histories of the candidate chosen later; '
        'its identity was not known at the start of those histories.',
        'Originality beyond existing research and source publication-time certification are not established.',
        'The Extension-2 learned scales retain their archived fitting status; freezing them does not establish that their original optimization converged.',
        'Compounded net-excess drawdown is a diagnostic index, not investor total-return wealth. Each period index '
        'starts at one, includes its initial peak and never resets at annual refits. Continuous-path drawdown retains '
        'earlier peaks. A return at or below −100% makes the relevant compound index unavailable.',
        'Failure of the declared criterion does not prove equivalence, universal inferiority or absence of any useful temporal signal.']
    if result['unavailable_position_metrics']:
        limitations.append('Some position/concentration statistics were unavailable in the supplied tables; missing values were not treated as zero.')
    additional = result['candidate_exposure_metadata']['additional_candidate_exposure']
    exposure_note = ('No additional candidate exposure beyond the supplied family inventories was declared.' if not additional
                     else f'{len(additional)} additional exposure records were declared and are retained in the source snapshot.')
    gate_table = pd.DataFrame([{'Completion check': name.replace('_', ' '),
                               'Status': 'Passed' if gates.get(name) is True else 'Not certified'}
                              for name in GATES+('report_visual_review_passed', 'final_study_acceptance_passed')])
    source_table = pd.DataFrame([{'Source artifact': name, 'SHA256': value}
                                for name, value in result['supplied_inputs_sha256'].items()])
    display_tables = dict(primary_comparisons=comparison_table, primary_metrics=metric_table,
                          per_seed_metrics=seed_table, robustness_comparisons=robustness_table,
                          selected_penalties=choice_table, candidate_exposure=exposure_table,
                          completion_gates=gate_table, source_artifacts=source_table)
    sections = []
    def section(title_text, paragraphs=(), table=None, caption=None, details=False):
        body = ''.join('<p>'+html.escape(text)+'</p>' for text in paragraphs)
        if table is not None:
            body += _html_table(table, caption or title_text)
        if details:
            sections.append('<details><summary>'+html.escape(title_text)+'</summary>'+body+'</details>')
        else:
            sections.append('<section><h2>'+html.escape(title_text)+'</h2>'+body+'</section>')
    section(headline, [summary, success_text], comparison_table,
            f'{primary_period}: Extension 3 minus each comparator; annual net utility')
    section('What changed', [implementation])
    section(f'{primary_period}: return, risk and trading costs', [evidence], metric_table,
            'Arithmetic means of separately computed seed statistics')
    section(f'{robustness_period}: previously exposed robustness period',
            ['These results check whether the mean utility differences reverse sign after the primary period.'],
            robustness_table)
    section('Results for each seed', [], seed_table, details=True)
    section('Penalty choices and research exposure',
            ['Each seed selects its penalty using the declared selection period; that choice stays fixed for later evaluation.', exposure_note],
            choice_table)
    section('Every declared candidate remains in the record', [], exposure_table, details=True)
    section('Annual fits and learned theme gains', [diagnostics_text, gains_text, coverage_text])
    if fit_diagnostics is not None:
        section('Inspect every supplied annual fit', [], fit_diagnostics, details=True)
    if theme_gains is not None:
        section('Inspect every supplied theme gain', [], theme_gains, details=True)
    section('Uncertainty and interpretation', [uncertainty]+limitations)
    section('Completion evidence', [status], gate_table, details=True)
    section('Sources and reproducibility',
            ['The report is generated from completed analysis tables and the supplied verified-input inventory. '
             'Source hashes and complete selection receipts are retained in report_source.json.'], source_table, details=True)
    css = '''
    :root{color-scheme:light;--ink:#18283b;--muted:#546575;--rule:#d9e1e7;--accent:#126c67}
    *{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1180px;margin:40px auto;padding:40px 44px;background:white;border:1px solid var(--rule);border-radius:16px}
    header{padding:0 0 30px;border-bottom:2px solid var(--accent)}h1{font-size:clamp(1.8rem,4vw,2.55rem);line-height:1.15;max-width:950px;margin:10px 0 18px}
    h2{font-size:1.35rem;line-height:1.3;margin:0 0 16px}p{max-width:960px;margin:12px 0}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.05em;text-transform:uppercase;font-size:.8rem}
    .status{color:var(--muted)}.synthetic{padding:14px 18px;background:#fff3cf;border:2px solid #a47508;font-weight:700;border-radius:8px;margin-bottom:24px}
    section{margin-top:36px}details{margin-top:24px;padding-top:16px;border-top:1px solid var(--rule)}summary{cursor:pointer;font-weight:650;font-size:1.06rem;padding:4px 0}details[open] summary{margin-bottom:16px}
    .table-wrap{overflow:auto;margin:20px 0;border:1px solid var(--rule);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:.875rem;font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:600;padding:14px 16px;background:#f2f6f7}
    th,td{padding:11px 13px;text-align:left;vertical-align:top;border-top:1px solid var(--rule)}th{background:#eaf1f2;font-weight:650;min-width:105px}th:first-child,td:first-child{min-width:165px}td{overflow-wrap:anywhere}tbody tr:nth-child(even){background:#f8fafb}
    a{color:var(--accent)}.downloads{margin-top:30px;display:flex;flex-wrap:wrap;gap:12px}.downloads a{padding:8px 13px;border:1px solid var(--rule);border-radius:6px;text-decoration:none}.muted,footer{color:var(--muted)}footer{margin-top:40px;font-size:.85rem}
    @media(max-width:650px){main{margin:0;padding:24px 18px;border:0;border-radius:0}body{font-size:15px}section{margin-top:28px}th,td{padding:9px 11px}}
    @media print{body{background:white}main{margin:0;border:0;padding:0;max-width:none}.downloads{display:none}details{break-inside:avoid}.table-wrap{overflow:visible}h2{break-after:avoid}tr{break-inside:avoid}}
    '''
    banner = '<div class="synthetic">SYNTHETIC QA FIXTURE — these are not historical research results.</div>' if synthetic else ''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>'+html.escape(title)+'</title><style>'+css+'</style></head><body><main>'+banner
                +'<header><div class="eyebrow">Portfolio-ML research · Extension 3</div><h1>'+html.escape(title)
                +'</h1><p class="status">'+html.escape(status)+'</p></header>'+''.join(sections)
                +'<nav class="downloads" aria-label="Report downloads"><a href="completion_report.md">Markdown report</a>'
                '<a href="report_source.json">Complete evidence snapshot</a><a href="primary_comparisons.csv">Comparison table</a></nav>'
                '<footer>Local research artifact · completed analysis and retained source receipts.</footer>'
                '</main></body></html>')
    markdown = ['# '+title, status, '## '+headline, summary, success_text, _md_table(comparison_table),
                '## What changed', implementation, '## Primary-period metrics', evidence, _md_table(metric_table),
                f'## {robustness_period}: previously exposed robustness', _md_table(robustness_table),
                '## Results for every seed', _md_table(seed_table), '## Penalty selection and research exposure',
                _md_table(choice_table), _md_table(exposure_table), exposure_note,
                '## Annual fits and theme gains', diagnostics_text, gains_text, coverage_text]
    if fit_diagnostics is not None:
        markdown.extend(['### Annual fit diagnostics', _md_table(fit_diagnostics)])
    if theme_gains is not None:
        markdown.extend(['### Annual theme gains', _md_table(theme_gains)])
    markdown.extend(['## Uncertainty and limitations', uncertainty, '\n'.join('- '+text for text in limitations),
                     '## Completion evidence', _md_table(gate_table), '## Sources', _md_table(source_table)])
    if synthetic:
        markdown.insert(1, '**SYNTHETIC QA FIXTURE — not historical research results.**')
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in display_tables.items():
        table.to_csv(folder/f'{name}.csv', index=False)
    for name, table in [('fit_diagnostics', fit_diagnostics), ('theme_gains', theme_gains)]:
        if table is not None:
            table.to_csv(folder/f'{name}.csv', index=False)
    snapshot = dict(synthetic_fixture=synthetic, study_completion_certified=study_complete,
                    numerical_historical_study_complete=numerical_complete,
                    completion_gates=gates, analysis_result=result,
                    display_tables={name: table.to_dict('records') for name, table in display_tables.items()},
                    full_analysis_tables={name: _records(table) for name, table in tables.items()})
    snapshot['fit_diagnostics'] = None if fit_diagnostics is None else _records(fit_diagnostics)
    snapshot['theme_gains'] = None if theme_gains is None else _records(theme_gains)
    (folder/'completion_report.html').write_text(document)
    (folder/'completion_report.md').write_text('\n\n'.join(markdown)+'\n')
    with (folder/'report_source.json').open('x') as handle:
        json.dump(_json(snapshot), handle, indent=2, allow_nan=False)
        handle.write('\n')
    manifest = dict(status='REPORT_RENDERED_PENDING_VISUAL_REVIEW', synthetic_fixture=synthetic,
                    study_completion_certified=study_complete,
                    numerical_historical_study_complete=numerical_complete,
                    generator_sha256=sha(Path(__file__)),
                    artifact_sha256={path.name: sha(path) for path in sorted(folder.iterdir()) if path.is_file()})
    with (folder/'report_manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    return dict(html=folder/'completion_report.html', markdown=folder/'completion_report.md',
                source=folder/'report_source.json', manifest=manifest)
