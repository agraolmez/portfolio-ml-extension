"""Frozen author-method comparisons; adverse findings do not expand the search."""
import math
import html
import pandas as pd
import numpy as np
from .common import atomic,now,ROOT,sha
from .verification import verify_statistics


def statistics(frame):
    x=frame.copy();x['return_eom']=pd.to_datetime(x.return_eom);x['year']=x.return_eom.dt.year
    rows=[]
    for first,last,label in [(1981,2020,'Primary'),(2020,2025,'Requested extension (2020 overlaps primary)'),(2021,2025,'Extension excluding overlap')]:
        views={}
        for method in ('B1','IND'):
            v=x[x.year.between(first,last)&x.method.eq(method)].sort_values('return_eom')
            expected=pd.date_range(f'{first}-01-31',f'{last}-12-31',freq='ME')
            if list(v.return_eom)!=list(expected):raise ValueError('Missing or duplicated portfolio month')
            views[method]=v
        def utility(v):return 12*(v.net_excess.mean()-5*v.gross_excess.var(ddof=1))
        def flow(v):return v.net_excess.to_numpy()-5*(v.gross_excess.to_numpy()-v.gross_excess.mean())**2
        delta=flow(views['IND'])-flow(views['B1']);sd=float(delta.std(ddof=1))
        if not np.isfinite(sd) or sd<=0:raise ValueError('Author inference has undefined paired standard error')
        z=float(delta.mean()/(sd/np.sqrt(len(delta))))
        rows.append(dict(label=label,first=first,last=last,months=len(delta),baseline=utility(views['B1']),
                    IND=utility(views['IND']),z=z,p=.5*math.erfc(z/np.sqrt(2))))
    return x,rows


def write_report(frame,out,verification,identity):
    x,rows=statistics(frame);independent=verify_statistics(x,rows)
    atomic(out/'statistical_verification.json',independent)
    report=out/'report_reviewed';report.mkdir(exist_ok=True)
    table='\n'.join(f"| {r['first']}–{r['last']} | {r['months']} | {100*r['baseline']:.3f}% | {100*r['IND']:.3f}% | {100*(r['IND']-r['baseline']):.3f} pp | {r['p']:.5g} |" for r in rows)
    text=f'''# Annual staged IND comparison

Numerical study completed {now()}. Final human-facing presentation review remains pending.

| Return period | Months | Baseline utility/year | IND utility/year | Difference | One-sided nominal p |
|---|---:|---:|---:|---:|---:|
{table}

The primary comparison is 1981–2020. The requested 2020–2025 extension overlaps it by one year; 2021–2025 is shown separately. These windows are not independent replications.

The baseline retains its original full-grid annual selection. The new staged algorithm fits characteristic scales at that selected coordinate, fits all nine temporal/conditional candidates, and chooses the next year's pair using only preceding validation returns. Initial 1971 parent settings use baseline-only 1966–1970 validation. Seed 1 is fixed. Annual training history expands. Portfolio holdings and trading costs continue uninterrupted across all 540 months. The fixed industry projection modifies incremental targets; executed positions need not be industry neutral.

Utility uses 12 × (mean net excess return − 5 × sample variance of gross excess return). The paired one-sided normal test uses the released authors' monthly centered utility-flow convention. These are nominal, unadjusted historical comparisons. Previously inspected data and prior model exploration remain limitations; chronological tuning does not create a fresh holdout. The comparison is against our reconstructed baseline on shared data and corrected wealth accounting, not the published table's original-vintage dataset. The conditional state retains the source-lag proxy timing limitation.

No other method, seed or candidate was searched after viewing these results. Negative or inconclusive primary findings complete the approved study.

Independent R coefficient solves and statistical checks, baseline path reproduction, direct validation-score checks, and continuous-accounting verification are saved with the study. Source lock: `{identity}`.

Reproduce/resume locally: `env/step3/bin/python -B scripts/annual_ind_staged_20261001.py` from the canonical project. The runner validates the immutable code/configuration lock and resumes completed layers/months. The same cumulative 48-active-hour limit applies to resumptions.
'''
    (report/'completion_report.md').write_text(text)
    trows=''.join(f"<tr><td>{r['first']}–{r['last']}</td><td>{r['months']}</td><td>{100*r['baseline']:.3f}%</td><td>{100*r['IND']:.3f}%</td><td>{100*(r['IND']-r['baseline']):.3f} pp</td><td>{r['p']:.5g}</td></tr>" for r in rows)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Annual staged IND results</title><style>body{{font:17px/1.55 system-ui;max-width:1050px;margin:48px auto;padding:0 20px;color:#192b31;background:#f5f7f4}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow-x:auto}}p{{max-width:85ch}}pre{{white-space:pre-wrap;font:inherit}}</style><h1>Annual staged IND results</h1><p>Numerical checks passed. Presentation review pending.</p><div class="scroll"><table><thead><tr><th>Period</th><th>Months</th><th>Baseline utility</th><th>IND utility</th><th>Difference</th><th>Nominal p</th></tr></thead><tbody>{trows}</tbody></table></div><pre>{html.escape(text[text.index('The primary comparison'):])}</pre></html>'''
    (report/'completion_report.html').write_text(body)
    result=dict(status='NUMERICALLY_COMPLETE_REVIEW_PENDING',generated_at=now(),months=540,comparisons=rows,
        independent_statistics=independent,verification=verification,report_reviewed=False,lock_sha256=identity,
        primary_significant_at_5_percent=bool(rows[0]['IND']>rows[0]['baseline'] and rows[0]['p']<.05))
    atomic(out/'result.json',result)
    return result
