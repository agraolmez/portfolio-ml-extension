"""Source-backed local report; numerical completion is distinct from final review."""
from html import escape
import json
from pathlib import Path
import pandas as pd


def render(analysis_folder, destination, *, artificial=False):
    analysis_folder,destination=Path(analysis_folder),Path(destination)
    result=json.loads((analysis_folder/'analysis_result.json').read_text())
    metrics=pd.read_csv(analysis_folder/'metrics.csv')
    intervals=pd.DataFrame(json.loads((analysis_folder/'intervals.json').read_text()))
    summary=metrics.groupby(['period','method'],sort=False)[['annual_net_utility','annual_net_mean','annual_net_volatility','annual_impact_cost']].mean().reset_index()
    for c in summary.columns[2:]:summary[c]=summary[c].map(lambda x:f'{100*x:.3f}%')
    display=intervals[['period','block_months','challenger','benchmark','mean_delta','lower','upper','all_seed_deltas_positive']].copy()
    for c in ['mean_delta','lower','upper']:display[c]=display[c].map(lambda x:f'{10000*x:.2f} bp')
    summary=summary.rename(columns=dict(period='Period',method='Model',annual_net_utility='Net utility / year',
        annual_net_mean='Net excess return / year',annual_net_volatility='Volatility / year',annual_impact_cost='Trading cost / year'))
    display=display.rename(columns=dict(period='Period',block_months='Block months',challenger='Model',benchmark='Compared with',
        mean_delta='Utility difference',lower='Lower bound',upper='Upper bound',all_seed_deltas_positive='Positive in all 3 seeds'))
    periods=dict(early_descriptive='1981–2000: descriptive',parent_selection='2001–2010: parent selection',
        extension4_selection='2011–2015: E4 selection',primary='2016–2020: primary',robustness='2021–2025: robustness',full_descriptive='1981–2025: descriptive')
    summary['Period']=summary.Period.map(periods);display['Period']=display.Period.map(periods)
    title='ARTIFICIAL SOFTWARE TEST — not historical research' if artificial else 'Extension 4: valuation-state conditioning'
    conclusion='Prespecified criterion met.' if result['prespecified_success'] else 'Prespecified improvement criterion not met.'
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Extension 4</title>
<style>body{font:17px/1.55 system-ui,sans-serif;color:#192e35;background:#f5f5ee;max-width:1200px;margin:auto;padding:30px}h1{font-size:2rem}table{border-collapse:collapse;font-size:14px;background:white;width:100%;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #ccd4d3;padding:10px}th{background:#e2ebe6}.scroll{overflow:auto}.notice{padding:18px;background:#fff3cd;border-left:5px solid #a77300}</style><main>'''
    html+=f'<h1>{escape(title)}</h1><p class="notice"><strong>{escape(conclusion)}</strong> Numerical results require final report review before scientific acceptance.</p>'
    html+='<p>Extension 4 freezes Extension 3’s learned characteristic scales, temporal gains and trading rule. One raw book-to-market dispersion state changes output feature coefficients. Controls test whether any change is explained by more fixed features or simple exposure timing.</p>'
    html+='<p><strong>Models:</strong> P3 = completed Extension 3; N4 = disabled Extension 4, which must reproduce P3; C4 = Extension 4; CAP4 = 512 extra fixed features with the E3 transform; SIZE4 = state-dependent exposure to P3; E2 = learned characteristic scales; F1024 = the earlier fixed 1,024-feature comparison.</p>'
    html+='<p>Selection: 2011–2015. Primary evaluation: 2016–2020. Separate robustness: 2021–2025. Earlier histories are descriptive under retrospectively selected settings. All periods were previously exposed.</p>'
    html+='<p class="notice">Source publication times and revision vintages are unavailable: this is a timing-uncertified source-lag study. The parent convention omits separate liquidation costs for disappearing stocks. Seed statistics are averaged; portfolios are not combined into an ensemble. Bootstrap inference conditions on fitted and selected models. Originality is not established.</p>'
    html+='<h2>Average of the three seed statistics</h2><div class="scroll">'+summary.to_html(index=False,escape=True,border=0)+'</div>'
    html+=f'<h2>Paired differences and adjusted intervals</h2><p>{result["comparison_family"]} comparisons, {result["bootstrap_draws"]:,} paired stationary-bootstrap draws, 12- and 24-month expected blocks. Utility uses net returns and population variance. Positive results require the complete prespecified criterion.</p><div class="scroll">'+display.to_html(index=False,escape=True,border=0)+'</div>'
    html+='<h2>Verification and reproduction</h2><p>Independent R recomputes utilities and intervals from saved returns and paired month indices. Actual-position accounting, neutral paths, annual fitting receipts, source hashes and execution logs accompany the study. Use the study execution plan and locked configuration for exact reproduction and resumption.</p></main></html>'
    with destination.open('x') as f:f.write(html)
    return dict(metrics_rows=len(metrics),interval_rows=len(intervals),artificial=artificial,final_visual_review_complete=False)
