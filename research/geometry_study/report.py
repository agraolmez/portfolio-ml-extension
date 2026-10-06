"""Generate the scientific comparison report and standalone uncertainty figure."""
import html,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import ROOT,OUT,locked,write
from .analyze import LABELS
from feature_geometry.history import sha

def table_md(frame):
    def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])

def report():
    plan=locked()['plan'];a=OUT/'analysis';result=json.loads((a/'analysis_result.json').read_text());manifest=json.loads((a/'analysis_manifest.json').read_text())
    for p,h in manifest['artifact_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed analysis evidence '+p)
    intervals=pd.read_csv(a/'primary_intervals.csv');means=pd.read_csv(a/'mean_seed_metrics.csv');seeds=pd.read_csv(a/'seed_contrasts.csv');fits=pd.read_csv(a/'fit_diagnostics.csv');parts=pd.read_csv(a/'utility_decomposition.csv')
    paths=json.loads((OUT/'paths_manifest.json').read_text());folder=OUT/'report';folder.mkdir()
    comparison=[x.split('-',1)[1] for x in plan['primary_contrasts']]
    fig,axes=plt.subplots(2,1,figsize=(11,10),sharex=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    lo=min(0,float(intervals.adjusted_lower.min()*10000));hi=max(0,float(intervals.adjusted_upper.max()*10000));pad=max((hi-lo)*.06,.1)
    for ax,period,title in zip(axes,['evaluation','exposed_robustness'],['2011–2020 retrospective evaluation · 120 months','2021–2025 exposed robustness · 60 months']):
        ax.axvline(0,color='#353b43',linewidth=1)
        for length,offset,color,marker in [(12,-.12,'#2367a1','o'),(24,.12,'#353b43','D')]:
            frame=intervals[(intervals.period==period)&(intervals.block_months==length)].set_index('contrast').loc[plan['primary_contrasts']]
            point=frame.mean_seed_delta.to_numpy()*10000;low=frame.adjusted_lower.to_numpy()*10000;high=frame.adjusted_upper.to_numpy()*10000;y=np.arange(6)+offset
            # Basic intervals need not contain the observed point; draw endpoints directly.
            ax.hlines(y,low,high,color=color,linewidth=1.8,label=f'Mean block {length} months')
            ax.plot(point,y,linestyle='',marker=marker,color=color,markersize=5,markerfacecolor=color if length==12 else 'white')
        ax.set_yticks(np.arange(6),[LABELS[m] for m in comparison]);ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=12,pad=12)
        ax.grid(axis='x',color='#e3e6ea',linewidth=.7);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.set_xlim(lo-pad,hi+pad)
    axes[0].legend(loc='lower right',frameon=False,fontsize=9);axes[1].set_xlabel('Full geometry minus control · annual net utility, basis points')
    fig.suptitle('Learned geometry: paired differences and uncertainty',x=.035,ha='left',fontsize=16,y=.98)
    fig.text(.035,.018,'Mean of three paired seed differences. Basic 95% intervals adjusted for six contrasts.\nBoth historical periods were previously exposed; inference conditions on fitted paths.',fontsize=10,color='#404650')
    fig.subplots_adjust(left=.31,right=.97,top=.91,bottom=.13,hspace=.3)
    fig.savefig(folder/'utility_intervals.png',dpi=180,facecolor='white');fig.savefig(folder/'utility_intervals.svg',facecolor='white');plt.close(fig)
    blocks=[]
    def heading(text):blocks.append(('heading',text))
    def paragraph(text):blocks.append(('paragraph',text))
    def table(frame):blocks.append(('table',frame))
    passed=result['primary_family_improvement_established']
    primary=seeds[seeds.period.eq('evaluation')].set_index('contrast');base=primary.loc['full-fixed512'];ib=intervals[(intervals.period=='evaluation')&(intervals.contrast=='full-fixed512')].set_index('block_months')
    heading('What the historical comparison found')
    paragraph(('The learned-geometry model satisfies the prespecified historical success criterion for this comparison family.' if passed else 'The learned-geometry model does not satisfy the prespecified historical success criterion for this comparison family. Improvement beyond the complete set of controls is not established.'))
    paragraph(f"Against the matched fixed 512-feature control, the mean annual net-utility difference in 2011–2020 is {base.mean_seed_delta*10000:+.2f} basis points. The adjusted 95% interval is [{ib.loc[12].adjusted_lower*10000:+.2f}, {ib.loc[12].adjusted_upper*10000:+.2f}] with 12-month blocks and [{ib.loc[24].adjusted_lower*10000:+.2f}, {ib.loc[24].adjusted_upper*10000:+.2f}] with 24-month blocks. These are retrospective, conditional estimates, not an untouched holdout or a forecast of future returns.")
    paragraph('Literature originality remains unestablished. The full Recursive Portfolio Machines methods were unavailable through the checked primary links, so this study does not establish an advantage over RPM. A completed historical comparison cannot resolve that missing comparison.')
    heading('What was implemented')
    paragraph('The comparison baseline is the accepted reconstruction of Portfolio-ML. Its historical results do not exactly reproduce the paper’s printed table. This study holds the reconstruction inputs and conventions fixed; it neither explains the earlier replication gap nor tunes results to the paper’s reported numbers.')
    paragraph('The original 115 characteristics enter fixed cosine/sine mixtures after multiplication by learned positive input scales. Zero-sum log-scales separate relative characteristic geometry from the global bandwidth; scales are bounded between 0.5 and 2. There are 114 independent scale parameters in addition to 513 output coefficients. This changes the finite feature representation and the aim portfolio. It does not change the original adjustment matrix m or the trading recursion.')
    paragraph('The main learner profiles the output coefficients under the original return, risk and quadratic-impact objective, then updates scales by projected gradient ascent. Every annual fit uses the entire expanding training history from November 1965 through November of the prior year. Outcomes must have matured by the preceding December decision. All scale learners start from zero each year and have eight maximum accepted updates; reaching that cap does not establish convergence.')
    paragraph('All 45 annual cycles, three predetermined frequency seeds, and 540 decisions per path are included. Each path starts from common accepted R1 pretrade holdings in December 1980, pays its first trade cost and then carries its own holdings through December 2025 returns. Annual and reporting-period boundaries do not reset holdings. The original 13-slice/12-term synthetic training approximation and source entry/exit accounting are retained.')
    heading('Primary evaluation: mean statistics across seeds')
    paragraph('These are arithmetic averages of each seed’s separately calculated statistics. They are not statistics of an ensemble portfolio. Utility is 12 × [mean monthly net excess return − 5 × population variance], with risk aversion 10. Only modeled quadratic impact is deducted.')
    for period,title in [('evaluation','2011–2020'),('exposed_robustness','2021–2025 exposed robustness')]:
        frame=means[means.period.eq(period)].set_index('method').loc[list(LABELS)]
        display=pd.DataFrame({'Method':[LABELS[m] for m in frame.index],'Net mean (%/yr)':frame.annual_net_excess_mean.map(lambda x:f'{x*100:.2f}'),'Net volatility (%/yr)':frame.annual_net_volatility.map(lambda x:f'{x*100:.2f}'),'Net utility (%/yr)':frame.annual_net_utility.map(lambda x:f'{x*100:.2f}'),'Impact (bp/yr)':frame.annual_impact_bps.map(lambda x:f'{x:.2f}'),'Monthly turnover':frame.monthly_absolute_turnover.map(lambda x:f'{x:.3f}')}).reset_index(drop=True)
        paragraph(title);table(display)
    heading('Paired differences and uncertainty')
    paragraph('The six prespecified comparisons use the mean of three within-seed utility differences. Each bootstrap resamples common dates across all seeds and methods, then recomputes means and population variances. Intervals are paired basic 95% intervals adjusted for six contrasts; 10,000 stationary circular-bootstrap draws are used at each block length. The bootstrap conditions on fitted paths and does not rerun model fitting, tuning or data revisions.')
    blocks.append(('figure',('utility_intervals.png','Annual net utility differences with six-comparison-adjusted intervals; positive values favor full geometry.')))
    for period in ['evaluation','exposed_robustness']:
        frame=intervals[intervals.period.eq(period)];display=[]
        for method in comparison:
            row=frame[frame.contrast.eq('full-'+method)].set_index('block_months');sr=seeds[(seeds.period==period)&(seeds.contrast=='full-'+method)].iloc[0]
            display.append({'Full geometry minus':LABELS[method],'Mean Δ (bp/yr)':f'{sr.mean_seed_delta*10000:+.2f}','Seed 1 / 2 / 3 (bp/yr)':f'{sr.seed1*10000:+.2f} / {sr.seed2*10000:+.2f} / {sr.seed3*10000:+.2f}','Adjusted interval, L=12':f'[{row.loc[12].adjusted_lower*10000:+.2f}, {row.loc[12].adjusted_upper*10000:+.2f}]','Adjusted interval, L=24':f'[{row.loc[24].adjusted_lower*10000:+.2f}, {row.loc[24].adjusted_upper*10000:+.2f}]'})
        paragraph('2011–2020 evaluation' if period=='evaluation' else '2021–2025 exposed robustness');table(pd.DataFrame(display))
    heading('Return, risk and cost accounting')
    paragraph('The utility difference decomposes exactly into the difference in gross mean returns, minus the difference in mean impact costs, minus five times the difference in annualized net-return variance. This is an accounting decomposition; it does not identify a causal economic mechanism.')
    frame=parts[(parts.period=='evaluation')&(parts.contrast=='full-fixed512')].mean(numeric_only=True)
    table(pd.DataFrame({'Contribution':['Gross mean return','Mean impact cost','Net-return variance','Total net utility'],'Annual utility contribution (bp)':[f'{frame[k]*10000:+.2f}' for k in ['gross_mean_contribution','cost_mean_contribution','net_variance_contribution','delta_utility']]}))
    heading('What the controls do and do not establish')
    paragraph('Risk-only geometry retains m while removing the explicit impact term from geometry supervision; it is not fully frictionless. Prediction geometry uses a trace-scaled ridge to match relative curvature, then receives the same full economic coefficient refit. The joint learner uses simultaneous projected scale and preconditioned coefficient gradients from the identity solution; both its raw coefficients and its final exact coefficient refit are evaluated. Larger fixed dictionaries contain the smaller dictionaries; both the original ridge and a prespecified double-ridge sensitivity are shown.')
    paragraph('All methods use the original past-selected annual bandwidth and economic ridge. This isolates the feature layer conditional on that tuning rule; it is not a claim that each comparator received its own best retuning. The update caps are common, but computation and achieved convergence need not be equal. No seed, geometry setting or favorable reporting period was selected from these results.')
    heading('Verification and scope')
    learned=fits[fits.method.isin(['full','risk_only','return_prediction','joint'])];counts=learned.status.value_counts()
    paragraph(f"Of the {len(learned):,} learned-geometry fits, {int(counts.get('ITERATION_BUDGET_REACHED',0)):,} reached the update cap, {int(counts.get('PROJECTED_GRADIENT_TOLERANCE',0)):,} reached the projected-gradient tolerance, and {int(counts.get('LINE_SEARCH_STALLED',0)):,} stopped when the line search stalled. These stopping outcomes do not establish global optimality. The diagnostic CSV retains each fit’s execution-version reference; engineering speedups were checked against saved complete annual fits.")
    paragraph(f"The original seed-1 fixed-feature path is recovered across all 540 months; its largest absolute position difference is {paths['original_recovery_max_weight_error']:.3g}. Direct scalar position accounting independently checks every monthly gross return and impact charge. Scalar utility calculations agree within {result['independent_scalar_utility_max_error']:.3g}. Independent R recomputation checks {result['independent_R']['bootstrap_utilities_checked']:,} bootstrap utilities, with maximum error {result['independent_R']['maximum_absolute_error']:.3g}.")
    paragraph('Engineering verification includes original operator/gradient/fitting equivalence, maturity and duplicate-date failures, input-scale constraints, cache invalidation, and model serialization. The first complete annual full-geometry fit exactly matches the preserved serial implementation. A tested parallel-solve issue in the local numerical backend is avoided by serializing LAPACK calls while processing independent feature/matrix work concurrently. The preserved initial attempt is not a separate financial-performance experiment.')
    paragraph('Current input vintage, unverified characteristic publication timing, source eligibility involving next-return availability, reconstructed wealth, impact-only costs and source exits remain limitations. All 1981–2025 outcomes were previously exposed to research. Confidence intervals that include zero do not demonstrate equivalence. No prospective observation, automation, order or publication was started; cancelled Step 11 remains cancelled.')
    heading('Earlier research and literature')
    paragraph('The accepted first-extension A0, A1 and B2 summaries remain in the retained-reference CSV. Their reporting blocks started from original R1 holdings, whereas the geometry paths carry their own holdings continuously from 1981. They are practical descriptive references with that initialization difference, not additional opportunistic tests of success.')
    paragraph('Recursive Portfolio Machines already learns characteristic geometry for a closed-form portfolio/SDF model. The accessible institutional and author records make it a close predecessor, but do not establish whether its full method includes this exact cost-aware construction. Deep Inception Networks studies learned features and positions under a portfolio objective with turnover/cost considerations. Generative random Fourier-feature kernel learning establishes learned Fourier representations outside this precise portfolio specification. Broad novelty claims about learned kernels or economic feature supervision are therefore unsupported.')
    blocks.append(('links',[('Study plan','../../../docs/FEATURE_GEOMETRY_STUDY_PLAN_20260914.md'),('Complete metrics','../analysis/strategy_summary.csv'),('Paired intervals','../analysis/primary_intervals.csv'),('Seed differences','../analysis/seed_contrasts.csv'),('Learned characteristic scales','../analysis/learned_scales.csv'),('Optimization diagnostics','../analysis/fit_diagnostics.csv'),('Earlier A0/A1/B2 references','../analysis/retained_first_extension_references.csv'),('Analysis verification','../analysis/analysis_result.json'),('Recursive Portfolio Machines institutional record','https://www.sfi.ch/en/publications/n-26-36-recursive-portfolio-machines'),('Deep Inception Networks','https://arxiv.org/abs/2307.05522'),('Generative random Fourier-feature learning','https://arxiv.org/abs/2009.04614')]))
    md=['# Learned characteristic geometry — historical results'];body=['<h1>Learned characteristic geometry — historical results</h1>']
    for kind,value in blocks:
        if kind=='heading':md.append('## '+value);body.append('<h2>'+html.escape(value)+'</h2>')
        elif kind=='paragraph':md.append(value);body.append('<p>'+html.escape(value)+'</p>')
        elif kind=='table':md.append(table_md(value));body.append('<div class="table-wrap">'+value.to_html(index=False,escape=True,border=0)+'</div>')
        elif kind=='figure':
            path,caption=value;md.append(f'![{caption}]({path})');body.append(f'<figure><img src="{path}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        elif kind=='links':
            md.append('\n'.join(f'- [{title}]({url})' for title,url in value));body.append('<ul>'+''.join(f'<li><a href="{html.escape(url)}">{html.escape(title)}</a></li>' for title,url in value)+'</ul>')
    (folder/'findings.md').write_text('\n\n'.join(md)+'\n')
    style='body{margin:0;background:#fafbfc;color:#222b35;font:17px/1.65 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:48px 32px}h1{font-size:34px;line-height:1.2}h2{margin-top:42px;font-size:24px;line-height:1.3}p{max-width:86ch}a{color:#175a91}table{border-collapse:collapse;min-width:660px;width:100%;font-size:14px;line-height:1.45}th,td{text-align:right;padding:11px 10px;border-bottom:1px solid #dce2e8}th:first-child,td:first-child{text-align:left}th{background:#eef2f6}.table-wrap{overflow:auto;margin:20px 0}figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#4c5968}li{margin:6px 0}@media(max-width:600px){main{padding:28px 18px}h1{font-size:28px}body{font-size:16px}}@media print{body{background:white}main{max-width:none;padding:0}.table-wrap{overflow:visible}table{font-size:10px}h2{break-after:avoid}tr,figure{break-inside:avoid}}'
    (folder/'completion_report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learned geometry historical results</title><style>'+style+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
    write(folder/'report_manifest.json',dict(status='GENERATED_PENDING_PRESENTATION_REVIEW',generated_at_utc=datetime.now(timezone.utc).isoformat(),analysis_manifest_sha256=sha(a/'analysis_manifest.json'),report_code_sha256=sha(Path(__file__)),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir() if p.is_file()}))
    print('Scientific report and standalone interval figure generated; presentation review required',flush=True)

if __name__=='__main__':
    report()
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_geometry_study.py')],check=True)
    rendering=OUT/'report_rendering';rendering.mkdir()
    node=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
    subprocess.run([str(node),str(ROOT/'scripts/geometry_report_preview.cjs'),(OUT/'report/completion_report.html').as_uri(),str(rendering)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/geometry_study_handoff.py')],check=True)
