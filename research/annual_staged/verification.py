"""Independent base-R solves/inference and direct accounting checks."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from .common import ROOT, finite_equal


def rscript(mode,folder):
    executable=shutil.which('Rscript')
    if not executable:raise RuntimeError('Independent R verifier unavailable')
    result=subprocess.run([executable,'--vanilla',str(ROOT/'annual_staged/verify.R'),mode,str(folder)],
                          capture_output=True,text=True,timeout=300)
    if result.returncode:raise ValueError('Independent R verification failed: '+result.stderr[-2000:])


def verify_system(curvature,score,coefficients,parent_width,ridge,multipliers):
    if tuple(multipliers)!=(100.,10.,1.):raise ValueError('Verifier candidate order changed')
    with tempfile.TemporaryDirectory(prefix='annual-staged-verify-') as temp:
        d=Path(temp)
        for name,array in [('A',curvature),('b',score),('beta',coefficients)]:
            np.asarray(array,dtype='<f8').ravel(order='F').tofile(d/(name+'.bin'))
        (d/'dimensions.txt').write_text(f'{len(score)} {parent_width} {ridge:.17g}\n')
        rscript('system',d)
        errors=np.loadtxt(d/'errors.txt')
    return dict(engine='base R solve',maximum_beta_error=float(np.max(errors)),passed=True)


def verify_statistics(frame,rows):
    with tempfile.TemporaryDirectory(prefix='annual-staged-statistics-') as temp:
        d=Path(temp);frame.to_csv(d/'monthly.csv',index=False);rscript('statistics',d)
        independent=pd.read_csv(d/'statistics.csv')
    for row,(_,r) in zip(rows,independent.iterrows()):
        for key in ('first','last','months','baseline','IND','z','p'):finite_equal(row[key],r[key])
    if len(rows)!=len(independent):raise ValueError('Missing independent comparison')
    return dict(engine='base R var, sd, pnorm',comparisons=len(rows),passed=True)


def verify_accounting(market,outcome,arrays,rows):
    # Independent expression of the recurrence, including the current holdings.
    expected=arrays['target']+market.m@(arrays['pretrade']-arrays['target'])
    finite_equal(expected,arrays['weight'])
    finite_equal(expected-arrays['pretrade'],arrays['trade'])
    drift=expected*(1+outcome.total_returns[:,None])/(1+outcome.market_growth)
    finite_equal(drift,arrays['drifted_weight'])
    next_values=pd.DataFrame(drift,index=arrays['ids']).reindex(arrays['next_ids'],fill_value=0.).to_numpy()
    finite_equal(next_values,arrays['next_pretrade'])
    gross=np.einsum('ij,i->j',expected,outcome.excess_returns)
    cost=.5*market.wealth*np.einsum('ij,i,ij->j',arrays['trade'],market.impact,arrays['trade'])
    finite_equal(gross,[r['gross_excess'] for r in rows]);finite_equal(cost,[r['impact_cost'] for r in rows])


def verify_first_path(history,book,c,temporal,coefficients,catalog,identity):
    from types import SimpleNamespace
    from .evaluation import CheckedPathSource
    from extension4.model import ConditionalModel
    from extension4.paths import trade_month
    from feature_geometry.features import FeatureMap
    from baseline.engine import feature_indices
    from scripts.industry_projection_kernels_20260928 import project_industry_targets,FACTORS
    source=CheckedPathSource(history);market=source.market('1980-12-31');bound=source.innovation(market)
    initial=source.common_initial();np.testing.assert_array_equal(initial.ids,market.ids)
    with np.load(catalog.path(f"data/prepared/step9_v1/baseline/coefficients/g{c['gi']}/1981_p512.npz"),allow_pickle=False) as z:beta=z['coefficients'][:,c['ridge_index']]
    with np.load(catalog.path(f"data/prepared/step9_v1/baseline/moments/g{c['gi']}/1980-12-31.npz"),allow_pickle=False) as z:baseline=z['signal_t'][:,feature_indices(512)]@beta
    raw=FeatureMap(temporal.feature_map.parent.frequencies).signals(market.characteristics[0],market.volatility[0],np.zeros(115))@beta
    finite_equal(raw,baseline)
    conditional=ConditionalModel(SimpleNamespace(year=1981,seed=1,model=temporal),'C4',coefficients[:,0],
        state_identity=book.identity,protocol_identity=identity,multiplier=100.)
    aim=conditional.aim(market,bound,book.bind(market.decision_date))
    with np.load(history.files.path('data/prepared/step6_v1/risk/1980-12-31.npz'),allow_pickle=False) as z:
        proj=project_industry_targets(baseline,aim,z['loadings'],z['factor_cov'],z['specific_variance'],market.impact,market.wealth,FACTORS)
    finite_equal(proj['industry_exposure_after'],np.zeros(12),atol=1e-9)
    outcome=source.realized(market)
    arrays,rows,_=trade_month(market,outcome,market.ids,np.repeat(initial.holdings[:,None],3,axis=1),
        np.column_stack((baseline,aim,proj['IND'])),[(1,'B1'),(1,'C4'),(1,'IND')],atol=1e-11,rtol=1e-10)
    verify_accounting(market,outcome,arrays,rows)
    archive=source._archive(market.decision_date);ix=np.flatnonzero((archive['seeds']==1)&(archive['methods']=='fixed512'))
    if len(ix)!=1:raise ValueError('Missing original baseline comparison')
    for name in ('pretrade','target','weight','trade','next_pretrade'):finite_equal(arrays[name][:,0],archive[name][:,ix[0]],atol=1e-9,rtol=1e-8)
    source.unchanged()
    return dict(passed=True,decision='1980-12-31',baseline_path_reproduced=True,independent_accounting=True,industry_projection=True)
