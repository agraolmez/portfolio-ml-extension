"""Verify the full prescribed bootstrap and independent R computations."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from inference.pipeline import ROOT,OUT,PRIOR,METHODS,VARIANTS,PERIODS,PROTOCOL,sha,save_json,preflight,load_returns
from inference.core import utility,stationary,bootstrap_utility,basic_interval,indices_from_primitives,success


def main():
    if (OUT/'review_summary.json').exists():raise ValueError('Completed review is immutable')
    preflight();checks=[];largest=0.
    def check(name,value,detail=None):
        checks.append(dict(check=name,passed=bool(value),detail=detail))
        if not value:raise AssertionError(name)
    def close(name,a,b,atol=2e-12):
        nonlocal largest
        a,b=np.asarray(a),np.asarray(b);error=float(np.max(np.abs(a-b)))
        largest=max(largest,error);check(name,np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=0),error)
    def rejects(name,call):
        try:call()
        except ValueError:check(name,True)
        else:check(name,False)
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    for rel,digest in manifest['artifact_sha256'].items():check('Locked result '+rel,sha(ROOT/rel)==digest)
    source=pd.read_csv(PRIOR/'strategy_summary.csv')
    p=pd.read_csv(OUT/'primary_intervals.csv',float_precision='round_trip')
    s=pd.read_csv(OUT/'secondary_intervals.csv',float_precision='round_trip')
    r=pd.read_csv(OUT/'reference/intervals.csv',float_precision='round_trip')
    ro=pd.read_csv(OUT/'reference/observed.csv',float_precision='round_trip')
    seeds=pd.read_csv(OUT/'seeds.csv')
    choice=json.loads((PRIOR/'frozen_selection.json').read_text())['choices']
    for variant in VARIANTS:check('R reference comparator '+variant,choice[variant]['C_star']=='R1')
    check('All primary rows',len(p)==54 and not p.duplicated(['period','variant','block_months','contrast']).any())
    check('All secondary rows',len(s)==378 and not s.duplicated(['period','variant','block_months','contrast']).any())
    check('Exactly three primary claim rows',p.primary_claim.sum()==3)
    index_sets=0;utility_values=0
    for period,(start,end,n) in PERIODS.items():
        for L in [12,24]:
            tag=f'{period}_L{L}';z=np.load(OUT/'bootstrap'/f'{tag}_indices.npz')
            idx,starts,restart,digests=stationary(PROTOCOL['protocol_id'],PROTOCOL['seeds']['extension'],L,start,n)
            for name,array in [('indices',idx),('starts',starts),('restart',restart)]:check('Full seed replay '+tag+'/'+name,np.array_equal(array,z[name]))
            saved=seeds[seeds.period.eq(period)&seeds.block_months.eq(L)].sort_values('replica')
            check('All seed digests '+tag,saved.seed_sha256.tolist()==digests)
            check('Interval index dimensions '+tag,idx.shape==(10000,n) and idx.min()>=0 and idx.max()<n)
            check('Expected restart frequency '+tag,abs(restart[:,1:].mean()-1/L)<.002)
            index_sets+=1
            for variant in VARIANTS:
                w,d,c=load_returns(period,variant);x=w.to_numpy();obs=utility(x)
                current=f'{period}_{variant}_L{L}';b=np.load(OUT/'bootstrap'/f'{current}_utility.npz')
                check('Method order '+current,b['methods'].tolist()==METHODS)
                close('Observed vector '+current,b['observed'],obs)
                rb=np.fromfile(OUT/'reference'/f'{current}_utility.bin',dtype='<f8').reshape(7,10000).T
                close('Every independent R bootstrap utility '+current,b['utility'],rb)
                utility_values+=rb.size
                for i,m in enumerate(METHODS):
                    name=(f'B1_{c["B1_kappa"]:g}' if m=='B1' else f'B3_{c["B3_rho"]:g}' if m=='B3' else m) if period=='selection' else m
                    old=source[source.period.eq(period)&source.variant.eq(variant)&source.method.eq(name)]
                    check('Unique Step 9 summary '+current+'/'+m,len(old)==1)
                    close('Accepted Step 9 utility '+current+'/'+m,obs[i],old.annual_net_utility.iloc[0])
                    oldr=ro[ro.period.eq(period)&ro.variant.eq(variant)&ro.block_months.eq(L)&ro.method.eq(m)]
                    close('Independent R observed '+current+'/'+m,obs[i],oldr.utility.iloc[0])
                check('Paired duplicate R1/B1 '+current,np.array_equal(b['utility'][:,0],b['utility'][:,1]))
    keys=['period','variant','block_months','contrast']
    for table,family in [(p,3),(s,1)]:
        merged=table.merge(r[r.family.eq(family)],on=keys,validate='one_to_one',suffixes=('_py','_r'))
        check('Every interval matched '+str(family),len(merged)==len(table))
        for col in ['estimate','lower','upper']:close('Every independent interval '+str(family)+'/'+col,merged[col+'_py'],merged[col+'_r'])
        check('Interval ordering '+str(family),(table.lower<=table.upper).all())
    for (period,variant,L),g in p.groupby(['period','variant','block_months']):
        a=g[g.contrast.eq('A1-R1')];b=g[g.contrast.eq('A1-C*')]
        close(f'Duplicate C* {period}/{variant}/{L}',a[['estimate','lower','upper']].to_numpy(),b[['estimate','lower','upper']].to_numpy())
    parts=pd.read_csv(OUT/'utility_decomposition.csv')
    close('Net mean plus variance penalty',parts.utility_difference,parts.net_mean_contribution+parts.variance_penalty_contribution)
    close('Gross mean less impact',parts.net_mean_contribution,parts.gross_mean_contribution+parts.impact_contribution)
    primary=p[p.primary_claim];claimed=json.loads((OUT/'inference_summary.json').read_text())
    check('Success rule exact',claimed['primary_success']==success(primary.estimate,primary.lower))
    check('No positive primary success',not claimed['primary_success'])
    synthetic_begin=len(checks)
    # Hand-computed sample: mean .02; population variance .0001; annual utility .234.
    close('Synthetic population variance and annualization',utility([.01,.03]),.234)
    x=np.array([[-.04,-.03],[.01,.02],[.02,.03],[.05,.06]])
    ids=np.array([[0,0,0,0],[1,1,2,2],[3,2,1,0]],dtype=int)
    u=bootstrap_utility(x,ids)
    close('Paired constant shift',u[:,1]-u[:,0],np.repeat(.12,3))
    close('Resampled sample mean and variance',u[:,0],[-.48,.1785,utility(x[:,0])])
    lo,hi=basic_interval(2.,np.array([0.,1.,2.,3.,10.]),alpha=.5)
    close('Basic reversed endpoints fixture',[lo,hi],[1.,3.])
    ordinary=basic_interval(0.,np.arange(100.),family=1);adjusted=basic_interval(0.,np.arange(100.),family=3)
    check('Bonferroni wider than ordinary',adjusted[0]<ordinary[0] and adjusted[1]>ordinary[1])
    start=np.array([[3,1,2,0],[0,3,1,2]],dtype=np.int32)
    restart=np.array([[1,0,0,1],[1,1,0,0]],bool)
    check('Explicit restart and circular wrap',np.array_equal(indices_from_primitives(start,restart),[[3,0,1,0],[0,3,0,1]]))
    idx,st,rs,_=stationary('fixture',1,1,'2000-01-31',4,25)
    check('L=1 independent starts',np.array_equal(idx,st) and rs.all())
    close('Identical methods have zero interval',basic_interval(0,np.zeros(100)),[0,0])
    check('Zero is not positive success',not success([0,1,1],[0,1,1]))
    check('Positive points need positive lowers',not success([1,1,1],[-1,1,1]))
    check('All positives succeed',success([1,1,1],[.1,.1,.1]))
    rejects('Missing/nonfinite returns rejected',lambda:utility([0,np.nan]))
    rejects('Empty returns rejected',lambda:utility([]))
    rejects('Short bootstrap interval rejected',lambda:bootstrap_utility(x,np.array([[0,1]])))
    rejects('Out of range indices rejected',lambda:bootstrap_utility(x,np.array([[0,1,2,4]])))
    rejects('Invalid first restart rejected',lambda:indices_from_primitives(start,np.zeros_like(start,bool)))
    rejects('Invalid start rejected',lambda:indices_from_primitives(start+4,np.ones_like(start,bool)))
    check('Utility cannot be bootstrapped as fixed centered monthly scores',not np.isclose(u[0,0],12*np.mean((x[:,0]-5*(x[:,0]-x[:,0].mean())**2)[ids[0]])))
    resources=[json.loads(v.read_text()) for v in OUT.glob('*_resources.json')]
    check('Completed measured prior jobs',all(v['within_limits'] and v['exit_code']==0 for v in resources))
    save_json(OUT/'review_summary.json',dict(status='PASS',checks_total=len(checks),checks_passed=sum(c['passed'] for c in checks),
        checks=checks,synthetic_checks=len(checks)-synthetic_begin-1,full_seed_index_replays=index_sets,
        independent_R_index_sequences=60000,independent_R_utility_values=utility_values,independent_R_intervals=len(r),
        max_numerical_discrepancy=largest,primary_success=claimed['primary_success'],
        interpretation='No demonstrated incremental utility benefit; not an equivalence test.',
        verification_code_sha256={str(v.relative_to(ROOT)):sha(v) for v in [ROOT/'scripts/step10_review.py',ROOT/'scripts/step10_reference.R']},
        reference_artifact_sha256={str(v.relative_to(ROOT)):sha(v) for v in (OUT/'reference').glob('*')}))
    print(f'PASS: {len(checks)} checks; {utility_values:,} independent R utility values and {len(r)} intervals.')


if __name__=='__main__':main()
