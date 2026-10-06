"""Small boundary fixtures for comparison-specific accounting and decisions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from comparisons.core import action,smoothing,select,metrics,family
from learner.core import remap

def run():
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    x=dict(date='2000-12-31',ids=np.array([1,2]),aim=np.array([.2,-.1]),m=np.diag([.8,.7]),loadings=np.eye(2),fc=np.diag([.02,.03]),specific=np.array([.01,.02]),impact=np.array([.002,.003]),wealth=100.,confidence={12:dict(mean=np.array([.3,-.2]),numerator=.003)})
    cap=dict(kappa_min=.5,kappa_max=1.2);h=np.array([.1,.1]);q=(np.eye(2)-x['m'])@(x['aim']-h)
    for speed in [.5,.75,1,1.25,1.5]:
        a=action(x,h,'B1',speed,None,12,cap);check('constant speed and binding cap '+str(speed),np.allclose(a['trade'],min(speed,1.2)*q))
    check('baseline zero trade remains zero',np.array_equal(action(x,x['aim'],'B1',1.5,None,12,cap)['trade'],np.zeros(2)))
    base=action(x,h,'R1',None,None,12,cap)
    check('R1 equals speed one',np.array_equal(base['trade'],action(x,h,'B1',1,None,12,cap)['trade']))
    dollars=x['wealth']*base['trade'];cost_dollars=.5*np.sum(x['impact']*dollars**2)
    check('dollar cost converted to return exactly',np.isclose(cost_dollars/x['wealth'],x['wealth']/2*np.sum(x['impact']*base['trade']**2)))
    value,dropped=remap(np.array([.3,-.4]),np.array([1,2]),np.array([2,3]))
    check('holdings survive by ID and entrants are zero',np.array_equal(value,[-.4,0]) and np.array_equal(dropped,[0]))
    target,drop,values=smoothing(np.array([.2,.6]),np.array([.3,-.4]),np.array([1,2]),np.array([1.1,.9]),np.array([2,3]),.5)
    check('smoothing uses growth and current entrant aim',np.allclose(target,[-.08,.6]))
    check('smoothing exit values preserved separately',np.array_equal(drop,[1]) and np.allclose(values,[.33]))
    initial,_,_=smoothing(x['aim'],None,None,None,x['ids'],.75);check('block-start smoothing is original aim',np.array_equal(initial,x['aim']))
    for method in ['B2','B3']:
        broken=dict(x,confidence={});a=action(broken,h,method,.5,None,12,cap,smoothed=np.array([np.nan,0]) if method=='B3' else None)
        check(method+' missing target fallback restores full baseline',np.array_equal(a['target'],x['aim']) and np.array_equal(a['trade'],base['trade']) and a['decision']['fallback_reason'] is not None)
        if method=='B3':
            # The pipeline stores exactly the returned target, including after fallback.
            nxt,_,_=smoothing(x['aim'],a['target'],x['ids'],np.ones(2),x['ids'],.5)
            check('B3 fallback resets the stored smoothing state',np.array_equal(nxt,x['aim']))
    for method in ['A0','A1','A2']:
        a=action(x,h,method,None,None,12,cap);check(method+' absent fit retains own-state baseline',np.array_equal(a['trade'],base['trade']) and a['decision']['fallback_reason'] is not None)
    broken=dict(x,aim=np.array([np.nan,0]))
    try:action(broken,h,'R1',None,None,12,cap)
    except ValueError:check('corrupt baseline input stops',True)
    else:check('corrupt baseline input stops',False)
    changed=dict(x,returns=np.array([100,-100]),growth=np.array([2,.5]),future_aim=np.array([99,99]))
    check('future outcomes cannot change current baseline decision',np.array_equal(action(changed,h,'R1',None,None,12,cap)['trade'],base['trade']))
    fallback=action(changed,h,'A1',None,None,12,cap);check('future outcomes cannot change missing-model fallback',np.array_equal(fallback['trade'],base['trade']))
    from learner.core import A0,A1
    def model(cols):return dict(features=cols,mean=[0.]*len(cols),scale=[1.]*len(cols),constant=[False]*len(cols),intercept=[.2,1.1],coefficients=[[.001,.0] for k in cols],minimum=[-100.]*len(cols),maximum=[100.]*len(cols))
    fitted=dict(status='FITTED',first_decision='2000-12-31',last_decision='2001-11-30',models={'A0':model(A0),'A1':model(A1)},uncertainty_status='FITTED',bootstrap=[dict(model(A1),intercept=[.001*i],coefficients=[[.001] for k in A1]) for i in range(100)])
    for method in ['A0','A1','A2']:
        a=action(x,h,method,None,fitted,12,cap);b=action(changed,h,method,None,fitted,12,cap)
        check(method+' fitted decision invariant to future outcomes',a['decision']==b['decision'] and a['features']==b['features'] and np.array_equal(a['trade'],b['trade']) and a['decision']['fallback_reason'] is None)
    failed=dict(fitted,models={'A1':dict(model(A1),intercept=[np.nan,1.])})
    a=action(x,h,'A1',None,failed,12,cap);check('nonfinite learned forecast restores own-state baseline',np.array_equal(a['trade'],base['trade']) and a['decision']['fallback_reason'] is not None)
    grid=pd.DataFrame([dict(method=name,annual_net_utility=0.) for name,_,_ in family()]);choice=select(grid)
    check('exact ties choose speed one smallest rho and R1',choice['B1_kappa']==1 and choice['B3_rho']==.25 and choice['C_star']=='R1')
    grid.loc[grid.method.eq('B1_1.25'),'annual_net_utility']=.9e-8;check('within-tolerance tie favors speed one',select(grid)['B1_kappa']==1)
    grid.loc[grid.method.eq('B1_1.25'),'annual_net_utility']=1.1e-8;check('outside-tolerance difference selects higher utility',select(grid)['B1_kappa']==1.25)
    grid.loc[:,'annual_net_utility']=-1.;grid.loc[grid.method.isin(['B1_0.75','B1_1.25']),'annual_net_utility']=0.;check('equal distance tie favors lower speed',select(grid)['B1_kappa']==.75)
    check('fixed family size',len(family())==13 and len(family(choice))==7)
    return dict(status='PASS',checks_total=len(checks),checks=checks,fixture='SYNTHETIC_FIXTURE_ONLY')

if __name__=='__main__':print(run())
