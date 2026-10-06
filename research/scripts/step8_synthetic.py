"""Boundary, chronology and independent optimization checks for the new machinery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from learner.core import *
from baseline.io import PROTOCOL,save_json
from learner.pipeline import OUT


def run():
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    rng=np.random.default_rng(20260905)
    impact=np.array([.3,1.,5.]);q,_=np.linalg.qr(rng.normal(size=(3,3)));sym=q@np.diag([.2,.5,.9])@q.T
    m=sym/np.sqrt(impact)[:,None]*np.sqrt(impact)[None,:]
    cap=feasible(m,impact);check('Non-diagonal impact similarity yields exact cap',abs(cap['kappa_max']-1.25)<1e-12)
    for name,bad in [('nonsymmetric',m+np.array([[0,.1,0],[0,0,0],[0,0,0]])),('negative spectrum',-np.eye(3)),('spectrum above one',1.01*np.eye(3))]:
        try:feasible(bad,impact)
        except ValueError:check(f'Reject {name}',True)
        else:check(f'Reject {name}',False)
    h=np.array([.1,-.2,.3]);aim=np.array([.4,.1,-.1]);trade=(np.eye(3)-m)@(aim-h)
    for k in [.5,1.,1.25]:
        effective=np.eye(3)-k*(np.eye(3)-m)
        check(f'Effective inertia identity {k}',np.allclose(h+k*trade,effective@h+(np.eye(3)-effective)@aim,atol=1e-14))
    values=np.array([[.1,.2],[.3,.4],[.5,.6]])
    mapped,dropped=remap(values,np.array([1,2,3]),np.array([3,1,4]))
    check('Unsorted next universe, zero entrant, explicit exit',np.array_equal(mapped,np.array([[.5,.6],[.1,.2],[0.,0.]])) and np.array_equal(dropped,[1]))
    x=rng.normal(size=(40,4));x[:,3]=7;y=np.column_stack([3+2*x[:,0],np.full(40,8.)]);w=np.repeat([1.,2.],20)
    f=fit_ridge(x,y,w,.1);check('Constant feature becomes zero column',f['constant'][3] and np.all(np.array(f['coefficients'])[3]==0))
    check('Unpenalized constant target is exact',np.max(abs(predict(f,x)[:,1]-8))<1e-12)
    doubled=fit_ridge(np.repeat(x,2,axis=0),np.repeat(y,2,axis=0),np.repeat(w/2,2),.1)
    check('Splitting date weight leaves fit unchanged',np.allclose(predict(f,x),predict(doubled,x),atol=1e-12))
    dates=pd.date_range('1990-01-31',periods=360,freq='ME')
    rows=pd.DataFrame([dict(decision_eom=d,state_id=str(s),outcome_end=d+pd.offsets.MonthEnd(12),label_available_at=d+pd.offsets.MonthEnd(12),usable=True,b=float(i%7),c=1.+i%5,**{k:float(i%11+s) for k in A1}) for i,d in enumerate(dates) for s in range(3)])
    allrows,tr,va,info=split(rows,pd.Timestamp('2010-12-31'))
    check('Exactly 36 consecutive validation months',va.decision_eom.nunique()==36 and np.all(np.diff(pd.DatetimeIndex(va.decision_eom.unique()).to_period('M').asi8)==1))
    check('Purged outcome and availability strict inequality',tr.outcome_end.max()<va.decision_eom.min() and tr.label_available_at.max()<va.decision_eom.min())
    check('At least 120 post-purge distinct months',info['status']=='FITTED' and tr.decision_eom.nunique()>=120)
    modified=rows.copy();modified.loc[modified.label_available_at.gt('2010-12-31'),['b','c']]=1e100
    later,_,_,_=split(modified,pd.Timestamp('2010-12-31'))
    check('Future target perturbation cannot enter annual data',allrows.equals(later))
    reduced=rows.drop(rows[(rows.decision_eom==dates[0]) & (rows.state_id=='2')].index)
    total=pd.Series(weights(reduced),index=reduced.index).groupby(reduced.decision_eom).sum()
    check('Usable states share exactly one unit per date',np.allclose(total,1))
    _,_,_,short=split(rows,pd.Timestamp('1992-01-31'));check('Insufficient history produces annual fallback',short['status']=='BASELINE_FALLBACK')
    # A constant target makes all penalties tied exactly; the larger penalty wins.
    tied=rows.copy();tied[['b','c']]=[0.,2.]
    eligible,t,v,_=split(tied,pd.Timestamp('2010-12-31'))
    selected=select_model(t,v,eligible,A0,[.01,.1,1,10]);check('Penalty ties choose largest, intercept unpenalized',selected['penalty']==10 and np.max(abs(predict(selected,eligible[A0])-[0,2]))<1e-12)
    gap_dates=pd.DatetimeIndex(list(dates[:24])+list(dates[30:60]))
    num,dig=seed(PROTOCOL,'primary','2010-12-31',1);starts,sampled,counts=blocks(gap_dates,12,num)
    check('Blocks never bridge a missing calendar span',all(np.all(np.diff(gap_dates[s:s+12].to_period('M').asi8)==1) for s in starts))
    check('Block counts preserve date sample length',counts.sum()==len(gap_dates) and len(sampled)==len(gap_dates))
    check('Component/date/replica seeds deterministic and distinct',seed(PROTOCOL,'primary','2010-12-31',1)==(num,dig) and seed(PROTOCOL,'blocks24','2010-12-31',1)[1]!=dig and seed(PROTOCOL,'primary','2010-12-31',2)[1]!=dig)
    try:blocks(dates[:5],12,num)
    except ValueError:check('No eligible contiguous block fails explicitly',True)
    else:check('No eligible contiguous block fails explicitly',False)
    for case in range(40):
        b=float(rng.normal()*3);c=float(rng.uniform(.2,6));u=float(rng.uniform(0,2));upper=float(rng.uniform(1,1.5))
        model=dict(mean=[0]*4,scale=[1]*4,constant=[False]*4,intercept=[b,c],coefficients=np.zeros((4,2)).tolist(),minimum=[-1]*4,maximum=[1]*4,features=A0)
        # Construct exactly the desired sample SD across 100 bootstrap predictions.
        standardized=np.linspace(-1,1,100);standardized/=standardized.std(ddof=1)
        reps=[dict(mean=[0]*4,scale=[1]*4,constant=[False]*4,intercept=[b+u*z],coefficients=np.zeros((4,1)).tolist()) for z in standardized]
        fitted=dict(status='FITTED',models={'A0':model,'A1':model},bootstrap=reps,uncertainty_status='FITTED')
        feat=dict.fromkeys(A0,0.);feat['initial_curvature']=1.
        result=decide(fitted,'A2',feat,.5,upper);chat=max(1,c)
        objective=lambda delta:-b*delta+.5*(chat+1)*delta**2+u*abs(delta)
        numerical=minimize_scalar(objective,bounds=(-.5,upper-1),method='bounded',options={'xatol':1e-13})
        check(f'Soft threshold matches independent convex minimizer {case}',objective(result['delta'])<=numerical.fun+1e-8 and abs(result['u']-u)<1e-12)
        if case==0:
            feat['initial_curvature']=0;check('Curvature-floor decision stays baseline',decide(fitted,'A2',feat,.5,upper)['kappa']==1)
            feat['initial_curvature']=1;fitted['uncertainty_status']='FAILED';check('Uncertainty failure stays baseline',decide(fitted,'A2',feat,.5,upper)['kappa']==1)
    value=dict(status='PASS',checks_total=len(checks),checks_passed=len(checks),checks=checks)
    save_json(OUT/'synthetic_checks.json',value);return value
if __name__=='__main__':print(run()['checks_total'])
