import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from annual_staged.fitting import YearInputs
from annual_staged.baseline import select_extensions,choose_baseline
from annual_staged.common import ORDER
from annual_staged.verification import verify_system,verify_accounting
from annual_staged.report import statistics
from annual_ind.conditional import solve_conditional
from extension4.paths import trade_month

class StagedTests(unittest.TestCase):
    def test_extended_training_cutoff(self):
        x=YearInputs(None,None,2025,lambda:None,lambda *a,**k:None,input_cache=None)
        self.assertEqual(x.cutoff,'2024-12-31');self.assertEqual(x.dates[-1],'2024-11-30');self.assertEqual(len(x),709)
        for year in (1970,2026):
            with self.assertRaises(ValueError):YearInputs(None,None,year,None,None,input_cache=None)
    def test_past_only_all_nine_and_ties(self):
        rows=[]
        for date in pd.date_range('1971-01-31','1981-12-31',freq='ME'):
            for c in ORDER:rows.append(dict(candidate=c,decision_eom=date-pd.offsets.MonthEnd(1),return_eom=date,
                fit_year=date.year,model_available_at=f'{date.year-1}-12-31',objective=1.))
        x=pd.DataFrame(rows);a=select_extensions(x,1981);self.assertEqual(a['candidate'],ORDER[0]);self.assertEqual(a['validation_months'],120)
        x.loc[x.return_eom.dt.year.eq(1981),'objective']=1e12;self.assertEqual(a,select_extensions(x,1981))
        with self.assertRaises(ValueError):select_extensions(x.iloc[1:],1981)
    def test_r_coefficient_oracle_width64_and_zero_ridge(self):
        rng=np.random.default_rng(42);n=129;u=rng.normal(size=(n,n));A=u.T@u+np.eye(n);b=rng.normal(size=n)
        for ridge in (0.,.2):
            beta=np.column_stack([solve_conditional(A,b,parent_width=65,parent_ridge=ridge,multiplier=m)[0] for m in (100.,10.,1.)])
            self.assertTrue(verify_system(A,b,beta,65,ridge,(100.,10.,1.))['passed'])
    def test_accounting_nontrivial_m_and_exit(self):
        market=SimpleNamespace(ids=np.array([1,2]),decision_date='1980-12-31',m=np.array([[.8,.1],[.2,.6]]),
            impact=np.array([.02,.03]),wealth=3.,covariance=np.eye(2)*.02)
        outcome=SimpleNamespace(ids=market.ids,decision_date=market.decision_date,next_ids=np.array([2,3]),
            total_returns=np.array([.1,-.2]),excess_returns=np.array([.08,-.22]),market_growth=.03)
        a,rows,_=trade_month(market,outcome,market.ids,np.array([[.1],[.3]]),np.array([[.4],[-.2]]),[(1,'B1')],atol=1e-11,rtol=1e-10)
        verify_accounting(market,outcome,a,rows)
        a['trade'][0,0]+=.01
        with self.assertRaises(AssertionError):verify_accounting(market,outcome,a,rows)
    def test_windows_overlap_explicitly(self):
        rng=np.random.default_rng(9);rows=[]
        for d in pd.date_range('1981-01-31','2025-12-31',freq='ME'):
            g=rng.normal(.01,.04)
            for m in ('B1','IND'):
                v=g+(rng.normal(.003,.005) if m=='IND' else 0)
                rows.append(dict(return_eom=d,method=m,gross_excess=v,impact_cost=.001,net_excess=v-.001))
        _,r=statistics(pd.DataFrame(rows));self.assertEqual([x['months'] for x in r],[480,72,60])
        from annual_staged.verification import verify_statistics
        x,r=statistics(pd.DataFrame(rows));self.assertTrue(verify_statistics(x,r)['passed'])
if __name__=='__main__':unittest.main()
