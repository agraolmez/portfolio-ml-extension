"""Synthetic basis tests only; no training or historical outcomes."""
import importlib.util
from functools import reduce
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('theme_policy_kernel_tests',ROOT/'scripts/theme_policy_kernels_20260928.py')
k=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=k;SPEC.loader.exec_module(k)
PROBE_SPEC=importlib.util.spec_from_file_location('theme_policy_probe_tests',ROOT/'scripts/theme_policy_probe_20260929.py')
probe=importlib.util.module_from_spec(PROBE_SPEC);sys.modules[PROBE_SPEC.name]=probe;PROBE_SPEC.loader.exec_module(probe)


def fixture():
    rows=[dict(position=i+1,characteristic=f'char{i}',cluster=k.THEMES[i%13],
               direction=(-1. if i%2 else 1.),rank_missing_value=.5,preserve_exact_zero=True)
          for i in range(115)]
    order=tuple(r['characteristic'] for r in rows)
    return rows,k.ThemeDictionary.from_rows(rows,order)


class ThemePolicyTests(unittest.TestCase):
    def setUp(self):self.rows,self.dictionary=fixture()

    def test_signed_group_mean_uses_all_members_and_preserves_zero(self):
        ranks=np.full((2,115),.5);ranks[0,0]=1.;ranks[0,13]=1.;ranks[0,26]=0.
        ranks[0,39]=np.nan
        actual=k.theme_scores(ranks,self.dictionary)
        # Group0 members0(+),13(-),26(+),39(-): contributions1,-1,-1,0.
        self.assertAlmostEqual(actual[0,0],-1/self.dictionary.sizes[0])
        np.testing.assert_array_equal(actual[0,1:],np.zeros(12))
        np.testing.assert_array_equal(actual[1],np.zeros(13))
        self.assertTrue(np.isnan(ranks[0,39]))
        self.assertEqual(ranks[0,26],0.)

    def test_missing_rank_neutrality_and_even_polynomial_value(self):
        q=k.theme_scores(np.full((2,115),np.nan),self.dictionary)
        np.testing.assert_array_equal(q,np.zeros((2,13)))
        raw=k.legendre_raw(q)
        np.testing.assert_array_equal(raw[:,:13],np.zeros((2,13)))
        np.testing.assert_array_equal(raw[:,13:26],np.full((2,13),-.5))
        np.testing.assert_array_equal(raw[:,26:],np.zeros((2,13)))

    def test_legendre_analytic_values_and_degree_major_names(self):
        q=np.repeat(np.array([-1.,-.5,0.,.5,1.])[:,None],13,axis=1)
        actual=k.legendre_raw(q)
        np.testing.assert_array_equal(actual[:,0],[-1.,-.5,0.,.5,1.])
        np.testing.assert_array_equal(actual[:,13],[1.,-.125,-.5,-.125,1.])
        np.testing.assert_array_equal(actual[:,26],[-1.,.4375,0.,-.4375,1.])
        names=k.signal_names(self.dictionary)
        self.assertEqual(len(names),40);self.assertEqual(names[0],'constant')
        self.assertEqual(names[1],'P1:accruals');self.assertEqual(names[14],'P2:accruals')
        self.assertEqual(names[27],'P3:accruals')

    def test_original_normalization_and_exact_nested_linear_control(self):
        rng=np.random.default_rng(920);ranks=rng.uniform(size=(20,115));vol=rng.uniform(.05,.5,20)
        raw=k.legendre_raw(k.theme_scores(ranks,self.dictionary))
        actual=k.theme_signals(ranks,vol,self.dictionary)
        centered=raw-raw.mean(axis=0);norm=np.sqrt(np.sum(centered*centered,axis=0))
        expected=np.column_stack((np.full(20,1/np.sqrt(20)),centered/norm))/vol[:,None]
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-14)
        np.testing.assert_array_equal(actual,k.normalize_signal(raw,vol))
        linear=k.theme_signals(ranks,vol,self.dictionary,maximum_degree=1)
        np.testing.assert_array_equal(actual[:,:14],linear)
        np.testing.assert_allclose((actual[:,1:]*vol[:,None]).sum(axis=0),0,atol=1e-14)
        np.testing.assert_allclose(np.sum((actual[:,1:]*vol[:,None])**2,axis=0),1,atol=1e-14)

    def test_zero_variance_is_rejected_without_dropping_columns(self):
        with self.assertRaises(ValueError):k.theme_signals(np.full((20,115),.5),np.ones(20),self.dictionary)

    def test_order_direction_and_missing_metadata_are_strict(self):
        order=self.dictionary.characteristics
        for changes in ({'direction':0},{'rank_missing_value':0},{'preserve_exact_zero':False},{'position':2}):
            bad=[dict(r) for r in self.rows];bad[0].update(changes)
            with self.assertRaises(ValueError):k.ThemeDictionary.from_rows(bad,order)
        with self.assertRaises(ValueError):k.ThemeDictionary.from_rows(self.rows,tuple(reversed(order)))

    def test_invalid_ranks_volatility_and_unapproved_api_fail(self):
        rng=np.random.default_rng(20);ranks=rng.uniform(size=(20,115))
        for value in (-.01,1.01,np.inf):
            bad=ranks.copy();bad[0,0]=value
            with self.assertRaises(ValueError):k.theme_scores(bad,self.dictionary)
        with self.assertRaises(ValueError):k.theme_signals(ranks,np.zeros(20),self.dictionary)
        with self.assertRaises(ValueError):k.legendre_raw(np.zeros((20,13)),maximum_degree=2)
        with self.assertRaises(TypeError):k.theme_signals(ranks,np.ones(20),self.dictionary,learned_scales=np.ones(115))
        with self.assertRaises(TypeError):k.theme_signals(ranks,np.ones(20),self.dictionary,realized_returns=np.zeros(20))

    def test_exact_moments_match_independent_dense_product_formula(self):
        """Direct finite-product algebra checks the unchanged economic operator."""
        from baseline.core import moment
        rng=np.random.default_rng(20260929);n=48
        signals=[k.theme_signals(rng.uniform(size=(n,115)),rng.uniform(.2,.7,n),self.dictionary)
                 for _ in range(13)]
        growth=[rng.uniform(.9,1.1,n) for _ in range(13)]
        # A non-diagonal and asymmetric M exposes multiplication-order mistakes.
        m=.25*np.eye(n)+rng.uniform(0,.02,(n,n))/n
        load=rng.normal(size=(n,5));sigma=.01*np.eye(n)+.003*(load@load.T)
        impact=rng.uniform(.01,.04,n);wealth=4.;ret=rng.normal(0,.03,n)
        actual=moment(signals,growth,m,sigma,impact,wealth,ret,return_operators=True)
        def direct_operator(offset):
            products=[np.eye(n)]+[
                reduce(np.matmul,[m@np.diag(growth[j]) for j in range(offset,offset+lag)])
                for lag in range(1,12)]
            denominator=np.add.reduce(products)
            numerator=np.add.reduce([a@signals[offset+lag] for lag,a in enumerate(products)])
            return np.linalg.solve(denominator,numerator)
        omega=direct_operator(0);prior=direct_operator(1)
        change=omega-np.diag(growth[0])@prior
        expected={
            'omega':omega,'omega_change':change,'r_tilde':omega.T@ret,
            'risk':10*omega.T@sigma@omega,
            'tc':wealth*change.T@np.diag(impact)@change,
        }
        expected['denom']=expected['risk']+expected['tc']
        for name,value in expected.items():
            np.testing.assert_allclose(actual[name],value,atol=2e-13,rtol=2e-12,err_msg=name)
        # Disabling higher polynomial coefficients recovers the same linear
        # operator exactly; this is a mathematical check, not a second fit.
        beta=rng.normal(size=40);beta[14:]=0
        np.testing.assert_allclose(omega@beta,omega[:,:14]@beta[:14],atol=1e-14)
        np.testing.assert_allclose(change@beta,change[:,:14]@beta[:14],atol=1e-14)

    def test_quadratic_objective_and_ridge_solution_match_portfolio_accounting(self):
        """Moment units/signs agree with direct return, risk and trade costs."""
        from baseline.core import moment,ridge_path
        rng=np.random.default_rng(614);n=50
        signals=[k.theme_signals(rng.uniform(size=(n,115)),rng.uniform(.2,.7,n),self.dictionary)
                 for _ in range(13)]
        growth=[rng.uniform(.95,1.05,n) for _ in range(13)]
        m=.3*np.eye(n);a=rng.normal(size=(n,7));sigma=.01*np.eye(n)+.002*a@a.T
        impact=rng.uniform(.01,.04,n);wealth=3.;ret=rng.normal(0,.03,n)
        result=moment(signals,growth,m,sigma,impact,wealth,ret,return_operators=True)
        penalty=.07
        coefficients,_=ridge_path(result['denom'],result['r_tilde'],[penalty])
        beta=coefficients[:,0];weights=result['omega']@beta;trades=result['omega_change']@beta
        direct=(weights@ret-5*(weights@sigma@weights)
                -wealth/2*np.sum(impact*trades**2)-penalty/2*(beta@beta))
        quadratic=(beta@result['r_tilde']-.5*beta@result['denom']@beta-penalty/2*(beta@beta))
        self.assertAlmostEqual(direct,quadratic,places=13)
        np.testing.assert_allclose((result['denom']+penalty*np.eye(40))@beta,
                                   result['r_tilde'],atol=1e-13,rtol=1e-12)
        # Every arbitrary perturbation must lower this strictly concave objective.
        for _ in range(4):
            perturbed=beta+rng.normal(0,.1,40)
            other=(perturbed@result['r_tilde']-.5*perturbed@result['denom']@perturbed
                   -penalty/2*(perturbed@perturbed))
            self.assertGreater(direct,other)

    def test_training_calendar_has_only_frozen_causal_prefixes(self):
        for year,count in ((1981,181),(1990,289),(2000,409)):
            dates=probe.training_calendar(year)
            self.assertEqual(len(dates),count)
            self.assertEqual(dates[0],pd.Timestamp('1965-11-30'))
            self.assertEqual(dates[-1],pd.Timestamp(f'{year-1}-11-30'))
            self.assertTrue(all(day.is_month_end for day in dates))
            self.assertTrue(all(a+pd.offsets.MonthEnd(1)==b for a,b in zip(dates,dates[1:])))
        for invalid in (1980,1982,2001,1981.,'1981',True):
            with self.assertRaises(ValueError):probe.training_calendar(invalid)

    def test_training_aggregation_never_accesses_poisoned_future_records(self):
        for year in (1981,1990,2000):
            keys=[str(day.date()) for day in probe.training_calendar(year)]
            positions={day:i for i,day in enumerate(keys)}
            class PoisonFuture:
                def __init__(self):self.accessed=[]
                def __iter__(self):raise AssertionError('Must not inspect full or future record inventory')
                def __getitem__(self,day):
                    if day not in positions:raise AssertionError('Future or unapproved record accessed')
                    self.accessed.append(day);i=positions[day]+1
                    return {'r_tilde':np.full(40,i),'risk':np.eye(40)*i,'tc':np.eye(40)*2*i}
            records=PoisonFuture();actual=probe.aggregate_training(records,year)
            self.assertEqual(records.accessed,keys);self.assertEqual(actual['count'],len(keys))
            mean=(len(keys)+1)/2
            np.testing.assert_array_equal(actual['score'],np.full(40,mean))
            np.testing.assert_array_equal(actual['risk'],np.eye(40)*mean)
            np.testing.assert_array_equal(actual['cost'],np.eye(40)*2*mean)

    def test_development_summary_requires_all_prespecified_months(self):
        frame=self.development_panel()
        for malformed in (frame.iloc[:-1],pd.concat([frame.iloc[:-1],frame.iloc[[0]]],ignore_index=True)):
            with self.assertRaises(ValueError):probe.summarize(malformed)
        substituted=frame.copy();substituted.loc[0,'return_eom']='1982-01-31'
        with self.assertRaises(ValueError):probe.summarize(substituted)
        _,result=probe.summarize(frame)
        self.assertTrue(result['development_gate_passed']);self.assertEqual(result['monthly_rows'],72)
        for key in ('automatic_full_study_authorized','statistically_significant_outperformance_established',
                    'scientific_study_complete','fresh_holdout','original_paper_outperformance_established'):
            self.assertFalse(result[key])

    def test_each_year_and_both_utility_definitions_must_pass_separately(self):
        for year in (1981,1990,2000):
            for losing_metric in ('source_gross_sample_variance_utility','net_population_utility'):
                frame=self.development_panel();mask=(frame.year==year)&(frame.method=='THEME40')
                if losing_metric=='net_population_utility':
                    # A tiny mean gain with fluctuating costs: source utility
                    # rises but the variance of net returns makes net utility fall.
                    frame.loc[mask,'gross_excess']=.01
                    frame.loc[mask,'impact_cost']=np.tile([0.,.005998],6)
                else:
                    # Costs track gross returns, leaving a higher constant net
                    # return while gross variance makes source utility fall.
                    gross=np.tile([.008,.028],6)
                    frame.loc[mask,'gross_excess']=gross
                    frame.loc[mask,'impact_cost']=gross-.0071
                _,result=probe.summarize(frame)
                row=next(row for row in result['candidate_minus_baseline'] if row['year']==year)
                other=('net_population_utility' if losing_metric=='source_gross_sample_variance_utility'
                       else 'source_gross_sample_variance_utility')
                self.assertLess(row[losing_metric],0);self.assertGreater(row[other],0)
                self.assertFalse(result['development_gate_passed'])

    @staticmethod
    def development_panel():
        rows=[]
        for year in (1981,1990,2000):
            for method in ('B1','THEME40'):
                for day in pd.date_range(f'{year}-01-31',f'{year}-12-31',freq='ME'):
                    rows.append(dict(year=year,method=method,return_eom=str(day.date()),
                                     decision_eom=str((day-pd.offsets.MonthEnd(1)).date()),
                                     gross_excess=.01+(.001 if method=='THEME40' else 0),impact_cost=.003))
        return pd.DataFrame(rows)


if __name__=='__main__':unittest.main()
