"""Artificial-data production integration, causality, accounting and restart tests."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from test_temporal_extension import temporal_fixture
from feature_geometry.features import fingerprint
from feature_geometry.moments import TrainingSample
from geometry_study.kernel import LeanBasis
from temporal_extension.history import BoundInnovation
from temporal_extension.model import TemporalModel
from temporal_extension.profile import evaluate
from temporal_study.paths import InitialState, RealizedMonth
from extension4.state import StateBook, build_states
from extension4.design import capacity_map, designs
from extension4.parent import SelectedParent
from extension4.training import collect, fit_moments
from extension4.model import ConditionalModel
from extension4.storage import commit_stage, verify_stage, write_json
from extension4.paths import replay
from extension4.selection import select, candidate
from extension4.authority import require_approval


def state_fixture(first='1997-01-31', last='2001-12-31'):
    dates=pd.date_range(first,last,freq='ME')
    raw=pd.DataFrame([dict(id=sid,eom=date,be_me=.1+(i%7+1)*sid/10,me=100.)
                      for i,date in enumerate(dates) for sid in range(1,5)])
    eligible=raw[['id','eom']].assign(valid=True)
    definition=dict(name='value_dispersion',field='be_me',quantiles=[.1,.9],quantile_method='linear',min_count=2,min_coverage=.5)
    normalizer=dict(previous_calendar_rows=4,min_finite_prior=2,exclude_current=True,preserve_gaps=True,ddof=0,inactive_value=0.,scale_floor=1e-12,clip=3.)
    metadata=dict(unranked=True,timing_mode='SOURCE_LAG_PROXY_UNCERTIFIED',source_sha256='artificial')
    return raw,eligible,dict(definition=definition,normalizer=normalizer,source_metadata=metadata,first=first,last=last)


def bound_fixture():
    rng,_,_,bases,settings,_,fmap,innovations=temporal_fixture()
    gains=np.linspace(.6,1.4,13)
    updated=[]; bound=[]
    for basis,history,date in zip(bases,innovations,pd.date_range('1999-01-31',periods=2,freq='ME')):
        market=replace(basis.sample.market,decision_date=str(date.date()))
        sample=TrainingSample(market,basis.sample.returns,str((date+pd.offsets.MonthEnd(1)).date()))
        updated.append(LeanBasis(sample))
        bound.append(tuple(BoundInnovation(i,str((date-pd.offsets.MonthEnd(lag)).date()),market.ids,
            fingerprint(z),'synthetic-source','synthetic-reference')
            for lag,(i,z) in enumerate(zip(history,market.characteristics))))
    result=evaluate(updated,innovations,fmap,gains,settings,rho=.1,objective_scale=1.,with_gradient=False)
    model=TemporalModel(fmap,gains,result['beta'],settings.cutoff,settings.cutoff,'synthetic-parent',
                        'synthetic-source','synthetic-reference',np.zeros(13),np.full(13,2.))
    parent=SelectedParent(2000,1,model,settings,{'rho':.1,'objective_scale':1.},'synthetic','1998-12-31')
    extras=capacity_map(model,np.column_stack((fmap.parent.frequencies,rng.normal(size=(115,256)))))
    raw,eligible,kw=state_fixture();book=build_states(raw,eligible,**kw)
    return parent,updated,bound,extras,book


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def test_actual_raw_values_future_perturbation_and_past_only_normalizer(self):
        raw,eligible,kw=state_fixture();before=build_states(raw,eligible,**kw)
        later=raw.copy();later.loc[later.eom>'1999-01-31','be_me']*=100.
        after=build_states(later,eligible,**kw)
        pd.testing.assert_frame_equal(before.frame[before.frame.eom<='1999-01-31'],after.frame[after.frame.eom<='1999-01-31'])
        # Changing current raw values does not change its historical mean or scale.
        current=raw.copy();current.loc[current.eom=='1999-01-31','be_me']*=2.
        changed=build_states(current,eligible,**kw)
        i=int(np.flatnonzero(before.frame.eom==pd.Timestamp('1999-01-31'))[0])
        self.assertEqual(before.frame.historical_mean[i],changed.frame.historical_mean[i])
        self.assertEqual(before.frame.historical_scale[i],changed.frame.historical_scale[i])

    def test_missing_calendar_rows_coverage_and_no_invented_publication_dates(self):
        raw,eligible,kw=state_fixture();missing=pd.Timestamp('1999-01-31')
        raw=raw[raw.eom!=missing];eligible=eligible[eligible.eom!=missing]
        book=build_states(raw,eligible,**kw)
        row=book.frame[book.frame.eom==missing].iloc[0]
        self.assertEqual(row.normalized,0.);self.assertFalse(row.active)
        next_row=book.frame[book.frame.eom==pd.Timestamp('1999-02-28')].iloc[0]
        self.assertEqual(next_row.historical_count,3)
        self.assertFalse(book.metadata['publication_timestamps_created'])
        self.assertEqual(book.metadata['timing_mode'],'SOURCE_LAG_PROXY_UNCERTIFIED')
        with self.assertRaises(ValueError):build_states(raw.iloc[1:],eligible,**kw)

    def test_source_publication_field_enforced_when_available(self):
        raw,eligible,kw=state_fixture()
        raw['published_at']=raw.eom
        raw.loc[raw.id==4,'published_at']=raw.loc[raw.id==4,'eom']+pd.offsets.MonthEnd(1)
        kw['source_metadata']['timing_mode']='OBSERVED_AVAILABILITY'
        with self.assertRaises(ValueError):build_states(raw,eligible,**kw)
        book=build_states(raw,eligible,availability_column='published_at',**kw)
        self.assertTrue(book.frame.unavailable_count.eq(1).all())
        self.assertTrue(book.frame.valid_count.eq(3).all())

    def test_state_binding_rejects_wrong_dates_changed_values_and_changed_book(self):
        raw,eligible,kw=state_fixture();book=build_states(raw,eligible,**kw)
        state=book.bind('1999-01-31',history=13)
        state.validate('1999-01-31',history=13,expected_identity=book.identity)
        with self.assertRaises(ValueError):state.validate('1999-02-28',history=13,expected_identity=book.identity)
        state.values.setflags(write=True);state.values[0,0]+=1e-15
        with self.assertRaises(ValueError):state.validate('1999-01-31',history=13,expected_identity=book.identity)
        book.frame.loc[12,'normalized']+=1e-15
        with self.assertRaises(ValueError):book.bind('1999-01-31')

    def test_full_width_controls_solve_neutrality_and_saved_model(self):
        parent,bases,bound,extras,book=bound_fixture()
        collected=collect(zip(bases,bound),parent,book,extras,expected_dates=['1999-01-31','1999-02-28'])
        fits=fit_moments(collected,parent,[100.,10.,1.],size_bounds=(-1/3,1/3),atol=1e-11,rtol=1e-10)
        self.assertEqual(len(fits),10)
        self.assertEqual({method for method,_,_,_ in fits},{'N4','C4','CAP4','SIZE4'})
        for method,multiplier,beta,receipt in fits:
            self.assertLessEqual(receipt['solve_residual'],1e-9)
            extra=extras if method=='CAP4' else None
            model=ConditionalModel(parent,method,beta,state_identity=book.identity,protocol_identity='toy',
                extras=extra,multiplier=multiplier,size_bounds=(-1/3,1/3) if method=='SIZE4' else None)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.npz';model.save(path)
                restored=ConditionalModel.load(path,parent,state_identity=book.identity,protocol_identity='toy',extras=extra)
                self.assertEqual(model.identity,restored.identity)
                with self.assertRaises(ValueError):ConditionalModel.load(path,parent,state_identity='changed',protocol_identity='toy',extras=extra)

    def test_affine_size_moments_match_direct_objective(self):
        parent,bases,bound,extras,book=bound_fixture()
        collected=collect(zip(bases,bound),parent,book,extras,expected_dates=['1999-01-31','1999-02-28'])
        for alpha in (-.3,0.,.2):
            values=[]
            for basis,history in zip(bases,bound):
                m=basis.sample.market
                result=designs(basis,history,parent.model,book.bind(m.decision_date,history=13),book.identity,extras)
                u=result['base'][0]@parent.model.beta+alpha*result['SIZE4'][0]
                q=result['base'][1]@parent.model.beta+alpha*result['SIZE4'][1]
                values.append(u@basis.sample.returns-.5*m.gamma*u@m.covariance@u-.5*m.wealth*q@(m.impact*q))
            expected=np.mean(values)-.5*parent.settings.ridge*(parent.model.beta@parent.model.beta)-.5*parent.settings.scale_penalty*(parent.model.feature_map.theta@parent.model.feature_map.theta)
            expected-=parent.receipt['objective_scale']*parent.receipt['rho']*np.mean((parent.model.gains-1.)**2)
            a=collected['affine'];actual=a['constant']+alpha*a['score']-.5*alpha*alpha*a['curvature']
            self.assertAlmostEqual(actual,expected,places=11)

    def test_wrong_innovation_order_and_unmatured_training_fail(self):
        parent,bases,bound,extras,book=bound_fixture()
        with self.assertRaises(ValueError):collect(zip(bases,[bound[0][::-1],bound[1]]),parent,book,extras,expected_dates=['1999-01-31','1999-02-28'])
        future=LeanBasis(replace(bases[0].sample,available_at='2000-01-31'))
        with self.assertRaises(ValueError):collect([(future,bound[0])],parent,book,extras,expected_dates=['1999-01-31'])

    def test_stage_resume_tamper_and_failed_attempt_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            self.assertTrue(commit_stage(path,{'id':'toy'},lambda p:write_json(p/'value.json',{'x':1})))
            self.assertFalse(commit_stage(path,{'id':'toy'},lambda p:self.fail('Committed stage reran')))
            with self.assertRaises(ValueError):verify_stage(path,{'id':'different'})
            (path/'value.json').write_text('{"x":2}')
            with self.assertRaises(ValueError):verify_stage(path)
            failed=Path(tmp)/'failed'
            def fail(p):
                write_json(p/'partial.json',{'x':1});raise RuntimeError('interrupt')
            with self.assertRaises(RuntimeError):commit_stage(failed,{},fail)
            self.assertFalse(failed.exists());self.assertEqual(len(list(Path(tmp).glob('.failed.attempt-*'))),1)

    def test_selection_excludes_later_returns_and_rejects_parent_leakage(self):
        rows=[]
        for date in pd.date_range('2011-01-31','2020-12-31',freq='ME'):
            for method in ('C4','CAP4','SIZE4'):
                for multiplier in [100.,10.,1.]:
                    rows.append(dict(return_eom=date,seed=1,method=candidate(method,multiplier),net_excess=multiplier*.0001))
        frame=pd.DataFrame(rows)
        kw=dict(first='2011-01-31',last='2015-12-31',seeds=[1],multipliers=[100.,10.,1.],tie_tolerance=1e-12,parent_selection_available_at='2010-12-31')
        original=select(frame,**kw);frame.loc[frame.return_eom>'2015-12-31','net_excess']=99.
        self.assertEqual(original,select(frame,**kw))
        with self.assertRaises(ValueError):select(frame,**dict(kw,parent_selection_available_at='2011-12-31'))

    def test_pending_proposal_cannot_launch_a_pilot(self):
        root=Path(__file__).resolve().parents[1]
        with self.assertRaises(PermissionError):require_approval(root,root/'config/extension4_study_v1_proposed.json',root/'config/nonexistent_extension4_approval.json')

    def test_continuous_accounting_annual_boundary_entries_exits_and_resume(self):
        parent,bases,bound,extras,book=bound_fixture()
        # Two months straddle the December refit; IDs change and holdings carry.
        dates=['1999-11-30','1999-12-31'];markets={};outcomes={}
        for i,(date,basis) in enumerate(zip(dates,bases)):
            n=len(basis.sample.market.ids);ids=np.arange(i,i+n)
            markets[date]=replace(basis.sample.market,decision_date=date,ids=ids)
            next_ids=np.arange(i+1,i+1+n+1)
            outcomes[date]=RealizedMonth(date,ids,np.linspace(-.03,.04,n),np.linspace(-.031,.039,n),.01,next_ids)
        class Source:
            def __init__(self):self.aims=0;self.visits=0
            def market(self,date):self.visits+=1;return markets[date]
            def innovation(self,m):return None
            def realized(self,m):
                if self.aims!=2*self.visits:raise AssertionError('Outcome accessed before all aims')
                return outcomes[m.decision_date]
            def unchanged(self):pass
        source=Source();initial=InitialState(dates[0],markets[dates[0]].ids,np.full(16,.01),'synthetic')
        def aims(seed,method,market,b):
            source.aims+=1
            return np.linspace(-.1,.1,len(market.ids))*(1. if method=='P3' else 2.)
        with tempfile.TemporaryDirectory() as tmp:
            options=dict(destination=Path(tmp)/'paths',lock_identity='toy',policy_identity='toy',atol=1e-11,rtol=1e-10)
            first=replay(source,dates[:1],[(1,'P3'),(1,'C4')],initial,aims,**options)
            pd.testing.assert_frame_equal(first,replay(source,dates[:1],[(1,'P3'),(1,'C4')],initial,aims,**options))
            result=replay(source,dates,[(1,'P3'),(1,'C4')],initial,aims,**options)
            self.assertEqual(len(first),2);self.assertEqual(len(result),4)
            with np.load(Path(tmp)/'paths'/dates[0]/'positions.npz') as a, np.load(Path(tmp)/'paths'/dates[1]/'positions.npz') as b:
                np.testing.assert_array_equal(a['next_pretrade'],b['pretrade'])
                self.assertFalse(np.allclose(b['pretrade'][:,0],b['pretrade'][:,1]))
                self.assertTrue(np.all(b['pretrade'][-1]==0))
            visits=source.visits
            pd.testing.assert_frame_equal(result,replay(source,dates,[(1,'P3'),(1,'C4')],initial,aims,**options))
            self.assertEqual(source.visits,visits)


if __name__=='__main__':unittest.main()
