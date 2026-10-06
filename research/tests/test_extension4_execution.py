"""Execution boundaries and independent verification; exclusively artificial returns."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from test_extension4_pipeline import state_fixture,bound_fixture
from extension4.state import build_states,StateBook
from extension4.training import collect,fit_moments
from extension4.verification import verify_states,independent_month
from extension4.paths import trade_month
from extension4.analysis import analyze
from extension4.report import render
from extension4.run import Study,validate_protocol,implementation_hashes,require_engineering
from extension4.run import main as run_main
from extension4.runtime import prior_seconds
from extension4.storage import write_json,verify_stage
from feature_geometry.history import sha
from temporal_study.paths import RealizedMonth

ROOT=Path(__file__).resolve().parents[1]


def artificial_returns(config):
    """Exactly paired synthetic returns; no financial data is read."""
    rng=np.random.default_rng(923);dates=pd.date_range('1981-01-31','2025-12-31',freq='ME')
    common=rng.normal(.004,.02,len(dates));rows=[]
    for seed in (1,2,3):
        for method in config['controls']:
            # A constant improvement gives an independently known utility difference.
            net=common+.0001*seed+(.0002 if method=='C4' else 0.)
            for date,value in zip(dates,net):
                rows.append(dict(return_eom=str(date.date()),seed=seed,method=method,net_excess=value,
                    impact_cost=.0001,turnover=.1,gross_exposure=1.2,net_exposure=.1,largest_absolute_weight=.05,
                    gross_weight_concentration=.02,modeled_annual_volatility=.1))
    return pd.DataFrame(rows)


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=threadpool_limits(limits=1)
    @classmethod
    def tearDownClass(cls):cls.threads.restore_original_limits()

    def test_independent_raw_state_and_corruption_detection(self):
        raw,eligible,kw=state_fixture();book=build_states(raw,eligible,**kw)
        result=verify_states(book,raw,eligible);self.assertLess(result['maximum_state_error'],1e-11)
        altered=book.frame.copy();altered.loc[20,'raw_state']+=.001
        bad=StateBook(altered,book.metadata)
        with self.assertRaises(AssertionError):verify_states(bad,raw,eligible)

    def test_independent_position_accounting_detects_changed_cost_and_exit(self):
        parent,bases,_,_,_=bound_fixture();market=bases[0].sample.market;n=len(market.ids)
        outcome=RealizedMonth(market.decision_date,market.ids,np.linspace(-.03,.04,n),np.linspace(-.031,.039,n),.01,market.ids[1:])
        arrays,rows,exits=trade_month(market,outcome,market.ids,np.full((n,2),.01),
            np.column_stack([np.linspace(-.1,.1,n),np.linspace(.03,-.03,n)]),[(1,'P3'),(1,'C4')],atol=1e-11,rtol=1e-10)
        self.assertLess(independent_month(market,outcome,arrays,rows,exits,atol=1e-11,rtol=1e-10),1e-11)
        wrong=deepcopy(rows);wrong[0]['impact_cost']+=.001
        with self.assertRaises(AssertionError):independent_month(market,outcome,arrays,wrong,exits,atol=1e-11,rtol=1e-10)
        with self.assertRaises(ValueError):independent_month(market,outcome,arrays,rows,exits[:-1],atol=1e-11,rtol=1e-10)

    def test_postselection_only_selected_candidates_are_fitted(self):
        parent,bases,bound,extras,book=bound_fixture()
        moments=collect(zip(bases,bound),parent,book,extras,expected_dates=['1999-01-31','1999-02-28'])
        result=fit_moments(moments,parent,[100.,10.,1.],size_bounds=(-1/3,1/3),atol=1e-11,rtol=1e-10,
            family_multipliers={'C4':[100.],'CAP4':[10.],'SIZE4':[1.]})
        self.assertEqual([(method,m) for method,m,_,_ in result],[('N4',None),('C4',100.),('CAP4',10.),('SIZE4',1.)])

    def test_saved_annual_runner_fit_is_reused_and_tamper_fails(self):
        parent,bases,bound,extras,book=bound_fixture()
        moments=collect(zip(bases,bound),parent,book,extras,expected_dates=['1999-01-31','1999-02-28'])
        config=json.loads((ROOT/'config/extension4_study_v1_proposed.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=Study.__new__(Study)
            study.root=ROOT;study.destination=root;study.config=config;study.lock='artificial';study.book=book
            study.config_path=root/'config.json';write_json(study.config_path,config)
            study.approval_path=root/'TEST_ONLY_NOT_AUTHORIZATION.json';write_json(study.approval_path,{'fixture':True})
            study.identity_context=dict(implementation_sha256=implementation_hashes(ROOT),
                config_sha256=sha(study.config_path),approval_sha256=sha(study.approval_path))
            study.parent=lambda year,seed:parent;study.extra=lambda p:extras;study.runtime=Mock()
            parent.receipt['parent_scale_optimizer_status']='synthetic'
            study.history=Mock();study.history.files.consumed={}
            study.cache=Mock();study.cache.diagnostics.return_value={'fixture':True}
            with patch('extension4.run.collect',return_value=moments) as calculate:
                receipt=study.fit(2000,1)
                self.assertEqual(len(receipt['models']),10)
                self.assertEqual(receipt,study.fit(2000,1));self.assertEqual(calculate.call_count,1)
            folder=root/'fits/2000/seed1';verify_stage(folder)
            (folder/'N4.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):study.fit(2000,1)

    def test_independent_R_pairing_intervals_and_report(self):
        config=json.loads((ROOT/'config/extension4_study_v1_proposed.json').read_text())
        config['inference']['draws']=31  # Software check, never the historical study.
        frame=artificial_returns(config)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'analysis'
            result=analyze(frame,config,folder,independent_script=ROOT/'scripts/extension4_independent.R')
            self.assertTrue(result['prespecified_success']);self.assertFalse(result['scientific_study_complete'])
            for row in json.loads((folder/'intervals.json').read_text()):
                np.testing.assert_allclose([row['mean_delta'],row['lower'],row['upper']],.0024,atol=1e-12)
            receipt=render(folder,Path(tmp)/'ARTIFICIAL.html',artificial=True)
            self.assertEqual(receipt['metrics_rows'],126);self.assertEqual(receipt['interval_rows'],12)
            html=(Path(tmp)/'ARTIFICIAL.html').read_text()
            self.assertIn('ARTIFICIAL SOFTWARE TEST',html);self.assertIn('31 paired stationary',html)
            self.assertNotIn('10,000 paired stationary',html)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):analyze(frame.iloc[:-1],config,Path(tmp),independent_script=ROOT/'scripts/extension4_independent.R')

    def test_resource_recovery_counts_all_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);attempts=root/'attempts';attempts.mkdir()
            write_json(attempts/'0001_start.json',dict(phase='pilot',sampling_interval_seconds=2.))
            write_json(attempts/'0001_end.json',dict(active_seconds=10.))
            write_json(attempts/'0002_start.json',dict(phase='pilot',sampling_interval_seconds=2.))
            (attempts/'0002_resources.jsonl').write_text('{"active_seconds":7}\n{"active_seconds":12}\n{interrupted')
            write_json(attempts/'0003_start.json',dict(phase='full',sampling_interval_seconds=2.))
            self.assertEqual(prior_seconds(root),(26.,{'pilot':24.,'full':2.}))

    def test_unsupported_protocol_cannot_be_silently_ignored(self):
        config=json.loads((ROOT/'config/extension4_study_v1_proposed.json').read_text());validate_protocol(config)
        config['penalty_search']['base_constant_penalized']=False
        with self.assertRaises(ValueError):validate_protocol(config)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):require_engineering(Path(tmp))

    def test_approved_passing_pilot_advances_and_failed_pilot_stops(self):
        # Mocked authority and study: no approval file or economic output is created.
        with tempfile.TemporaryDirectory() as tmp:
            study=Mock();study.destination=Path(tmp);study.config={}
            config=json.loads((ROOT/'config/extension4_study_v1_proposed.json').read_text())
            with patch('sys.argv',['extension4_study.py','run']),patch('extension4.run.require_approval',return_value=(config,{})),\
                    patch('extension4.run.Study',return_value=study),patch('extension4.run.Runtime') as runtime:
                run_main()
                self.assertEqual([call[0] for call in study.method_calls],['pilot','full'])
                self.assertEqual([call.args[-1] for call in runtime.call_args_list],['pilot','full'])
            study.reset_mock();study.pilot.side_effect=RuntimeError('technical failure')
            with patch('sys.argv',['extension4_study.py','run']),patch('extension4.run.require_approval',return_value=(config,{})),\
                    patch('extension4.run.Study',return_value=study),patch('extension4.run.Runtime'):
                with self.assertRaisesRegex(RuntimeError,'technical failure'):run_main()
            study.full.assert_not_called()


if __name__=='__main__':unittest.main()
