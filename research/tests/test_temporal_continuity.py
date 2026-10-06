"""ID-representation regression and fail-closed preserved-artifact recovery."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_temporal_paths as fixtures
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from temporal_study.paths import run_paths as original_paths
from temporal_study.selection import Candidate, CandidatePath, SelectionPlan, select_candidates
from temporal_continuity.identity import canonical_ids, holdings_fingerprint, annual_volatility
from temporal_continuity.paths import run_paths
from temporal_continuity.repair import derive_candidate_metadata, verify_repair
from temporal_fast_v4.study import write_new


class ContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_hash_losslessly_unifies_float_and_integer_ids_without_changing_holdings(self):
        h = np.array([.1, -.2, 0.])
        for dtype in (np.int32, np.int64, np.float32, np.float64, np.uint64):
            self.assertEqual(holdings_fingerprint(np.array([1, 4, 9], dtype=dtype), h),
                             fingerprint(np.array([1, 4, 9], dtype=np.int64), h))
        self.assertNotEqual(holdings_fingerprint([1, 4, 9], h), holdings_fingerprint([4, 1, 9], h))
        altered = h.copy(); altered[0] = np.nextafter(altered[0], np.inf)
        self.assertNotEqual(holdings_fingerprint([1, 4, 9], h), holdings_fingerprint([1, 4, 9], altered))

    def test_bad_ids_rejected_without_rounding(self):
        for ids in ([1.5, 2], [np.nan, 2], [np.inf, 2], [1, 1], ['1', '2'],
                    [True, False], [float(2**54), 2], np.array([2**63, 2], dtype=np.uint64), [[1, 2]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                canonical_ids(ids)

    def test_bad_holdings_rejected_without_coercion(self):
        for h in (np.array([1], dtype=float), np.array([1, 2], dtype=np.float32), np.array([np.nan, 1])):
            with self.assertRaises(ValueError): holdings_fingerprint([1, 2], h)

    def test_risk_diagnostic_strided_vectors_and_real_overflow(self):
        rng = np.random.default_rng(21)
        a = rng.normal(size=(17, 17)); cov = a@a.T
        weights = rng.normal(size=(17, 3))
        for w in weights.T:
            self.assertAlmostEqual(annual_volatility(w, cov), np.sqrt(12*w@cov@w), places=12)
        self.assertEqual(annual_volatility(np.zeros(17), cov), 0.)
        for w,c in [(np.array([np.inf]), np.eye(1)), (np.ones(2), -np.eye(2))]:
            with self.assertRaises(ValueError): annual_volatility(w,c)
        with self.assertRaises((ValueError, FloatingPointError)):
            annual_volatility(np.full(2, 1e308), np.eye(2))

    def problem(self):
        source, initial, policies, options = fixtures.PathTests().problem()
        realized = source.realized
        source.realized = lambda market: replace(realized(market), next_ids=realized(market).next_ids.astype(float))
        return source, initial, policies, options

    def test_new_paths_equal_original_economics_and_pass_original_selection(self):
        source, initial, policies, options = self.problem()
        old = original_paths(source, source.dates, policies, initial, **options)
        new = run_paths(source, source.dates, policies, initial, **options)
        hashes = ['pretrade_sha256','next_pretrade_sha256']
        pd.testing.assert_frame_equal(old['monthly'].drop(columns=hashes+['modeled_annual_volatility']),
                                      new['monthly'].drop(columns=hashes+['modeled_annual_volatility']), check_exact=True)
        np.testing.assert_allclose(old['monthly'].modeled_annual_volatility, new['monthly'].modeled_annual_volatility,
                                   atol=1e-12, rtol=1e-11)
        for a,b in zip(old['positions'],new['positions']):
            for key in a: np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(new['manifest']['neutral_maximum_position_error'], 0.)
        plan = SelectionPlan(protocol_id='fixture',family_id='test',seeds=(1,2),
            candidates=(Candidate('learned', {'rho': .1}),),initial_decision=source.dates[0],
            validation_first_return='2001-01-31',validation_last_return='2001-02-28',
            selection_available_at='2001-02-28',first_evaluation_decision='2001-02-28',
            max_available_decision='2001-01-31',max_available_outcome='2001-02-28',
            initial_pretrade_sha256=holdings_fingerprint(initial.ids,initial.holdings),gamma=10.,
            exposure_label='artificial regression only')
        def candidates(result):
            frame=result['monthly']
            return [CandidatePath(seed,'learned',frame[(frame.seed==seed)&(frame.method=='learned')],
                                  {'fixture': 'a'*64},True) for seed in (1,2)]
        with self.assertRaisesRegex(ValueError,'carry continuously'):
            select_candidates(plan=plan,paths=candidates(old))
        selected=select_candidates(plan=plan,paths=candidates(new))
        self.assertEqual(len(selected['selected_by_seed']),2)

    def saved_fixture(self, root):
        source, initial, policies, options = self.problem()
        folder=root/'output/original';options['destination']=folder
        folder.parent.mkdir(parents=True,exist_ok=True)
        result=original_paths(source,source.dates,policies,initial,**options)
        inventory={str(p.relative_to(root)):sha(p) for p in folder.iterdir() if p.is_file()}
        write_new(folder/'stage_commit.json',dict(status='COMPLETE',study_lock_sha256='a'*64,artifact_sha256=inventory))
        return folder,result

    def test_derivative_preserves_original_files_and_all_economic_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,result=self.saved_fixture(root)
            before={p.name:sha(p) for p in folder.iterdir()}
            repaired=derive_candidate_metadata(root,folder,root/'output/repaired','a'*64)
            self.assertEqual(before,{p.name:sha(p) for p in folder.iterdir()})
            self.assertEqual(repaired['audit']['adjacent_path_links'],12)
            self.assertEqual(repaired['audit']['raw_fingerprint_mismatches'],12)
            fields=['pretrade_sha256','next_pretrade_sha256']
            pd.testing.assert_frame_equal(result['monthly'].drop(columns=fields),repaired['monthly'].drop(columns=fields),check_exact=True)
            with self.assertRaises(FileExistsError):
                derive_candidate_metadata(root,folder,root/'output/repaired','a'*64)

    def test_changed_committed_file_is_rejected_before_derivative_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,_=self.saved_fixture(root)
            with (folder/'monthly.parquet').open('ab') as f:f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'Changed locked dependency'):
                derive_candidate_metadata(root,folder,root/'output/repaired','a'*64)
            self.assertFalse((root/'output/repaired').exists())

    def test_legitimately_hashed_but_discontinuous_holdings_still_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,_=self.saved_fixture(root)
            path=folder/'2000-12-31.npz'
            with np.load(path) as d:arrays={k:d[k] for k in d.files}
            arrays['pretrade'][0,0]+=1e-10
            np.savez_compressed(path,**arrays)
            table=pd.read_parquet(folder/'monthly.parquet')
            match=(table.decision_eom==pd.Timestamp('2000-12-31'))&(table.seed==arrays['seeds'][0])&(table.method==arrays['methods'][0])
            table.loc[match,'pretrade_sha256']=fingerprint(arrays['ids'],arrays['pretrade'][:,0])
            table.to_parquet(folder/'monthly.parquet',index=False)
            manifest=json.loads((folder/'paths_manifest.json').read_text())
            for name in ('2000-12-31.npz','monthly.parquet'):manifest['artifact_sha256'][name]=sha(folder/name)
            (folder/'paths_manifest.json').write_text(json.dumps(manifest))
            commit=json.loads((folder/'stage_commit.json').read_text())
            for name in commit['artifact_sha256']:commit['artifact_sha256'][name]=sha(root/name)
            (folder/'stage_commit.json').write_text(json.dumps(commit))
            with self.assertRaisesRegex(ValueError,'discontinuous'):
                derive_candidate_metadata(root,folder,root/'output/repaired','a'*64)
            self.assertFalse((root/'output/repaired').exists())

    def test_versioned_path_code_diff_is_limited_to_hashes_and_diagnostic(self):
        import inspect
        source=inspect.getsource(original_paths)
        source=source.replace('fingerprint(market.ids, h[:, j])','holdings_fingerprint(market.ids, h[:, j])')
        source=source.replace("fingerprint(realized.next_ids, result['next_pretrade'])","holdings_fingerprint(realized.next_ids, result['next_pretrade'])")
        source=source.replace('fingerprint(initial.ids, initial.holdings)','holdings_fingerprint(initial.ids, initial.holdings)')
        source=source.replace('float(np.sqrt(12*weights@market.covariance@weights))','annual_volatility(weights, market.covariance)')
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(run_paths))))

    def test_fitting_loading_and_authorization_functions_remain_identical(self):
        root=Path(__file__).resolve().parents[1]
        def functions(path):
            tree=ast.parse(path.read_text())
            return {node.name:ast.dump(node) for node in tree.body
                    if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        old=functions(root/'temporal_fast_v4/study.py')
        new=functions(root/'temporal_continuity/study.py')
        self.assertEqual(set(old),set(new))
        changed={name for name in old if old[name]!=new[name]}
        self.assertEqual(changed,{'_select','_external_benchmark','run_study'})

    def test_repair_receipt_binds_code_authority_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);study=root/'output/study';study.mkdir(parents=True)
            (study/'study_lock.json').write_text('{}')
            (root/'config.json').write_text('{}');(root/'approval.json').write_text('{}')
            package=root/'temporal_continuity';package.mkdir()
            (package/'identity.py').write_text('# fixture')
            (root/'scripts').mkdir();(root/'scripts/extension3_continuity_study.py').write_text('# fixture')
            artifact=study/'evidence.json';artifact.write_text('{}')
            receipt=dict(status='VERIFIED_CONTINUITY_REPAIR',original_study_output='output/study',
                original_study_lock_sha256=sha(study/'study_lock.json'),
                original_approval_sha256=sha(root/'approval.json'),config_sha256=sha(root/'config.json'),
                scientific_settings_changed=False,fitting_engine_changed=False,original_results_overwritten=False,
                resource_limits_changed=False,paid_resources=False,user_confirmation='authorized fixture',
                gates=[dict(passed=True)],code_sha256={str(p.relative_to(root)):sha(p) for p in
                    [package/'identity.py',root/'scripts/extension3_continuity_study.py']},
                artifact_sha256={'output/study/evidence.json':sha(artifact)})
            path=root/'output/repair.json';path.write_text(json.dumps(receipt))
            def check():return verify_repair(root,study,root/'config.json',root/'approval.json',path)
            self.assertEqual(check(),receipt)
            for field in ('scientific_settings_changed','fitting_engine_changed','original_results_overwritten','resource_limits_changed','paid_resources'):
                changed=dict(receipt);changed[field]=True;path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'exact unchanged'):check()
            path.write_text(json.dumps(receipt));artifact.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed locked dependency'):check()


if __name__=='__main__':unittest.main()
