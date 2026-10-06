"""Synthetic authorization, immutable reuse and staged-job planning checks."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from feature_geometry.history import sha
from temporal_numerics.study import (_attempt_seconds, _budget_authorizations, _load_fit, _merge_inventories, candidate_name, fit_specs,
                                verify_approval, write_new)

ROOT = Path(__file__).resolve().parents[1]


class RevisedStudyOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT/'config/extension3_study_v2.json').read_text())
        self.config_path = self.root/'config/study.json'
        write_new(self.config_path, self.config)
        self.pilot = self.root/'output/pilot'
        self.pilot.mkdir(parents=True)
        self.source = self.root/'fixture.py'
        self.source.write_text('synthetic fixture\n')
        self.result = dict(status='PILOT_COMPLETE', all_attempted_fits_converged=True,
            numerical_results={'temporal':dict(status='converged', converged=True,
                neutral_gate={'passed':True}, solve_residual=1e-15)})
        write_new(self.pilot/'pilot_result.json', self.result)
        import scipy
        write_new(self.pilot/'temporal/fit.json', dict(optimizer_method='L-BFGS-B',
            scipy_version=scipy.__version__, optimizer_settings=self.config['optimizer']))
        write_new(self.pilot/'pilot_lock.json', dict(config=self.config,
            code_sha256={'fixture.py':sha(self.source)}))
        self.approval_path = self.root/'config/approval.json'
        self.approval = dict(full_historical_experiment=True, paid_resources=False,
            overwrite_existing_results=False, config_sha256=sha(self.config_path),
            user_confirmation='Synthetic approval fixture; no historical execution',
            maximum_total_wall_seconds=100.,
            pilot_result_path='output/pilot/pilot_result.json',
            pilot_manifest_path='output/pilot/pilot_manifest.json')
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def refresh(self):
        # Test fixtures deliberately mutate their own synthetic receipts to
        # exercise gates; production writers never overwrite saved artifacts.
        manifest = dict(status='COMPLETE_PILOT_ONLY', artifact_sha256={
            str(p.relative_to(self.root)):sha(p) for p in self.pilot.rglob('*')
            if p.is_file() and p.name != 'pilot_manifest.json'})
        (self.pilot/'pilot_manifest.json').write_text(json.dumps(manifest))
        self.approval.update(pilot_result_sha256=sha(self.pilot/'pilot_result.json'),
                             pilot_manifest_sha256=sha(self.pilot/'pilot_manifest.json'))
        self.approval_path.write_text(json.dumps(self.approval))

    def verify(self):
        return verify_approval(self.root, self.config_path, self.approval_path)

    def selections(self):
        return {kind:{'selected_by_seed':[dict(seed=seed, parameters={'rho':rho},
                    candidate=kind+'__'+candidate_name(rho)) for seed,rho in zip((1,2,3),(.1,.01,.001))]}
                for kind in ('temporal', 'current_only')}

    def test_full_run_receipt_binds_exact_successful_pilot_and_config(self):
        config, approval = self.verify()
        self.assertFalse(config['approval']['full_historical_experiment'])
        self.assertTrue(approval['full_historical_experiment'])
        self.assertEqual(approval['maximum_total_wall_seconds'], 100.)

    def test_pilot_permission_alone_does_not_authorize_full_run(self):
        self.approval['full_historical_experiment'] = False
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Explicit full-run approval'):
            self.verify()

    def test_missing_total_budget_or_paid_permission_rejected(self):
        for value in (None, 0, -1, float('inf'), True):
            self.approval['maximum_total_wall_seconds'] = value
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'wall-clock budget'):
                self.verify()
        self.approval['maximum_total_wall_seconds'] = 100
        self.approval['paid_resources'] = True
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Explicit full-run approval'):
            self.verify()

    def test_unsuccessful_or_empty_pilot_cannot_launch_study(self):
        for change in ({'converged':False}, {'neutral_gate':{'passed':False}}, {'solve_residual':.1}):
            result = dict(self.result, numerical_results={'temporal':dict(self.result['numerical_results']['temporal'], **change)})
            (self.pilot/'pilot_result.json').write_text(json.dumps(result))
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'Pilot convergence'):
                self.verify()
        (self.pilot/'pilot_result.json').write_text(json.dumps(dict(self.result, numerical_results={})))
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'include the temporal'):
            self.verify()

    def test_changed_config_pilot_and_source_hashes_rejected(self):
        self.approval['config_sha256'] = '0'*64
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'exact configuration'):
            self.verify()
        self.approval['config_sha256'] = sha(self.config_path)
        self.refresh()
        self.source.write_text('changed fixture\n')
        with self.assertRaisesRegex(ValueError, 'Changed locked dependency'):
            self.verify()

    def test_different_pilot_solver_runtime_cannot_authorize_revision(self):
        path = self.pilot/'temporal/fit.json'
        fit = json.loads(path.read_text())
        fit['scipy_version'] = 'different-version'
        path.write_text(json.dumps(fit))
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'SciPy version differs'):
            self.verify()

    def test_future_fits_exist_only_after_per_seed_selection(self):
        initial = fit_specs(self.config)
        self.assertEqual(len(initial), 630)
        self.assertEqual({r['year'] for r in initial}, set(range(1981,2011)))
        complete = fit_specs(self.config, self.selections())
        self.assertEqual(len(complete), 765)
        later = [r for r in complete if r['year'] >= 2011]
        self.assertEqual(len(later), 135)
        for row in later:
            if row['kind'] != 'fixed_temporal':
                self.assertEqual(row['rho'], (.1,.01,.001)[row['seed']-1])
        self.assertEqual(len([r for r in complete if r['kind']=='fixed_temporal']), 135)

    def test_later_stage_requires_one_declared_penalty_per_seed(self):
        selections = self.selections()
        selections['temporal']['selected_by_seed'][0]['parameters']['rho'] = 9.
        with self.assertRaisesRegex(ValueError, 'uniquely selected declared'):
            fit_specs(self.config, selections)
        selections = self.selections()
        selections['current_only']['selected_by_seed'].append(selections['current_only']['selected_by_seed'][0])
        with self.assertRaisesRegex(ValueError, 'uniquely selected declared'):
            fit_specs(self.config, selections)

    def test_elapsed_budget_survives_resumes_and_interruption(self):
        output = self.root/'output/study'
        start = output/'attempts/0001_start.json'
        write_new(start, dict(started_at_utc=datetime.now(timezone.utc).isoformat()))
        write_new(start.with_name('0001_end.json'), dict(start_sha256=sha(start), seconds=12.5))
        self.assertEqual(_attempt_seconds(output), 12.5)
        write_new(output/'attempts/0002_start.json', dict(
            started_at_utc=(datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat()))
        self.assertGreaterEqual(_attempt_seconds(output), 32.5)
        start.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed prior attempt'):
            _attempt_seconds(output)

    def test_new_receipts_never_overwrite_prior_results(self):
        with self.assertRaises(FileExistsError):
            write_new(self.pilot/'pilot_result.json', {'status':'replacement'})
        self.assertEqual(json.loads((self.pilot/'pilot_result.json').read_text()), self.result)

    def saved_fit(self, *, converged=True):
        from tests.test_temporal_paths import parent_fixture, fitted_policy
        from temporal_extension.training import save_result
        parent = parent_fixture(2000)
        model, receipt = fitted_policy(parent)
        spec = dict(year=2000, seed=1, kind='temporal', rho=.1, candidate='rho_0p1')
        lock_hash = 'a'*64
        import scipy
        receipt.update(optimizer_method='L-BFGS-B', scipy_version=scipy.__version__, optimizer_settings=self.config['optimizer'], objective_scale=parent.objective_scale,
            parent_settings=asdict(parent.settings), parent_artifacts_sha256=dict(parent.hashes),
            parent_scale_optimizer_status=parent.metadata['status'], rho=.1,
            fit_spec=spec, study_lock_sha256=lock_hash)
        if not converged:
            receipt.update(status='step_limit', converged=False, eligible_for_portfolio_evaluation=False)
        folder = self.root/'output/synthetic_fit'
        save_result(folder, model, receipt)
        commit = dict(study_lock_sha256=lock_hash, fit_spec=spec,
            artifact_sha256={str(p.relative_to(self.root)):sha(p) for p in folder.iterdir() if p.is_file()})
        write_new(folder/'fit_commit.json', commit)
        return folder, spec, parent, lock_hash

    def test_verified_complete_fit_is_reused_and_exact_settings_are_checked(self):
        folder, spec, parent, lock_hash = self.saved_fit()
        model, receipt = _load_fit(self.root, folder, spec, parent, lock_hash, self.config)
        self.assertEqual(receipt['rho'], .1)
        self.assertEqual(model.parent_fingerprint, parent.fingerprint)
        receipt['optimizer_settings']['tolerance'] = 1.
        (folder/'fit.json').write_text(json.dumps(receipt))
        commit = json.loads((folder/'fit_commit.json').read_text())
        commit['artifact_sha256'][str((folder/'fit.json').relative_to(self.root))] = sha(folder/'fit.json')
        (folder/'fit_commit.json').write_text(json.dumps(commit))
        with self.assertRaisesRegex(ValueError, 'optimizer settings differ'):
            _load_fit(self.root, folder, spec, parent, lock_hash, self.config)

    def test_nonconverged_candidate_stops_instead_of_being_removed(self):
        folder, spec, parent, lock_hash = self.saved_fit(converged=False)
        with self.assertRaisesRegex(ValueError, 'nonconverged candidate stops'):
            _load_fit(self.root, folder, spec, parent, lock_hash, self.config)
        self.assertTrue((folder/'diagnostic_model.npz').exists())

    def test_modified_committed_fit_or_partial_folder_never_reused(self):
        folder, spec, parent, lock_hash = self.saved_fit()
        (folder/'model.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Changed locked dependency'):
            _load_fit(self.root, folder, spec, parent, lock_hash, self.config)
        (folder/'fit_commit.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Preserved partial fit'):
            _load_fit(self.root, folder, spec, parent, lock_hash, self.config)

    def test_resumed_dependency_inventory_retains_reused_training_and_paths(self):
        fit = {'training-risk': 'a'*64, 'ranks': 'b'*64}
        path = {'evaluation-risk': 'c'*64, 'ranks': 'b'*64}
        first = _merge_inventories(fit, path, {'ranks':'b'*64, 'benchmark':'d'*64})
        resumed = _merge_inventories(fit, path, {'benchmark':'d'*64})
        self.assertEqual(first, resumed)
        self.assertIn('training-risk', resumed)
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            _merge_inventories(fit, {'ranks':'9'*64})

    def budget_problem(self):
        destination = self.root/'output/budget_fixture'
        write_new(destination/'study_lock.json', dict(config=self.config,
            approval_sha256=sha(self.approval_path), source_lock='unchanged synthetic code'))
        return destination

    def extension(self, name, ceiling, *, previous_path=None, previous_ceiling=100., **changes):
        body = dict(extend_full_run_budget=True, original_approval_sha256=sha(self.approval_path),
            config_sha256=sha(self.config_path),
            previous_authorization_sha256=sha(previous_path or self.approval_path),
            previous_maximum_total_wall_seconds=previous_ceiling,
            maximum_total_wall_seconds=ceiling,
            user_confirmation='Explicit synthetic increase to the total run budget',
            paid_resources=False, overwrite_existing_results=False, scientific_settings_unchanged=True)
        body.update(changes)
        path = self.root/'config'/name
        write_new(path, body)
        return path

    def budgets(self, destination, *paths, allow_new=True):
        return _budget_authorizations(self.root, destination, self.config_path,
            self.approval_path, self.approval, paths, allow_new=allow_new)

    def test_initial_budget_is_unchanged_without_extension(self):
        destination = self.budget_problem()
        result = self.budgets(destination)
        self.assertEqual(result['maximum_total_wall_seconds'], 100.)
        self.assertEqual(result['latest_authorization_sha256'], sha(self.approval_path))
        self.assertEqual(result['extension_receipts'], [])
        self.assertFalse((destination/'budget_authorizations').exists())

    def test_explicit_extensions_append_without_changing_original_approval_or_lock(self):
        destination = self.budget_problem()
        original_approval, original_lock = sha(self.approval_path), sha(destination/'study_lock.json')
        first = self.extension('increase1.json', 200.)
        second = self.extension('increase2.json', 500., previous_path=first, previous_ceiling=200.)
        result = self.budgets(destination, first, second)
        self.assertEqual(result['maximum_total_wall_seconds'], 500.)
        self.assertEqual(len(result['extension_receipts']), 2)
        self.assertEqual(sha(self.approval_path), original_approval)
        self.assertEqual(sha(destination/'study_lock.json'), original_lock)
        journal = destination/'budget_authorizations'
        old_bytes = {p.name:p.read_bytes() for p in journal.iterdir()}
        self.assertEqual(self.budgets(destination), result)  # no flags on resume
        self.assertEqual(self.budgets(destination, first, second), result)  # repeated explicit chain
        self.assertEqual({p.name:p.read_bytes() for p in journal.iterdir()}, old_bytes)

    def test_extension_cannot_grant_missing_original_full_run_permission(self):
        self.approval['full_historical_experiment'] = False
        self.refresh()
        destination = self.budget_problem()
        extension = self.extension('unapproved.json', 200.)
        with self.assertRaisesRegex(ValueError, 'original full-run approval'):
            self.budgets(destination, extension)
        self.assertFalse((destination/'budget_authorizations').exists())

    def test_bad_chain_or_scientific_change_is_not_partially_persisted(self):
        cases = [dict(previous_authorization_sha256='0'*64),
                 dict(previous_maximum_total_wall_seconds=99.),
                 dict(original_approval_sha256='0'*64), dict(config_sha256='0'*64),
                 dict(scientific_settings_unchanged=False), dict(paid_resources=True),
                 dict(overwrite_existing_results=True), dict(user_confirmation=''),
                 dict(optimizer={'max_steps':999})]
        destination = self.budget_problem()
        first = self.extension('valid_first.json', 200.)
        for i, change in enumerate(cases):
            second = self.extension(f'invalid{i}.json', 400., previous_path=first,
                                    previous_ceiling=200., **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'chained budget-only'):
                self.budgets(destination, first, second)
            self.assertFalse((destination/'budget_authorizations').exists())

    def test_ceiling_must_increase_and_cannot_reset_spent_time(self):
        destination = self.budget_problem()
        for i, value in enumerate((100., 50., True)):
            path = self.extension(f'bad_ceiling{i}.json', value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'strictly larger'):
                self.budgets(destination, path)
        nonfinite = self.extension('nonfinite_ceiling.json', 200.)
        invalid_body = json.loads(nonfinite.read_text())
        invalid_body['maximum_total_wall_seconds'] = float('inf')
        # Deliberately malformed external input; the production writer itself
        # correctly refuses to serialize non-finite approval values.
        nonfinite.write_text(json.dumps(invalid_body))
        with self.assertRaisesRegex(ValueError, 'strictly larger'):
            self.budgets(destination, nonfinite)
        start = destination/'attempts/0001_start.json'
        write_new(start, dict(started_at_utc=datetime.now(timezone.utc).isoformat()))
        write_new(start.with_name('0001_end.json'), dict(start_sha256=sha(start), seconds=125.))
        extension = self.extension('later_budget.json', 200.)
        budget = self.budgets(destination, extension)
        self.assertEqual(_attempt_seconds(destination), 125.)
        self.assertEqual(budget['maximum_total_wall_seconds']-_attempt_seconds(destination), 75.)

    def test_locked_receipt_bytes_and_journal_chain_are_checked_on_resume(self):
        destination = self.budget_problem()
        path = self.extension('bound_receipt.json', 200.)
        self.budgets(destination, path)
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'locked budget-extension receipt changed'):
            self.budgets(destination)
        path.write_bytes(original)
        record_path = destination/'budget_authorizations/0001.json'
        record = json.loads(record_path.read_text())
        record['study_lock_sha256'] = '0'*64
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'another study lock or chain'):
            self.budgets(destination)

    def test_extra_wall_budget_does_not_make_a_diagnostic_fit_reusable(self):
        destination = self.budget_problem()
        folder, spec, parent, lock_hash = self.saved_fit(converged=False)
        preserved = {p.name:p.read_bytes() for p in folder.iterdir()}
        extension = self.extension('budget_only_not_fit_retry.json', 200.)
        self.assertEqual(self.budgets(destination, extension)['maximum_total_wall_seconds'], 200.)
        with self.assertRaisesRegex(ValueError, 'nonconverged candidate stops'):
            _load_fit(self.root, folder, spec, parent, lock_hash, self.config)
        self.assertEqual({p.name:p.read_bytes() for p in folder.iterdir()}, preserved)

    def test_chain_cannot_be_skipped_reordered_or_added_after_numerical_completion(self):
        destination = self.budget_problem()
        first = self.extension('first_link.json', 200.)
        second = self.extension('second_link.json', 300., previous_path=first, previous_ceiling=200.)
        with self.assertRaisesRegex(ValueError, 'chained budget-only'):
            self.budgets(destination, second, first)
        with self.assertRaisesRegex(ValueError, 'completed study'):
            self.budgets(destination, first, allow_new=False)
        self.assertFalse((destination/'budget_authorizations').exists())
        self.budgets(destination, first, second)
        (destination/'budget_authorizations/0001.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or reordered'):
            self.budgets(destination)


if __name__ == '__main__':
    unittest.main()
