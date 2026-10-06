"""Optimized study cannot bypass provenance, full-run scope or fit identity."""
from pathlib import Path
import json,tempfile,unittest
from feature_geometry.history import sha
from temporal_fast_v4.settings import EXECUTION,environment_identity
from temporal_fast_v4.validation import verify_optimized_approval
from temporal_fast_v4.study import _load_fit,verify_approval
import test_temporal_numerical_study as fixtures


class FastStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=self.root/'config/extension3_speed_v4.json';self.config.parent.mkdir()
        self.config.write_text(json.dumps(dict(execution=dict(EXECUTION))))
        self.sources=['temporal_fast_v4/'+p for p in ('kernels.py','kernels.c','native.py','streaming.py','settings.py','training.py','validation.py','study.py','source_checks.py','source_checks.c','history.py','reference.py','storage.py','cache.py','deferred.py','controls.py','risk.py','panels.py','memory.py')]
        for name in self.sources:
            path=self.root/name;path.parent.mkdir(exist_ok=True);path.write_text('# synthetic\n')
        real_root=Path(__file__).resolve().parents[1]
        for name in {name for key in ('native_build','compact_build','source_checks_build') for name in environment_identity()[key]['artifact_sha256']}:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((real_root/name).read_bytes())
            self.sources.append(name)
        self.path=self.root/'output/verification.json';self.path.parent.mkdir()
        self.receipt=dict(status='VERIFIED_OPTIMIZATION_ONLY',execution_settings=dict(EXECUTION),
            execution_environment=environment_identity(),full_pilot_profiles_passed=True,
            both_neutral_coefficients_match_parent=True,both_saved_pilots_still_converged=True,
            maximum_history_profiles_passed=True,disk_and_reference_equivalence_passed=True,
            portfolio_evaluation_run=False,gain_optimization_run=False,
            code_sha256={p:sha(self.root/p) for p in self.sources})
        self.approval=dict(optimized_profile_engine=True,optimization_config_sha256=sha(self.config),
                           speed_verification_path='output/verification.json')
        self.refresh()
    def tearDown(self):self.temp.cleanup()
    def refresh(self):
        self.path.write_text(json.dumps(self.receipt));self.approval['speed_verification_sha256']=sha(self.path)
    def verify(self):return verify_optimized_approval(self.root,self.approval)

    def test_successful_fixed_profile_evidence_is_bound(self):
        self.assertEqual(self.verify(),self.receipt)
        self.approval['optimized_profile_engine']=False
        with self.assertRaisesRegex(ValueError,'approval must bind'):self.verify()

    def test_missing_gates_or_changed_runtime_cannot_activate_engine(self):
        original=dict(self.receipt)
        for change in [dict(full_pilot_profiles_passed=False),dict(both_saved_pilots_still_converged=False),
                       dict(both_neutral_coefficients_match_parent=False),dict(execution_environment={}),
                       dict(portfolio_evaluation_run=True),dict(maximum_history_profiles_passed=False),dict(disk_and_reference_equivalence_passed=False)]:
            self.receipt=dict(original,**change);self.refresh()
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'evidence is incomplete'):self.verify()

    def test_changed_verified_implementation_is_rejected(self):
        (self.root/self.sources[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError,'implementation changed'):self.verify()

    def test_native_binary_and_build_record_are_bound(self):
        artifacts={name for key in ('native_build','compact_build','source_checks_build') for name in environment_identity()[key]['artifact_sha256']}
        for name in artifacts:
            original=(self.root/name).read_bytes()
            (self.root/name).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'implementation changed'):self.verify()
            (self.root/name).write_bytes(original)
            expected=self.receipt['code_sha256'].pop(name);self.refresh()
            with self.assertRaisesRegex(ValueError,'native build missing'):self.verify()
            self.receipt['code_sha256'][name]=expected;self.refresh()

    def test_native_source_and_wrapper_cannot_be_omitted(self):
        for name in ('kernels.c','native.py','settings.py'):
            path='temporal_fast_v4/'+name
            expected=self.receipt['code_sha256'].pop(path);self.refresh()
            with self.assertRaisesRegex(ValueError,'implementation missing'):self.verify()
            self.receipt['code_sha256'][path]=expected;self.refresh()

    def test_legacy_fit_cannot_be_relabelled_as_optimized_execution(self):
        fixture=fixtures.RevisedStudyOrchestrationTests();fixture.setUp()
        try:
            folder,spec,parent,lock=fixture.saved_fit()
            with self.assertRaisesRegex(ValueError,'optimized execution'):
                _load_fit(fixture.root,folder,spec,parent,lock,fixture.config)
            receipt=json.loads((folder/'fit.json').read_text())
            receipt.update(execution_settings=dict(EXECUTION),execution_environment=environment_identity())
            (folder/'fit.json').write_text(json.dumps(receipt))
            commit=json.loads((folder/'fit_commit.json').read_text())
            commit['artifact_sha256'][str((folder/'fit.json').relative_to(fixture.root))]=sha(folder/'fit.json')
            (folder/'fit_commit.json').write_text(json.dumps(commit))
            _load_fit(fixture.root,folder,spec,parent,lock,fixture.config)
        finally:fixture.tearDown()

    def test_optimization_authorization_is_not_full_study_authorization(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'output') as tmp:
            path=Path(tmp)/'no_full_approval.json';path.write_text(json.dumps(self.approval))
            with self.assertRaisesRegex(ValueError,'Explicit full-run approval'):
                verify_approval(root,root/'config/extension3_study_v3_proposed.json',path)


if __name__=='__main__':unittest.main()
