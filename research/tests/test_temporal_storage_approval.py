"""Measured engineering changes cannot silently activate in a historical run."""
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from feature_geometry.history import sha
from temporal_storage.validation import verify_storage_approval
from temporal_storage.configuration import pilot_configuration, settings, ROOT


class StorageApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root/'config.json'; self.config.write_text('{}')
        self.codes = ['temporal_numerics/spill.py', 'temporal_storage/cache.py']
        for name in self.codes:
            p = self.root/name; p.parent.mkdir(exist_ok=True); p.write_text('# synthetic\n')
        self.path = self.root/'output/storage_check.json'; self.path.parent.mkdir()
        self.receipt = dict(status='PASS', config_sha256=sha(self.config), real_accepted_data=True,
            exact_roundtrip=True, parent_neutral_signals_exact=True, portfolio_evaluation_run=False,
            gain_optimization_run=False, code_sha256={p:sha(self.root/p) for p in self.codes})
        self.approval = dict(exact_overflow_disk_reuse=True, storage_check_path='output/storage_check.json')
        self.refresh()

    def tearDown(self): self.tmp.cleanup()
    def refresh(self):
        self.path.write_text(json.dumps(self.receipt)); self.approval['storage_check_sha256']=sha(self.path)
    def verify(self): return verify_storage_approval(self.root, self.config, self.approval)

    def test_measured_equivalence_with_explicit_scope_passes(self):
        self.assertEqual(self.verify(), self.receipt)
        self.approval['exact_overflow_disk_reuse']=False
        with self.assertRaisesRegex(ValueError, 'Explicit full-run approval'): self.verify()

    def test_synthetic_failed_or_wrong_scientific_check_cannot_authorize(self):
        original = dict(self.receipt)
        for key, value in [('real_accepted_data',False), ('status','FAIL'), ('exact_roundtrip',False),
                           ('config_sha256','wrong'), ('parent_neutral_signals_exact',False),
                           ('portfolio_evaluation_run',True), ('gain_optimization_run',True)]:
            self.receipt=dict(original, **{key:value}); self.refresh()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'equivalence'): self.verify()

    def test_receipt_code_and_inventory_integrity_are_bound(self):
        self.path.write_text(self.path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'bound measured'): self.verify()
        self.refresh(); (self.root/self.codes[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'implementation changed'): self.verify()
        self.receipt['code_sha256']={}; self.refresh()
        with self.assertRaisesRegex(ValueError, 'absent'): self.verify()


class ProposedTimeAllowanceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/extension3_study_v3_proposed.json').read_text())

    def test_only_wall_allowance_differs_and_does_not_authorize_run(self):
        parent=pilot_configuration(self.config)
        self.assertEqual(parent['optimizer']['wall_seconds'], 3600.)
        self.assertEqual(settings(self.config).wall_seconds, 10800.)
        self.assertFalse(self.config['approval']['full_historical_experiment'])
        from temporal_storage.study import verify_approval
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as tmp:
            path=Path(tmp)/'unapproved.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Explicit full-run approval'):
                verify_approval(ROOT, ROOT/'config/extension3_study_v3_proposed.json', path)

    def test_more_time_cannot_silently_weaken_accuracy_or_change_model(self):
        for group,key,value in [('optimizer','tolerance',1e-3),('optimizer','max_steps',101),
                                ('resources','threads',8),('optimizer','wall_seconds',99999)]:
            config=deepcopy(self.config);config[group][key]=value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'Only the proposed'):
                settings(config)
        config=deepcopy(self.config);config['frozen_parent_scales']=False
        with self.assertRaisesRegex(ValueError,'Only the proposed'):settings(config)


if __name__ == '__main__': unittest.main()
