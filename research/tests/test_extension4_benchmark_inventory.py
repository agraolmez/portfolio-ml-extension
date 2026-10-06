"""Fail-closed nested acceptance and catalog integrity checks, with toy files."""
import json
from pathlib import Path
import tempfile
import unittest

from extension4.parent import ParentCatalog
from extension4_execution.benchmark_inventory_v1 import GEOMETRY, include_accepted_benchmarks
from feature_geometry.history import sha


class BenchmarkInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.catalog = ParentCatalog.__new__(ParentCatalog)
        self.catalog.root = self.root
        self.catalog.consumed = {}; self.catalog._stats = {}
        self.catalog.acceptance_sha256 = 'original-e3-identity'
        self.artifacts = {}
        for year in range(1981, 2026):
            for seed in (1, 2, 3):
                folder = self.root / GEOMETRY / 'fits' / str(year) / f'seed{seed}' / 'fixed1024'
                folder.mkdir(parents=True)
                (folder/'model.npz').write_bytes(b'toy comparison only')
                receipt = dict(status='EXACT_FIXED_FEATURE_SOLVE', method='fixed1024', year=year,
                    seed=seed, cutoff=f'{year-1}-12-31', files_sha256={'model.npz':sha(folder/'model.npz')})
                (folder/'fit.json').write_text(json.dumps(receipt))
                for p in folder.iterdir():self.artifacts[str(p.relative_to(self.root))] = sha(p)
        self.acceptance = dict(status='ACCEPTED_COMPLETED_HISTORICAL_GEOMETRY_STUDY', accepted=True,
            study_complete=True, years=[1981,2025], frequency_seeds=[1,2,3], gates=[dict(passed=True)],
            artifact_sha256=self.artifacts)
        self.manifest = self.root / GEOMETRY / 'acceptance_manifest.json'
        self.rebind()

    def rebind(self):
        self.manifest.write_text(json.dumps(self.acceptance))
        self.catalog.inventory = {GEOMETRY+'/acceptance_manifest.json':sha(self.manifest)}
        self.catalog.consumed.clear();self.catalog._stats.clear()

    def test_only_explicit_comparisons_added_and_original_identity_preserved(self):
        original = dict(self.catalog.inventory)
        self.acceptance['artifact_sha256']['unrelated.npz'] = '0'*64
        self.rebind();original = dict(self.catalog.inventory)
        receipt = include_accepted_benchmarks(self.catalog)
        self.assertEqual(len(receipt['artifact_sha256']),270)
        self.assertEqual(self.catalog.acceptance_sha256,'original-e3-identity')
        self.assertEqual({k:self.catalog.inventory[k] for k in original},original)
        self.assertNotIn('unrelated.npz',self.catalog.inventory)
        self.assertEqual(receipt,include_accepted_benchmarks(self.catalog))

    def test_changed_acceptance_rejected(self):
        self.manifest.write_text('{}')
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)

    def test_unaccepted_manifest_rejected(self):
        self.acceptance['accepted']=False;self.rebind()
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)

    def test_missing_benchmark_rejected_without_partial_admission(self):
        self.artifacts.pop(next(iter(self.artifacts)));self.rebind()
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)
        self.assertEqual(len(self.catalog.inventory),1)

    def test_changed_model_rejected(self):
        (self.root/next(iter(self.artifacts))).write_bytes(b'tampered')
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)

    def test_conflicting_existing_hash_rejected(self):
        self.catalog.inventory[next(iter(self.artifacts))]='bad'
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)

    def test_bad_year_receipt_rejected(self):
        rel=next(k for k in self.artifacts if k.endswith('fit.json'))
        p=self.root/rel;d=json.loads(p.read_text());d['year']=2099;p.write_text(json.dumps(d))
        self.artifacts[rel]=sha(p);self.rebind()
        with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)

    def test_changed_after_admission_still_rejected(self):
        include_accepted_benchmarks(self.catalog)
        rel=next(iter(self.artifacts));(self.root/rel).write_bytes(b'changed after admission')
        with self.assertRaises(ValueError):self.catalog.path(rel)

    def test_symlink_outside_project_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            p=Path(outside)/'model';p.write_bytes(b'toy comparison only')
            rel=next(iter(self.artifacts));local=self.root/rel;local.unlink();local.symlink_to(p)
            with self.assertRaises(ValueError):include_accepted_benchmarks(self.catalog)


if __name__=='__main__':unittest.main()
