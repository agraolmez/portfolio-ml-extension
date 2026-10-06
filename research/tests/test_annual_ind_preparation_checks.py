import unittest
from concurrent.futures import ThreadPoolExecutor
from annual_ind.preparation_checks import PreparationFiles

class FakeFiles:
    def __init__(self):self.changed=False;self.scans=0;self.reads=0
    def unchanged(self):
        self.scans+=1
        if self.changed:raise ValueError('Accepted source changed')
    def path(self,name):
        self.reads+=1
        if self.changed:raise ValueError('Actual file read changed')
        return name

class PreparationCheckTests(unittest.TestCase):
    def test_redundant_scans_grouped_but_reads_and_end_checks_remain(self):
        raw=FakeFiles();f=PreparationFiles(raw)
        with f.preparation():
            for _ in range(32):f.unchanged();self.assertEqual(f.path('input'),'input')
            self.assertEqual(raw.scans,1)
        self.assertEqual(raw.scans,2);self.assertEqual(raw.reads,32)
        self.assertEqual(f.completed_batches,1);self.assertEqual(f.deferred_checks,32)
        f.unchanged();self.assertEqual(raw.scans,3)

    def test_preexisting_source_change_prevents_any_preparation(self):
        raw=FakeFiles();raw.changed=True;f=PreparationFiles(raw)
        with self.assertRaisesRegex(ValueError,'source changed'):
            with f.preparation():self.fail('Must not prepare changed source')
        raw.changed=False
        with f.preparation():pass
        self.assertEqual(f.completed_batches,1)

    def test_change_during_preparation_cannot_escape_as_success(self):
        raw=FakeFiles();f=PreparationFiles(raw)
        def work():
            with f.preparation():
                raw.changed=True;f.unchanged();return 'unsafe result'
        with self.assertRaisesRegex(ValueError,'source changed'):work()
        self.assertEqual(f.completed_batches,0);self.assertIsNone(f._owner)

    def test_individual_actual_file_check_is_never_deferred(self):
        raw=FakeFiles();f=PreparationFiles(raw)
        with self.assertRaisesRegex(ValueError,'source changed'):
            with f.preparation():
                raw.changed=True
                with self.assertRaisesRegex(ValueError,'file read changed'):f.path('input')

    def test_failure_still_checks_sources_and_releases_scope(self):
        raw=FakeFiles();f=PreparationFiles(raw)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with f.preparation():raise RuntimeError('deliberate')
        self.assertEqual(raw.scans,2);self.assertIsNone(f._owner)
        with f.preparation():pass
        self.assertEqual(f.completed_batches,1)

    def test_nested_or_cross_thread_use_rejected(self):
        raw=FakeFiles();f=PreparationFiles(raw)
        with f.preparation():
            with self.assertRaisesRegex(RuntimeError,'Nested'):
                with f.preparation():pass
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaisesRegex(RuntimeError,'across threads'):pool.submit(f.unchanged).result()
        self.assertIsNone(f._owner)

if __name__=='__main__':unittest.main()
