import sqlite3
import tempfile
from pathlib import Path
import unittest
import numpy as np
from annual_ind.checkpoints import CheckpointStore


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='annual-ind-checkpoints-')
        self.path = Path(self.temporary.name)/'models.sqlite'
        self.store = CheckpointStore(self.path, run_identity='synthetic-run')
        self.context = dict(cutoff='1970-12-31', source='fixture', frequency_key='fixed-g0')
        self.arrays = dict(theta=np.array([.1, -.1]), beta=np.array([1., 2., 3.]))

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_roundtrip_reopen_and_idempotent_resume(self):
        expected = self.store.put('parent', self.context, self.arrays)
        self.store.close()
        self.store = CheckpointStore(self.path, run_identity='synthetic-run')
        self.assertEqual(self.store.put('parent', self.context, self.arrays), expected)
        restored = self.store.get('parent', context=self.context)
        for name in self.arrays:
            np.testing.assert_array_equal(restored[name], self.arrays[name])
        self.assertEqual(len(self.store.inventory()), 1)

    def test_cannot_replace_or_relabel_parent(self):
        self.store.put('parent', self.context, self.arrays)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.store.put('parent', self.context, dict(theta=self.arrays['theta'], beta=np.zeros(3)))
        with self.assertRaisesRegex(ValueError, 'context'):
            self.store.get('parent', context=dict(self.context, cutoff='1980-12-31'))
        np.testing.assert_array_equal(self.store.get('parent')['beta'], self.arrays['beta'])

    def test_another_run_cannot_resume(self):
        with self.assertRaisesRegex(ValueError, 'another run'):
            CheckpointStore(self.path, run_identity='different-run')

    def test_tampered_payload_fails_hash_gate(self):
        self.store.put('parent', self.context, self.arrays)
        with sqlite3.connect(self.path) as conn:
            conn.execute('UPDATE bundles SET payload=? WHERE key=?', (b'changed', 'parent'))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.store.get('parent')

    def test_nonfinite_and_object_arrays_rejected(self):
        for value in [np.array([np.nan]), np.array([{'x': 1}], dtype=object)]:
            with self.assertRaises(ValueError):
                self.store.put('bad', self.context, dict(value=value))
        self.assertEqual(self.store.inventory(), [])

    def test_scalar_shape_and_dtype_preserved(self):
        self.store.put('scalar', self.context, dict(value=np.array(7, dtype=np.int32)))
        value = self.store.get('scalar')['value']
        self.assertEqual(value.shape, ())
        self.assertEqual(value.dtype, np.dtype('int32'))

    def test_ambiguous_array_names_rejected(self):
        for name in ('../beta', 'file', '', 7):
            with self.assertRaises(ValueError):
                self.store.put('bad', self.context, {name: np.ones(2)})


if __name__ == '__main__':
    unittest.main()
