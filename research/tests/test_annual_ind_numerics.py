"""Exercise actual native BLAS thread policy and ordered resource failures."""
import time
import unittest
from annual_ind.numerics import ordered_months
from temporal_fast.backend import accelerate


class NumericalWorkerTests(unittest.TestCase):
    def test_each_worker_enters_policy_and_preserves_order(self):
        def work(i):
            time.sleep((7-i)*.001)
            return i, accelerate().BLASGetThreading()
        for workers in (1, 2, 4, 8):
            with self.subTest(workers=workers):
                actual = list(ordered_months(work, range(8), workers=workers, guard=lambda: None))
                self.assertEqual(actual, [(i, 1) for i in range(8)])

    def test_resource_error_propagates(self):
        def stop():
            raise MemoryError('reserve breached')
        with self.assertRaisesRegex(MemoryError, 'reserve breached'):
            list(ordered_months(lambda i: i, range(4), workers=2, guard=stop))

    def test_empty_input_and_invalid_worker_count(self):
        self.assertEqual(list(ordered_months(str, [], workers=2, guard=lambda: None)), [])
        with self.assertRaises(ValueError):
            list(ordered_months(str, [], workers=0, guard=lambda: None))


if __name__ == '__main__':
    unittest.main()
