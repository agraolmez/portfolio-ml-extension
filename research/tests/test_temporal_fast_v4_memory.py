"""Resource gates account for macOS compressed memory as well as resident RSS."""
from pathlib import Path
from tempfile import TemporaryDirectory
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from temporal_fast_v4.memory import usage,_Usage
from temporal_fast_v4.study import Runtime

class ChargedMemoryTests(unittest.TestCase):
    def test_native_layout_and_process_observation(self):
        import ctypes
        self.assertEqual(ctypes.sizeof(_Usage),296)
        self.assertEqual(_Usage.phys_footprint.offset,72)
        self.assertEqual(_Usage.lifetime_max_phys_footprint.offset,240)
        row=usage();self.assertGreater(row['physical_footprint_bytes'],0)
        self.assertGreaterEqual(row['peak_physical_footprint_bytes'],row['physical_footprint_bytes'])
        with self.assertRaises(OSError):usage(-1)

    def test_compressed_memory_cannot_hide_a_resource_violation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);config={'resources':dict(process_memory_gib=24,minimum_free_disk_gib=0,maximum_new_storage_gib=10)}
            runtime=Runtime(root,root,config,dict(maximum_total_wall_seconds=1000),StringIO(),0)
            runtime.process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=2**30))
            with patch('temporal_fast_v4.study.memory_usage',return_value=dict(physical_footprint_bytes=25*2**30,peak_physical_footprint_bytes=25*2**30)):
                with self.assertRaisesRegex(MemoryError,'memory'):runtime.guard()
            with patch('temporal_fast_v4.study.memory_usage',return_value=dict(physical_footprint_bytes=10*2**30,peak_physical_footprint_bytes=10*2**30)):
                runtime.guard();self.assertEqual(runtime.peak_rss,2**30)

if __name__=='__main__':unittest.main()
