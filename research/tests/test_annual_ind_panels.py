import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from annual_ind.panels import CalendarSourcePanels


class AcceptedFile:
    def __init__(self, path):
        self.file = path
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def path(self, name):
        if name != 'source' or hashlib.sha256(self.file.read_bytes()).hexdigest() != self.digest:
            raise ValueError('Changed accepted input')
        return self.file


class CalendarPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/'source.parquet'
        dates = pd.date_range('1998-01-31','2001-12-31',freq='ME')
        rows = pd.DataFrame(dict(id=np.repeat(np.arange(4),len(dates)),
            eom=np.tile(dates,4),x=np.arange(4*len(dates),dtype=float),valid=True))
        rows.loc[3,'x'] = np.nan
        rows.to_parquet(self.path,index=False,row_group_size=25)
        self.files = AcceptedFile(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_requested_month_order_types_masks_and_columns_match_direct_read(self):
        panel = CalendarSourcePanels(self.files,{'source':('id','eom','x','valid')},2**20)
        for date in ('1998-04-30','2000-01-31','1999-12-31','2001-11-30','2000-01-31','2002-01-31'):
            columns = ['eom','x','id']
            actual = panel.read('source',columns,date)
            expected = pd.read_parquet(self.path,columns=columns,filters=[('eom','==',pd.Timestamp(date))])
            pd.testing.assert_frame_equal(actual,expected)
            self.assertTrue(actual.eom.eq(pd.Timestamp(date)).all())
        self.assertEqual(panel.hits,1)

    def test_eviction_bound_and_release(self):
        panel = CalendarSourcePanels(self.files,{'source':('id','eom','x','valid')},1800)
        for year in (1998,1999,2000,2001,1998):
            panel.read('source',['id','x'],f'{year}-01-31')
            self.assertLessEqual(panel.retained_bytes,1800)
        self.assertGreater(panel.evictions,0)
        panel.release()
        self.assertEqual(panel.retained_bytes,0)
        self.assertFalse(panel.blocks)

    def test_invalid_requests_and_changed_source_rejected_even_on_hit(self):
        panel = CalendarSourcePanels(self.files,{'source':('id','eom','x','valid')},2**20)
        panel.read('source',['id'],'1998-01-31')
        with self.assertRaises(ValueError):
            panel.read('source',['missing'],'1998-01-31')
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Changed accepted'):
            panel.read('source',['id'],'1998-01-31')


if __name__ == '__main__':
    unittest.main()
