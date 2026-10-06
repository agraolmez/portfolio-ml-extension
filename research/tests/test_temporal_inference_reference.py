"""Independent local R arithmetic on synthetic analysis artifacts only."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from scripts.verify_extension3_inference import export_inputs, run_reference
from temporal_study.analysis import analyze_paths
from test_temporal_analysis import analysis_fixture


class IndependentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rscript = shutil.which('Rscript')
        if cls.rscript is None:
            raise unittest.SkipTest('Local Rscript is unavailable; independent R check not run')
        cls.temporary = tempfile.TemporaryDirectory(prefix='extension3-inference-reference-')
        cls.folder = Path(cls.temporary.name)
        cls.frame, cls.plan, kwargs = analysis_fixture()
        cls.analysis_folder = cls.folder/'analysis'
        analyze_paths(cls.frame, cls.plan, **kwargs, destination=cls.analysis_folder)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'temporary'):
            cls.temporary.cleanup()

    def test_R_recomputes_utilities_seed_averages_and_four_family_intervals(self):
        destination = self.folder/'verified'
        export_inputs(self.frame, self.plan, self.analysis_folder, destination)
        result = run_reference(destination, rscript=self.rscript)
        receipt = result['receipt']
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['cases_checked'], 4)
        self.assertEqual(receipt['bootstrap_utilities_checked'], 4*64*15)
        self.assertEqual(receipt['contrast_intervals_checked'], 64)
        self.assertEqual(receipt['comparison_family'], 4)
        self.assertLess(receipt['maximum_utility_error'], 1e-11)
        self.assertLess(receipt['maximum_interval_error'], 1e-11)
        self.assertFalse(receipt['historical_fit_run'])
        self.assertFalse(receipt['originality_or_performance_claim'])
        with self.assertRaises(FileExistsError):
            export_inputs(self.frame, self.plan, self.analysis_folder, destination)
        with self.assertRaises(subprocess.CalledProcessError):
            run_reference(destination, rscript=self.rscript)

    def test_R_independently_rejects_changed_saved_bootstrap_utility(self):
        destination = self.folder/'wrong-utility'
        export_inputs(self.frame, self.plan, self.analysis_folder, destination)
        path = destination/'evaluation_L3_utilities.csv'
        matrix = np.loadtxt(path, delimiter=',')
        matrix[0, 0] += .01
        np.savetxt(path, matrix, delimiter=',', fmt='%.17g')
        # Invoke R directly to test independent arithmetic rather than the
        # wrapper's separate exported-input hash guard.
        script = Path(__file__).resolve().parents[1]/'scripts/extension3_inference_reference.R'
        result = subprocess.run([self.rscript, str(script), str(destination)],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('bootstrap utility mismatch', result.stderr)
        self.assertFalse((destination/'R_verification.json').exists())

    def test_R_independently_rejects_changed_reported_interval(self):
        import pandas as pd
        destination = self.folder/'wrong-interval'
        export_inputs(self.frame, self.plan, self.analysis_folder, destination)
        path = destination/'evaluation_L3_intervals.csv'
        table = pd.read_csv(path)
        table.loc[0, 'adjusted_lower'] += .01
        table.to_csv(path, index=False, float_format='%.17g')
        script = Path(__file__).resolve().parents[1]/'scripts/extension3_inference_reference.R'
        result = subprocess.run([self.rscript, str(script), str(destination)],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('basic interval mismatch', result.stderr)
        self.assertFalse((destination/'R_verification.json').exists())

    def test_wrapper_rejects_changed_artifacts_and_exported_inputs(self):
        destination = self.folder/'changed-export'
        export_inputs(self.frame, self.plan, self.analysis_folder, destination)
        path = destination/'evaluation_L3_returns.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Changed exported'):
            run_reference(destination, rscript=self.rscript)
        bad = self.folder/'bad-analysis'
        shutil.copytree(self.analysis_folder, bad)
        path = bad/'primary_intervals.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Changed or invalid supplied analysis'):
            export_inputs(self.frame, self.plan, bad, self.folder/'must-not-exist')
        self.assertFalse((self.folder/'must-not-exist').exists())


if __name__ == '__main__':
    unittest.main()
