"""Source reconciliation, completion boundaries and immutable report checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from test_temporal_analysis import analysis_fixture
from temporal_study.analysis import analyze_paths
from temporal_study.report import GATES, write_report, verify_report_inputs


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, plan, kwargs = analysis_fixture()
        cls.analysis = analyze_paths(frame, plan, **kwargs)

    def test_synthetic_fixture_marked_source_backed_and_never_study_complete(self):
        source = dict(self.analysis, synthetic_fixture=True,
                       completion_gates={gate: True for gate in GATES})
        diagnostics = pd.DataFrame([dict(year=2000, seed=1, kind='temporal', status='converged',
                                        coverage={'valid': 100, 'bypassed': 20})])
        gains = pd.DataFrame([dict(year=2000, seed=1, kind='temporal', theme='profitability', gain=.8)])
        with tempfile.TemporaryDirectory(prefix='temporal-report-check-') as temporary:
            folder = Path(temporary)/'synthetic'
            built = write_report(source, folder, fit_diagnostics=diagnostics, theme_gains=gains)
            text = built['html'].read_text()
            self.assertIn('SYNTHETIC QA FIXTURE', text)
            self.assertIn('not historical research results', text)
            self.assertIn('100% historical coverage', text)
            self.assertNotIn('<script', text)
            self.assertFalse(built['manifest']['study_completion_certified'])
            self.assertFalse(built['manifest']['numerical_historical_study_complete'])
            saved = json.loads(built['source'].read_text())
            self.assertEqual(saved['analysis_result'], source['result'])
            self.assertEqual(saved['theme_gains'][0]['method'], 'temporal')
            self.assertEqual(verify_report_inputs(folder, source, fit_diagnostics=diagnostics,
                                                   theme_gains=gains), built['manifest'])
            resumed = dict(source)
            table_names = ('strategy_summary', 'mean_seed_metrics', 'seed_contrasts',
                           'primary_intervals', 'selection_links', 'utility_decomposition')
            for name in table_names:
                csv = Path(temporary)/(name+'.csv')
                source[name].to_csv(csv, index=False, float_format='%.17g')
                resumed[name] = pd.read_csv(csv, float_precision='round_trip')
            self.assertEqual(verify_report_inputs(folder, resumed, fit_diagnostics=diagnostics,
                                                   theme_gains=gains), built['manifest'])
            with self.assertRaises(FileExistsError):
                write_report(source, folder, fit_diagnostics=diagnostics, theme_gains=gains)

    def test_analysis_alone_and_numerics_do_not_claim_full_study_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='temporal-report-gates-') as temporary:
            pending = write_report(self.analysis, Path(temporary)/'pending')
            self.assertIn('full-study completion has not been certified', pending['html'].read_text())
            self.assertFalse(pending['manifest']['study_completion_certified'])
            numerical = dict(self.analysis, completion_gates={gate: True for gate in GATES})
            completed = write_report(numerical, Path(temporary)/'numerical')
            self.assertIn('visual review and final study acceptance are pending', completed['html'].read_text())
            self.assertTrue(completed['manifest']['numerical_historical_study_complete'])
            self.assertFalse(completed['manifest']['study_completion_certified'])

    def test_changed_source_bytes_analysis_or_incomplete_status_rejected(self):
        with tempfile.TemporaryDirectory(prefix='temporal-report-integrity-') as temporary:
            folder = Path(temporary)/'report'
            built = write_report(self.analysis, folder)
            changed = deepcopy(self.analysis)
            changed['result']['protocol_id'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'analysis or completion'):
                verify_report_inputs(folder, changed)
            built['html'].write_text(built['html'].read_text()+' changed')
            with self.assertRaisesRegex(ValueError, 'Changed or escaped'):
                verify_report_inputs(folder, self.analysis)
            incomplete = dict(self.analysis, result=dict(self.analysis['result'], status='RUNNING'))
            with self.assertRaisesRegex(ValueError, 'completed analysis'):
                write_report(incomplete, Path(temporary)/'incomplete')
            inconsistent = deepcopy(self.analysis)
            inconsistent['mean_seed_metrics'].loc[0, 'annual_net_utility'] += 1.
            # Edit the actual primary parent row to exercise displayed-number reconciliation.
            mask = inconsistent['mean_seed_metrics'].period.eq('evaluation') & inconsistent['mean_seed_metrics'].method.eq('parent')
            inconsistent['mean_seed_metrics'].loc[mask, 'annual_net_utility'] += 1.
            with self.assertRaisesRegex(ValueError, 'does not reconcile'):
                write_report(inconsistent, Path(temporary)/'inconsistent')


if __name__ == '__main__':
    unittest.main()
