"""Selection cannot obtain an advantage from later returns or missing candidates."""
import unittest
import numpy as np
import pandas as pd
from annual_ind.selection import author_objective, select_prior_history


class AnnualSelectionTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for year in (1971, 1972, 1973):
            for date in pd.date_range(f'{year}-01-31', f'{year}-12-31', freq='ME'):
                for candidate in ('strong', 'weak'):
                    rows.append(dict(candidate=candidate, decision_eom=date-pd.offsets.MonthEnd(1),
                        return_eom=date, fit_year=year, model_available_at=f'{year-1}-12-31',
                        objective=(1. if candidate == 'strong' else 2.) if year == 1971 else
                                  (5. if candidate == 'strong' else 0.)))
        return pd.DataFrame(rows)

    def select(self, frame, year=1972):
        return select_prior_history(frame, year=year, candidate_order=('strong', 'weak'),
                                    first_return='1971-01-31')

    def test_released_formula_for_vector_and_coefficient_bank(self):
        r = np.array([.2, .3]); d = np.array([[2., .1], [.1, 3.]])
        b = np.array([[.4, .5], [-.1, .7]])
        expected = np.array([r@v - .5*v@d@v for v in b.T])
        np.testing.assert_allclose(author_objective(r, d, b), expected)
        self.assertAlmostEqual(author_objective(r, d, b[:, 0]), expected[0])

    def test_selection_uses_only_prior_returns(self):
        frame = self.fixture(); a = self.select(frame)
        frame.loc[frame.return_eom.dt.year.ge(1972), 'objective'] = 1e12
        self.assertEqual(a, self.select(frame))
        self.assertEqual(a['candidate'], 'weak'); self.assertEqual(a['validation_months'], 12)

    def test_expanding_history_changes_next_year_choice(self):
        a = self.select(self.fixture(), 1973)
        self.assertEqual(a['candidate'], 'strong'); self.assertEqual(a['validation_months'], 24)

    def test_missing_month_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.select(self.fixture().drop(index=0))

    def test_missing_candidate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'family'):
            self.select(self.fixture().query("candidate == 'strong'"))

    def test_duplicate_rejected(self):
        frame = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.select(pd.concat([frame, frame.iloc[:1]]))

    def test_later_selected_parent_rejected(self):
        frame = self.fixture(); frame.loc[0, 'model_available_at'] = '2010-12-31'
        with self.assertRaisesRegex(ValueError, 'later'):
            self.select(frame)

    def test_same_year_fitting_rejected(self):
        frame = self.fixture(); frame.loc[0, 'fit_year'] = 1972
        with self.assertRaisesRegex(ValueError, 'annual model'):
            self.select(frame)

    def test_ties_follow_frozen_order(self):
        frame = self.fixture(); frame['objective'] = 1.
        self.assertEqual(self.select(frame)['candidate'], 'strong')

    def test_no_prior_history_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No prior'):
            self.select(self.fixture(), 1971)

    def test_nonfinite_prior_score_rejected(self):
        frame = self.fixture(); frame.loc[0, 'objective'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.select(frame)

    def test_midmonth_return_rejected(self):
        frame = self.fixture(); frame.loc[0, 'return_eom'] = pd.Timestamp('1971-01-15')
        # This becomes a missing required first month rather than an admitted row.
        with self.assertRaises(ValueError):
            self.select(frame)

    def test_misaligned_decision_rejected(self):
        frame = self.fixture(); frame.loc[0, 'decision_eom'] = pd.Timestamp('1970-11-30')
        with self.assertRaisesRegex(ValueError, 'align'):
            self.select(frame)

    def test_future_only_candidate_does_not_change_choice(self):
        frame = self.fixture(); extra = frame[frame.return_eom.dt.year.eq(1973)].copy()
        extra['candidate'] = 'future_idea'; extra['objective'] = 1e15
        self.assertEqual(self.select(frame), self.select(pd.concat([frame, extra])))


if __name__ == '__main__':
    unittest.main()
