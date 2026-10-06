import unittest
import numpy as np
from annual_ind.conditional import solve_conditional
from extension4.estimation import solve_coefficients


class ConditionalCandidateTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[2., .2, .1], [.2, 3., .7], [.1, .7, 4.]])
        self.score = np.array([1., 2., 3.])

    def test_zero_ridge_matches_independent_dense_solve(self):
        beta, residual = solve_conditional(self.matrix, self.score, parent_width=2, parent_ridge=0., multiplier=100.)
        np.testing.assert_allclose(beta, np.linalg.solve(self.matrix, self.score), atol=1e-14, rtol=1e-13)
        self.assertLess(residual, 1e-12)

    def test_zero_ridge_multipliers_are_identical(self):
        values = [solve_conditional(self.matrix, self.score, parent_width=2, parent_ridge=0., multiplier=m)[0]
                  for m in (100., 10., 1.)]
        for value in values[1:]:
            np.testing.assert_array_equal(value, values[0])

    def test_positive_grid_preserves_accepted_equations(self):
        for ridge in (.001, 1., 100.):
            for multiplier in (100., 10., 1.):
                beta, residual = solve_conditional(self.matrix, self.score, parent_width=2,
                                                    parent_ridge=ridge, multiplier=multiplier)
                original = solve_coefficients(self.matrix, self.score, parent_width=2,
                    parent_ridge=ridge, conditional_ridge=multiplier*ridge, disabled=False)[0]
                np.testing.assert_array_equal(beta, original)
                expected = np.linalg.solve(self.matrix+np.diag([ridge, ridge, multiplier*ridge]), self.score)
                np.testing.assert_allclose(beta, expected, atol=1e-14, rtol=1e-13)

    def test_singular_zero_ridge_fails_without_substitution(self):
        with self.assertRaisesRegex(ValueError, 'no silent repair'):
            solve_conditional(np.diag([1., 1., 0.]), self.score, parent_width=2, parent_ridge=0., multiplier=1.)

    def test_invalid_systems_rejected(self):
        for width, ridge, multiplier in [(1, 1., 1.), (2, -1., 1.), (2, 1., 0.), (True, 1., 1.)]:
            with self.assertRaises(ValueError):
                solve_conditional(self.matrix, self.score, parent_width=width, parent_ridge=ridge, multiplier=multiplier)


if __name__ == '__main__':
    unittest.main()
