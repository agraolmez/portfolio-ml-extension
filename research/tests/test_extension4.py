"""Native bridge checks; synthetic examples do not approve an economic study."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

from test_temporal_extension import temporal_fixture
from temporal_extension.model import TemporalModel
from temporal_extension.profile import evaluate as parent_profile
from feature_geometry.training import _solve as parent_solve
from extension4.features import dynamic_design, parent_columns
from extension4.estimation import month_moments, solve_coefficients
from extension4.parent import SelectedParent


def fixture():
    rng, _, dense, bases, settings, _, fmap, innovations = temporal_fixture()
    gains = np.linspace(.6, 1.4, 13)  # E4 must preserve non-neutral E3 gains.
    result = parent_profile(bases, innovations, fmap, gains, settings,
                            rho=.1, objective_scale=1., with_gradient=False)
    model = TemporalModel(fmap, gains, result['beta'], settings.cutoff,
                          settings.cutoff, 'synthetic-parent', 'synthetic-source',
                          'synthetic-reference', np.zeros(13), np.full(13, 2.))
    return rng, dense, bases, settings, innovations, model, result


class NativeExtension4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_disabled_moments_and_actual_parent_solver_at_full_width(self):
        _, _, bases, settings, innovations, model, expected = fixture()
        before = model._content_hash()
        results = [month_moments(b, i, model, np.empty((13, 0)),
                                state_names=(), interaction_columns=())
                   for b, i in zip(bases, innovations)]
        means = {key: np.mean([r[key] for r in results], axis=0)
                 for key in ('score', 'risk', 'cost')}
        for key, value in means.items():
            np.testing.assert_allclose(value, expected[key], atol=1e-11, rtol=1e-10)
        beta, residual = solve_coefficients(means['risk'] + means['cost'], means['score'],
            parent_width=513, parent_ridge=settings.ridge, conditional_ridge=None, disabled=True)
        np.testing.assert_allclose(beta, expected['beta'], atol=1e-11, rtol=1e-10)
        self.assertLess(residual, 1e-9)
        self.assertEqual(before, model._content_hash())

    def test_historical_states_enter_before_parent_filter_and_keep_amplitude(self):
        _, dense, bases, _, innovations, model, _ = fixture()
        states = np.linspace(-1., 1., 13)[:, None]
        basis, history = bases[0], innovations[0]
        arguments = dict(state_names=('spread',), interaction_columns=tuple(range(1, 513)))
        actual = dynamic_design(basis, history, model, states, **arguments)
        double = dynamic_design(basis, history, model, 2 * states, **arguments)
        m = basis.sample.market
        signals = [model.feature_map.signals(z, v, i, model.gains)
                   for z, v, i in zip(m.characteristics, m.volatility, history)]
        # Independent dense operator evaluates the explicitly expanded sequence.
        expected = dense[0].operators([np.column_stack((b, c[0] * b[:, 1:]))
                                       for b, c in zip(signals, states)])
        for key, reference in zip(('omega', 'change'), expected):
            np.testing.assert_allclose(actual[key], reference, atol=1e-11, rtol=1e-10)
            np.testing.assert_array_equal(actual[key][:, :513], double[key][:, :513])
            np.testing.assert_allclose(double[key][:, 513:], 2 * actual[key][:, 513:],
                                       atol=1e-11, rtol=1e-10)
        wrongly_timed = states[0, 0] * basis.operators(signals)[0][:, 1:]
        self.assertGreater(np.linalg.norm(actual['omega'][:, 513:] - wrongly_timed), .01)
        self.assertEqual(len(actual['columns']), 1025)
        self.assertEqual(actual['columns'][513], 'state[spread]::cos_0')
        self.assertNotIn('state[spread]::constant', actual['columns'])

    def test_zero_states_positive_penalty_recover_parent_joint_solve(self):
        _, _, bases, settings, innovations, model, expected = fixture()
        results = [month_moments(b, i, model, np.zeros((13, 1)),
                    state_names=('zero',), interaction_columns=tuple(range(1, 513)))
                   for b, i in zip(bases, innovations)]
        curvature = np.mean([r['risk'] + r['cost'] for r in results], axis=0)
        score = np.mean([r['score'] for r in results], axis=0)
        beta, residual = solve_coefficients(curvature, score, parent_width=513,
            parent_ridge=settings.ridge, conditional_ridge=7., disabled=False)
        np.testing.assert_allclose(beta[:513], expected['beta'], atol=1e-11, rtol=1e-10)
        np.testing.assert_array_equal(beta[513:], np.zeros(512))
        self.assertLess(residual, 1e-9)

    def test_block_ridge_units_and_constant_penalty(self):
        curvature = np.array([[2., .2, .1], [.2, 3., .7], [.1, .7, 4.]])
        score = np.array([1., 2., 3.])
        beta, _ = solve_coefficients(curvature, score, parent_width=2,
            parent_ridge=.5, conditional_ridge=2., disabled=False)
        np.testing.assert_allclose(beta, np.linalg.solve(curvature + np.diag([.5, .5, 2.]), score),
                                   atol=1e-14, rtol=1e-13)
        # Zeroing conditional beta after fitting is not a valid neutrality test.
        restricted, _ = parent_solve(curvature[:2, :2], score[:2], .5)
        self.assertGreater(np.linalg.norm(beta[:2] - restricted), .001)

    def test_invalid_systems_do_not_silently_gain_jitter_or_new_columns(self):
        for curvature, score, width, penalty, disabled in [
            (-np.eye(3), np.ones(3), 2, .5, False),
            (np.eye(3), np.ones(3), 2, 0., False),
            (np.eye(3), np.ones(3), 2, 1., True),
            (np.array([[1., 1.], [0., 1.]]), np.ones(2), 1, 1., False),
        ]:
            with self.subTest(disabled=disabled, penalty=penalty):
                with self.assertRaises(ValueError):
                    solve_coefficients(curvature, score, parent_width=width,
                        parent_ridge=.1, conditional_ridge=penalty, disabled=disabled)

    def test_parent_selection_chronology_is_distinct_from_fit_cutoff(self):
        p = SelectedParent(1981, 1, None, None, {}, 'synthetic', '2010-12-31')
        with self.assertRaisesRegex(ValueError, 'not known'):
            p.assert_legal_selection('2000-12-31')
        p.assert_legal_selection('2010-12-31')

    def test_column_labels_and_frozen_parent_tampering(self):
        _, _, bases, _, innovations, model, _ = fixture()
        self.assertEqual(parent_columns(model)[0], 'constant')
        # The model makes read-only copies; a deliberate mutation is still detected.
        model.gains.setflags(write=True)
        model.gains[0] += .01
        with self.assertRaisesRegex(ValueError, 'changed'):
            dynamic_design(bases[0], innovations[0], model, np.empty((13, 0)),
                           state_names=(), interaction_columns=())


if __name__ == '__main__':
    unittest.main()
