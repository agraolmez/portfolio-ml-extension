"""Independent formula, metadata and rejection checks for the public E4 helper."""
import unittest
import numpy as np
from public_support.conditional_features import augment_parent_features

class PublicInteractionTests(unittest.TestCase):
    def test_formula_and_nonmutation_across_dimensions(self):
        rng = np.random.default_rng(20261006)
        for n in (1, 9, 37):
            for p in (1, 4, 513):
                for k in (0, 1, 3):
                    x = rng.normal(size=(n, p)); saved = x.copy()
                    c = rng.normal(size=k); j = tuple(range(p-1, -1, -1))
                    pn = tuple(f'p{i}' for i in range(p)); sn = tuple(f's{i}' for i in range(k))
                    result = augment_parent_features(x, c, parent_names=pn,
                        state_names=sn, interaction_columns=j)
                    expected = np.concatenate([x] + [c[z]*x[:,j] for z in range(k)], axis=1)
                    np.testing.assert_array_equal(result.design, expected)
                    np.testing.assert_array_equal(x, saved)
                    self.assertFalse(np.shares_memory(x, result.design))
                    self.assertEqual(result.parent_width, p); self.assertEqual(result.state_count, k)
                    self.assertEqual(result.interaction_columns, j)
                    self.assertEqual(result.columns, pn + tuple(f'state[{s}]::{pn[i]}' for s in sn for i in j))
    def test_rejects_invalid_inputs_and_overflow(self):
        args = dict(parent_features=np.ones((2,2)), state=np.ones(1),
                    parent_names=('a','b'), state_names=('s',), interaction_columns=(1,))
        invalid = [dict(parent_features=np.ones(2)), dict(parent_features=np.empty((0,2))),
                   dict(parent_features=np.full((2,2),np.nan)), dict(state=np.array([np.inf])),
                   dict(state=np.ones((1,1))), dict(parent_names=('a','a')),
                   dict(state_names=()), dict(interaction_columns=()),
                   dict(interaction_columns=(True,)), dict(interaction_columns=(1.,)),
                   dict(interaction_columns=(1,1)), dict(interaction_columns=(-1,)),
                   dict(interaction_columns=(2,)),
                   dict(parent_names=('a','state[s]::a'), interaction_columns=(0,)),
                   dict(parent_features=np.full((2,2),1e308),state=np.array([1e308]))]
        for change in invalid:
            with self.subTest(change=list(change)), np.errstate(over='ignore'):
                with self.assertRaises(ValueError): augment_parent_features(**(args|change))
    def test_empty_state_identity_and_zero_interaction(self):
        x = np.array([[1.,2.],[3.,4.]])
        result = augment_parent_features(x, [],parent_names=('a','b'),state_names=(),interaction_columns=())
        np.testing.assert_array_equal(result.design,x)
        result = augment_parent_features(x,[0.],parent_names=('a','b'),state_names=('s',),interaction_columns=(1,))
        np.testing.assert_array_equal(result.design[:,:2],x)
        np.testing.assert_array_equal(result.design[:,2],np.zeros(2))
