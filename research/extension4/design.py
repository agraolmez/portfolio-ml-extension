"""Bound production design and controls on the frozen annual E3 representation."""
import numpy as np
import pandas as pd
from feature_geometry.features import FeatureMap
from temporal_extension.features import TemporalFeatureMap
from .features import parent_columns


def capacity_map(parent_model, all_frequencies):
    frequencies = np.asarray(all_frequencies)
    base = parent_model.feature_map
    if frequencies.shape != (115, 512): raise ValueError('Exact archived nested frequency dictionary required')
    np.testing.assert_array_equal(frequencies[:, :256], base.parent.frequencies)
    return TemporalFeatureMap(FeatureMap(frequencies[:, 256:], 'archived_nested_extra_pairs'),
                              base.theta, base.groups)


def validate_history(market, bound, parent_model):
    parent_model._check_unchanged()
    if len(bound) != 13: raise ValueError('Thirteen source-bound temporal slices required')
    return tuple(item.validate(pd.Timestamp(market.decision_date)-pd.offsets.MonthEnd(lag), market.ids, z,
                               parent_model.source_fingerprint, parent_model.reference_fingerprint)
                 for lag, (item, z) in enumerate(zip(bound, market.characteristics)))


def signals(model, z, vol, innovation, state, extras):
    base = model.feature_map.signals(z, vol, innovation, model.gains)
    # One extra scalar column represents state times the complete parent aim.
    # It includes the parent's constant only for the pure-exposure control.
    c = float(state)
    extra = extras.signals(z, vol, innovation, model.gains)[:, 1:]
    return np.column_stack((base, c*base[:, 1:], c*(base@model.beta), extra))


def designs(basis, bound_history, parent_model, bound_states, state_identity, extras):
    market = basis.sample.market
    innovation = validate_history(market, bound_history, parent_model)
    c = bound_states.validate(market.decision_date, history=13, expected_identity=state_identity)
    if c.shape != (13, 1): raise ValueError('One approved state required')
    sequence = [signals(parent_model, z, vol, i, state[0], extras)
                for z, vol, i, state in zip(market.characteristics, market.volatility, innovation, c)]
    omega, change = basis.operators(sequence)
    p = parent_model.feature_map.parent.columns
    q = p-1
    if omega.shape[1] != p+2*q+1: raise ValueError('Augmented column dimensions changed')
    return dict(base=(omega[:, :p], change[:, :p]),
                C4=(omega[:, :p+q], change[:, :p+q]),
                CAP4=(np.column_stack((omega[:, :p], omega[:, p+q+1:])),
                      np.column_stack((change[:, :p], change[:, p+q+1:]))),
                SIZE4=(omega[:, p+q], change[:, p+q]))


def columns(model, method):
    names = parent_columns(model)
    if method == 'N4': return names
    if method == 'C4': return names + tuple('value_dispersion::'+n for n in names[1:])
    if method == 'CAP4': return names + tuple('extra::'+n for n in names[1:])
    if method == 'SIZE4': return ('parent_aim_state_gain',)
    raise ValueError('Unknown model family')
