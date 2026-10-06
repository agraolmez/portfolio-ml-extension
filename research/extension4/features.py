"""Use parent-normalized inputs, then interact at each historical slice."""
import numpy as np
from feature_geometry.features import fingerprint
from public_support.conditional_features import augment_parent_features


def parent_columns(model):
    k=model.feature_map.parent.frequencies.shape[1]
    return ('constant',)+tuple(f'cos_{j}' for j in range(k))+tuple(f'sin_{j}' for j in range(k))


def dynamic_design(basis, innovations, parent_model, states, *, state_names, interaction_columns):
    """states rows must be current date, previous month, ..., twelve months ago.

    The runner must bind dates/source identity before calling this low-level
    function. No state definition or historical protocol is selected here.
    """
    parent_model._check_unchanged()
    m=basis.sample.market
    if len(innovations)!=13 or m.source_id=='': raise ValueError('Aligned parent history required')
    states=np.asarray(states,dtype=float)
    if states.shape!=(13,len(state_names)) or not np.isfinite(states).all():
        raise ValueError('Thirteen aligned finite state slices required')
    signals=[parent_model.feature_map.signals(z,v,i,parent_model.gains)
             for z,v,i in zip(m.characteristics,m.volatility,innovations)]
    augmented=[augment_parent_features(b,c,parent_names=parent_columns(parent_model),
                 state_names=state_names,interaction_columns=interaction_columns)
               for b,c in zip(signals,states)]
    # The same audited finite-history operator receives the complete sequence.
    omega,change=basis.operators([a.design for a in augmented])
    return dict(omega=omega,change=change,signal=augmented[0].design,
                columns=augmented[0].columns,
                parent_identity=parent_model._content_hash(),
                state_fingerprint=fingerprint(states,tuple(state_names),tuple(interaction_columns)))


def disabled_aim(parent_model, market, bound_innovation):
    """Explicit exact parent path, without a conditional fit or new selection."""
    return parent_model.aim(market.decision_date,market.ids,market.characteristics[0],
                           market.volatility[0],bound_innovation)
