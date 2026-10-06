"""Explicit training integration and new-destination artifacts; no automatic runs."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from feature_geometry.history import sha
from temporal_numerics.optimizer import optimize
from .streaming import OrderedObjective as StreamingObjective
from .settings import EXECUTION, environment_identity
from temporal_extension.model import TemporalModel


def train_profile(batch_factory, feature_map, parent_settings, *, expected_neutral_beta,
                  expected_parent_feature_fingerprint, parent_fingerprint,
                  source_fingerprint, reference_fingerprint, rho, objective_scale,
                  optimizer_settings, progress=None):
    """Fit gains with an obligatory archived-neutral coefficient gate.

    Every scientific/numerical setting is supplied by the caller. This function
    cannot select a history window, penalty, validation plan or compute budget.
    Nonconverged models are returned as diagnostics, explicitly marked ineligible
    for portfolio evaluation by this engineering interface.
    """
    if feature_map.fingerprint != expected_parent_feature_fingerprint:
        raise ValueError("Temporal map differs from the verified frozen parent")
    expected_neutral_beta = np.asarray(expected_neutral_beta, dtype=float).copy()
    if (expected_neutral_beta.shape != (feature_map.parent.columns,)
            or not np.isfinite(expected_neutral_beta).all()):
        raise ValueError("Aligned archived parent coefficients are required")
    objective = StreamingObjective(batch_factory, feature_map, parent_settings,
                                   rho=rho, objective_scale=objective_scale,
                                   workers=EXECUTION["workers"],trig_backend=EXECUTION["trig_backend"])
    neutral = {}
    last_identity = {}
    def checked(gains, with_gradient=True):
        result = objective(gains, with_gradient=with_gradient)
        if not neutral:
            if not np.array_equal(gains, np.ones(feature_map.gain_count)):
                raise ValueError("The first solve must be the neutral parent")
            np.testing.assert_allclose(result['beta'], expected_neutral_beta,
                                       atol=1e-11, rtol=1e-10,
                                       err_msg='Neutral coefficients differ from archived parent')
            neutral.update(passed=True, maximum_beta_error=float(np.max(abs(result['beta']-expected_neutral_beta))),
                           absolute_tolerance=1e-11, relative_tolerance=1e-10)
        last_identity.update(training_months=result['training_months'],
                             training_source_fingerprint=result['training_source_fingerprint'],
                             coverage=result['coverage'], coverage_unit=result['coverage_unit'])
        return result
    result = optimize(checked, feature_map.gain_count, optimizer_settings, progress=progress)
    if feature_map.fingerprint != expected_parent_feature_fingerprint:
        raise ValueError("Frozen parent changed during optimization")
    model = None
    if result['beta'] is not None and neutral.get('passed'):
        lower = np.broadcast_to(np.asarray(optimizer_settings.lower, dtype=float), result['gains'].shape)
        upper = np.broadcast_to(np.asarray(optimizer_settings.upper, dtype=float), result['gains'].shape)
        model = TemporalModel(feature_map, result['gains'], result['beta'], parent_settings.cutoff,
                              parent_settings.cutoff, parent_fingerprint, source_fingerprint,
                              reference_fingerprint, lower, upper)
    receipt = dict(result, neutral_gate=neutral, **last_identity,
                   execution_settings=dict(EXECUTION), execution_environment=environment_identity(),
                   parent_fingerprint=parent_fingerprint,
                   parent_feature_fingerprint=expected_parent_feature_fingerprint,
                   source_fingerprint=source_fingerprint, reference_fingerprint=reference_fingerprint,
                   parent_settings=asdict(parent_settings), rho=float(rho),
                   objective_scale=float(objective_scale), optimizer_settings=asdict(optimizer_settings),
                   eligible_for_portfolio_evaluation=bool(result['converged'] and model is not None),
                   historical_run_authorized_by_function=False)
    return model, receipt


from temporal_study.controls import (CurrentOnlyFeatureMap, CurrentOnlyModel, _parent_map_fingerprint,
    current_only_reference_fingerprint, current_only_batch_factory)

def train_current_only_profile(basis_factory, feature_map, parent_settings, *,
                               expected_neutral_beta, expected_parent_feature_fingerprint,
                               parent_fingerprint, source_fingerprint, rho, objective_scale,
                               optimizer_settings, progress=None):
    """Fit the diagnostic with the common exact objective and neutral gate.

    All stopping, bound and penalty settings are explicit and may be matched to
    the temporal challenger. This function selects no scientific protocol.
    """
    from .training import train_profile
    if (type(feature_map) is not CurrentOnlyFeatureMap
            or _parent_map_fingerprint(feature_map) != expected_parent_feature_fingerprint):
        raise ValueError("Current-only map differs from the verified frozen parent")
    reference = current_only_reference_fingerprint(source_fingerprint)
    raw_model, receipt = train_profile(
        current_only_batch_factory(basis_factory), feature_map, parent_settings,
        expected_neutral_beta=expected_neutral_beta,
        expected_parent_feature_fingerprint=feature_map.fingerprint,
        parent_fingerprint=parent_fingerprint, source_fingerprint=source_fingerprint,
        reference_fingerprint=reference, rho=rho, objective_scale=objective_scale,
        optimizer_settings=optimizer_settings, progress=progress)
    model = None
    if raw_model is not None:
        model = CurrentOnlyModel(feature_map, raw_model.gains, raw_model.beta, raw_model.cutoff,
                                 raw_model.parent_cutoff, parent_fingerprint, source_fingerprint,
                                 reference, raw_model.lower, raw_model.upper)
    receipt.update(model_kind='learned_current_only',
                   parent_feature_fingerprint=expected_parent_feature_fingerprint,
                   control_feature_fingerprint=feature_map.fingerprint,
                   coverage_interpretation='current input cells; no historical reference or coverage requirement')
    return model, receipt


from feature_geometry.features import array
from temporal_extension.features import TemporalFeatureMap

def fit_fixed_profile(batch_factory, feature_map, parent_settings, *, gains, lower, upper,
                      expected_neutral_beta, expected_parent_feature_fingerprint,
                      parent_fingerprint, source_fingerprint, reference_fingerprint,
                      rho, objective_scale):
    """Exactly solve beta for a prespecified temporal or current-only control.

    The all-one parent solve is checked before the non-neutral solve. This is
    a fixed configuration, so it makes no gain-optimizer convergence claim.
    ``batch_factory`` must provide the common (basis, thirteen slices) interface;
    use ``current_only_batch_factory`` for the no-history control.
    """
    from dataclasses import asdict
    from .streaming import OrderedObjective as StreamingObjective
    if type(feature_map) not in (TemporalFeatureMap, CurrentOnlyFeatureMap):
        raise ValueError("An explicit temporal or current-only transform is required")
    if _parent_map_fingerprint(feature_map) != expected_parent_feature_fingerprint:
        raise ValueError("Control map differs from the verified frozen parent")
    gains = array(gains, 1)
    lower = np.broadcast_to(array(lower), gains.shape).copy()
    upper = np.broadcast_to(array(upper), gains.shape).copy()
    if (gains.shape != (feature_map.gain_count,) or np.any(lower > upper)
            or np.any(lower > 1.) or np.any(upper < 1.)
            or np.any(gains < lower) or np.any(gains > upper)):
        raise ValueError("Explicit fixed gains must respect bounds containing neutral one")
    current_only = type(feature_map) is CurrentOnlyFeatureMap
    if current_only and reference_fingerprint != current_only_reference_fingerprint(source_fingerprint):
        raise ValueError("Current-only controls must record no historical reference")
    objective = StreamingObjective(batch_factory, feature_map, parent_settings,
                                   rho=rho, objective_scale=objective_scale,
                                   workers=EXECUTION["workers"],trig_backend=EXECUTION["trig_backend"])
    neutral = objective(np.ones(feature_map.gain_count), with_gradient=False)
    expected_neutral_beta = array(expected_neutral_beta, 1)
    if expected_neutral_beta.shape != (feature_map.parent.columns,):
        raise ValueError("Aligned archived parent coefficients are required")
    np.testing.assert_allclose(neutral['beta'], expected_neutral_beta, atol=1e-11, rtol=1e-10,
                               err_msg='Neutral coefficients differ from archived parent')
    result = neutral if np.all(gains == 1.) else objective(gains, with_gradient=False)
    model_type = CurrentOnlyModel if current_only else TemporalModel
    model = model_type(feature_map, gains, result['beta'], parent_settings.cutoff,
                       parent_settings.cutoff, parent_fingerprint, source_fingerprint,
                       reference_fingerprint, lower, upper)
    receipt = dict(result, execution_settings=dict(EXECUTION), execution_environment=environment_identity(), status='fixed_configuration_solved', model_kind=(
        'fixed_current_only' if current_only else 'fixed_temporal'),
        gains=gains, fixed_gains=gains, prescribed_gains=gains, lower=lower, upper=upper,
        neutral_gate=dict(passed=True, maximum_beta_error=float(np.max(np.abs(
            neutral['beta'] - expected_neutral_beta))), absolute_tolerance=1e-11, relative_tolerance=1e-10),
        parent_fingerprint=parent_fingerprint,
        parent_feature_fingerprint=expected_parent_feature_fingerprint,
        control_feature_fingerprint=feature_map.fingerprint,
        source_fingerprint=source_fingerprint, reference_fingerprint=reference_fingerprint,
        parent_settings=asdict(parent_settings), rho=float(rho), objective_scale=float(objective_scale),
        eligible_for_portfolio_evaluation=True, historical_run_authorized_by_function=False,
        coverage_interpretation=('current input cells; no historical reference or coverage requirement'
                                 if current_only else 'calendar history availability for temporal adjustment'))
    return model, receipt
