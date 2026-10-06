"""Explicit training integration and new-destination artifacts; no automatic runs."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from feature_geometry.history import sha
from .optimizer import optimize
from .streaming import StreamingObjective
from .model import TemporalModel


def train_profile(batch_factory, feature_map, parent_settings, *, expected_neutral_beta,
                  expected_parent_feature_fingerprint, parent_fingerprint,
                  source_fingerprint, reference_fingerprint, rho, objective_scale,
                  optimizer_settings):
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
                                   rho=rho, objective_scale=objective_scale)
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
    result = optimize(checked, feature_map.gain_count, optimizer_settings)
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
                   parent_fingerprint=parent_fingerprint,
                   parent_feature_fingerprint=expected_parent_feature_fingerprint,
                   source_fingerprint=source_fingerprint, reference_fingerprint=reference_fingerprint,
                   parent_settings=asdict(parent_settings), rho=float(rho),
                   objective_scale=float(objective_scale), optimizer_settings=asdict(optimizer_settings),
                   eligible_for_portfolio_evaluation=bool(result['converged'] and model is not None),
                   historical_run_authorized_by_function=False)
    return model, receipt


def fit_annual(parent, history, *, rho, objective_scale, optimizer_settings):
    """Connect the complete archived annual training range; never shorten it.

    Calling this function on actual historical data is a historical fit and
    requires the separately approved protocol/pilot. Merely importing it runs
    nothing. The root task's software checks use synthetic train_profile only.
    """
    from .history import VerifiedParent, AcceptedTemporalHistory
    if not isinstance(parent, VerifiedParent) or not isinstance(history, AcceptedTemporalHistory):
        raise ValueError("Verified archived parent and accepted history adapter required")
    parent.assert_cutoff(parent.settings.cutoff)
    before = parent.fingerprint
    factory = history.batch_factory(parent, parent.metadata['training_first'], parent.metadata['training_last'])
    model, receipt = train_profile(factory, parent.feature_map, parent.settings,
                                  expected_neutral_beta=parent.model.beta,
                                  expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
                                  parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
                                  reference_fingerprint=history.reference_fingerprint,
                                  rho=rho, objective_scale=objective_scale, optimizer_settings=optimizer_settings)
    history.files.unchanged()
    if parent.fingerprint != before:
        raise ValueError("Archived parent changed during the annual fit")
    receipt.update(year=parent.year, seed=parent.seed,
                   training_first=parent.metadata['training_first'], training_last=parent.metadata['training_last'],
                   accepted_inputs_sha256=dict(history.files.consumed), parent_artifacts_sha256=dict(parent.hashes),
                   parent_scale_optimizer_status=parent.metadata['status'])
    return model, receipt


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported receipt value {type(value).__name__}')


def save_result(destination, model, receipt):
    """Write a complete diagnostic/fit receipt only to an unused directory."""
    destination = Path(destination)
    # Validate serializability before reserving the destination.
    body = json.loads(json.dumps(receipt, default=_json_value, allow_nan=False))
    if model is not None:
        model._check_unchanged()
        bounds_shape = model.gains.shape
        lower = np.broadcast_to(np.asarray(body['optimizer_settings']['lower'], dtype=float), bounds_shape)
        upper = np.broadcast_to(np.asarray(body['optimizer_settings']['upper'], dtype=float), bounds_shape)
        if (body['parent_fingerprint'] != model.parent_fingerprint
                or body['parent_feature_fingerprint'] != model.feature_map.fingerprint
                or body['source_fingerprint'] != model.source_fingerprint
                or body['reference_fingerprint'] != model.reference_fingerprint
                or body['parent_settings']['cutoff'] != model.cutoff
                or body['parent_settings']['cutoff'] != model.parent_cutoff
                or not np.array_equal(lower, model.lower) or not np.array_equal(upper, model.upper)
                or not np.array_equal(body['gains'], model.gains)
                or not np.array_equal(body['beta'], model.beta)):
            raise ValueError("Model parameters or provenance differ from its fitting receipt")
    if body.get('eligible_for_portfolio_evaluation'):
        if (model is None or not body.get('converged') or body.get('status') != 'converged'
                or body.get('projected_gradient_norm') is None
                or not np.isfinite(body['projected_gradient_norm'])
                or body['projected_gradient_norm'] > body['optimizer_settings']['tolerance']):
            raise ValueError("A converged model is required for evaluation eligibility")
    destination.mkdir(parents=True, exist_ok=False)
    name = 'model.npz' if body.get('eligible_for_portfolio_evaluation') else 'diagnostic_model.npz'
    if model is not None:
        model.save(destination / name)
        body['model_file'] = name
        body['model_sha256'] = sha(destination / name)
    with (destination / 'fit.json').open('x') as handle:
        json.dump(body, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return body
