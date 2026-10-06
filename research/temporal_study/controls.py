"""Capacity-matched current-only control for the frozen Extension-2 parent.

The diagnostic uses one gain per supplied theme on current inputs, without a
historical reference: control_z[:, j] = z[:, j] * gains[group[j]]. The parent's
learned scales and frequency matrix stay fixed. No bounds, penalty, optimizer
budget or historical experiment is selected by this module.

``CurrentOnlyFeatureMap`` implements the map interface used by the existing
profiled objective and gain optimizer. Its history argument is deliberately
ignored. ``current_only_batch_factory`` can supply the common objective with
placeholder slices while requiring only ordinary parent training bases.
"""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np

from feature_geometry.features import FeatureMap, array, fingerprint
from feature_geometry.moments import MomentBasis
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.reference import Innovation
from temporal_extension.model import TemporalModel


CURRENT_ONLY_KIND = "current_only_theme_gain_v1"


@dataclass(frozen=True)
class CurrentOnlyFeatureMap(TemporalFeatureMap):
    """Same number of theme gains and features; current inputs only.

    ``innovation`` is accepted for compatibility with the common exact-profile
    interface, but neither its values nor its observation mask enter this map.
    Thus missing historical observations do not disable current-only rescaling.
    Bounds and anchoring strength must be supplied explicitly to the common
    optimizer/objective, just as for the temporal challenger.
    """

    @property
    def kind(self):
        return CURRENT_ONLY_KIND

    @property
    def fingerprint(self):
        # Distinguish the control even when its neutral outputs coincide.
        return fingerprint(CURRENT_ONLY_KIND, super().fingerprint)

    def inputs(self, z, innovation, gains):
        z, gains = array(z, 2), array(gains, 1)
        if (z.shape[1] != self.parent.dimensions
                or gains.shape != (self.gain_count,)):
            raise ValueError("Input or current-only theme-gain shape mismatch")
        if np.all(gains == 1.):
            return z
        return array(z * gains[self.groups], 2)

    def vjp(self, z, vol, innovation, gains, upstream):
        control_z = self.inputs(z, innovation, gains)
        z = array(z, 2)
        _, vol, scale, cosine, sine, normalized, norm = self.parent._parts(
            control_z, vol, self.theta)
        upstream = array(upstream, 2)
        if upstream.shape != (len(z), self.parent.columns):
            raise ValueError("Signal adjoint shape mismatch")
        adjoint = upstream[:, 1:] / vol[:, None]
        raw_gradient = (adjoint - adjoint.mean(axis=0)
                        - normalized * np.sum(normalized * adjoint, axis=0)) / norm
        k = self.parent.frequencies.shape[1]
        angle_gradient = -sine * raw_gradient[:, :k] + cosine * raw_gradient[:, k:]
        input_gradient = (angle_gradient @ self.parent.frequencies.T) * scale
        by_characteristic = np.sum(input_gradient * z, axis=0)
        return np.bincount(self.groups, weights=by_characteristic,
                           minlength=self.gain_count)


def current_only_batch_factory(basis_factory):
    """Adapt a repeatable parent-basis factory without reading any history.

    The shared streaming objective expects thirteen ``Innovation`` containers.
    These zero-valued placeholders are ignored by ``CurrentOnlyFeatureMap``;
    they carry all-true eligibility because every finite parent input is used.
    Coverage from this control means current-input coverage, not availability
    of past source observations. It must not be reported as temporal coverage.
    A caller must use this factory with ``CurrentOnlyFeatureMap`` only.
    """
    if not callable(basis_factory):
        raise ValueError("A repeatable parent-basis factory is required")

    def batches():
        for basis in basis_factory():
            if not isinstance(basis, MomentBasis):
                raise ValueError("An audited parent moment basis is required")
            market = basis.sample.market
            if len(market.characteristics) != 13:
                raise ValueError("All thirteen parent characteristic slices are required")
            placeholders = tuple(
                Innovation(np.zeros_like(z), np.ones_like(z, dtype=bool),
                           dict(current_unobserved=0, prior_unobserved_or_absent=0,
                                prior_security_month_absent=0,
                                interpretation="current-only input; no history required"))
                for z in market.characteristics)
            yield basis, placeholders

    return batches


def current_only_reference_fingerprint(source_fingerprint):
    """Explicitly record that this diagnostic has no historical reference."""
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("Current-input source provenance is required")
    return fingerprint(CURRENT_ONLY_KIND, "no_historical_reference", source_fingerprint)


@dataclass(frozen=True)
class CurrentOnlyModel(TemporalModel):
    """Persisted control with an explicit transform discriminator.

    The common temporal loader is deliberately incompatible with this format:
    it must never silently interpret current-only gains as innovation gains.
    The annual policy is responsible for supplying accepted current market data
    and enforcing the annual parent's decision interval. No historical binding
    is needed to evaluate this current-only map.
    """

    def __post_init__(self):
        if type(self.feature_map) is not CurrentOnlyFeatureMap:
            raise ValueError("A current-only feature map is required")
        if self.reference_fingerprint != current_only_reference_fingerprint(self.source_fingerprint):
            raise ValueError("Current-only models must record no historical reference")
        super().__post_init__()

    def _metadata(self):
        return dict(super()._metadata(), format_version=2, transform_kind=CURRENT_ONLY_KIND)

    def aim(self, decision_date, ids, z, vol, bound_innovation=None):
        self._check_unchanged()
        day = date.fromisoformat(decision_date).isoformat()
        if day < self.cutoff:
            raise ValueError("Current-only model would use future training information")
        ids = np.asarray(ids)
        z = array(z, 2)
        if ids.ndim != 1 or len(ids) != len(z) or len(np.unique(ids)) != len(ids):
            raise ValueError("Unique ordered security IDs must match current inputs")
        # A caller may supply a temporal binding, but it has no role in this
        # no-history diagnostic. Accepted-market provenance belongs to policy.
        return self.feature_map.signals(z, vol, None, self.gains) @ self.beta

    @classmethod
    def load(cls, path, *, expected_parent_fingerprint):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
            expected = metadata.pop('parameter_fingerprint')
            names = ['beta', 'frequencies', 'gains', 'groups', 'lower', 'theta', 'upper']
            actual = fingerprint(*(data[k] for k in names), json.dumps(metadata, sort_keys=True))
            if (metadata.get('format_version') != 2 or expected != actual
                    or metadata.get('transform_kind') != CURRENT_ONLY_KIND
                    or metadata.get('parent_fingerprint') != expected_parent_fingerprint):
                raise ValueError("Changed current-only model, transform kind or parent artifact")
            fmap = CurrentOnlyFeatureMap(FeatureMap(data['frequencies'], metadata['frequency_source']),
                                         data['theta'], data['groups'])
            return cls(fmap, data['gains'], data['beta'], metadata['cutoff'],
                       metadata['parent_cutoff'], metadata['parent_fingerprint'],
                       metadata['source_fingerprint'], metadata['reference_fingerprint'],
                       data['lower'], data['upper'])


def _parent_map_fingerprint(feature_map):
    return TemporalFeatureMap(feature_map.parent, feature_map.theta, feature_map.groups).fingerprint


def train_current_only_profile(basis_factory, feature_map, parent_settings, *,
                               expected_neutral_beta, expected_parent_feature_fingerprint,
                               parent_fingerprint, source_fingerprint, rho, objective_scale,
                               optimizer_settings):
    """Fit the diagnostic with the common exact objective and neutral gate.

    All stopping, bound and penalty settings are explicit and may be matched to
    the temporal challenger. This function selects no scientific protocol.
    """
    from temporal_extension.training import train_profile
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
        optimizer_settings=optimizer_settings)
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


def fit_current_only_annual(parent, history, *, rho, objective_scale, optimizer_settings):
    """Use the full archived annual training range without constructing innovations.

    This function performs a historical fit when called on actual inputs and
    requires the same explicit study authorization as the temporal annual fit.
    """
    import pandas as pd
    from geometry_study.kernel import LeanBasis
    from temporal_extension.history import VerifiedParent, AcceptedTemporalHistory
    if not isinstance(parent, VerifiedParent) or not isinstance(history, AcceptedTemporalHistory):
        raise ValueError("Verified archived parent and accepted history adapter required")
    parent.assert_cutoff(parent.settings.cutoff)
    before = parent.fingerprint
    first, last = parent.metadata['training_first'], parent.metadata['training_last']
    def bases():
        for day in pd.date_range(first, last, freq='ME'):
            yield LeanBasis(history.training_sample(day, parent.settings.cutoff))
    fmap = CurrentOnlyFeatureMap(parent.feature_map.parent, parent.feature_map.theta,
                                 parent.feature_map.groups)
    model, receipt = train_current_only_profile(
        bases, fmap, parent.settings, expected_neutral_beta=parent.model.beta,
        expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
        parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
        rho=rho, objective_scale=objective_scale, optimizer_settings=optimizer_settings)
    history.files.unchanged()
    if parent.fingerprint != before:
        raise ValueError("Archived parent changed during the annual current-only fit")
    receipt.update(year=parent.year, seed=parent.seed, training_first=first, training_last=last,
                   accepted_inputs_sha256=dict(history.files.consumed),
                   parent_artifacts_sha256=dict(parent.hashes),
                   parent_scale_optimizer_status=parent.metadata['status'])
    return model, receipt


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
    from temporal_extension.streaming import StreamingObjective
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
                                   rho=rho, objective_scale=objective_scale)
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
    receipt = dict(result, status='fixed_configuration_solved', model_kind=(
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


def fit_fixed_annual(parent, history, *, gains, lower, upper, rho, objective_scale, kind):
    """Exactly solve a declared fixed control over the full parent's training range."""
    import pandas as pd
    from geometry_study.kernel import LeanBasis
    from temporal_extension.history import VerifiedParent, AcceptedTemporalHistory
    if not isinstance(parent, VerifiedParent) or not isinstance(history, AcceptedTemporalHistory):
        raise ValueError("Verified archived parent and accepted history adapter required")
    if kind not in ('fixed_temporal', 'fixed_current_only'):
        raise ValueError("An explicit fixed control kind is required")
    parent.assert_cutoff(parent.settings.cutoff)
    before = parent.fingerprint
    first, last = parent.metadata['training_first'], parent.metadata['training_last']
    if kind == 'fixed_current_only':
        fmap = CurrentOnlyFeatureMap(parent.feature_map.parent, parent.feature_map.theta,
                                     parent.feature_map.groups)
        def bases():
            for day in pd.date_range(first, last, freq='ME'):
                yield LeanBasis(history.training_sample(day, parent.settings.cutoff))
        factory = current_only_batch_factory(bases)
        reference = current_only_reference_fingerprint(history.source_fingerprint)
    else:
        fmap = parent.feature_map
        factory = history.batch_factory(parent, first, last)
        reference = history.reference_fingerprint
    model, receipt = fit_fixed_profile(
        factory, fmap, parent.settings, gains=gains, lower=lower, upper=upper,
        expected_neutral_beta=parent.model.beta,
        expected_parent_feature_fingerprint=parent.feature_map.fingerprint,
        parent_fingerprint=before, source_fingerprint=history.source_fingerprint,
        reference_fingerprint=reference, rho=rho, objective_scale=objective_scale)
    history.files.unchanged()
    if parent.fingerprint != before:
        raise ValueError("Archived parent changed during the annual fixed-control fit")
    receipt.update(year=parent.year, seed=parent.seed, training_first=first, training_last=last,
                   accepted_inputs_sha256=dict(history.files.consumed),
                   parent_artifacts_sha256=dict(parent.hashes),
                   parent_scale_optimizer_status=parent.metadata['status'])
    return model, receipt


def save_fixed_result(destination, model, receipt):
    """Save exact fixed-control coefficients to a new output directory only."""
    from feature_geometry.history import sha
    from temporal_extension.training import _json_value
    body = json.loads(json.dumps(receipt, default=_json_value, allow_nan=False))
    expected_type = {'fixed_temporal': TemporalModel, 'fixed_current_only': CurrentOnlyModel}.get(
        body.get('model_kind'))
    if expected_type is None or type(model) is not expected_type:
        raise ValueError("A matching fixed-control model and receipt are required")
    model._check_unchanged()
    if (body.get('status') != 'fixed_configuration_solved'
            or body.get('eligible_for_portfolio_evaluation') is not True
            or body.get('neutral_gate', {}).get('passed') is not True
            or body.get('solve_residual') is None or not np.isfinite(body['solve_residual'])
            or not 0. <= body['solve_residual'] <= 1e-9):
        raise ValueError("A neutral-verified exact conditional solve is required")
    if (body['parent_fingerprint'] != model.parent_fingerprint
            or body['parent_feature_fingerprint'] != _parent_map_fingerprint(model.feature_map)
            or body['control_feature_fingerprint'] != model.feature_map.fingerprint
            or body['source_fingerprint'] != model.source_fingerprint
            or body['reference_fingerprint'] != model.reference_fingerprint
            or body['parent_settings']['cutoff'] != model.cutoff
            or body['parent_settings']['cutoff'] != model.parent_cutoff
            or not np.array_equal(body['lower'], model.lower)
            or not np.array_equal(body['upper'], model.upper)
            or not np.array_equal(body['gains'], model.gains)
            or not np.array_equal(body['fixed_gains'], model.gains)
            or not np.array_equal(body['prescribed_gains'], model.gains)
            or not np.array_equal(body['beta'], model.beta)):
        raise ValueError("Fixed-control parameters or provenance differ from fitting receipt")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.save(destination / 'model.npz')
    body.update(model_file='model.npz', model_sha256=sha(destination / 'model.npz'))
    with (destination / 'fit.json').open('x') as handle:
        json.dump(body, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return body


def save_current_only_result(destination, model, receipt):
    """Save the learned current-only fit without conflating its map with parent D."""
    from feature_geometry.history import sha
    from temporal_extension.training import _json_value
    body = json.loads(json.dumps(receipt, default=_json_value, allow_nan=False))
    if body.get('model_kind') != 'learned_current_only':
        raise ValueError("A learned current-only fitting receipt is required")
    if model is not None:
        if type(model) is not CurrentOnlyModel:
            raise ValueError("A current-only model is required")
        model._check_unchanged()
        shape = model.gains.shape
        lower = np.broadcast_to(np.asarray(body['optimizer_settings']['lower'], dtype=float), shape)
        upper = np.broadcast_to(np.asarray(body['optimizer_settings']['upper'], dtype=float), shape)
        if (body['parent_fingerprint'] != model.parent_fingerprint
                or body['parent_feature_fingerprint'] != _parent_map_fingerprint(model.feature_map)
                or body['control_feature_fingerprint'] != model.feature_map.fingerprint
                or body['source_fingerprint'] != model.source_fingerprint
                or body['reference_fingerprint'] != model.reference_fingerprint
                or body['parent_settings']['cutoff'] != model.cutoff
                or body['parent_settings']['cutoff'] != model.parent_cutoff
                or not np.array_equal(lower, model.lower) or not np.array_equal(upper, model.upper)
                or not np.array_equal(body['gains'], model.gains)
                or not np.array_equal(body['beta'], model.beta)):
            raise ValueError("Current-only parameters or provenance differ from fitting receipt")
    if body.get('eligible_for_portfolio_evaluation'):
        if (model is None or not body.get('converged') or body.get('status') != 'converged'
                or not body.get('neutral_gate', {}).get('passed')
                or body.get('projected_gradient_norm') is None
                or not np.isfinite(body['projected_gradient_norm'])
                or body['projected_gradient_norm'] > body['optimizer_settings']['tolerance']):
            raise ValueError("A converged current-only model is required for evaluation eligibility")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    name = 'model.npz' if body.get('eligible_for_portfolio_evaluation') else 'diagnostic_model.npz'
    if model is not None:
        model.save(destination / name)
        body.update(model_file=name, model_sha256=sha(destination / name))
    with (destination / 'fit.json').open('x') as handle:
        json.dump(body, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return body
