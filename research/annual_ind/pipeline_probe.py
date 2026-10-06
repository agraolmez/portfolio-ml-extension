"""Bounded fresh E2 -> E3 -> C4 integration, without historical selection.

Coordinates are timing fixtures. The caller supplies newly fitted E2 parameters,
an outcome cutoff and resource guards. No archived fitted parent is loaded here.
"""
import json
import time
import numpy as np
import pandas as pd
from feature_geometry.features import FeatureMap, fingerprint
from feature_geometry.training import _solve
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.model import TemporalModel
from temporal_extension.history import RANKS, RAW
from temporal_numerics.optimizer import LBFGSBSettings
from temporal_fast_v4.training import train_profile
from temporal_fast.backend import single_thread_blas
from extension4.state import build_states
from extension4.design import validate_history
from extension4.estimation import month_moments
from extension4.storage import write_json, commit_stage
from annual_ind.conditional import solve_conditional


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def run_layers(root, out, history, bases, geometry_model, settings, normalizer, runtime):
    started = time.monotonic()
    dictionary = pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'))
    if (dictionary.characteristic.tolist() != list(history.features)
            or dictionary.cluster.nunique() != 13 or dictionary.cluster.isna().any()):
        raise ValueError('Exact accepted characteristic/theme dictionary required')
    groups = dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
    fmap = TemporalFeatureMap(FeatureMap(geometry_model.feature_map.frequencies, geometry_model.feature_map.source),
                              geometry_model.theta, groups)
    batches = []
    for basis in bases:
        runtime.guard()
        bound = history.innovations(basis.sample.market)
        innovation = history.validate_market(basis.sample.market, bound)
        batches.append((basis, innovation, bound))
    history.panels.release()
    def factory():
        for basis, innovation, _ in batches:
            runtime.guard()
            yield basis, innovation
    optimizer = json.loads((root/'config/extension3_study_v3_proposed.json').read_text())['optimizer']
    parent_identity = fingerprint(geometry_model.theta, geometry_model.beta, fmap.fingerprint, settings.cutoff)
    start = time.monotonic()
    with single_thread_blas():
        temporal, receipt = train_profile(factory, fmap, settings,
            expected_neutral_beta=geometry_model.beta,
            expected_parent_feature_fingerprint=fmap.fingerprint, parent_fingerprint=parent_identity,
            source_fingerprint=history.source_fingerprint, reference_fingerprint=history.reference_fingerprint,
            rho=.1, objective_scale=normalizer, optimizer_settings=LBFGSBSettings(**optimizer),
            progress=lambda row: runtime.event('temporal_optimizer', details=plain(row)))
    temporal_seconds = time.monotonic()-start
    receipt = plain(receipt)
    write_json(out/'temporal_attempt.json', receipt)
    if not receipt['eligible_for_portfolio_evaluation'] or temporal is None:
        raise ValueError('Fresh temporal candidate failed the inherited convergence gate; no candidate filtering')
    def save_temporal(folder):
        temporal.save(folder/'model.npz')
        restored = TemporalModel.load(folder/'model.npz', expected_parent_fingerprint=parent_identity)
        np.testing.assert_array_equal(restored.gains, temporal.gains)
        np.testing.assert_array_equal(restored.beta, temporal.beta)
        write_json(folder/'fit.json', receipt)
    commit_stage(out/'fresh_temporal', dict(parent=parent_identity, rho=.1), save_temporal)
    runtime.event('temporal_fit_complete', seconds=temporal_seconds, status=receipt['status'])

    # Recompute the causal state from accepted raw inputs; saved E4 states are not inputs.
    config = json.loads((root/'config/extension4_study_v1_proposed.json').read_text())
    cutoff = pd.Timestamp(settings.cutoff)
    filters = [('eom', '<=', cutoff)]
    raw = pd.read_parquet(history.files.path(RAW), columns=['id','eom','be_me','me'], filters=filters)
    eligible = pd.read_parquet(history.files.path(RANKS), columns=['id','eom','valid'], filters=filters)
    definition = config['state_definitions'][0]
    metadata = dict(unranked=True, timing_mode='SOURCE_LAG_PROXY_UNCERTIFIED',
        source_sha256=history.files.consumed[RAW], eligibility_sha256=history.files.consumed[RANKS],
        source_lag_months=definition['source_lag_months'], source_max_age_months=definition['source_max_age_months'],
        actual_accounting_age_observed=False, revision_vintages_available=False)
    book = build_states(raw, eligible, definition=definition, normalizer=config['state_standardization'],
        source_metadata=metadata, first=str(raw.eom.min().date()), last=str(cutoff.date()))
    del raw, eligible
    book.frame.to_parquet(out/'fresh_state_history.parquet', index=False)
    write_json(out/'fresh_state_metadata.json', book.metadata)
    p = len(temporal.beta)
    def moments(basis, bound):
        m = basis.sample.market
        innovation = validate_history(m, bound, temporal)
        states = book.bind(m.decision_date, history=13).validate(m.decision_date, history=13, expected_identity=book.identity)
        return month_moments(basis, innovation, temporal, states,
            state_names=book.names, interaction_columns=tuple(range(1,p)))
    score = np.zeros(2*p-1)
    curvature = np.zeros((2*p-1, 2*p-1))
    start = time.monotonic()
    with single_thread_blas():
        for basis, _, bound in batches:
            runtime.guard()
            if basis.sample.available_at > settings.cutoff:
                raise ValueError('Conditional training outcome after cutoff')
            result = moments(basis, bound)
            score += result['score']; curvature += result['risk']+result['cost']
        score /= len(batches); curvature /= len(batches)
        beta, neutral_residual = _solve(curvature[:p,:p], score[:p], settings.ridge)
        np.testing.assert_allclose(beta, temporal.beta, atol=1e-11, rtol=1e-10)
        fitted = {}
        for multiplier in (100., 10., 1.):
            value, residual = solve_conditional(curvature, score, parent_width=p,
                                                 parent_ridge=settings.ridge, multiplier=multiplier)
            fitted[str(multiplier)] = dict(beta=value, residual=residual)
    conditional_seconds = time.monotonic()-start
    def save_conditional(folder):
        with (folder/'moments_and_coefficients.npz').open('xb') as f:
            np.savez_compressed(f, score=score, curvature=curvature,
                                **{k:v['beta'] for k,v in fitted.items()})
        write_json(folder/'fit.json', dict(parent=temporal._content_hash(), base_ridge=settings.ridge,
            multipliers=list(fitted), neutral_maximum_beta_error=float(np.max(abs(beta-temporal.beta))),
            neutral_residual=neutral_residual, residuals={k:v['residual'] for k,v in fitted.items()},
            training_months=len(batches), cutoff=settings.cutoff, seconds=conditional_seconds))
    commit_stage(out/'fresh_conditional', dict(parent=temporal._content_hash(), state=book.identity), save_conditional)
    history.files.unchanged()
    return dict(status='FRESH_THREE_LAYER_FIT_COMPLETE_NOT_HISTORICAL_STUDY',
        temporal_seconds=temporal_seconds, conditional_seconds=conditional_seconds,
        complete_layer_seconds=time.monotonic()-started, temporal_status=receipt['status'],
        temporal_converged=receipt['converged'], conditional_candidates=3,
        training_months=len(batches), width=p-1, ridge=settings.ridge,
        newly_fitted_scales=True, newly_fitted_gains=True, neutral_gates_passed=True,
        state_recomputed_from_raw=True, historical_selection_run=False, new_portfolio_paths=0)
