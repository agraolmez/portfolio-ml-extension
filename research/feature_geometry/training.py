"""Profiled economic training with a closed coefficient solve and bounded scales."""
from dataclasses import asdict, dataclass
from datetime import date
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .features import array, fingerprint, project_scales

OBJECTIVES = ("full", "risk_only", "return_prediction")


@dataclass(frozen=True)
class FitSettings:
    cutoff: str
    ridge: float = 1.
    scale_penalty: float = .001
    bound: float = float(np.log(2.))
    max_steps: int = 8
    max_backtracks: int = 16
    initial_step: float = .25
    armijo: float = 1e-4
    gradient_tolerance: float = 1e-8

    def __post_init__(self):
        object.__setattr__(self, "cutoff", date.fromisoformat(self.cutoff).isoformat())
        values = [self.ridge, self.scale_penalty, self.bound, self.initial_step,
                  self.armijo, self.gradient_tolerance]
        if (not np.isfinite(values).all() or self.ridge < 0 or self.scale_penalty < 0
                or self.bound <= 0 or self.initial_step <= 0 or not 0 < self.armijo < 1
                or self.gradient_tolerance <= 0 or self.max_steps < 0 or self.max_backtracks < 1
                or not isinstance(self.max_steps, int) or not isinstance(self.max_backtracks, int)):
            raise ValueError("Invalid fixed fitting settings")


def validate_training(bases, feature_map, settings):
    if not bases:
        raise ValueError("Training observations are required")
    decisions = [b.sample.market.decision_date for b in bases]
    if decisions != sorted(set(decisions)):
        raise ValueError("Training dates must be unique and ordered")
    for basis in bases:
        sample = basis.sample
        if sample.available_at > settings.cutoff:
            raise ValueError("Unmatured outcome at fitting cutoff")
        if sample.market.characteristics[0].shape[1] != feature_map.dimensions:
            raise ValueError("Characteristic dictionary mismatch")


def _solve(matrix, score, ridge):
    system = (matrix + matrix.T) / 2 + ridge * np.eye(len(score))
    # Penalize the constant too, matching Portfolio-ML rather than the A0/A1 learner.
    beta = cho_solve(cho_factor(system, lower=True, check_finite=False), score, check_finite=False)
    residual = np.linalg.norm(system @ beta - score) / max(np.linalg.norm(score), 1e-30)
    if not np.isfinite(beta).all() or residual > 1e-9:
        raise ValueError("Coefficient solve failed; no silent jitter or ridge substitution")
    return beta, float(residual)


def profile(bases, feature_map, theta, settings, objective="full", gradient=True):
    """Recompute geometry-dependent moments, profile beta, and apply the envelope gradient.

    Return prediction is date-averaged cross-sectional squared-loss fitting of
    current normalized signals, with equal 1/n stock weights per date. Its
    theta-independent minus-y-squared constant is omitted from the displayed value.
    """
    validate_training(bases, feature_map, settings)
    theta = array(theta, 1)
    if theta.shape != (feature_map.dimensions,) or objective not in OBJECTIVES:
        raise ValueError("Invalid scale shape or training objective")
    # Unconstrained evaluations are useful for derivative checks. fit_geometry projects.
    records = [basis.matrices(feature_map, theta) for basis in bases]
    risk = np.mean([r["risk"] for r in records], axis=0)
    cost = np.mean([r["cost"] for r in records], axis=0)
    score = np.mean([r["score"] for r in records], axis=0)
    if objective == "return_prediction":
        fit_score = np.mean([r["signal"].T @ b.sample.returns / len(b.sample.returns)
                             for b, r in zip(bases, records)], axis=0)
        matrix = np.mean([r["signal"].T @ r["signal"] / len(r["signal"]) for r in records], axis=0)
    else:
        fit_score, matrix = score, risk + (cost if objective == "full" else 0)
    beta, residual = _solve(matrix, fit_score, settings.ridge)
    value = .5 * fit_score @ beta - .5 * settings.scale_penalty * (theta @ theta)
    result = {"value": float(value), "beta": beta, "score": score, "risk": risk,
              "cost": cost, "objective_score": fit_score, "objective_matrix": matrix,
              "solve_residual": residual, "gradient": None}
    if gradient:
        result["gradient"] = _theta_gradient(bases, feature_map, theta, beta, records, settings, objective)
    if not np.isfinite(value) or (gradient and not np.isfinite(result["gradient"]).all()):
        raise ValueError("Nonfinite profile objective or gradient")
    return result


def _theta_gradient(bases, feature_map, theta, beta, records, settings, objective):
    grad = -settings.scale_penalty * theta.copy()
    count = len(bases)
    for basis, record in zip(bases, records):
        sample = basis.sample
        market = sample.market
        if objective == "return_prediction":
            residual = sample.returns - record["signal"] @ beta
            grad += feature_map.vjp(market.characteristics[0], market.volatility[0], theta,
                                    np.outer(residual / (count * len(residual)), beta))
            continue
        v, u = record["omega"] @ beta, record["change"] @ beta
        omega_gradient = (sample.returns - market.gamma * (market.covariance @ v)) / count
        change_gradient = -market.wealth * market.impact * u / count if objective == "full" else np.zeros_like(u)
        adjoints = basis.pullback(omega_gradient, change_gradient)
        for z, vol, adjoint in zip(market.characteristics, market.volatility, adjoints):
            grad += feature_map.vjp(z, vol, theta, np.outer(adjoint, beta))
    return grad


def joint_value_gradient(bases, feature_map, theta, beta, settings, objective="full"):
    """Same joint theta/beta family for an ordinary gradient-training comparator."""
    validate_training(bases, feature_map, settings)
    theta, beta = array(theta, 1), array(beta, 1)
    if objective not in ("full", "risk_only") or theta.shape != (feature_map.dimensions,) or beta.shape != (feature_map.columns,):
        raise ValueError("Joint gradient requires aligned parameters and an economic objective")
    records = [basis.matrices(feature_map, theta) for basis in bases]
    score = np.mean([r["score"] for r in records], axis=0)
    matrix = np.mean([r["risk"] + (r["cost"] if objective == "full" else 0) for r in records], axis=0)
    system = matrix + settings.ridge * np.eye(len(beta))
    value = score @ beta - .5 * beta @ system @ beta - .5 * settings.scale_penalty * theta @ theta
    return {"value": float(value), "theta_gradient": _theta_gradient(bases, feature_map, theta, beta, records, settings, objective),
            "beta_gradient": score - system @ beta}


def fit_geometry(bases, feature_map, settings, objective="full", initial_theta=None, progress=None):
    """Projected ascent with Armijo backtracking; iteration limits are not convergence."""
    validate_training(bases, feature_map, settings)
    if objective not in OBJECTIVES:
        raise ValueError("Unknown geometry-training objective")
    start = time.monotonic()
    theta = project_scales(np.zeros(feature_map.dimensions) if initial_theta is None else initial_theta, settings.bound)
    trace, evaluations = [], 0
    current = profile(bases, feature_map, theta, settings, objective)
    evaluations += 1
    trace.append({"iteration": 0, "value": current["value"], "accepted": True, "max_abs_theta": float(np.max(abs(theta)))})
    if progress:
        progress(dict(trace[-1], objective=objective))
    status = "ITERATION_BUDGET_REACHED"
    for iteration in range(1, settings.max_steps + 1):
        grad = current["gradient"]
        projected_norm = float(np.max(abs(project_scales(theta + grad, settings.bound) - theta)))
        if projected_norm <= settings.gradient_tolerance:
            status = "PROJECTED_GRADIENT_TOLERANCE"
            break
        step = settings.initial_step / max(float(np.max(abs(grad))), 1e-30)
        accepted = False
        for backtrack in range(settings.max_backtracks):
            trial_theta = project_scales(theta + step * grad, settings.bound)
            displacement = trial_theta - theta
            slope = float(grad @ displacement)
            if slope <= 0:
                break
            trial = profile(bases, feature_map, trial_theta, settings, objective, gradient=False)
            evaluations += 1
            if trial["value"] >= current["value"] + settings.armijo * slope:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = "LINE_SEARCH_STALLED"
            break
        theta = trial_theta
        current = profile(bases, feature_map, theta, settings, objective, gradient=True)
        evaluations += 1
        trace.append({"iteration": iteration, "value": current["value"], "accepted": True,
                      "backtracks": backtrack, "step": step, "projected_gradient_norm_before": projected_norm,
                      "max_abs_theta": float(np.max(abs(theta)))})
        if progress:
            progress(dict(trace[-1], objective=objective))
    full_beta, full_residual = _solve(current["risk"] + current["cost"], current["score"], settings.ridge)
    full_value = current["score"] @ full_beta - .5 * full_beta @ (current["risk"] + current["cost"]) @ full_beta
    source_key = fingerprint(*(b.sample.fingerprint for b in bases))
    key = fingerprint("feature_geometry_v1", source_key, feature_map.fingerprint, theta,
                      asdict(settings), objective, settings.cutoff)
    return {"theta": theta, "beta": full_beta, "geometry_beta": current["beta"], "objective": objective,
            "status": status, "trace": trace, "profile_evaluations": evaluations,
            "seconds": time.monotonic() - start, "settings": asdict(settings),
            "training_first_decision": bases[0].sample.market.decision_date,
            "training_last_decision": bases[-1].sample.market.decision_date,
            "training_months": len(bases), "latest_outcome_availability": max(b.sample.available_at for b in bases),
            "source_fingerprint": source_key, "geometry_cache_key": key,
            "frequency_fingerprint": feature_map.fingerprint,
            "final_geometry_profile_value": current["value"], "full_training_economic_value_before_ridge": float(full_value),
            "full_solve_residual": full_residual, "score": current["score"], "risk": current["risk"], "cost": current["cost"],
            "beta_rule": "Full risk-and-impact refit after geometry is frozen, for all geometry objectives"}
