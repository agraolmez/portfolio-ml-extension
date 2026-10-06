"""Deterministic box-constrained ascent for an explicitly configured profile.

This is optimization machinery, not a historical fitting policy. Every numerical
setting is mandatory. The callback must rebuild its affected moments and solve
the conditional coefficients on *every* call. Wall time is checked between calls;
an in-flight evaluation cannot be interrupted by this module.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from time import monotonic

import numpy as np


@dataclass(frozen=True)
class OptimizerSettings:
    lower: object
    upper: object
    tolerance: float
    max_steps: int
    max_evaluations: int
    max_backtracks: int
    initial_step: float
    armijo: float
    wall_seconds: float

    def __post_init__(self):
        for name in ("max_steps", "max_evaluations", "max_backtracks"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be an explicit nonnegative integer")
        for name in ("tolerance", "initial_step", "armijo", "wall_seconds"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be an explicit finite positive number")
        if self.armijo >= 1:
            raise ValueError("armijo must be strictly between zero and one")


def _bounds(value, count, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        raise ValueError(f"{name} must contain finite real bounds")
    result = np.full(count, raw.item(), dtype=float) if raw.ndim == 0 else raw.astype(float, copy=True)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be scalar or have one finite value per gain")
    return result


def _checked_profile(result, count):
    if not isinstance(result, dict):
        raise ValueError("Profile callback must return a dictionary")
    value = result.get("value")
    residual = result.get("solve_residual")
    for name, scalar in (("value", value), ("solve_residual", residual)):
        if (isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Real)
                or not np.isfinite(scalar)):
            raise ValueError(f"Profile {name} must be finite")
    if residual < 0:
        raise ValueError("Profile solve_residual must be nonnegative")
    arrays = {}
    for name in ("gradient", "beta"):
        raw = np.asarray(result.get(name))
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not raw.size:
            raise ValueError(f"Profile {name} must be a finite nonempty real vector")
        arrays[name] = raw.astype(float, copy=True)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Profile {name} must be finite")
    if arrays["gradient"].shape != (count,):
        raise ValueError("Profile gradient must have one entry per gain")
    return dict(value=float(value), solve_residual=float(residual), **arrays)


def optimize(evaluate, gain_count, settings):
    """Maximize an exact profiled criterion from all-one temporal gains.

    ``evaluate(gains, with_gradient=True)`` supplies value, gradient, beta and
    solve_residual. Each proposed point receives one fresh value/gradient solve;
    rejected evaluations count against the same budget as accepted evaluations.
    max_steps counts accepted updates; max_backtracks permits that many halvings
    after the first trial. A unit-step projected-gradient infinity norm certifies
    constrained stationarity. Failures and exhausted caps never mean convergence.

    The returned dictionary retains the last valid accepted point if evaluation
    fails, but sets converged=False. No valid point exists when its value is None.
    The caller must gate reporting/deployment on status and convergence explicitly.
    """
    if not isinstance(settings, OptimizerSettings) or not callable(evaluate):
        raise ValueError("Explicit OptimizerSettings and callable profile required")
    if (isinstance(gain_count, (bool, np.bool_)) or not isinstance(gain_count, Integral)
            or gain_count <= 0):
        raise ValueError("gain_count must be a positive integer")
    lower = _bounds(settings.lower, gain_count, "lower")
    upper = _bounds(settings.upper, gain_count, "upper")
    if np.any(lower > upper) or np.any(lower > 1.) or np.any(upper < 1.):
        raise ValueError("Every gain's ordered bounds must contain neutral one")
    gains = np.ones(gain_count)
    current = None
    steps = 0
    calls = 0
    trials, evaluations, objective_trace, projected_gradient_trace = [], [], [], []
    start = monotonic()

    def elapsed():
        return max(0., monotonic() - start)

    def pg_norm(point, gradient):
        with np.errstate(over="ignore"):
            projected = np.clip(point + gradient, lower, upper)
        return float(np.max(np.abs(projected - point)))

    def boundary(point):
        return dict(at_lower=(point == lower).tolist(), at_upper=(point == upper).tolist())

    def finish(status):
        return dict(status=status, converged=status == "converged", gains=gains.copy(),
                    value=None if current is None else current["value"],
                    beta=None if current is None else current["beta"].copy(),
                    gradient=None if current is None else current["gradient"].copy(),
                    solve_residual=None if current is None else current["solve_residual"],
                    projected_gradient_norm=None if current is None else pg_norm(gains, current["gradient"]),
                    accepted_steps=steps, evaluation_count=calls, elapsed_seconds=elapsed(),
                    objective_trace=objective_trace, projected_gradient_trace=projected_gradient_trace,
                    trials=trials, evaluations=evaluations, boundary_flags=boundary(gains),
                    wall_time_enforcement="checked_between_evaluations_only")

    def call(point, phase):
        nonlocal calls
        if elapsed() >= settings.wall_seconds:
            return None, "wall_time_limit"
        if calls >= settings.max_evaluations:
            return None, "evaluation_limit"
        calls += 1
        receipt = dict(index=calls, phase=phase, gains=point.tolist())
        evaluations.append(receipt)
        supplied = point.copy()
        supplied.setflags(write=False)
        try:
            checked = _checked_profile(evaluate(supplied, with_gradient=True), gain_count)
            if current is not None and checked["beta"].shape != current["beta"].shape:
                raise ValueError("Profile coefficient dimension changed between evaluations")
        except Exception as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            return None, "evaluation_failed"
        receipt.update(status="complete", value=checked["value"],
                       solve_residual=checked["solve_residual"],
                       projected_gradient_norm=pg_norm(point, checked["gradient"]))
        if elapsed() >= settings.wall_seconds:
            receipt["status"] = "completed_after_time_limit"
            return None, "wall_time_limit"
        return checked, None

    current, error = call(gains, "initial")
    if error:
        return finish(error)
    objective_trace.append(current["value"])
    projected_gradient_trace.append(pg_norm(gains, current["gradient"]))

    while True:
        # Test final accepted point before the step/evaluation cap: reaching an
        # optimum on the last available update is certified, not misclassified.
        if projected_gradient_trace[-1] <= settings.tolerance:
            return finish("converged")
        if elapsed() >= settings.wall_seconds:
            return finish("wall_time_limit")
        if steps >= settings.max_steps:
            return finish("step_limit")
        if calls >= settings.max_evaluations:
            return finish("evaluation_limit")
        accepted = False
        step_size = settings.initial_step
        for backtrack in range(settings.max_backtracks + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = np.clip(gains + step_size * current["gradient"], lower, upper)
                direction = candidate - gains
                directional_gain = float(current["gradient"] @ direction)
                required = current["value"] + settings.armijo * directional_gain
            trial = dict(update=steps + 1, backtrack=backtrack, step_size=float(step_size),
                         gains=candidate.tolist(), accepted=False, boundary_flags=boundary(candidate))
            trials.append(trial)
            if (not np.isfinite(candidate).all() or not np.isfinite(directional_gain)
                    or not np.isfinite(required)):
                trial["reason"] = "nonfinite_proposal"
                return finish("numerical_failure")
            if directional_gain <= 0 or np.array_equal(candidate, gains):
                trial["reason"] = "no_representable_ascent_step"
                return finish("line_search_stalled")
            trial["armijo_required"] = float(required)
            previous_calls = calls
            proposal, error = call(candidate, "trial")
            trial["evaluation_index"] = calls if calls > previous_calls else None
            if error:
                trial["reason"] = error
                return finish(error)
            trial.update(value=proposal["value"], solve_residual=proposal["solve_residual"],
                         projected_gradient_norm=pg_norm(candidate, proposal["gradient"]))
            if proposal["value"] >= required:
                trial.update(accepted=True, reason="armijo_satisfied")
                gains, current = candidate, proposal
                steps += 1
                objective_trace.append(current["value"])
                projected_gradient_trace.append(trial["projected_gradient_norm"])
                accepted = True
                break
            trial["reason"] = "armijo_rejected"
            step_size *= .5
        if not accepted:
            return finish("line_search_stalled")
