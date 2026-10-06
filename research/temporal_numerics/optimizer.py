"""Bounded L-BFGS-B with independent stationarity and hard evaluation gates."""
from dataclasses import dataclass
from numbers import Integral, Real
from time import monotonic

import numpy as np
import scipy
from scipy.optimize import minimize
from temporal_extension.optimizer import _bounds, _checked_profile


@dataclass(frozen=True)
class LBFGSBSettings:
    lower: object
    upper: object
    tolerance: float
    max_steps: int
    max_evaluations: int
    max_line_searches: int
    history_size: int
    relative_ftol: float
    wall_seconds: float

    def __post_init__(self):
        for key in ('max_steps', 'max_evaluations', 'max_line_searches', 'history_size'):
            value = getattr(self, key)
            minimum = 1 if key in ('max_line_searches', 'history_size') else 0
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(key+' must be an explicit valid integer')
        for key in ('tolerance', 'relative_ftol', 'wall_seconds'):
            value = getattr(self, key)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0
                    or (key != 'relative_ftol' and value == 0)):
                raise ValueError(key+' must be an explicit valid real number')


class _Halt(Exception):
    def __init__(self, status):
        self.status = status


def optimize(evaluate, gain_count, settings, *, progress=None):
    """Maximize the unchanged profile, retaining only accepted iterate states.

    Initial/neutral and rejected line-search calls all count against max_evaluations.
    SciPy's iteration-delayed maxfun check is not used to enforce that limit.
    The initial exact profile is reused once when SciPy requests its starting point.
    Every later evaluator invocation rebuilds the profile. Only a callback-confirmed
    accepted point can replace the saved gains/beta/gradient on interruption.
    """
    if not isinstance(settings, LBFGSBSettings) or not callable(evaluate):
        raise ValueError('Explicit LBFGSBSettings and callable evaluator required')
    if isinstance(gain_count, (bool, np.bool_)) or not isinstance(gain_count, Integral) or gain_count <= 0:
        raise ValueError('A positive gain count is required')
    if progress is not None and not callable(progress):
        raise ValueError('Progress must be callable')
    lower = _bounds(settings.lower, gain_count, 'lower')
    upper = _bounds(settings.upper, gain_count, 'upper')
    if np.any(lower > upper) or np.any(lower > 1) or np.any(upper < 1):
        raise ValueError('Bounds must contain neutral one')
    started = monotonic()
    gains = np.ones(gain_count)
    current = None
    last_trial = None
    calls = steps = 0
    evaluations, trials, objective_trace, projected_gradient_trace = [], [], [], []
    library = {}

    def elapsed():
        return max(0., monotonic()-started)

    def pg(point, gradient):
        with np.errstate(over='ignore'):
            return float(np.max(np.abs(np.clip(point+gradient, lower, upper)-point)))

    def notify(event, **values):
        if progress:
            progress(dict(event=event, elapsed_seconds=elapsed(), **values))

    def guard():
        if elapsed() >= settings.wall_seconds:
            raise _Halt('wall_time_limit')

    def call(point, phase):
        nonlocal calls, last_trial
        guard()
        if calls >= settings.max_evaluations:
            raise _Halt('evaluation_limit')
        point = np.asarray(point, dtype=float).copy()
        if point.shape != gains.shape or not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
            raise _Halt('invalid_solver_proposal')
        calls += 1
        entry = dict(index=calls, phase=phase, gains=point.tolist(), accepted=False)
        evaluations.append(entry)
        notify('evaluation_started', index=calls, phase=phase)
        supplied = point.copy()
        supplied.setflags(write=False)
        try:
            checked = _checked_profile(evaluate(supplied, with_gradient=True), gain_count)
            if checked['solve_residual'] > 1e-9:
                raise ValueError('Coefficient solve residual exceeds 1e-9')
            if current is not None and checked['beta'].shape != current['beta'].shape:
                raise ValueError('Coefficient dimension changed')
        except Exception as exc:
            entry.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise _Halt('wall_time_limit' if isinstance(exc, TimeoutError) else 'evaluation_failed') from exc
        entry.update(status='complete', value=checked['value'], solve_residual=checked['solve_residual'],
                     projected_gradient_norm=pg(point, checked['gradient']))
        if elapsed() >= settings.wall_seconds:
            entry['status'] = 'completed_after_time_limit'
            raise _Halt('wall_time_limit')
        last_trial = (point, checked, entry)
        notify('evaluation_complete', index=calls, value=checked['value'],
               projected_gradient_norm=entry['projected_gradient_norm'], solve_residual=checked['solve_residual'])
        return checked

    def finish(status):
        return dict(status=status, converged=status == 'converged', optimizer_method='L-BFGS-B',
            scipy_version=scipy.__version__, library_result=library, gains=gains.copy(),
            value=None if current is None else current['value'],
            beta=None if current is None else current['beta'].copy(),
            gradient=None if current is None else current['gradient'].copy(),
            solve_residual=None if current is None else current['solve_residual'],
            projected_gradient_norm=None if current is None else pg(gains, current['gradient']),
            accepted_steps=steps, evaluation_count=calls, elapsed_seconds=elapsed(),
            objective_trace=objective_trace, projected_gradient_trace=projected_gradient_trace,
            trials=trials, evaluations=evaluations,
            boundary_flags=dict(at_lower=(gains == lower).tolist(), at_upper=(gains == upper).tolist()),
            wall_time_enforcement='before_and_after_evaluations_plus_caller_month_guards',
            evaluation_limit_enforcement='hard_before_each_expensive_call_including_neutral_and_rejected_trials')

    try:
        current = call(gains, 'initial')
        last_trial[2]['accepted'] = True
        objective_trace.append(current['value'])
        projected_gradient_trace.append(pg(gains, current['gradient']))
        if projected_gradient_trace[-1] <= settings.tolerance:
            return finish('converged')
        if settings.max_steps == 0:
            return finish('step_limit')
        initial_request = True

        def objective(point):
            nonlocal initial_request
            guard()
            if initial_request:
                initial_request = False
                if not np.array_equal(point, gains):
                    raise _Halt('invalid_solver_initial_point')
                value = current
            else:
                value = call(point, 'trial')
            return -value['value'], -value['gradient']

        def accepted(point):
            nonlocal gains, current, steps
            guard()
            if steps >= settings.max_steps:
                raise _Halt('step_limit')
            if last_trial is None or not np.array_equal(point, last_trial[0]):
                raise _Halt('accepted_point_has_no_matching_profile')
            proposed, entry = last_trial[1:]
            rounding = 32*np.finfo(float).eps*max(1., abs(current['value']))
            if proposed['value'] < current['value']-rounding:
                raise _Halt('nonmonotone_accepted_point')
            gains, current = np.asarray(point).copy(), proposed
            entry['accepted'] = True
            steps += 1
            objective_trace.append(current['value'])
            projected_gradient_trace.append(pg(gains, current['gradient']))
            trials.append(dict(update=steps, evaluation_index=entry['index'], accepted=True,
                               value=current['value'], projected_gradient_norm=projected_gradient_trace[-1]))
            notify('accepted_update', update=steps, evaluation_index=entry['index'],
                   value=current['value'], projected_gradient_norm=projected_gradient_trace[-1], gains=gains.tolist())
            if projected_gradient_trace[-1] <= settings.tolerance:
                raise _Halt('converged')
            if steps >= settings.max_steps:
                raise _Halt('step_limit')

        result = minimize(objective, gains.copy(), jac=True, method='L-BFGS-B',
            bounds=list(zip(lower, upper)), callback=accepted,
            options=dict(maxiter=settings.max_steps, maxfun=settings.max_evaluations,
                         maxls=settings.max_line_searches, maxcor=settings.history_size,
                         gtol=settings.tolerance, ftol=settings.relative_ftol))
        library.update(success=bool(result.success), status=int(result.status), message=str(result.message),
                       iterations=int(result.nit), evaluations=int(result.nfev))
        # A library stop cannot promote an unaccepted trial or loosen stationarity.
        if not np.array_equal(result.x, gains):
            return finish('solver_result_not_last_accepted_point')
        if pg(gains, current['gradient']) <= settings.tolerance:
            return finish('converged')
        return finish('library_stopped_before_stationarity')
    except _Halt as exc:
        return finish(exc.status)
    except Exception as exc:
        library.update(error_type=type(exc).__name__, error=str(exc))
        return finish('solver_failed')
