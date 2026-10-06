"""Reuse only identity-geometry moments across ridge candidates.

The matrix work at theta=0 does not depend on the ridge or scale penalty.
Nonzero theta always recomputes the economic moments. Each fit still starts
from zero and takes the inherited projected-ascent/backtracking steps.
"""
from dataclasses import dataclass
import time
import numpy as np
from feature_geometry.features import fingerprint, project_scales
from feature_geometry.training import _solve, validate_training
from geometry_study import kernel


def context(bases, feature_map, cutoff):
    validate_training(bases, feature_map, type('Cutoff', (), {'cutoff': cutoff})())
    return fingerprint('annual_ind_neutral_geometry_v1', cutoff,
                       feature_map.fingerprint,
                       *(b.sample.fingerprint for b in bases))


@dataclass(frozen=True)
class NeutralMoments:
    identity: str
    score: np.ndarray
    risk: np.ndarray
    cost: np.ndarray

    def __post_init__(self):
        for key in ('score', 'risk', 'cost'):
            value = np.array(getattr(self, key), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError('Finite neutral moments required')
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        p = len(self.score)
        if (not self.identity or self.score.shape != (p,)
                or self.risk.shape != (p, p) or self.cost.shape != (p, p)):
            raise ValueError('Aligned neutral moments and explicit identity required')

    @classmethod
    def build(cls, bases, feature_map, cutoff):
        identity = context(bases, feature_map, cutoff)
        values = kernel.sums(bases, feature_map, np.zeros(feature_map.dimensions), 'full')
        return cls(identity, *values)

    def evaluate(self, bases, feature_map, settings):
        if self.identity != context(bases, feature_map, settings.cutoff):
            raise ValueError('Neutral moments belong to another training context')
        beta, residual = _solve(self.risk + self.cost, self.score, settings.ridge)
        theta = np.zeros(feature_map.dimensions)
        # Preserve the accepted operation order, including the zero penalty.
        value = (self.score @ beta - .5 * beta @
                 (self.risk + self.cost + settings.ridge * np.eye(len(beta))) @ beta
                 - .5 * settings.scale_penalty * theta @ theta)
        return dict(value=float(value), beta=beta, score=self.score, risk=self.risk,
                    cost=self.cost, solve_residual=residual)


def fit_full(bases, feature_map, settings, *, neutral, objective_scale, progress=None):
    """Accepted full-objective matrix_fit arithmetic, with neutral reuse only."""
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError('Positive fixed objective normalizer required')
    validate_training(bases, feature_map, settings)
    start = time.monotonic()
    theta = np.zeros(feature_map.dimensions)
    current = neutral.evaluate(bases, feature_map, settings)
    evaluations = 1
    trace = [dict(iteration=0, value=current['value'])]
    status = 'ITERATION_BUDGET_REACHED'
    for iteration in range(1, settings.max_steps + 1):
        raw_grad = kernel.gradient(bases, feature_map, theta, current['beta'], settings, 'full')
        grad = raw_grad / objective_scale
        pg = float(np.max(abs(project_scales(theta + grad, settings.bound) - theta)))
        if pg <= settings.gradient_tolerance:
            status = 'PROJECTED_GRADIENT_TOLERANCE'
            break
        step = settings.initial_step / max(float(np.max(abs(grad))), 1e-30)
        accepted = False
        for backtrack in range(settings.max_backtracks):
            trial_theta = project_scales(theta + step * grad, settings.bound)
            displacement = trial_theta - theta
            slope = float(raw_grad @ displacement)
            if slope <= 0:
                break
            trial = kernel.evaluate(bases, feature_map, trial_theta, settings, 'full')
            evaluations += 1
            if trial['value'] >= current['value'] + settings.armijo * slope:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = 'LINE_SEARCH_STALLED'
            break
        theta, current = trial_theta, trial
        trace.append(dict(iteration=iteration, value=current['value'], backtracks=backtrack,
                          step=step, projected_gradient_before=pg))
        if progress:
            progress(trace[-1])
    return dict(theta=theta, beta=current['beta'], geometry_beta=current['beta'].copy(),
                status=status, trace=trace, profile_evaluations=evaluations,
                seconds=time.monotonic()-start, solve_residual=current['solve_residual'],
                final_training_value=current['value'])
