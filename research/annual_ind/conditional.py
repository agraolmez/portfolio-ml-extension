"""Fresh C4 candidate systems, including the authors' explicit zero penalty.

This separate adapter leaves accepted Extension-4 code and results unchanged.
Singular systems fail; no floor, jitter, pseudoinverse or candidate removal.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from extension4.estimation import solve_coefficients as accepted_positive_solve


def solve_conditional(curvature, score, *, parent_width, parent_ridge, multiplier):
    if (type(parent_width) is not int or not np.isfinite(parent_ridge)
            or parent_ridge < 0 or not np.isfinite(multiplier) or multiplier <= 0):
        raise ValueError('Explicit nonnegative base ridge and positive multiplier required')
    score = np.asarray(score, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    if (score.ndim != 1 or len(score) != 2*parent_width-1 or parent_width < 2
            or curvature.shape != (len(score), len(score))
            or not np.isfinite(score).all() or not np.isfinite(curvature).all()
            or not np.allclose(curvature, curvature.T, atol=1e-12, rtol=1e-10)):
        raise ValueError('Finite symmetric aligned C4 system required')
    if parent_ridge > 0:
        return accepted_positive_solve(curvature, score, parent_width=parent_width,
            parent_ridge=parent_ridge, conditional_ridge=multiplier*parent_ridge, disabled=False)
    system = (curvature+curvature.T)/2
    try:
        beta = cho_solve(cho_factor(system, lower=True, check_finite=False), score, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError('Singular or nonpositive zero-ridge C4 system; no silent repair') from error
    residual = float(np.linalg.norm(system@beta-score)/max(np.linalg.norm(score), 1e-30))
    if not np.isfinite(beta).all() or not np.isfinite(residual) or residual > 1e-9:
        raise ValueError('Zero-ridge C4 residual check failed')
    return beta, residual
