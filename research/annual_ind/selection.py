"""Calendar-causal selection using the released Portfolio-ML validation score.

These functions do not fit models, read performance files, or authorize a run.
Candidate tables must contain genuinely prior-year-fitted validation scores.
The full runner must additionally verify the model/source hash lineage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def author_objective(score, denominator, coefficients):
    """r_tilde' beta - beta' denominator beta / 2, excluding fitting penalties."""
    score = np.asarray(score, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    if (score.ndim != 1 or denominator.shape != (len(score), len(score))
            or beta.ndim not in (1, 2) or beta.shape[0] != len(score)
            or not all(np.isfinite(x).all() for x in (score, denominator, beta))):
        raise ValueError('Finite, aligned score, denominator and coefficients required')
    if not np.allclose(denominator, denominator.T, atol=1e-12, rtol=1e-10):
        raise ValueError('Economic denominator must be symmetric')
    return score @ beta - .5 * np.sum(beta * (denominator @ beta), axis=0)


def _month_end(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None or result != result.normalize() or not result.is_month_end:
        raise ValueError('A timezone-free calendar month-end is required')
    return result


def select_prior_history(scores, *, year, candidate_order, first_return,
                         tie_tolerance=1e-12):
    """Choose for year Y using complete candidate scores through December Y-1.

    Additional later rows are deliberately ignored. Malformed prior rows,
    incomplete candidates, and models estimated with their own validation
    outcomes are rejected. Ties follow the supplied, externally frozen order.
    """
    required = {'candidate', 'decision_eom', 'return_eom', 'fit_year',
                'model_available_at', 'objective'}
    if not required.issubset(scores.columns):
        raise ValueError('Missing validation chronology columns')
    candidates = tuple(candidate_order)
    if (type(year) is not int or not candidates or len(candidates) != len(set(candidates))
            or any(not isinstance(c, str) or not c for c in candidates)
            or not np.isfinite(tie_tolerance) or tie_tolerance < 0):
        raise ValueError('Explicit year, unique candidates and nonnegative tie tolerance required')
    first = _month_end(first_return)
    cutoff = pd.Timestamp(year - 1, 12, 31)
    expected = pd.date_range(first, cutoff, freq='ME')
    if not len(expected):
        raise ValueError('No prior validation months available')
    frame = scores.copy()
    frame['return_eom'] = pd.to_datetime(frame.return_eom, errors='raise')
    if frame.return_eom.isna().any():
        raise ValueError('Undated validation row')
    frame = frame[frame.return_eom.between(first, cutoff)].copy()
    if set(frame.candidate) != set(candidates):
        raise ValueError('Candidate family differs from the frozen family')
    if frame.duplicated(['candidate', 'return_eom']).any():
        raise ValueError('Duplicate candidate-month score')
    for column in ('decision_eom', 'return_eom', 'model_available_at'):
        values = pd.to_datetime(frame[column], errors='raise')
        if (values.dt.tz is not None or values.isna().any()
                or not values.dt.is_month_end.all() or not values.eq(values.dt.normalize()).all()):
            raise ValueError('A timezone-free calendar month-end is required')
        frame[column] = values
    if not (frame.decision_eom + pd.offsets.MonthEnd(1)).eq(frame.return_eom).all():
        raise ValueError('Decision and return months do not align')
    if not frame.fit_year.eq(frame.return_eom.dt.year).all():
        raise ValueError('Validation score has the wrong annual model')
    fitted_cutoff = pd.to_datetime((frame.fit_year.astype(int) - 1).astype(str) + '-12-31')
    if (frame.model_available_at > fitted_cutoff).any():
        raise ValueError('Validation model uses later estimation or selection information')
    if not np.isfinite(frame.objective.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite candidate score; failed candidates cannot be dropped')
    # A unique month-end in [first, cutoff] can only be one of the expected
    # calendar months. Equal counts therefore establish full calendars without
    # rescanning the whole table once for every member of a large candidate grid.
    grouped = frame.groupby('candidate', sort=False).objective.agg(['size', 'mean'])
    if not grouped['size'].eq(len(expected)).all():
        raise ValueError('Incomplete or noncalendar validation history')
    means = {candidate: float(grouped.loc[candidate, 'mean']) for candidate in candidates}
    maximum = max(means.values())
    chosen = next(c for c in candidates if maximum - means[c] <= tie_tolerance)
    return dict(year=year, candidate=chosen, selection_available_at=str(cutoff.date()),
                first_validation_return=str(first.date()), last_validation_return=str(cutoff.date()),
                validation_months=len(expected), candidate_scores=means,
                criterion='mean_author_economic_objective_excluding_fitting_penalties',
                tie_order=list(candidates), tie_tolerance=tie_tolerance)
