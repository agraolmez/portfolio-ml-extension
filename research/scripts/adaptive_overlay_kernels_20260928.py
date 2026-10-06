"""Pure, fixed agreement overlay; no fitting, outcome inputs or file access.

Seed dispersion is an algorithmic stability diagnostic, not a calibrated
forecast variance. This coordinatewise overlay has no utility guarantee.
"""
from __future__ import annotations

import numpy as np


def agreement_targets(component_baseline, component_c4, *, force_neutral=False):
    """Return stock-specific and amplitude-matched targets from three seeds.

    Inputs have shape (stocks, 3), with matched baseline/C4 seed columns.
    The fixed stock gate is mean_departure**2 divided by that quantity plus
    sample variance across seeds. A zero denominator maps to one. ASIZE has
    the same L1 incremental target exposure as AGREE, before trading.

    ``force_neutral`` is an equivalence-test switch, never a fitted candidate.
    """
    if not isinstance(force_neutral, (bool, np.bool_)):
        raise ValueError('force_neutral must be a boolean equivalence-test switch')
    baseline = np.asarray(component_baseline, dtype=float)
    c4 = np.asarray(component_c4, dtype=float)
    if (baseline.ndim != 2 or baseline.shape != c4.shape or baseline.shape[1] != 3
            or baseline.shape[0] == 0 or not np.isfinite(baseline).all()
            or not np.isfinite(c4).all()):
        raise ValueError('Matching, nonempty, finite stock-by-three target arrays required')
    # Fail closed on numerically unrepresentable input differences or squares;
    # do not insert an arbitrary numerical floor into the scientific formula.
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            departures = c4-baseline
            bens, cens = baseline.mean(axis=1), c4.mean(axis=1)
            dbar = departures.mean(axis=1)
            v = departures.var(axis=1, ddof=1)
            numerator = dbar*dbar
            denominator = numerator+v
            g = np.divide(numerator, denominator, out=np.ones_like(dbar), where=denominator != 0)
            if force_neutral:
                g = np.ones_like(dbar)
            # Preserve exact neutral targets instead of relying on cancellation
            # between separately rounded means.
            agree = np.where(g == 1, cens, bens+g*dbar)
            total_departure = np.abs(dbar).sum()
            h = float(np.abs(g*dbar).sum()/total_departure) if total_departure != 0 else 1.
            asize = cens.copy() if h == 1 else bens+h*dbar
    except FloatingPointError as error:
        raise ValueError('Target arithmetic exceeds finite floating-point range') from error
    arrays = (bens, cens, dbar, v, g, agree, asize)
    if not all(np.isfinite(x).all() for x in arrays) or not np.isfinite(h):
        raise ValueError('Nonfinite overlay arithmetic')
    if np.any(g < 0) or np.any(g > 1) or not 0 <= h <= 1:
        raise ValueError('Agreement multiplier outside its mathematical bounds')
    return dict(AGREE=agree, ASIZE=asize, BENS=bens, CENS=cens,
                g=g, dbar=dbar, v=v, h=h)


def risk_scale(decision_date, return_dates, gross_returns, predicted_variances):
    """Fixed 60-month calibration using only outcomes matured by the decision.

    The reference portfolio is external to the overlay. Future observations
    are excluded before validating or using numerical outcomes. The scale is
    capped at one and never increases target leverage.
    """
    dates = np.asarray(return_dates, dtype='datetime64[D]')
    decision = np.datetime64(decision_date, 'D')
    gross = np.asarray(gross_returns, dtype=float)
    predicted = np.asarray(predicted_variances, dtype=float)
    if (dates.ndim != 1 or gross.shape != dates.shape or predicted.shape != dates.shape
            or np.isnat(dates).any() or np.isnat(decision)
            or np.any(np.diff(dates) <= np.timedelta64(0, 'D'))):
        raise ValueError('Unique ordered dates and aligned reference observations required')
    month_ends = (dates.astype('datetime64[M]')+np.timedelta64(1,'M')).astype('datetime64[D]')-np.timedelta64(1,'D')
    decision_end = (decision.astype('datetime64[M]')+np.timedelta64(1,'M')).astype('datetime64[D]')-np.timedelta64(1,'D')
    if np.any(dates != month_ends) or decision != decision_end:
        raise ValueError('Reference returns and decisions must use calendar month ends')
    eligible = np.flatnonzero(dates <= decision)
    if len(eligible) < 60:
        raise ValueError('Sixty matured reference observations required')
    indices = eligible[-60:]
    if (dates[indices[-1]] != decision
            or np.any(np.diff(dates[indices].astype('datetime64[M]').astype(int)) != 1)):
        raise ValueError('The latest sixty calendar months must be complete and matured')
    g, v = gross[indices], predicted[indices]
    if not np.isfinite(g).all() or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('Finite returns and nonnegative forecast variances required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        realized = float(g.var(ddof=1))
        forecast = float(v.mean())
        k = float(min(1., np.sqrt(forecast/realized))) if realized > 0 else 1.
    if not all(np.isfinite(x) for x in (realized, forecast, k)):
        raise ValueError('Nonfinite reference calibration')
    return dict(k=k, realized_variance=realized, predicted_variance_mean=forecast,
                indices=indices, window_first=str(dates[indices[0]]),
                window_last=str(dates[indices[-1]]), months=60)
