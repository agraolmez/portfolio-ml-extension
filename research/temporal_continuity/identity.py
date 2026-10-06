"""Lossless security-ID identity and finite, BLAS-independent risk diagnostics."""
import numpy as np
from feature_geometry.features import fingerprint


def canonical_ids(ids):
    """Keep order; reject ambiguous, fractional, missing or duplicate IDs."""
    values = np.asarray(ids)
    if values.ndim != 1 or values.dtype.kind not in 'iuf':
        raise ValueError('One-dimensional numeric security IDs required')
    if values.dtype.kind == 'f':
        if (not np.isfinite(values).all() or np.any(abs(values) > 2**53)
                or np.any(values != np.trunc(values))):
            raise ValueError('Security IDs must be lossless, finite integers')
    elif values.dtype.kind == 'u' and np.any(values >= 2**63):
        raise ValueError('Security IDs exceed the signed integer range')
    result = values.astype(np.int64)
    if not np.array_equal(result, values) or len(np.unique(result)) != len(result):
        raise ValueError('Security IDs must be unique and losslessly represented')
    return result


def holdings_fingerprint(ids, holdings):
    """Canonicalize IDs only; retain exact float64 holdings bytes and order."""
    ids = canonical_ids(ids)
    values = np.asarray(holdings)
    if values.dtype != np.dtype('float64') or values.shape != ids.shape or not np.isfinite(values).all():
        raise ValueError('Aligned finite float64 holdings required')
    return fingerprint(ids, values)


def annual_volatility(weights, covariance):
    """Same 12*w' Sigma w diagnostic using an explicit checked contraction.

    This diagnostic never participates in fitting, trading or selection.
    Avoid the prior vector/matrix BLAS expression's unexplained transient
    overflow flag; genuine nonfinite arithmetic still raises.
    """
    weights, covariance = np.asarray(weights), np.asarray(covariance)
    if (weights.ndim != 1 or covariance.shape != (len(weights), len(weights))
            or not np.isfinite(weights).all() or not np.isfinite(covariance).all()):
        raise ValueError('Aligned finite weights and covariance required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        variance = 12 * np.einsum('i,ij,j->', weights, covariance, weights, optimize=False)
        if not np.isfinite(variance) or variance < 0:
            raise ValueError('Nonfinite or negative modeled portfolio variance')
        return float(np.sqrt(variance))
