"""Protocol v1 moving-block moments and state-dependent trade disagreement."""
import hashlib
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def replica_seed(protocol, block_months, first_decision, replica):
    """Canonical JSON -> SHA256 -> first 128 bits, big endian -> PCG64."""
    if replica not in range(1, protocol['aim_resampling']['replicas'] + 1):
        raise ValueError('Replica ID outside the approved count')
    payload = dict(protocol_id=protocol['protocol_id'], master_seed=protocol['seeds']['extension'],
                   component=f'aim_moving_blocks_{block_months}', date=str(first_decision), replica=replica)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return int(digest[:32], 16), digest


def moving_blocks(n_months, length, seed):
    if n_months < length or length < 1:
        raise ValueError('Insufficient contiguous estimation months for the fixed block length')
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n_months - length + 1, size=(n_months + length - 1) // length)
    sampled = (starts[:, None] + np.arange(length)).ravel()[:n_months]
    counts = np.bincount(sampled, minlength=n_months)
    return starts, sampled, counts


def fit_coefficients(matrix, score, penalty):
    a = np.asarray(matrix, dtype=float)
    r = np.asarray(score, dtype=float)
    if a.shape != (len(r), len(r)) or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError('Invalid coefficient moments')
    if penalty < 0 or not np.isfinite(penalty) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10):
        raise ValueError('Invalid penalty or nonsymmetric moments')
    system = (a + a.T) / 2 + penalty * np.eye(len(r))
    factor = cho_factor(system, lower=True, check_finite=False)
    coef = cho_solve(factor, r, check_finite=False)
    residual = np.linalg.norm(system @ coef - r) / max(np.linalg.norm(r), 1e-30)
    if not np.isfinite(coef).all() or residual > 1e-10:
        raise ValueError('Failed coefficient solve; all replicas are required')
    return coef, float(residual)


def state_features(central_aim, replica_aims, holdings, m, sigma, impact, wealth, gamma=10., floor=1e-12):
    """Recompute from this state's holdings; do not reuse another state's D.

    Replica aims have stocks in rows and exactly 100 coefficient replicas in
    columns. The ensemble mean is a denominator/dispersion diagnostic only.
    The original central aim is the target used for q0 and the A0 predictors.
    """
    h, aim, replicas = map(lambda x: np.asarray(x, dtype=float), (holdings, central_aim, replica_aims))
    n = len(h)
    if aim.shape != (n,) or replicas.shape != (n, 100) or m.shape != (n, n) or sigma.shape != (n, n):
        raise ValueError('Misaligned state or fewer than 100 replicas')
    if any(not np.isfinite(x).all() for x in [h, aim, replicas, m, sigma, impact]) or wealth <= 0:
        raise ValueError('Invalid state inputs')
    adjustment = np.eye(n) - m
    q0 = adjustment @ (aim - h)
    q = adjustment @ (replicas - h[:, None])
    qbar = q.mean(axis=1)
    centered = q - qbar[:, None]
    numerator = float(np.sum(centered * (sigma @ centered)) / 100)
    denominator = float(qbar @ sigma @ qbar)
    qrisk = float(q0 @ sigma @ q0)
    hrisk = float(h @ sigma @ h)
    irisk = float(wealth * np.sum(np.asarray(impact) * q0**2))
    if min(numerator, denominator, qrisk, hrisk, irisk) < 0:
        raise ValueError('Negative quadratic risk or impact; diagnose without silent repair')
    dispersion = numerator / max(denominator, floor)
    alignment_floor = qrisk <= floor or hrisk <= floor
    alignment = 0. if alignment_floor else float(q0 @ sigma @ h / np.sqrt(qrisk * hrisk))
    result = dict(dispersion=dispersion, dispersion_numerator=numerator,
                  mean_trade_risk=denominator, denominator_floor=denominator <= floor,
                  log1p_aim_dispersion=float(np.log1p(dispersion)),
                  trade_risk=qrisk, trade_impact_curvature=irisk, holdings_risk=hrisk,
                  risk_alignment=alignment, alignment_floor=alignment_floor,
                  log_trade_risk=float(np.log(max(qrisk, floor))),
                  log_trade_impact=float(np.log(max(irisk, floor))),
                  log_holdings_risk=float(np.log(max(hrisk, floor))),
                  alignment_times_log1p_dispersion=float(alignment * np.log1p(dispersion)),
                  initial_curvature=float(gamma * qrisk + irisk),
                  mean_aim_distance_from_central=float(np.linalg.norm(replicas.mean(axis=1) - aim)))
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite confidence feature')
    return result, q0, qbar
