"""Full conditional economic solve and envelope derivative; no fitting policy.

All numerical settings are explicit. This module does not choose a penalty,
lookback, bounds, stopping rule, validation split or historical run budget.
"""
import numpy as np
from feature_geometry.training import _solve, validate_training


def evaluate(bases, innovations, feature_map, gains, settings, *, rho, objective_scale,
             with_gradient=True):
    validate_training(bases, feature_map.parent, settings)
    if (len(innovations) != len(bases) or any(len(x) != 13 for x in innovations)
            or not np.isfinite([rho, objective_scale]).all() or rho < 0 or objective_scale <= 0):
        raise ValueError("Thirteen innovations per sample and explicit finite objective settings required")
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (feature_map.gain_count,) or not np.isfinite(gains).all():
        raise ValueError("Invalid temporal gains")
    p = feature_map.parent.columns
    score = np.zeros(p)
    risk = np.zeros((p, p))
    cost = np.zeros((p, p))
    records = []
    for basis, history in zip(bases, innovations):
        market = basis.sample.market
        signals = [feature_map.signals(z, vol, innov, gains)
                   for z, vol, innov in zip(market.characteristics, market.volatility, history)]
        omega, change = basis.operators(signals)
        rr = market.gamma * omega.T @ market.covariance @ omega
        cc = market.wealth * change.T @ (market.impact[:, None] * change)
        score += omega.T @ basis.sample.returns
        risk += (rr + rr.T) / 2
        cost += (cc + cc.T) / 2
        records.append((omega, change))
    count = len(bases)
    score, risk, cost = score / count, risk / count, cost / count
    beta, residual = _solve(risk + cost, score, settings.ridge)
    # Preserve the parent's theta penalty as a constant: theta is never refitted.
    parent_value = (score @ beta - .5 * beta @ (risk + cost + settings.ridge * np.eye(p)) @ beta
                    - .5 * settings.scale_penalty * feature_map.theta @ feature_map.theta)
    value = parent_value / objective_scale - rho * np.mean((gains - 1.) ** 2)
    grad = None
    if with_gradient:
        grad = np.zeros(feature_map.gain_count)
        for basis, history, (omega, change) in zip(bases, innovations, records):
            market = basis.sample.market
            v, u = omega @ beta, change @ beta
            gv = (basis.sample.returns - market.gamma * (market.covariance @ v)) / count
            gu = -market.wealth * market.impact * u / count
            adjoints = basis.pullback(gv, gu)
            for z, vol, innov, adj in zip(market.characteristics, market.volatility, history, adjoints):
                grad += feature_map.vjp(z, vol, innov, gains, np.outer(adj, beta))
        grad = grad / objective_scale - 2. * rho * (gains - 1.) / feature_map.gain_count
    if not np.isfinite(value) or (grad is not None and not np.isfinite(grad).all()):
        raise ValueError("Nonfinite profiled value or derivative")
    return dict(value=float(value), parent_value=float(parent_value), beta=beta, score=score,
                risk=risk, cost=cost, solve_residual=residual, gradient=grad)
