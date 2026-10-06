"""Two-pass profiled objective with at most one month's feature operators retained.

The batch factory must be repeatable and yield (basis, thirteen innovations).
Historical adapters validate date/ID/source bindings before returning a batch.
No lookback, penalty, stopping policy or historical execution is selected here.
"""
from dataclasses import asdict
import hashlib
import threading
import numpy as np
from feature_geometry.features import array, fingerprint
from feature_geometry.training import _solve
from feature_geometry.moments import MomentBasis
from .reference import Innovation


class StreamingObjective:
    def __init__(self, batch_factory, feature_map, settings, *, rho, objective_scale):
        if (not callable(batch_factory) or not np.isfinite([rho, objective_scale]).all()
                or rho < 0 or objective_scale <= 0):
            raise ValueError("Repeatable batch factory and explicit finite objective settings required")
        self.batch_factory = batch_factory
        self.feature_map = feature_map
        self.settings = settings
        self.rho, self.objective_scale = float(rho), float(objective_scale)
        self._source_identity = None
        self._model_identity = self._model_hash()
        self._lock = threading.Lock()

    def _model_hash(self):
        return fingerprint(self.feature_map.fingerprint, asdict(self.settings),
                           self.rho, self.objective_scale)

    def _batches(self):
        previous = None
        for basis, innovations in self.batch_factory():
            if not isinstance(basis, MomentBasis):
                raise ValueError("An audited parent moment-basis implementation is required")
            sample = basis.sample
            market = sample.market
            if previous is not None and market.decision_date <= previous:
                raise ValueError("Training dates must be unique and ordered")
            if sample.available_at > self.settings.cutoff:
                raise ValueError("Unmatured outcome at fitting cutoff")
            if (market.characteristics[0].shape[1] != self.feature_map.parent.dimensions
                    or len(innovations) != 13
                    or any(not isinstance(x, Innovation) or x.values.shape != z.shape
                           for x, z in zip(innovations, market.characteristics))):
                raise ValueError("Misaligned characteristic dictionary or temporal slices")
            previous = market.decision_date
            market_identity = fingerprint(market.decision_date, market.source_id, market.ids,
                                          *market.characteristics, *market.volatility, *market.growth,
                                          market.m, market.covariance, market.impact,
                                          market.wealth, market.gamma)
            sample_identity = fingerprint(market_identity, sample.returns, sample.available_at)
            if market_identity != market.fingerprint or sample_identity != sample.fingerprint:
                raise ValueError("Market or training sample arrays changed after construction")
            operators = [*basis.current_lu, *basis.prior_lu,
                         *getattr(basis, 'current', ()), *getattr(basis, 'prior', ())]
            identity = fingerprint(sample_identity, *operators,
                                   *(value for x in innovations for value in (x.values, x.valid)))
            yield basis, innovations, identity

    def _finish_pass(self, count, digest, *, expected=None):
        if count == 0:
            raise ValueError("Training observations are required")
        identity = (count, digest.hexdigest())
        if expected is not None and identity != expected:
            raise ValueError("Source, mask or sample sequence changed between objective passes")
        if self._source_identity is not None and identity != self._source_identity:
            raise ValueError("Source, mask or sample sequence changed across gain evaluations")
        if self._model_hash() != self._model_identity:
            raise ValueError("Frozen model or objective configuration changed")
        return identity

    def __call__(self, gains, with_gradient=True):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A streaming objective cannot be evaluated concurrently")
        try:
            return self._evaluate(gains, with_gradient)
        finally:
            self._lock.release()

    def _evaluate(self, gains, with_gradient):
        fmap = self.feature_map
        gains = array(gains, 1)
        if gains.shape != (fmap.gain_count,):
            raise ValueError("Invalid temporal gain vector")
        if self._model_hash() != self._model_identity:
            raise ValueError("Frozen model or objective configuration changed")
        p = fmap.parent.columns
        score, risk, cost = np.zeros(p), np.zeros((p, p)), np.zeros((p, p))
        count = 0
        digest = hashlib.sha256()
        coverage = {'all_13_slices': {}, 'current_slice': {}, 'reason_counts_complete': True}
        reason_keys = ('current_unobserved', 'prior_unobserved_or_absent', 'prior_security_month_absent')
        for basis, history, identity in self._batches():
            market = basis.sample.market
            signals = [fmap.signals(z, vol, innov, gains)
                       for z, vol, innov in zip(market.characteristics, market.volatility, history)]
            omega, change = basis.operators(signals)
            rr = market.gamma * omega.T @ market.covariance @ omega
            cc = market.wealth * change.T @ (market.impact[:, None] * change)
            score += omega.T @ basis.sample.returns
            risk += (rr + rr.T) / 2
            cost += (cc + cc.T) / 2
            count += 1
            digest.update(identity.encode())
            for lag, innovation in enumerate(history):
                counts = dict(cells=innovation.values.size, adjusted_eligible=int(innovation.valid.sum()),
                              bypassed=int((~innovation.valid).sum()))
                for key in reason_keys:
                    if key in innovation.reasons:
                        value = innovation.reasons[key]
                        if not isinstance(value, int) or not 0 <= value <= counts['cells']:
                            raise ValueError('Invalid source-mask coverage reason count')
                        counts[key] = value
                    else:
                        coverage['reason_counts_complete'] = False
                for scope in ('all_13_slices', 'current_slice') if lag == 0 else ('all_13_slices',):
                    for key, value in counts.items():
                        coverage[scope][key] = coverage[scope].get(key, 0) + value
            del signals, omega, change, rr, cc, market, basis, history
        source_identity = self._finish_pass(count, digest)
        score, risk, cost = score / count, risk / count, cost / count
        beta, residual = _solve(risk + cost, score, self.settings.ridge)
        parent_value = (score @ beta - .5 * beta @ (risk + cost + self.settings.ridge * np.eye(p)) @ beta
                        - .5 * self.settings.scale_penalty * fmap.theta @ fmap.theta)
        value = parent_value / self.objective_scale - self.rho * np.mean((gains - 1.) ** 2)
        grad = None
        if with_gradient:
            grad = np.zeros(fmap.gain_count)
            second_count, second_digest = 0, hashlib.sha256()
            for basis, history, identity in self._batches():
                market = basis.sample.market
                # Only this month's operators are rebuilt; no history-sized cache.
                signals = [(fmap.signals(z, vol, innov, gains) @ beta)[:, None]
                           for z, vol, innov in zip(market.characteristics, market.volatility, history)]
                omega, change = basis.operators(signals)
                v, u = omega[:, 0], change[:, 0]
                gv = (basis.sample.returns - market.gamma * (market.covariance @ v)) / count
                gu = -market.wealth * market.impact * u / count
                adjoints = basis.pullback(gv, gu)
                for z, vol, innov, adj in zip(market.characteristics, market.volatility, history, adjoints):
                    grad += fmap.vjp(z, vol, innov, gains, np.outer(adj, beta))
                second_count += 1
                second_digest.update(identity.encode())
                del signals, omega, change, v, u, gv, gu, adjoints, market, basis, history
            self._finish_pass(second_count, second_digest, expected=source_identity)
            grad = grad / self.objective_scale - 2. * self.rho * (gains - 1.) / fmap.gain_count
        if not np.isfinite(value) or (grad is not None and not np.isfinite(grad).all()):
            raise ValueError("Nonfinite profiled objective or gradient")
        self._source_identity = source_identity
        return dict(value=float(value), parent_value=float(parent_value), beta=beta, score=score,
                    risk=risk, cost=cost, solve_residual=residual, gradient=grad,
                    training_months=count, training_source_fingerprint=source_identity[1],
                    frozen_model_fingerprint=self._model_identity, coverage=coverage,
                    coverage_unit='decision-security-characteristic cells; all_13_slices repeats overlapping history')
