"""Exact source lookback with fixed transition operators and reverse derivatives."""
from dataclasses import dataclass
from datetime import date
import warnings
import numpy as np
from scipy.linalg import LinAlgWarning, cho_factor, lu_factor, lu_solve
from .features import array, fingerprint


@dataclass(frozen=True)
class MarketMoment:
    decision_date: str
    ids: np.ndarray
    characteristics: tuple
    volatility: tuple
    growth: tuple
    m: np.ndarray
    covariance: np.ndarray
    impact: np.ndarray
    wealth: float
    gamma: float = 10.
    source_id: str = "synthetic"

    def __post_init__(self):
        object.__setattr__(self, "decision_date", date.fromisoformat(self.decision_date).isoformat())
        ids = np.array(self.ids, copy=True)
        if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
            raise ValueError("Unique ordered security IDs are required")
        ids.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        for name in ("characteristics", "volatility", "growth"):
            blocks = tuple(array(x, 2 if name == "characteristics" else 1) for x in getattr(self, name))
            if len(blocks) != 13:
                raise ValueError("Exactly thirteen aligned lookback slices are required")
            object.__setattr__(self, name, blocks)
        n, d = self.characteristics[0].shape
        if n != len(ids) or any(z.shape != (n, d) for z in self.characteristics):
            raise ValueError("Current-ID characteristic alignment failed")
        if any(x.shape != (n,) for x in self.volatility + self.growth):
            raise ValueError("Current-ID growth/volatility alignment failed")
        if any(np.any(x <= 0) for x in self.volatility):
            raise ValueError("Volatility must be positive")
        for name in ("m", "covariance", "impact"):
            object.__setattr__(self, name, array(getattr(self, name), 1 if name == "impact" else 2))
        if self.m.shape != (n, n) or self.covariance.shape != (n, n) or self.impact.shape != (n,):
            raise ValueError("Misaligned economic inputs")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Covariance must be symmetric")
        cho_factor(self.covariance, lower=True, check_finite=False)
        if (not np.isfinite([self.wealth, self.gamma]).all() or self.wealth <= 0
                or self.gamma <= 0 or np.any(self.impact <= 0)):
            raise ValueError("Invalid wealth, risk aversion or impact")
        object.__setattr__(self, "fingerprint", fingerprint(
            self.decision_date, self.source_id, self.ids, *self.characteristics,
            *self.volatility, *self.growth, self.m, self.covariance,
            self.impact, self.wealth, self.gamma))


@dataclass(frozen=True)
class TrainingSample:
    market: MarketMoment
    returns: np.ndarray
    available_at: str

    def __post_init__(self):
        object.__setattr__(self, "available_at", date.fromisoformat(self.available_at).isoformat())
        if date.fromisoformat(self.available_at) <= date.fromisoformat(self.market.decision_date):
            raise ValueError("Outcome availability must follow the decision")
        returns = array(self.returns, 1)
        if returns.shape != self.market.ids.shape:
            raise ValueError("Misaligned training outcomes")
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "fingerprint", fingerprint(self.market.fingerprint, returns, self.available_at))


class MomentBasis:
    """Caches only geometry-independent lookback matrices; never old feature moments."""
    def __init__(self, sample):
        self.sample = sample
        market = sample.market
        n = len(market.ids)
        current = [np.eye(n)]
        prior = [np.eye(n)]
        for i in range(11):
            current.append(current[-1] @ (market.m * market.growth[i][None, :]))
            prior.append(prior[-1] @ (market.m * market.growth[i + 1][None, :]))
        self.current, self.prior = tuple(current), tuple(prior)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            self.current_lu = lu_factor(sum(current), check_finite=False)
            self.prior_lu = lu_factor(sum(prior), check_finite=False)
        for x in self.current + self.prior:
            x.setflags(write=False)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.current + self.prior) + self.current_lu[0].nbytes + self.prior_lu[0].nbytes

    def operators(self, signals):
        if len(signals) != 13 or any(x.shape != signals[0].shape for x in signals):
            raise ValueError("Thirteen aligned signal slices are required")
        omega = lu_solve(self.current_lu, sum(c @ s for c, s in zip(self.current, signals[:12])), check_finite=False)
        lag = lu_solve(self.prior_lu, sum(c @ s for c, s in zip(self.prior, signals[1:])), check_finite=False)
        return omega, omega - self.sample.market.growth[0][:, None] * lag

    def pullback(self, omega_gradient, change_gradient):
        """Adjoints for vector-valued S_i beta; original current-date m stays fixed."""
        g = self.sample.market.growth[0]
        cur = lu_solve(self.current_lu, omega_gradient + change_gradient, trans=1, check_finite=False)
        lag = lu_solve(self.prior_lu, -g * change_gradient, trans=1, check_finite=False)
        result = [np.zeros_like(cur) for _ in range(13)]
        for i in range(12):
            result[i] += self.current[i].T @ cur
            result[i + 1] += self.prior[i].T @ lag
        return result

    def signals(self, feature_map, theta):
        market = self.sample.market
        return [feature_map.signals(z, vol, theta) for z, vol in zip(market.characteristics, market.volatility)]

    def matrices(self, feature_map, theta):
        signals = self.signals(feature_map, theta)
        omega, change = self.operators(signals)
        market = self.sample.market
        risk = market.gamma * omega.T @ market.covariance @ omega
        cost = market.wealth * change.T @ (market.impact[:, None] * change)
        return {"score": omega.T @ self.sample.returns, "risk": (risk + risk.T) / 2,
                "cost": (cost + cost.T) / 2, "omega": omega, "change": change,
                "signal": signals[0]}
