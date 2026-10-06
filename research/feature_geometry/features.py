"""Fixed Fourier directions with differentiable, normalized input scales."""
from dataclasses import dataclass
import hashlib
import numpy as np


def array(value, ndim=None):
    result = np.array(value, dtype=np.float64, copy=True)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError("Invalid dimensions or nonfinite numerical input")
    result.setflags(write=False)
    return result


def fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def project_scales(theta, bound=np.log(2.)):
    """Euclidean projection onto the zero-sum box, not clip-then-center."""
    theta = array(theta, 1)
    if not len(theta) or not np.isfinite(bound) or bound <= 0:
        raise ValueError("A nonempty scale vector and positive finite bound are required")
    if abs(theta.sum()) <= 1e-14 and np.max(np.abs(theta)) <= bound:
        return theta.copy()
    lo, hi = float(theta.min() - bound), float(theta.max() + bound)
    for _ in range(100):
        midpoint = lo / 2 + hi / 2
        trial = np.clip(theta - midpoint, -bound, bound)
        if trial.sum() > 0:
            lo = midpoint
        else:
            hi = midpoint
    result = np.clip(theta - (lo / 2 + hi / 2), -bound, bound)
    if abs(result.sum()) > 1e-10:
        raise ValueError("Projection failed its zero-sum constraint")
    return result


@dataclass(frozen=True)
class FeatureMap:
    frequencies: np.ndarray
    source: str = "supplied fixed directions"

    def __post_init__(self):
        frequencies = array(self.frequencies, 2)
        if min(frequencies.shape) < 1:
            raise ValueError("Empty frequency matrix")
        object.__setattr__(self, "frequencies", frequencies)

    @property
    def dimensions(self):
        return self.frequencies.shape[0]

    @property
    def columns(self):
        return 1 + 2 * self.frequencies.shape[1]

    @property
    def fingerprint(self):
        return fingerprint(self.frequencies, self.source)

    def _parts(self, characteristics, volatility, theta):
        z, vol, theta = map(np.asarray, (characteristics, volatility, theta))
        if (z.ndim != 2 or z.shape[1] != self.dimensions or z.shape[0] < 2
                or vol.shape != (len(z),) or theta.shape != (self.dimensions,)
                or any(not np.isfinite(x).all() for x in (z, vol, theta))
                or np.any(vol <= 0)):
            raise ValueError("Invalid characteristics, scales or volatility")
        with np.errstate(over="raise", invalid="raise"):
            scale = np.exp(theta)
            angles = (z * scale) @ self.frequencies
        cosine, sine = np.cos(angles), np.sin(angles)
        raw = np.column_stack((cosine, sine))
        centered = raw - raw.mean(axis=0)
        norm = np.sqrt(np.sum(centered**2, axis=0))
        if np.any(norm == 0) or not np.isfinite(norm).all():
            raise ValueError("Zero or invalid cross-sectional Fourier-feature norm")
        return z, vol, scale, cosine, sine, centered / norm, norm

    def signals(self, characteristics, volatility, theta):
        z, vol, _, _, _, normalized, _ = self._parts(characteristics, volatility, theta)
        return np.column_stack((np.full(len(z), 1 / np.sqrt(len(z))), normalized)) / vol[:, None]

    def vjp(self, characteristics, volatility, theta, upstream):
        """Gradient of <upstream, signals>, including normalization; m is external."""
        z, vol, scale, cosine, sine, normalized, norm = self._parts(characteristics, volatility, theta)
        upstream = np.asarray(upstream)
        if upstream.shape != (len(z), self.columns) or not np.isfinite(upstream).all():
            raise ValueError("Misaligned or nonfinite signal adjoint")
        adjoint = upstream[:, 1:] / vol[:, None]
        raw_gradient = (adjoint - adjoint.mean(axis=0)
                        - normalized * np.sum(normalized * adjoint, axis=0)) / norm
        k = self.frequencies.shape[1]
        angle_gradient = -sine * raw_gradient[:, :k] + cosine * raw_gradient[:, k:]
        return scale * np.sum(z * (angle_gradient @ self.frequencies.T), axis=0)
