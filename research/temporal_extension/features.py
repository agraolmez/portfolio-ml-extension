"""Temporal gains followed by the unchanged, once-only frozen parent scale map."""
from dataclasses import dataclass
import numpy as np
from feature_geometry.features import FeatureMap, array, fingerprint
from .reference import Innovation


@dataclass(frozen=True)
class TemporalFeatureMap:
    parent: FeatureMap
    theta: np.ndarray
    groups: np.ndarray

    def __post_init__(self):
        theta = array(self.theta, 1)
        groups = np.asarray(self.groups)
        if (theta.shape != (self.parent.dimensions,) or groups.shape != theta.shape
                or groups.dtype.kind not in "iu" or np.any(groups < 0)
                or not np.array_equal(np.unique(groups), np.arange(groups.max() + 1))):
            raise ValueError("Frozen scales and complete contiguous theme indices required")
        groups = groups.copy()
        groups.setflags(write=False)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "groups", groups)

    @property
    def gain_count(self):
        return int(self.groups.max()) + 1

    @property
    def fingerprint(self):
        return fingerprint(self.parent.fingerprint, self.theta, self.groups)

    def inputs(self, z, innovation, gains):
        z, gains = array(z, 2), array(gains, 1)
        if (not isinstance(innovation, Innovation) or innovation.values.shape != z.shape
                or z.shape[1] != self.parent.dimensions or gains.shape != (self.gain_count,)):
            raise ValueError("Input, innovation or theme-gain shape mismatch")
        if np.all(gains == 1.):
            return z  # Bit-for-bit parent input, including imputed current values.
        return array(z + (gains[self.groups] - 1.) * innovation.values, 2)

    def signals(self, z, vol, innovation, gains):
        # The parent applies exp(theta) internally. Do not pre-scale this input.
        return self.parent.signals(self.inputs(z, innovation, gains), vol, self.theta)

    def vjp(self, z, vol, innovation, gains, upstream):
        temporal_z = self.inputs(z, innovation, gains)
        _, vol, scale, cosine, sine, normalized, norm = self.parent._parts(temporal_z, vol, self.theta)
        upstream = array(upstream, 2)
        if upstream.shape != (len(z), self.parent.columns):
            raise ValueError("Signal adjoint shape mismatch")
        adjoint = upstream[:, 1:] / vol[:, None]
        raw_gradient = (adjoint - adjoint.mean(axis=0)
                        - normalized * np.sum(normalized * adjoint, axis=0)) / norm
        k = self.parent.frequencies.shape[1]
        angle_gradient = -sine * raw_gradient[:, :k] + cosine * raw_gradient[:, k:]
        input_gradient = (angle_gradient @ self.parent.frequencies.T) * scale
        by_characteristic = np.sum(input_gradient * innovation.values, axis=0)
        return np.bincount(self.groups, weights=by_characteristic, minlength=self.gain_count)
