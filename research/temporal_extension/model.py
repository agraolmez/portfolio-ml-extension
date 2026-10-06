"""Saved temporal coefficients and gains with dated, provenance-checked decisions."""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import numpy as np
from feature_geometry.features import FeatureMap, array, fingerprint
from .features import TemporalFeatureMap


@dataclass(frozen=True)
class TemporalModel:
    feature_map: TemporalFeatureMap
    gains: np.ndarray
    beta: np.ndarray
    cutoff: str
    parent_cutoff: str
    parent_fingerprint: str
    source_fingerprint: str
    reference_fingerprint: str
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        cutoff = date.fromisoformat(self.cutoff).isoformat()
        parent_cutoff = date.fromisoformat(self.parent_cutoff).isoformat()
        if parent_cutoff > cutoff:
            raise ValueError("Parent was learned after the temporal training cutoff")
        object.__setattr__(self, 'cutoff', cutoff)
        object.__setattr__(self, 'parent_cutoff', parent_cutoff)
        for name in ('gains', 'beta', 'lower', 'upper'):
            object.__setattr__(self, name, array(getattr(self, name), 1))
        shape = (self.feature_map.gain_count,)
        if (self.gains.shape != shape or self.lower.shape != shape or self.upper.shape != shape
                or self.beta.shape != (self.feature_map.parent.columns,)
                or np.any(self.lower > self.upper) or np.any(self.lower > 1.)
                or np.any(self.upper < 1.) or np.any(self.gains < self.lower)
                or np.any(self.gains > self.upper)):
            raise ValueError("Invalid saved temporal model shape, bounds or coefficients")
        if any(not isinstance(x, str) or not x for x in
               (self.parent_fingerprint, self.source_fingerprint, self.reference_fingerprint)):
            raise ValueError("Parent, input-source and reference provenance are required")
        object.__setattr__(self, '_content_identity', self._content_hash())

    def _content_hash(self):
        return fingerprint(self.feature_map.fingerprint, self.gains, self.beta,
                           self.lower, self.upper, json.dumps(self._metadata(), sort_keys=True))

    def _check_unchanged(self):
        if self._content_hash() != self._content_identity:
            raise ValueError("Frozen temporal model content changed after construction")

    def aim(self, decision_date, ids, z, vol, bound_innovation):
        self._check_unchanged()
        day = date.fromisoformat(decision_date).isoformat()
        if day < self.cutoff:
            raise ValueError("Temporal model would use future training information")
        innovation = bound_innovation.validate(day, ids, z,
                                              source_fingerprint=self.source_fingerprint,
                                              reference_fingerprint=self.reference_fingerprint)
        return self.feature_map.signals(z, vol, innovation, self.gains) @ self.beta

    def _metadata(self):
        return dict(format_version=1, cutoff=self.cutoff, parent_cutoff=self.parent_cutoff,
                    parent_fingerprint=self.parent_fingerprint,
                    source_fingerprint=self.source_fingerprint,
                    reference_fingerprint=self.reference_fingerprint,
                    frequency_source=self.feature_map.parent.source)

    def save(self, path):
        self._check_unchanged()
        metadata = self._metadata()
        arrays = dict(frequencies=self.feature_map.parent.frequencies, theta=self.feature_map.theta,
                      groups=self.feature_map.groups, gains=self.gains, beta=self.beta,
                      lower=self.lower, upper=self.upper)
        metadata['parameter_fingerprint'] = fingerprint(
            *(arrays[k] for k in sorted(arrays)), json.dumps(metadata, sort_keys=True))
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path, *, expected_parent_fingerprint):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
            expected = metadata.pop('parameter_fingerprint')
            names = ['beta', 'frequencies', 'gains', 'groups', 'lower', 'theta', 'upper']
            actual = fingerprint(*(data[k] for k in names), json.dumps(metadata, sort_keys=True))
            if (metadata['format_version'] != 1 or expected != actual
                    or metadata['parent_fingerprint'] != expected_parent_fingerprint):
                raise ValueError("Changed temporal model or mismatched parent artifact")
            fmap = TemporalFeatureMap(FeatureMap(data['frequencies'], metadata['frequency_source']),
                                      data['theta'], data['groups'])
            return cls(fmap, data['gains'], data['beta'], metadata['cutoff'],
                       metadata['parent_cutoff'], metadata['parent_fingerprint'],
                       metadata['source_fingerprint'], metadata['reference_fingerprint'],
                       data['lower'], data['upper'])


def decide(model, market, bound_innovation, holding_ids, holdings):
    """Original rule from caller-supplied own holdings; there is no reset branch."""
    h = array(holdings, 1)
    if not np.array_equal(holding_ids, market.ids) or h.shape != market.ids.shape:
        raise ValueError("Inherited holdings must match current ordered security IDs")
    aim = model.aim(market.decision_date, market.ids, market.characteristics[0],
                    market.volatility[0], bound_innovation)
    weights = market.m @ h + (np.eye(len(h)) - market.m) @ aim
    return dict(ids=market.ids.copy(), aim=aim, weights=weights, trades=weights-h)
