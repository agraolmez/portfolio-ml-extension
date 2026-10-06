"""Saved coefficients bind to one accepted annual parent and one state history."""
import json
import numpy as np
import pandas as pd
from feature_geometry.features import array, fingerprint
from .design import columns
from .features import disabled_aim


class ConditionalModel:
    def __init__(self, parent, method, beta, *, state_identity, protocol_identity,
                 extras=None, multiplier=None, size_bounds=None):
        self.parent, self.method = parent, method
        self.beta = array(beta, 1)
        self.state_identity, self.protocol_identity = state_identity, protocol_identity
        self.extras, self.multiplier = extras, multiplier
        self.size_bounds = None if size_bounds is None else tuple(size_bounds)
        if method not in ('N4', 'C4', 'CAP4', 'SIZE4') or self.beta.shape != (len(columns(parent.model, method)),):
            raise ValueError('Model family and coefficient column order must match')
        if not state_identity or not protocol_identity: raise ValueError('State/protocol identities required')
        if method == 'CAP4' and extras is None: raise ValueError('Capacity dictionary is missing')
        if method == 'N4':
            np.testing.assert_allclose(self.beta, parent.model.beta, atol=1e-11, rtol=1e-10)
        elif multiplier is None or multiplier <= 0:
            raise ValueError('Explicit positive candidate multiplier required')
        if method == 'SIZE4' and (self.size_bounds is None or not self.size_bounds[0] <= self.beta[0] <= self.size_bounds[1]):
            raise ValueError('Exposure parameter outside its approved bounds')
        self._identity = self._snapshot()

    def metadata(self):
        return dict(format_version=1, year=self.parent.year, seed=self.parent.seed, method=self.method,
                    parent_identity=self.parent.model._content_hash(), state_identity=self.state_identity,
                    protocol_identity=self.protocol_identity, multiplier=self.multiplier,
                    size_bounds=self.size_bounds, columns=columns(self.parent.model, self.method),
                    extra_identity=None if self.extras is None else self.extras.fingerprint)

    def _snapshot(self): return fingerprint(self.beta, json.dumps(self.metadata(), sort_keys=True))

    @property
    def identity(self):
        self.parent.model._check_unchanged()
        if self._snapshot() != self._identity: raise ValueError('Frozen conditional model was changed')
        return self._identity

    def aim(self, market, bound_innovation, bound_state):
        self.identity
        day = pd.Timestamp(market.decision_date)
        if (day < pd.Timestamp(self.parent.model.cutoff) or
                (day+pd.offsets.MonthEnd(1)).year != self.parent.year):
            raise ValueError('Wrong annual model or future training information')
        if self.method == 'N4': return disabled_aim(self.parent.model, market, bound_innovation)
        c = float(bound_state.validate(day, history=1, expected_identity=self.state_identity)[0, 0])
        model = self.parent.model
        z, vol = market.characteristics[0], market.volatility[0]
        innovation = bound_innovation.validate(day, market.ids, z, model.source_fingerprint, model.reference_fingerprint)
        base = model.feature_map.signals(z, vol, innovation, model.gains)
        if self.method == 'SIZE4':
            return (1.+self.beta[0]*c)*(base@model.beta)
        conditional = (c*base[:, 1:] if self.method == 'C4'
                       else self.extras.signals(z, vol, innovation, model.gains)[:, 1:])
        return np.column_stack((base, conditional))@self.beta

    def save(self, path):
        identity = self.identity
        with path.open('xb') as f:
            np.savez_compressed(f, beta=self.beta, metadata=np.array(json.dumps(self.metadata(), sort_keys=True)),
                                identity=np.array(identity))

    @classmethod
    def load(cls, path, parent, *, state_identity, protocol_identity, extras=None):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata'].item()))
            if (meta['format_version'] != 1 or meta['parent_identity'] != parent.model._content_hash()
                    or meta['state_identity'] != state_identity or meta['protocol_identity'] != protocol_identity
                    or meta['year'] != parent.year or meta['seed'] != parent.seed):
                raise ValueError('Saved model parent/year/state/protocol mismatch')
            model = cls(parent, meta['method'], saved['beta'], state_identity=state_identity,
                        protocol_identity=protocol_identity, extras=extras, multiplier=meta['multiplier'],
                        size_bounds=meta['size_bounds'])
            if model.identity != str(saved['identity'].item()): raise ValueError('Saved coefficient/column identity differs')
            return model
