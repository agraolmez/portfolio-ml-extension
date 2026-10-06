"""Bounded exact-profile reuse for the three temporal restraint candidates.

Only the restraint term changes between candidates with the same annual parent.
Profiles are keyed by the complete gain vector, including rejected line-search
points. Fresh optimizations and evaluation counts are unchanged. Cache hits
recheck the full input/operator fingerprint, rather than trusting object IDs.
"""
from collections import OrderedDict
import copy
import hashlib
import numpy as np
from annual_ind.parallel_layers import ParallelTemporalObjective


def _bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes(v) for v in value)
    return 0


class SharedTemporalProfiles:
    def __init__(self, *args, maximum_bytes, guard, **kwargs):
        if (type(maximum_bytes) is not int or not 0 <= maximum_bytes <= 128*2**20
                or not callable(guard) or 'rho' in kwargs):
            raise ValueError('Bounded cache, guard and no shared restraint required')
        self.engine = ParallelTemporalObjective(*args, rho=0., **kwargs)
        self.args, self.kwargs = args, kwargs
        self.maximum_bytes, self.guard = maximum_bytes, guard
        self.cache = OrderedDict()
        self.retained_bytes = self.hits = self.misses = 0
        self.identity = self._configuration()

    def _configuration(self):
        e = self.engine
        return (e._model_hash(), e.workers, e.trig_backend,
                e.parallel_checks, e.rank_one, e.packed, e.shared,
                e.fused_parts, e.fused_gradient, e.deferred_loads, e._options)

    def _audit_hit(self, expected):
        digest = hashlib.sha256()
        count = 0
        for batch in self.engine._batches():
            self.guard()
            _, _, identity = self.engine._audit(batch)
            digest.update(identity.encode())
            count += 1
        if (count != expected['training_months']
                or digest.hexdigest() != expected['training_source_fingerprint']):
            raise ValueError('Cached temporal profile source/operator identity changed')

    def for_restraint(self, rho):
        if not np.isfinite(rho) or rho < 0:
            raise ValueError('Finite nonnegative restraint required')
        model = ParallelTemporalObjective(*self.args, rho=rho, **self.kwargs)

        def evaluate(gains, with_gradient=True):
            self.guard()
            if self._configuration() != self.identity:
                raise ValueError('Shared temporal configuration changed')
            point = np.asarray(gains, dtype=np.float64)
            if point.shape != (self.engine.feature_map.gain_count,) or not np.isfinite(point).all():
                raise ValueError('Finite aligned gain vector required')
            key = (point.tobytes(), bool(with_gradient))
            if key in self.cache:
                payload = self.cache[key]
                self._audit_hit(payload)
                self.hits += 1
                self.cache.move_to_end(key)
            else:
                payload = self.engine(point, with_gradient=with_gradient)
                self.misses += 1
                size = _bytes(payload)
                if size <= self.maximum_bytes:
                    while self.cache and self.retained_bytes + size > self.maximum_bytes:
                        _, removed = self.cache.popitem(last=False)
                        self.retained_bytes -= _bytes(removed)
                    self.cache[key] = copy.deepcopy(payload)
                    self.retained_bytes += size
            result = copy.deepcopy(payload)
            result['value'] = float(result['parent_value']/self.engine.objective_scale
                                    - rho*np.mean((point-1.)**2))
            if with_gradient:
                result['gradient'] = result['gradient'] - 2.*rho*(point-1.)/len(point)
            result['frozen_model_fingerprint'] = model._model_identity
            self.guard()
            return result
        return evaluate

    def diagnostics(self):
        return dict(hits=self.hits, misses=self.misses,
                    retained_array_bytes=self.retained_bytes,
                    maximum_array_bytes=self.maximum_bytes,
                    full_input_audit_on_cache_hit=True)
