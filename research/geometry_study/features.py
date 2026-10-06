"""Per-worker 13-slice feature-part cache; no cross-parameter reuse."""
from collections import OrderedDict
import threading
import numpy as np
from feature_geometry.features import FeatureMap
class CachedFeatureMap(FeatureMap):
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self,'_local',threading.local())
    def _parts(self,z,vol,theta):
        key=(id(z),id(vol),np.asarray(theta).tobytes())
        if not hasattr(self._local,'cache'):self._local.cache=OrderedDict()
        cache=self._local.cache
        if key not in cache:
            cache[key]=super()._parts(z,vol,theta)
            while len(cache)>13:cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]
