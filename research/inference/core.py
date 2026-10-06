"""Stationary paired resampling and the approved empirical utility statistic."""
import hashlib
import json
import numpy as np


def seed(protocol_id, master, length, date, replica):
    payload = dict(protocol=protocol_id, master_seed=master,
                   component=f'step10_stationary_L{length}', date=date, replica=replica)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return int(digest[:32], 16), digest


def indices_from_primitives(starts, restart):
    starts = np.asarray(starts)
    restart = np.asarray(restart, dtype=bool)
    if starts.ndim != 2 or starts.shape != restart.shape or starts.shape[1] < 1:
        raise ValueError('Matching nonempty replica/month arrays required')
    n = starts.shape[1]
    if not np.issubdtype(starts.dtype, np.integer) or np.any(starts < 0) or np.any(starts >= n):
        raise ValueError('Start index outside observed interval')
    if not restart[:, 0].all():
        raise ValueError('First position must start uniformly')
    idx = np.empty_like(starts, dtype=np.int32)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t-1] + 1) % n)
    return idx


def stationary(protocol_id, master, length, date, n, replicas=10000):
    if n < 2 or length < 1 or replicas < 1:
        raise ValueError('Invalid stationary bootstrap dimensions')
    starts = np.empty((replicas, n), dtype=np.int32)
    restart = np.empty((replicas, n), dtype=bool)
    digests = []
    for b in range(replicas):
        integer, digest = seed(protocol_id, master, length, date, b + 1)
        rng = np.random.Generator(np.random.PCG64(integer))
        starts[b] = rng.integers(0, n, size=n, dtype=np.int32)
        restart[b] = rng.random(n) < 1 / length
        restart[b, 0] = True
        digests.append(digest)
    return indices_from_primitives(starts, restart), starts, restart, digests


def utility(x, axis=0):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('Finite nonempty returns required')
    return 12 * (x.mean(axis=axis) - 5 * x.var(axis=axis, ddof=0))


def bootstrap_utility(x, indices):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite date/method matrix required')
    if indices.ndim != 2 or indices.shape[1] != len(x) or np.any(indices < 0) or np.any(indices >= len(x)):
        raise ValueError('Indices must resample the complete interval')
    result = np.empty((len(indices), x.shape[1]))
    # Bounded allocations; every sample uses its own mean and population variance.
    for first in range(0, len(indices), 250):
        result[first:first+250] = utility(x[indices[first:first+250]], axis=1)
    return result


def basic_interval(observed, draws, alpha=.05, family=1):
    if not 0 < alpha < 1 or family < 1:
        raise ValueError('Invalid interval probabilities')
    lo, hi = np.quantile(draws, [alpha/(2*family), 1-alpha/(2*family)], method='linear')
    return float(2*observed-hi), float(2*observed-lo)


def success(points, lower):
    return bool(np.all(np.asarray(points) > 0) and np.all(np.asarray(lower) > 0))


def acf(x, lag):
    z = np.asarray(x, dtype=float) - np.mean(x)
    denominator = z @ z
    return float(z[lag:] @ z[:-lag] / denominator) if denominator > 0 else None
