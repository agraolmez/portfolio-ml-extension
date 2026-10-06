"""Versioned execution identity; unchanged scientific settings are external."""
from types import MappingProxyType
import platform

EXECUTION=MappingProxyType(dict(workers=4,trig_backend='accelerate',blas_threads_per_worker=1,
    native_thread_control='BLASSetThreading(SINGLE_THREADED) on each thread',
    aggregation='original chronological month and slice order',
    neutral_features='original NumPy transformation',
    arithmetic='float64; original economic equations and conditional coefficient solve'))


def environment_identity():
    return dict(system=platform.platform(),machine=platform.machine(),python=platform.python_version())
