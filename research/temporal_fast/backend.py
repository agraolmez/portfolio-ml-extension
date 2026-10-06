"""Explicit Apple Accelerate control, verified against the installed SDK headers."""
from contextlib import contextmanager
import ctypes
from functools import lru_cache
import platform
import numpy as np
from threadpoolctl import threadpool_limits


@lru_cache(maxsize=1)
def accelerate():
    if platform.system() != 'Darwin': raise RuntimeError('Apple Accelerate requires macOS')
    lib=ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
    lib.BLASSetThreading.argtypes=[ctypes.c_uint];lib.BLASSetThreading.restype=ctypes.c_int
    lib.BLASGetThreading.argtypes=[];lib.BLASGetThreading.restype=ctypes.c_uint
    pointer=ctypes.POINTER(ctypes.c_double)
    lib.vvsincos.argtypes=[pointer,pointer,pointer,ctypes.POINTER(ctypes.c_int)]
    lib.vvsincos.restype=None
    return lib


def worker_init():
    if platform.system() == 'Darwin':
        lib=accelerate()
        if lib.BLASSetThreading(1) != 0 or lib.BLASGetThreading() != 1:
            raise RuntimeError('Unable to enforce one BLAS thread per monthly worker')


@contextmanager
def single_thread_blas():
    # The native Accelerate setting is thread-local, so every pool worker must
    # also call worker_init. threadpoolctl does not discover this backend.
    if platform.system() == 'Darwin':
        lib=accelerate();previous=lib.BLASGetThreading();worker_init()
        try:yield
        finally:
            if lib.BLASSetThreading(previous) != 0:raise RuntimeError('Cannot restore BLAS policy')
    else:
        with threadpool_limits(limits=1):yield


def sincos(angles):
    angles=np.ascontiguousarray(angles,dtype=np.float64)
    if not np.isfinite(angles).all() or angles.size>np.iinfo(np.int32).max:
        raise ValueError('Finite bounded-size double-precision angles required')
    sine=np.empty_like(angles);cosine=np.empty_like(angles)
    pointer=ctypes.POINTER(ctypes.c_double);count=ctypes.c_int(angles.size)
    accelerate().vvsincos(sine.ctypes.data_as(pointer),cosine.ctypes.data_as(pointer),
                         angles.ctypes.data_as(pointer),ctypes.byref(count))
    return cosine,sine
