"""macOS charged memory, including compressed pages hidden by resident RSS."""
import ctypes
import os
import platform
import psutil
from functools import lru_cache


_FIELDS=('user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size '
         'phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time '
         'child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime '
         'diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance '
         'cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated '
         'cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes '
         'lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy '
         'interval_max_phys_footprint runnable_time').split()


class _Usage(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in _FIELDS]


@lru_cache(maxsize=1)
def _function():
    if platform.system()!='Darwin':raise RuntimeError('Verified physical footprint requires macOS')
    library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    fn=library.proc_pid_rusage;fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
    return fn


def usage(pid=None):
    pid=os.getpid() if pid is None else pid
    result=_Usage()
    if _function()(pid,4,ctypes.byref(result))!=0:raise OSError(ctypes.get_errno(),'Cannot read process physical footprint')
    return dict(physical_footprint_bytes=result.phys_footprint,peak_physical_footprint_bytes=result.lifetime_max_phys_footprint,
        resident_bytes=result.resident_size,process_start=result.proc_start_abstime)


def charged_bytes():
    current=usage()
    return max(current['physical_footprint_bytes'],psutil.Process().memory_info().rss)
