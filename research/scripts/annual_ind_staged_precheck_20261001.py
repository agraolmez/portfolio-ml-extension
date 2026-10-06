"""Approved small numerical integration checks, charged to the new 48-hour budget."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from temporal_fast.backend import single_thread_blas
from annual_staged.common import *
from annual_staged.baseline import initial_parent,parent_choices,check_original_selections
from annual_ind.runtime_expanded import Runtime,prior_seconds
from extension4.storage import write_json
import traceback

runtime=Runtime(OUT,read(CONFIG),'integration');error=None;status='STOPPED'
try:
    with threadpool_limits(limits=1),single_thread_blas():
        catalog=Catalog();first=initial_parent(catalog,OUT/'initial_parent',sha(CONFIG),runtime)
        choices=parent_choices(catalog,first)
        check=check_original_selections(catalog,choices,runtime)
        atomic(OUT/'initial_parent_check.json',dict(first=first,original_selections=check,passed=True))
        runtime.event('initial_parent_passed',coordinate={k:first[k] for k in ('gi','width','ridge_index')})
        status='COMPLETE'
except BaseException as exc:
    error=exc;atomic(OUT/'precheck_stop.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
finally:
    runtime.close(status,error)
    charged,_=prior_seconds(OUT);atomic(OUT/'resource_ledger.json',dict(charged_seconds=charged,cumulative_engineering_cap_seconds=172800))
