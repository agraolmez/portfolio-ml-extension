"""Financial source verification and bounded production inputs."""
import pandas as pd
import numpy as np
from baseline.full import preflight
from temporal_fast_v4.history import StudyHistory
from annual_ind.panels import CalendarSourcePanels
from annual_ind.preparation_checks import PreparationFiles
from annual_ind.cache_expanded import OperatorCache, ExpandedInputCache
from annual_ind.cached_inputs import open_parent_store
from annual_ind.coordinate_expanded import causal_state
from extension4.state import StateBook
from scripts.industry_projection_kernels_20260928 import FACTORS
from .common import ROOT,E3,E4,read,sha,atomic,finite_equal


def setup(config,catalog,out,runtime):
    runtime.event('phase_progress',label='Verifying financial inputs and source dates',completed=None,total=None)
    receipt,_=preflight();atomic(out/'financial_preflight.json',receipt)
    history=StudyHistory(ROOT,receipt,weights=[1/6]*6)
    history.panels=CalendarSourcePanels(history.files,history.panels.columns,config['calendar_source_bytes'])
    history.files.inventory.update(read(ROOT/'output/step6_20260911/preparation_manifest.json')['artifact_sha256'])
    history.files=PreparationFiles(history.files);history.panels.files=history.files
    cache=OperatorCache(maximum_bytes=config['operator_cache_bytes'],guard=runtime.guard)
    disk_store=open_parent_store(ROOT,history,sha(ROOT/(E3+'/study_lock.json')))
    inputs=ExpandedInputCache(config['input_cache_bytes'],runtime.guard,disk_store)
    book=StateBook(pd.read_parquet(catalog.path(E4+'/states/history.parquet')),catalog.read(E4+'/states/metadata.json'))
    if book.identity!=catalog.read(E4+'/states/identity.json')['identity']:raise ValueError('State identity changed')
    with np.load(catalog.path('data/prepared/step4_v1/risk/factor_covariances.npz'),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['factors'],np.array(FACTORS))
    return history,cache,inputs,book


def verify_prefix(history,book,runtime):
    records=[]
    for cutoff in ('1970-12-31','2024-12-31'):
        runtime.guard();runtime.event('phase_progress',label=f'Checking state chronology through {cutoff}',completed=None,total=None)
        prefix=causal_state(ROOT,history,cutoff)
        expected=book.frame[book.frame.eom<=pd.Timestamp(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.frame,expected,check_exact=True)
        records.append(dict(cutoff=cutoff,months=len(expected),exact=True))
    return records
