"""Source/chronology and independent coefficient checks before historical fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from annual_staged.common import *
from annual_staged.setup import setup,verify_prefix
from annual_staged.baseline import parent_choices
from annual_staged.fitting import Archived
from annual_staged.verification import verify_system,verify_first_path
from annual_staged.runtime import Runtime
from feature_geometry.training import FitSettings
from dataclasses import replace
from temporal_numerics.optimizer import LBFGSBSettings
from annual_ind.conditional import solve_conditional
from threadpoolctl import threadpool_limits
import pandas as pd,numpy as np,traceback
runtime=Runtime(OUT,read(CONFIG),'integration');error=None;status='STOPPED'
try:
 with threadpool_limits(limits=1):
  catalog=Catalog();history,cache,inputs,book=setup(read(CONFIG),catalog,OUT,runtime)
  causal=verify_prefix(history,book,runtime)
  choices=parent_choices(catalog,read(OUT/'initial_parent/selection.json'));c=choices[1981]
  archive=Archived(catalog,history,book)
  frequency=np.fromfile(history.files.path(f"{E2}/directions/seed1_g{c['gi']}.bin"),dtype='<f8').reshape(115,512)[:,:256]
  gp=read(ROOT/'config/feature_geometry_study_v1.json')['geometry_settings']
  settings=FitSettings('1980-12-31',ridge=c['ridge'],scale_penalty=0.,**{k:v for k,v in gp.items() if k!='relative_scale_penalty'})
  geometry,receipt,_=archive.geometry(c,frequency,settings);settings=replace(settings,scale_penalty=.01*receipt['objective_normalizer'])
  dictionary=pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'));groups=dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
  optimizer=LBFGSBSettings(**read(ROOT/'config/extension3_study_v3_proposed.json')['optimizer'])
  temporal,tfit,_=archive.temporal(c,.1,geometry.theta,frequency,groups,settings,optimizer)
  score,curvature,_=archive.conditional(c,.1,temporal)
  betas=np.column_stack([solve_conditional(curvature,score,parent_width=len(temporal.beta),parent_ridge=c['ridge'],multiplier=m)[0] for m in MULTIPLIERS])
  oracle=verify_system(curvature,score,betas,len(temporal.beta),c['ridge'],MULTIPLIERS)
  path_check=verify_first_path(history,book,c,temporal,betas,catalog,sha(CONFIG))
  atomic(OUT/'archive_integration_check.json',dict(passed=True,causal_prefixes=causal,independent=oracle,first_path=path_check,consumed=catalog.consumed))
  runtime.event('archive_integration_passed');status='COMPLETE'
except BaseException as exc:
 error=exc;atomic(OUT/f'archive_check_stop_{__import__("time").time_ns()}.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
finally:runtime.close(status,error)
