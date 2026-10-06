"""Bounded engineering check of saved production models and two actual parent months.

No E4 raw state is constructed and no new historical coefficient is fitted.
The zero-state operator check is an explicitly artificial disabled-state check.
"""
import fcntl
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.full import preflight
from feature_geometry.history import sha
from geometry_study.kernel import LeanBasis
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_study.paths import AcceptedPathSource
from extension4.parent import ParentCatalog,PARENT
from extension4.model import ConditionalModel
from extension4.state import BoundState
from extension4.design import designs,capacity_map
from extension4.paths import replay
from extension4.storage import write_json
from extension4.verification import independent_month


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/extension4_engineering_20260925/actual_neutral')
    out=parser.parse_args().output.resolve()
    if not out.is_relative_to(ROOT/'output/extension4_engineering_20260925'):
        raise ValueError('Use a new folder inside the engineering evidence directory')
    out.mkdir()  # A rerun must use reviewed, versioned evidence rather than overwrite.
    pa.set_cpu_count(1);pa.set_io_thread_count(1);started=time.monotonic()
    with (ROOT/'output/.extension3_heavy_job.lock').open('a+') as lock,single_thread_blas():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt,_=preflight();catalog=ParentCatalog(ROOT)
        if receipt!=catalog.read(PARENT+'/financial_preflight.json'):raise ValueError('Parent financial inputs changed')
        write_json(out/'financial_preflight.json',receipt)
        print('Financial preflight unchanged.',flush=True)
        history=StudyHistory(ROOT,receipt,weights=[1/6]*6);source=AcceptedPathSource(history)
        parents={s:catalog.load(1981,s) for s in (1,2,3)};models={};errors=[]
        for seed,parent in parents.items():
            model=ConditionalModel(parent,'N4',parent.model.beta,state_identity='NO_E4_STATE_ENGINEERING_NEUTRAL',
                protocol_identity='TWO_MONTH_ENGINEERING_CHECK_ONLY')
            path=out/f'N4_seed{seed}.npz';model.save(path)
            models[seed]=ConditionalModel.load(path,parent,state_identity=model.state_identity,protocol_identity=model.protocol_identity)
        def aim(seed,method,market,bound):return models[seed].aim(market,bound,None)
        def compare(date,arrays,rows):
            with np.load(catalog.path(PARENT+'/selected_paths/'+date+'.npz'),allow_pickle=False) as saved:
                for i,seed in enumerate((1,2,3)):
                    col=np.flatnonzero((saved['seeds']==seed)&(saved['methods']=='selected_temporal'))[0]
                    maximum=0.
                    for name in ('ids','next_ids'):np.testing.assert_array_equal(arrays[name],saved[name])
                    for name in ('pretrade','target','weight','trade','drifted_weight','next_pretrade'):
                        np.testing.assert_allclose(arrays[name][:,i],saved[name][:,col],atol=1e-11,rtol=1e-10)
                        maximum=max(maximum,float(np.max(abs(arrays[name][:,i]-saved[name][:,col]))))
                    errors.append(dict(date=date,seed=seed,maximum_position_error=maximum))
        dates=['1980-12-31','1981-01-31'];initial=source.common_initial()
        options=dict(destination=out/'paths',lock_identity='ENGINEERING_ONLY',policy_identity='N4_ARCHIVED_PARENT',
                     atol=1e-11,rtol=1e-10,compare=compare)
        frame=replay(source,dates,[(s,'N4') for s in (1,2,3)],initial,aim,**options)
        pd.testing.assert_frame_equal(frame,replay(source,dates,[(s,'N4') for s in (1,2,3)],initial,aim,**options))
        import json
        for date in dates:
            folder=out/'paths'/date
            with np.load(folder/'positions.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
            market=source.market(date);outcome=source.realized(market)
            independent_month(market,outcome,arrays,json.loads((folder/'monthly.json').read_text()),
                json.loads((folder/'exits.json').read_text()),atol=1e-11,rtol=1e-10)
        print('Saved production N4 models reproduce all three accepted paths across two months.',flush=True)
        parent=parents[1];date='1965-11-30'
        sample=history.training_sample(date,parent.model.cutoff);basis=LeanBasis(sample);bound=history.innovations(sample.market)
        meta=catalog.read('output/feature_geometry_study_20260914/fits/1981/seed1/full/fit.json')
        freq=np.fromfile(catalog.path(f"output/feature_geometry_study_20260914/directions/seed1_g{meta['bandwidth_index']}.bin"),dtype='<f8').reshape(115,512)
        extras=capacity_map(parent.model,freq)
        states=BoundState(tuple(str((pd.Timestamp(date)-pd.offsets.MonthEnd(i)).date()) for i in range(13)),
            np.zeros((13,1)),'ARTIFICIAL_ZERO_STATE',('value_dispersion',))
        result=designs(basis,bound,parent.model,states,states.book_identity,extras)
        direct=basis.operators([parent.model.feature_map.signals(z,v,b.innovation,parent.model.gains)
            for z,v,b in zip(sample.market.characteristics,sample.market.volatility,bound)])
        for a,b,c,size in zip(result['base'],direct,result['C4'],result['SIZE4']):
            np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-10)
            np.testing.assert_allclose(c[:,:513],b,atol=1e-11,rtol=1e-10)
            np.testing.assert_array_equal(c[:,513:],np.zeros_like(c[:,513:]))
            np.testing.assert_array_equal(size,np.zeros_like(size))
        source.unchanged();history.files.unchanged()
        write_json(out/'result.json',dict(status='PASS',actual_parent_months=2,actual_parent_seeds=3,results=errors,
            saved_model_roundtrip=True,saved_path_restart=True,independent_accounting=True,
            one_actual_training_slice_zero_state_operator=True,zero_conditional_columns=512,
            historical_E4_fit=False,raw_E4_state_constructed=False,complete_135_refit_neutrality=False,
            input_sha256={**catalog.consumed,**history.files.consumed},elapsed_seconds=time.monotonic()-started))
        print('PASS: production neutrality, independent accounting, zero-state operator and restart.',flush=True)


if __name__=='__main__':main()
