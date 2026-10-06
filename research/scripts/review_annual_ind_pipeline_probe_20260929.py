"""Independent dense R solves and state reconstruction for the two bounded probes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from feature_geometry.history import sha
from extension4.storage import verify_stage, write_json
from extension4.state import StateBook
from extension4.verification import verify_states
from extension4.runtime import prior_seconds
from temporal_extension.history import RAW, RANKS


def main():
    out=ROOT/'output/annual_ind_pipeline_review_20260929'
    out.mkdir()
    started=time.monotonic()
    results=[]
    for index in (50,0):
        folder=ROOT/f'output/annual_ind_fresh_probe_pipeline_r{index}_20260929'
        record=json.loads((folder/'three_layer_result.json').read_text())
        lock=json.loads((folder/'lock.json').read_text())
        for key in ('code_sha256','protected_sha256'):
            for rel,h in lock[key].items():
                if sha(ROOT/rel)!=h: raise ValueError('Changed dependency: '+rel)
        for stage in ('fresh_parent','fresh_temporal','fresh_conditional'):
            verify_stage(folder/stage)
        book=StateBook(pd.read_parquet(folder/'fresh_state_history.parquet'),
                       json.loads((folder/'fresh_state_metadata.json').read_text()))
        cutoff=book.metadata['last'];filters=[('eom','<=',pd.Timestamp(cutoff))]
        raw=pd.read_parquet(ROOT/RAW,columns=['id','eom','be_me','me'],filters=filters)
        eligible=pd.read_parquet(ROOT/RANKS,columns=['id','eom','valid'],filters=filters)
        state_result=verify_states(book,raw,eligible)
        with np.load(folder/'fresh_conditional/moments_and_coefficients.npz',allow_pickle=False) as saved:
            matrix=saved['curvature'];score=saved['score'];p=(len(score)+1)//2
            transfer=out/f'r{index}';transfer.mkdir()
            np.asarray(matrix.T,dtype='<f8').tofile(transfer/'curvature.bin')
            np.asarray(score,dtype='<f8').tofile(transfer/'score.bin')
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
            command=['/usr/local/bin/Rscript',str(ROOT/'scripts/annual_ind_probe_solve_review.R'),
                     str(transfer),str(len(score)),str(p),str(record['ridge'])]
            with (transfer/'R.log').open('x') as stream:
                subprocess.run(command,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=60)
            other=np.loadtxt(transfer/'R_coefficients.csv',delimiter=',')
            expected=np.column_stack([saved[str(v)] for v in (100.,10.,1.)])
            np.testing.assert_allclose(other,expected,atol=1e-11,rtol=1e-10)
            error=float(np.max(abs(other-expected)))
        telemetry=[json.loads(line) for pth in (folder/'attempts').glob('*_resources.jsonl') for line in pth.read_text().splitlines()]
        if any(r['physical_footprint_gib']>16 or r['system_available_gib']<4 or r['swap_growth_bytes']>0 for r in telemetry):
            raise ValueError('Recorded resource safeguard failure')
        results.append(dict(ridge_index=index,independent_R_max_error=error,state_verification=state_result,
            temporal_converged=record['temporal_converged'],neutral_gates_passed=record['neutral_gates_passed'],
            peak_memory_gib=max(r['physical_footprint_gib'] for r in telemetry),
            minimum_available_gib=min(r['system_available_gib'] for r in telemetry),
            active_seconds=prior_seconds(folder)[0]))
    write_json(out/'review.json',dict(status='PASS_BOUNDED_FIT_INTEGRATION_ONLY',results=results,
        review_seconds=time.monotonic()-started,new_portfolio_paths=0,full_study_started=False,
        complete_pipeline_pilot=False,maximum_history_tested=False))
    print(json.dumps(dict(status='PASS_BOUNDED_FIT_INTEGRATION_ONLY',results=results)))


if __name__=='__main__':main()
