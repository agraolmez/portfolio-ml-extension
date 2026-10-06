"""Finite local feasibility batch; no AI/API calls and no full-study launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/annual_ind_thread_benchmarks_20260929'


def save(name, body):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def main():
    if OUT.exists():raise FileExistsError('Existing batch is preserved; inspect its status before a new attempt')
    OUT.mkdir()
    code=ROOT/'scripts/annual_ind_fit_benchmark_v2_20260929.py'
    digest=hashlib.sha256(code.read_bytes()).hexdigest()
    # At most eight monthly workers; one fitting process at a time, original memory safeguards.
    specs=[(512,w) for w in (1,2,4,8)]
    results=[]
    for p,w in specs:
        status=dict(state='RUNNING_NUMERICAL_BENCHMARKS_ONLY',width=p,workers=w,
            completed=len(results),planned=len(specs),updated_at_utc=datetime.now(timezone.utc).isoformat(),
            full_study_dispatched=False,AI_calls=False,benchmark_script_sha256=digest)
        save('status.json',status)
        if hashlib.sha256(code.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Benchmark source changed while batch was running')
        folder=ROOT/f'output/annual_ind_fresh_probe_threads_p{p}_w{w}_20260929'
        command=[str(ROOT/'env/step3/bin/python'),'-B',str(code),'--output',str(folder),
                 '--width',str(p),'--workers',str(w)]
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                                       VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
        with (OUT/f'p{p}_w{w}.log').open('x') as log:
            run=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:
            save('status.json',dict(status,state='STOPPED_FOR_REVIEW',exit_code=run.returncode,
                failed_output=str(folder.relative_to(ROOT)),completed=len(results)))
            return 1
        result=json.loads((folder/'probe_result.json').read_text())
        result['output']=str(folder.relative_to(ROOT));results.append(result)
        if w!=1:
            other=ROOT/f'output/annual_ind_fresh_probe_threads_p{p}_w1_20260929/fresh_parent/model.npz'
            with np.load(other,allow_pickle=False) as a, np.load(folder/'fresh_parent/model.npz',allow_pickle=False) as b:
                for key in ('theta','beta','frequencies'):
                    np.testing.assert_allclose(a[key],b[key],atol=1e-11,rtol=1e-10)
        save('completed_results.json',results)
    fastest={p:min((r for r in results if r['p']==p),key=lambda r:r['geometry_fit_seconds']) for p in (512,)}
    scenario=sum(r['this_width_parent_hours_at_probe_rate'] for r in fastest.values())
    save('summary.json',dict(status='BOUNDED_BENCHMARKS_COMPLETE_NOT_FULL_STUDY',
        results=results,worker_equivalence_passed=True,
        width_512_parent_hours_at_this_early_probe_rate=scenario,
        all_widths_measured=False,
        temporal_and_conditional_fits_included=False,full_run_ETA_established=False,
        conservative_72_hour_full_launch_gate_passed=False,
        reason='Early-coordinate throughput scenarios are not a conservative complete-pipeline projection; no automatic full launch is authorized by this benchmark.',
        AI_calls=False,full_study_dispatched=False))
    save('status.json',dict(state='BENCHMARKS_COMPLETE_FULL_STUDY_NOT_STARTED',completed=len(results),
        planned=len(specs),updated_at_utc=datetime.now(timezone.utc).isoformat(),AI_calls=False,
        full_study_dispatched=False))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if OUT.exists():save('status.json',dict(state='STOPPED_FOR_REVIEW',error=str(exc),
                                              AI_calls=False,full_study_dispatched=False))
        raise
