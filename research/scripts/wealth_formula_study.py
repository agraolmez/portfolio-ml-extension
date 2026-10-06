"""One-change wealth recurrence diagnostic. Accepted baseline files are read-only.

Run with the project's Python: scripts/wealth_formula_study.py supervise
Stages and monthly checkpoints are hash validated on every resume.
"""
import argparse
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# Hashing the immutable historical evidence should not evict working memory.
_open = io.open
def uncached_open(*args, **kwargs):
    handle = _open(*args, **kwargs)
    if 'b' in getattr(handle, 'mode', '') and handle.readable():
        fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
    return handle
io.open = uncached_open

import numpy as np
import pandas as pd
from baseline import engine as machinery
from baseline.full import DATES, MOMENT_DATES, PROTECTED, dependencies
from baseline.io import DATA, FEATURES, SETTINGS, PROTOCOL, sha, save_json

OUT = ROOT / 'output/wealth_formula_study_20260928'
PREP = OUT / 'prepared'
REF = OUT / 'reference'
OLD = ROOT / 'data/prepared/step6_v1'
OLD_REF = ROOT / 'output/step6_20260911/reference'
OLD_RUN = ROOT / 'output/step6_baseline_1981_2020'
SAMPLES = ['1980-12-31', '1990-12-31', '2000-12-31', '2020-11-30']
STAGES = ['preflight', 'neutral', 'moments', 'fit', 'path', 'export', 'r_fit', 'r_path', 'review']

def event(name, **fields):
    row = dict(at_utc=pd.Timestamp.now(tz='UTC').isoformat(), event=name, **fields)
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, default=str) + '\n'); f.flush(); os.fsync(f.fileno())
    save_json(OUT / 'status.json', row)
    print(json.dumps(row, default=str), flush=True)

def inventory(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}

def verify(receipt):
    for name, digest in receipt.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed checkpoint/dependency: ' + name)

def complete(stage, paths):
    save_json(OUT / f'checkpoints/{stage}.json', dict(stage=stage, artifact_sha256=inventory(paths)))
    event('stage_complete', stage=stage)

def wealth():
    w = pd.read_parquet(DATA / 'market_wealth.parquet')
    w = w[w.eom <= pd.Timestamp('2020-12-31')].copy().sort_values('eom')
    assert w.eom.is_unique and list(w.eom) == list(pd.date_range(w.eom.min(), w.eom.max(), freq='ME'))
    r = w.market_total_return.to_numpy(); values = np.empty(len(w)); values[-1] = 1e10
    assert np.isfinite(r).all() and ((r > -1) & (r < 1)).all()
    for j in range(len(w)-1, 0, -1): values[j-1] = values[j]*(1-r[j])
    assert np.isfinite(values).all() and (values > 0).all()
    w['author_formula_wealth'] = values
    return w

def configure(neutral=False):
    prep = OUT / 'neutral_prepared' if neutral else PREP
    for p in [prep / 'matrices', REF]: p.mkdir(parents=True, exist_ok=True)
    if not (prep / 'risk').exists(): (prep / 'risk').symlink_to(OLD / 'risk', target_is_directory=True)
    machinery.PREP, machinery.OUT = prep, OUT
    # The separate supervisor owns resource limits; no change to protected protocol.
    machinery.guard = lambda: {'resource_guard': 'external supervisor, every two seconds'}
    e = machinery.Engine(end=DATES[-1], verify=False)
    w = wealth().set_index('eom')
    series = w.wealth_usd if neutral else w.author_formula_wealth
    assert e.data.eom.isin(series.index).all()
    e.data['wealth_usd'] = e.data.eom.map(series)
    for d, block in e.months.items(): block['wealth_usd'] = float(series.loc[d])
    start = json.loads((OUT / 'start.json').read_text())
    e.verification = {'checksums_sha256': hashlib.sha256((start['dependency_fingerprint'] + sha(OUT / 'wealth.csv') + str(neutral)).encode()).hexdigest()}
    return e

def preflight():
    receipt = dependencies()
    # Bind the complete accepted baseline preparation and its completed review.
    parent = ROOT / 'output/step6_20260911'
    for name in ['preparation_manifest.json', 'acceptance_manifest.json']:
        m = json.loads((parent / name).read_text())
        verify(m['artifact_sha256'])
    protected = inventory([ROOT / p for p in PROTECTED])
    sources = [ROOT / 'baseline' / p for p in ['engine.py', 'core.py', 'risk.py', 'io.py', 'full.py', 'wealth.py', 'reference_helpers.R']]
    sources += [ROOT / 'scripts' / p for p in ['wealth_formula_study.py', 'step6_reference_fit.R', 'step6_reference_path.R']]
    sources += [parent / 'preparation_manifest.json', parent / 'acceptance_manifest.json', ROOT / 'output/step2_20260910/source_settings.json']
    bound = inventory(sources)
    w = wealth(); w.to_csv(OUT / 'wealth.csv', index=False, float_format='%.17g')
    for gi in [0, 1]:
        REF.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD_REF / f'rff_g{gi}.bin', REF / f'rff_g{gi}.bin')
        assert sha(REF / f'rff_g{gi}.bin') == sha(OLD_REF / f'rff_g{gi}.bin')
    start = dict(status='DIAGNOSTIC_AUTHORIZED_NOT_A_SCIENTIFIC_AMENDMENT', dependencies=receipt,
        protected=protected, sources=bound, dependency_fingerprint=hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest(),
        only_changed_input='Backward wealth: W_previous = W_current * (1 - current market total return)',
        fixed='Same R1 data, market total returns, risk, seed 1, full original tuning grid, accounting, calendar and m formula; refit and recompute m',
        samples=SAMPLES, decisions=480, moment_months=661, source_utility_gamma=10,
        author_paper_utility_rounded=0.086, limitation='Conditional effect on current accepted inputs; authors frozen data unavailable')
    save_json(OUT / 'start.json', start)
    # Read the actual author definition, supplying exactly our audited total-return series.
    rcode = """root<-normalizePath(commandArgs(TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'));setDTthreads(1)
load_functions(file.path(author,'1 - Prepare Data.R'),'wealth_func')
out<-file.path(root,'output/wealth_formula_study_20260928');w<-fread(file.path(out,'wealth.csv'));w[,eom:=as.Date(eom)]
# Zero rf plus the already audited total return prevents changing return inputs.
answer<-wealth_func(1e10,as.Date('2020-12-31'),w[,.(eom_ret=eom,mkt_vw_exc=market_total_return)],w[,.(eom,rf=0)])
fwrite(answer,file.path(out,'r_wealth.csv'))
"""
    (OUT / 'source_wealth.R').write_text(rcode)
    subprocess.run(['/usr/local/bin/Rscript', str(OUT / 'source_wealth.R'), str(ROOT)], check=True)
    rw = pd.read_csv(OUT / 'r_wealth.csv', parse_dates=['eom'], float_precision='round_trip')
    joined = w.merge(rw[['eom','wealth']], on='eom', validate='one_to_one')
    assert len(joined) == len(w) and np.allclose(joined.author_formula_wealth, joined.wealth, rtol=1e-12, atol=0)
    assert (1e10*(1-.1)*(1-.2)) == 7.2e9
    zero = np.cumprod(1-np.zeros(12)); assert (zero==1).all()
    save_json(OUT / 'preflight.json', dict(source_R_wealth_matches=True, wealth_months=len(w), protected_unchanged=True,
        parent_verified=True, relative_error=float(abs(joined.author_formula_wealth/joined.wealth-1).max())))
    complete('preflight', [OUT / p for p in ['start.json','wealth.csv','r_wealth.csv','source_wealth.R','preflight.json']] + list(REF.glob('rff*')))

def neutral():
    e = configure(neutral=True); checks=[]
    for day in ['1965-11-30','1980-12-31','2020-11-30']:
        for gi in [0,1]:
            result=e.inputs(pd.Timestamp(day),gi)
            with np.load(OLD/f'moments/g{gi}/{day}.npz') as z:
                for key in ['ids','signal_t','r_tilde','denom']:
                    a,b=result[key],z[key]
                    ok=np.allclose(a,b,atol=1e-8,rtol=1e-6)
                    checks.append(dict(date=day,g=gi,field=key,passed=ok,max_abs=float(np.max(abs(a-b)))))
                    if not ok: raise ValueError(str(checks[-1]))
        event('neutral_date_passed',date=day)
    save_json(OUT/'neutral_checks.json',dict(passed=all(x['passed'] for x in checks),checks=checks))
    complete('neutral',[OUT/'neutral_checks.json',*list((OUT/'neutral_prepared/matrices').glob('*.npz'))])

def moments():
    e=configure(); checkpoint=OUT/'checkpoints/moments_progress.json'
    done=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    verify(done)
    for gi in [0,1]:
        folder=PREP/f'moments/g{gi}';folder.mkdir(parents=True,exist_ok=True)
        for i,d in enumerate(MOMENT_DATES):
            p=folder/f'{d:%Y-%m-%d}.npz'; rel=str(p.relative_to(ROOT))
            if rel in done: continue
            result=e.inputs(d,gi)
            # These are precisely the fields consumed by fitting and execution.
            # Risk and cost components are not separately duplicated on disk.
            tmp=p.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**{k:result[k] for k in ['ids','signal_t','r_tilde','denom']})
            tmp.replace(p);done[rel]=sha(p)
            mp=PREP/f'matrices/{d:%Y-%m-%d}.npz';done[str(mp.relative_to(ROOT))]=sha(mp)
            save_json(checkpoint,done)
            if i%12==0 or i==len(MOMENT_DATES)-1: event('moments',bandwidth=gi,month=i+1,total=661,through=str(d.date()))
    complete('moments',[ROOT/p for p in done])

def fit():
    e=configure();e.fit(2020)
    complete('fit',list((PREP/'coefficients').rglob('*.npz'))+list(PREP.glob('*.parquet'))+[OUT/'fit_build.json',OUT/'coefficient_diagnostics.csv'])

def path():
    e=configure();aims=e.aims(DATES)
    assert not aims.duplicated(['eom','id']).any() and set(aims.eom)==set(DATES)
    for year,b in aims.groupby('fit_year'):
        assert b.selection_return_year.eq(year-1).all() and b[['g_index','p','lambda_index']].drop_duplicates().shape[0]==1
    m,p,x=e.run(DATES,aims,require_step4=False)
    for name,df in [('monthly',m),('positions',p),('aims',aims),('exits',x)]:df.to_parquet(OUT/f'{name}.parquet',index=False)
    complete('path',[OUT/f'{name}.parquet' for name in ['monthly','positions','aims','exits']])

def export():
    for gi in [0,1]:
        folder=REF/f'fit/g{gi}';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'denom.bin').open('wb') as a,(folder/'rtilde.bin').open('wb') as r:
            for d in MOMENT_DATES:
                with np.load(PREP/f'moments/g{gi}/{d:%Y-%m-%d}.npz') as z:
                    np.asarray(z['denom'],dtype='<f8').tofile(a);np.asarray(z['r_tilde'],dtype='<f8').tofile(r)
                    if d in DATES:
                        old=np.fromfile(OLD_REF/f'signals/g{gi}/{d:%Y-%m-%d}.bin',dtype='<f8').reshape(z['signal_t'].shape)
                        assert np.array_equal(old,z['signal_t'])
        save_json(folder/'dates.json',[str(d.date()) for d in MOMENT_DATES])
    # These inputs are mathematically independent of wealth, checked above for every signal.
    for name in ['risk','signals','rates.csv']:
        p=REF/name
        if not p.exists():p.symlink_to(OLD_REF/name,target_is_directory=(OLD_REF/name).is_dir())
    series=wealth().set_index('eom').author_formula_wealth
    with (REF/'data.csv').open('w') as dest:
        first=True
        for block in pd.read_csv(OLD_REF/'data.csv',chunksize=20000,float_precision='round_trip'):
            block['wealth_usd']=pd.to_datetime(block.eom).map(series)
            assert block.wealth_usd.notna().all()
            block.to_csv(dest,index=False,header=first,float_format='%.17g');first=False
    meta=json.loads((OLD_REF/'metadata.json').read_text());meta['sample_input_dates']=SAMPLES
    meta['source_function_checks']='All grid fits and holdings; 4 predeclared dates x 2 source input rebuilds'
    save_json(REF/'metadata.json',meta)
    for stage in ['fit','path']:
        text=(ROOT/f'scripts/step6_reference_{stage}.R').read_text().replace('output/step6_20260911/reference','output/wealth_formula_study_20260928/reference')
        text=text.replace("source(file.path(root,'baseline/reference_helpers.R'))","source(file.path(root,'baseline/reference_helpers.R'));setDTthreads(1)")
        if stage=='path':
            text=text.replace('  lower<-min(samples)','  if(length(samples)){\n  lower<-min(samples)').replace('  rm(inputs,sub);gc()','  rm(inputs,sub);gc()\n  }')
            text=text.replace('81 predeclared dates','4 predeclared dates')
        (OUT/f'reference_{stage}.R').write_text(text)
    complete('export',list((REF/'fit').rglob('*'))+[REF/p for p in ['metadata.json','data.csv','rates.csv']]+[OUT/'reference_fit.R',OUT/'reference_path.R'])

def r_fit():
    subprocess.run(['/usr/local/bin/Rscript',str(OUT/'reference_fit.R'),str(ROOT)],check=True)
    complete('r_fit',list((REF/'fit').rglob('r_coef*'))+[REF/p for p in ['r_validation.csv','r_annual_selection.csv','fit_status.json']])

def r_path():
    subprocess.run(['/usr/local/bin/Rscript',str(OUT/'reference_path.R'),str(ROOT)],check=True)
    complete('r_path',list((REF/'sample_inputs').glob('*'))+[REF/p for p in ['r_aims.csv','r_weights.csv','r_performance.csv','path_status.json','R_session.txt']])

def review():
    subprocess.run([sys.executable,str(ROOT/'scripts/wealth_formula_review.py')],check=True)
    complete('review',[OUT/'review.json',OUT/'comparison.csv',OUT/'independent_monthly.parquet'])

def pressure():
    level=ctypes.c_int();size=ctypes.c_size_t(ctypes.sizeof(level))
    result=ctypes.CDLL(None,use_errno=True).sysctlbyname(b'kern.memorystatus_vm_pressure_level',ctypes.byref(level),ctypes.byref(size),None,0)
    if result or level.value!=1:raise MemoryError(f'Native memory pressure is not normal: {level.value}, syscall {result}')

def supervise(stop_after):
    from extension4.runtime import Runtime
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    budget=dict(total_active_seconds=14400,max_physical_footprint_gib=16,min_system_headroom_gib=4,max_additional_disk_gib=20,min_free_disk_gib=50)
    pressure();runtime=Runtime(OUT,dict(compute_budget=budget),'wealth_diagnostic');child=None
    try:
        for stage in STAGES:
            cp=OUT/f'checkpoints/{stage}.json'
            if cp.exists():verify(json.loads(cp.read_text())['artifact_sha256']);event('stage_reused',stage=stage)
            else:
                if (OUT/'start.json').exists():
                    start=json.loads((OUT/'start.json').read_text());verify(start['protected']);verify(start['sources'])
                event('stage_started',stage=stage)
                with (OUT/f'{stage}.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        runtime.guard();pressure();time.sleep(2)
                    if child.returncode:raise RuntimeError(f'{stage} failed ({child.returncode}); inspect {stage}.log')
                child=None
                if not cp.exists():raise RuntimeError('Missing completed stage receipt: '+stage)
            if stage==stop_after:break
        runtime.close('COMPLETED_THROUGH_'+stop_after);event('supervisor_complete',through=stop_after)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            import signal
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        runtime.close('STOPPED',exc);event('stopped',error=str(exc),resume='env/step3/bin/python scripts/wealth_formula_study.py supervise')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['supervise']+STAGES)
    parser.add_argument('--stop-after',choices=STAGES,default='review');args=parser.parse_args()
    if args.stage=='supervise':supervise(args.stop_after)
    else:globals()[args.stage]()
