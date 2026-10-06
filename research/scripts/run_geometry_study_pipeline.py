"""One-time detached study runner with a process lock and durable stage receipts."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/feature_geometry_study_20260914'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def now():return datetime.now(timezone.utc).isoformat()

def run(attempt):
    if not attempt or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in attempt):
        raise ValueError('Attempt must be a simple unique identifier')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(signum,frame):raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    lock=(OUT/'pipeline.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another study pipeline or its child still holds the execution lock')
    folder=OUT/'attempts'/attempt;folder.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(folder/'matplotlib_cache'))
    files=[Path(__file__),ROOT/'scripts/feature_geometry_study.py',ROOT/'scripts/verify_geometry_study.py',ROOT/'scripts/geometry_study_handoff.py',ROOT/'scripts/geometry_report_preview.cjs',ROOT/'geometry_study/analyze.py',ROOT/'geometry_study/report.py',OUT/'plan_lock.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    launch=dict(attempt=attempt,pid=os.getpid(),parent_pid=os.getppid(),started_at_utc=now(),code_sha256=hashes,numerical_threads=4,veclib_threads_per_worker=1,one_time_execution=True,scheduled_automation=False)
    (folder/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    state=dict(launch,status='STARTING',stage='preflight')
    def save(**values):
        state.update(values,updated_at_utc=now());temp=folder/'state.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');os.replace(temp,folder/'state.json')
        print(json.dumps({k:state[k] for k in ['status','stage','updated_at_utc']},sort_keys=True),flush=True)
    child=None
    try:
        sys.path.insert(0,str(ROOT))
        from baseline.full import preflight
        from geometry_study.run import locked
        locked();fresh,_=preflight()
        if fresh!=json.loads((OUT/'financial_preflight.json').read_text()):raise ValueError('Financial inputs differ from the locked study')
        (folder/'financial_preflight.json').write_text(json.dumps(fresh,indent=2)+'\n')
        stages=[('fits_and_paths',[sys.executable,'-u','scripts/feature_geometry_study.py','all']),('analysis',[sys.executable,'-u','-m','geometry_study.analyze']),('report_and_verification',[sys.executable,'-u','-m','geometry_study.report'])]
        for stage,command in stages:
            for path,digest in hashes.items():
                if sha(ROOT/path)!=digest:raise ValueError('Pipeline source changed before stage '+stage+': '+path)
            save(status='RUNNING',stage=stage)
            with (folder/f'{stage}.log').open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                save(child_pid=child.pid)
                code=child.wait();child=None
            if code!=0:raise RuntimeError(f'{stage} exited with status {code}; preserve logs and completed outputs before resuming')
            (folder/f'{stage}_complete.json').write_text(json.dumps(dict(stage=stage,status='COMPLETE',at_utc=now()),indent=2)+'\n')
        save(status='COMPUTATION_COMPLETE_FINAL_VISUAL_REVIEW_PENDING',stage='complete',child_pid=None)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:pass
        save(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args();run(args.attempt)
