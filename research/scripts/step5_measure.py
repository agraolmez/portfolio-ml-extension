"""Run one pilot/review job with wall-time, process-group RSS and storage records."""
import argparse,json,os,resource,shutil,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/step5_20260911';OUT.mkdir(parents=True,exist_ok=True)


def storage():
    return sum(p.stat().st_size for folder in ['data','output','env'] for p in (ROOT/folder).rglob('*') if p.is_file())/2**30


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','reference']);args=parser.parse_args()
    protocol=json.loads((ROOT/'config/protocol_v1.json').read_text());limits=protocol['resources']
    command=[str(ROOT/'env/step3/bin/python'),'-m','baseline.pilot'] if args.stage=='pilot' else ['/usr/local/bin/Rscript',str(ROOT/'scripts/step5_reference.R'),str(ROOT)]
    output=OUT/f'{args.stage}_resources.json'
    if output.exists():raise ValueError('Existing measured execution must be reviewed before rerunning')
    before=storage();free_before=shutil.disk_usage(ROOT).free/2**30
    if before>limits['new_storage_gib'] or free_before<limits['minimum_free_disk_gib']:raise RuntimeError('Pre-run storage boundary')
    env=os.environ.copy();env.update(VECLIB_MAXIMUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    peak=0.;samples=0;minimum_free=free_before;aborted=None;maximum_storage=before
    start=time.monotonic();started=datetime.now(timezone.utc).isoformat();last_disk=start
    with (OUT/f'{args.stage}_run.log').open('w') as log:
        process=subprocess.Popen(['/usr/bin/time','-l','-o',str(OUT/f'{args.stage}_system_time.txt'),*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            listing=subprocess.check_output(['ps','-axo','pgid=,rss='],text=True)
            rss=sum(int(row.split()[1]) for row in listing.splitlines() if len(row.split())==2 and int(row.split()[0])==process.pid)/2**20
            supervisor=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
            peak=max(peak,rss+supervisor);samples+=1
            now=time.monotonic()
            if now-last_disk>10:
                minimum_free=min(minimum_free,shutil.disk_usage(ROOT).free/2**30)
                maximum_storage=max(maximum_storage,storage());last_disk=now
            if peak>limits['aggregate_memory_gib'] or minimum_free<limits['minimum_free_disk_gib'] or maximum_storage>limits['new_storage_gib']:
                aborted='Configured resource boundary exceeded';os.killpg(process.pid,signal.SIGTERM);break
            time.sleep(.25)
        code=process.wait()
    after=storage();minimum_free=min(minimum_free,shutil.disk_usage(ROOT).free/2**30);maximum_storage=max(maximum_storage,after)
    system=(OUT/f'{args.stage}_system_time.txt').read_text();os_peak=None
    for line in system.splitlines():
        if 'maximum resident set size' in line:os_peak=int(line.split()[0])/2**30
    record={'stage':args.stage,'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'wall_seconds':time.monotonic()-start,'sampled_process_group_plus_supervisor_peak_gib':peak,'os_peak_main_job_gib':os_peak,'memory_sample_interval_seconds':.25,'memory_samples':samples,'minimum_free_disk_gib':minimum_free,'fresh_storage_before_gib':before,'fresh_storage_after_gib':after,'maximum_fresh_storage_gib':maximum_storage,'storage_scope':'Complete fresh data, output and env folders; includes Step 5, excludes pre-reset archive','single_heavy_job':True,'thread_limit':4,'aborted':aborted,'limits':limits,'command':command}
    record['within_limits']=not aborted and code==0 and max(peak,os_peak or 0)<=limits['aggregate_memory_gib'] and minimum_free>=limits['minimum_free_disk_gib'] and maximum_storage<=limits['new_storage_gib']
    output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
    if not record['within_limits']:raise SystemExit(1)

if __name__=='__main__':main()
