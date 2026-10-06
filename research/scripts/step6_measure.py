"""Measure one Step 6 job and enforce the approved memory/disk boundaries."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/step6_20260911'


def storage():
    # Count physical files once; risk symlinks are retained dependencies, not copies.
    seen, total = set(), 0
    for name in ['data', 'output', 'env']:
        for path in (ROOT / name).rglob('*'):
            if path.is_file():
                s = path.stat(); key = (s.st_dev, s.st_ino)
                if key not in seen:
                    seen.add(key); total += s.st_size
    return total / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'export', 'reference_fit', 'reference_path', 'review'])
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    limits = json.loads((ROOT / 'config/protocol_v1.json').read_text())['resources']
    commands = {'prepare': ['-m', 'baseline.full', 'prepare'], 'run': ['-m', 'baseline.full', 'run'],
        'export': ['scripts/step6_export_reference.py'], 'review': ['scripts/step6_review.py']}
    command = ([str(ROOT / 'env/step3/bin/python'), *commands[args.stage]] if args.stage in commands else
        ['/usr/local/bin/Rscript', str(ROOT / f'scripts/step6_{args.stage}.R'), str(ROOT)])
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'{args.stage}_attempt{args.attempt}'
    record_path = OUT / f'{stem}_resources.json'
    if record_path.exists() or (OUT / f'{stem}.log').exists():
        raise ValueError('Measured attempt already exists; preserve it and choose a new attempt')
    lock_path = OUT / 'heavy_job.lock'
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
    before = storage(); free = shutil.disk_usage(ROOT).free / 2**30
    if before > limits['new_storage_gib'] or free < limits['minimum_free_disk_gib']:
        lock_path.unlink(); raise RuntimeError('Pre-run storage boundary')
    env = os.environ.copy(); env.update(VECLIB_MAXIMUM_THREADS='4', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    peak, samples, max_storage, min_free = 0., 0, before, free
    start = time.monotonic(); last_disk = start; aborted = None
    started = datetime.now(timezone.utc).isoformat()
    try:
        with (OUT / f'{stem}.log').open('w') as log:
            proc = subprocess.Popen(['/usr/bin/time', '-l', '-o', str(OUT / f'{stem}_system_time.txt'), *command],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                rows = subprocess.check_output(['ps', '-axo', 'pgid=,rss='], text=True)
                rss = sum(int(row.split()[1]) for row in rows.splitlines()
                    if len(row.split()) == 2 and int(row.split()[0]) == proc.pid) / 2**20
                peak = max(peak, rss + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
                samples += 1
                if time.monotonic() - last_disk >= 10:
                    min_free = min(min_free, shutil.disk_usage(ROOT).free / 2**30)
                    max_storage = max(max_storage, storage()); last_disk = time.monotonic()
                if peak > limits['aggregate_memory_gib'] or min_free < limits['minimum_free_disk_gib'] or max_storage > limits['new_storage_gib']:
                    aborted = 'Configured resource boundary exceeded'
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(.25)
            code = proc.wait()
    finally:
        lock_path.unlink(missing_ok=True)
    after = storage(); max_storage = max(max_storage, after)
    min_free = min(min_free, shutil.disk_usage(ROOT).free / 2**30)
    system = (OUT / f'{stem}_system_time.txt').read_text()
    os_peak = next((int(line.split()[0]) / 2**30 for line in system.splitlines() if 'maximum resident set size' in line), 0.)
    record = {'stage': args.stage, 'attempt': args.attempt, 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic()-start,
        'exit_code': code, 'command': command, 'aborted': aborted, 'thread_limit': 4,
        'single_heavy_job_lock': True, 'sample_interval_seconds': .25, 'memory_samples': samples,
        'sampled_process_group_plus_supervisor_peak_gib': peak, 'os_peak_main_job_gib': os_peak,
        'fresh_storage_before_gib': before, 'fresh_storage_after_gib': after,
        'maximum_fresh_storage_gib': max_storage, 'minimum_free_disk_gib': min_free,
        'storage_scope': 'Unique files in fresh data, output and env; recovery archive excluded', 'limits': limits}
    record['within_limits'] = not aborted and max(peak, os_peak) <= limits['aggregate_memory_gib'] and max_storage <= limits['new_storage_gib'] and min_free >= limits['minimum_free_disk_gib']
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    if code or not record['within_limits']: raise SystemExit(1)


if __name__ == '__main__': main()
