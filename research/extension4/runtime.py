"""Continuous local CPU/memory/disk telemetry, awake-time budgets and restart ledger."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import time
import psutil
from temporal_fast_v4.memory import usage
from .storage import write_json


def now():return datetime.now(timezone.utc).isoformat()


def prior_seconds(folder):
    total=0.;by_phase={}
    for start in sorted((Path(folder)/'attempts').glob('*_start.json')):
        s=json.loads(start.read_text());end=start.with_name(start.name.replace('_start','_end'))
        if end.exists():seconds=json.loads(end.read_text())['active_seconds']
        else:
            latest=0.
            telemetry=start.with_name(start.name.replace('_start.json','_resources.jsonl'))
            if telemetry.exists():
                with telemetry.open() as source:
                    for line in source:
                        try:latest=max(latest,float(json.loads(line)['active_seconds']))
                        except (ValueError,KeyError):continue
            # An interrupted sampling interval is conservatively charged.
            seconds=latest+s['sampling_interval_seconds']
        total+=seconds;by_phase[s['phase']]=by_phase.get(s['phase'],0.)+seconds
    return total,by_phase


class Runtime:
    def __init__(self, folder, config, phase):
        if time.get_clock_info('monotonic').implementation!='mach_absolute_time()':
            raise RuntimeError('Approved local awake-time accounting requires the audited macOS clock')
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.budget=config['compute_budget'];self.phase=phase
        self.previous,self.previous_phases=prior_seconds(folder)
        attempts=self.folder/'attempts';attempts.mkdir(exist_ok=True)
        number=len(list(attempts.glob('*_start.json')))+1
        self.stem=attempts/f'{number:04d}';self.start=time.monotonic();self.process=psutil.Process()
        self.start_swap=psutil.swap_memory().used;self.error=None
        self._stop=threading.Event();self._lock=threading.Lock();self.last_disk=0.;self.disk_bytes=0
        write_json(Path(str(self.stem)+'_start.json'),dict(phase=phase,started_at_utc=now(),pid=os.getpid(),
            process_create_time=self.process.create_time(),prior_active_seconds=self.previous,
            clock='mach_absolute_time()',sampling_interval_seconds=2.,budget=self.budget))
        self.sample()
        self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start()

    def _watch(self):
        while not self._stop.wait(2.):
            try:self.sample()
            except Exception as error:self.error=error;return

    def sample(self):
        with self._lock:
            elapsed=time.monotonic()-self.start
            processes=[self.process]+self.process.children(recursive=True)
            memory=0;cpu=0.
            for process in processes:
                try:
                    info=usage(process.pid);memory+=max(info['physical_footprint_bytes'],info['resident_bytes'])
                    times=process.cpu_times();cpu+=times.user+times.system
                except (psutil.NoSuchProcess,ProcessLookupError):continue
            available=psutil.virtual_memory().available;swap=psutil.swap_memory().used
            if elapsed-self.last_disk>=10. or not self.disk_bytes:
                self.disk_bytes=sum(p.stat().st_size for p in self.folder.rglob('*') if p.is_file())
                self.last_disk=elapsed
            row=dict(at_utc=now(),phase=self.phase,active_seconds=elapsed,cumulative_active_seconds=self.previous+elapsed,
                     cpu_seconds=cpu,physical_footprint_gib=memory/2**30,system_available_gib=available/2**30,
                     swap_used_gib=swap/2**30,swap_growth_bytes=max(0,swap-self.start_swap),
                     free_disk_gib=shutil.disk_usage(self.folder).free/2**30,study_disk_gib=self.disk_bytes/2**30)
            with Path(str(self.stem)+'_resources.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            self.last_sample=row
            if self.previous+elapsed>self.budget['total_active_seconds']:raise TimeoutError('Approved cumulative active-time budget exhausted')
            if self.phase=='pilot' and self.previous_phases.get('pilot',0)+elapsed>self.budget['pilot_active_seconds']:raise TimeoutError('Approved cumulative pilot budget exhausted')
            if row['physical_footprint_gib']>self.budget['max_physical_footprint_gib']:raise MemoryError('Process tree exceeds approved memory budget')
            if row['system_available_gib']<self.budget['min_system_headroom_gib']:raise MemoryError('System headroom below approved reserve')
            if row['swap_growth_bytes']>0:raise MemoryError('System swap grew during this attempt; preserve checkpoint and review pressure')
            if row['free_disk_gib']<self.budget['min_free_disk_gib'] or row['study_disk_gib']>self.budget['max_additional_disk_gib']:raise OSError('Approved disk limit reached')
            return row

    def guard(self):
        if self.error is not None:raise self.error
        if time.monotonic()-self.start>self.budget['total_active_seconds']-self.previous:
            raise TimeoutError('Approved cumulative budget exhausted')

    def event(self,event,**values):
        self.guard()
        row=dict(at_utc=now(),event=event,phase=self.phase,active_seconds=time.monotonic()-self.start,**values)
        with (self.folder/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)

    def close(self,status,error=None):
        self._stop.set()
        if hasattr(self,'thread'):self.thread.join(timeout=5.)
        write_json(Path(str(self.stem)+'_end.json'),dict(status=status,ended_at_utc=now(),active_seconds=time.monotonic()-self.start,
                   error=None if error is None else str(error),last_sample=getattr(self,'last_sample',None)))
