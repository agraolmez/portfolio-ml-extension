"""Two fixed integration probes, sharing the existing cumulative two-hour cap."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/annual_ind_pipeline_batch_20260929'


def save(name, value):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous batch; use a separately reviewed recovery version')
    OUT.mkdir()
    script = ROOT/'scripts/annual_ind_pipeline_probe_20260929.py'
    identity = hashlib.sha256(script.read_bytes()).hexdigest()
    results = []
    for ridge_index in (50, 0):
        status = dict(state='RUNNING_BOUNDED_THREE_LAYER_FIT', ridge_index=ridge_index,
            completed=len(results), planned=2, at_utc=datetime.now(timezone.utc).isoformat(),
            AI_calls=False, full_study_dispatched=False, script_sha256=identity)
        save('status.json', status)
        if hashlib.sha256(script.read_bytes()).hexdigest() != identity:
            raise ValueError('Probe source changed during batch')
        folder = ROOT/f'output/annual_ind_fresh_probe_pipeline_r{ridge_index}_20260929'
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
        with (OUT/f'r{ridge_index}.log').open('x') as log:
            result = subprocess.run([str(ROOT/'env/step3/bin/python'), '-B', str(script),
                '--output', str(folder), '--width', '512', '--workers', '8',
                '--ridge-index', str(ridge_index)], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save('status.json', dict(status, state='STOPPED_FOR_REVIEW', exit_code=result.returncode,
                failed_output=str(folder.relative_to(ROOT))))
            return 1
        record = json.loads((folder/'three_layer_result.json').read_text())
        record['output'] = str(folder.relative_to(ROOT))
        results.append(record)
        save('completed_results.json', results)
    save('status.json', dict(state='THREE_LAYER_FIT_PROBES_COMPLETE_NO_HISTORICAL_RUN',
        completed=2, planned=2, at_utc=datetime.now(timezone.utc).isoformat(), AI_calls=False,
        full_study_dispatched=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if OUT.exists():
            save('status.json', dict(state='STOPPED_FOR_REVIEW', error=str(error),
                AI_calls=False, full_study_dispatched=False))
        raise
