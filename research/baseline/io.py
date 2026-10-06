"""Fail-closed input and artifact contracts for the baseline machinery."""
import hashlib
import json
import os
import resource
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/prepared/step3_author_v3'
SOURCE=ROOT/'output/step3_author_conventions_20260911'
OUT=ROOT/'output/step4_20260911'
PREP=ROOT/'data/prepared/step4_v1'
SETTINGS=json.loads((ROOT/'output/step2_20260910/source_settings.json').read_text())
PROTOCOL=json.loads((ROOT/'config/protocol_v1.json').read_text())
FEATURES=SETTINGS['features']
for p in [OUT,PREP,OUT/'reference',PREP/'risk',PREP/'matrices']:
    p.mkdir(parents=True,exist_ok=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if v is pd.NaT:return None
        return v
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(clean(value),indent=2,default=str,allow_nan=False)+'\n')


def verify_data(root=ROOT):
    src=root/'output/step3_author_conventions_20260911'
    m=json.loads((src/'acceptance_manifest.json').read_text())
    status=json.loads((root/'data/prepared/step3_author_v3/STATUS.json').read_text())
    expected='ACCEPTED_AUTHOR_CONVENTION_RECONSTRUCTION'
    if m.get('status')!=expected or status.get('status')!=expected or not m.get('step3_complete') or not all(g.get('passed') is True for g in m['gates']):
        raise ValueError('Accepted Step 3 author-convention bundle required')
    checksums=json.loads((src/'bundle_checksums.json').read_text())
    for rel,digest in checksums.items():
        p=root/rel
        if not p.is_file() or sha(p)!=digest:raise ValueError(f'Stale/missing accepted input or preparation dependency: {rel}')
    return {'status':expected,'files_verified':len(checksums),'manifest_sha256':sha(src/'acceptance_manifest.json'),'checksums_sha256':sha(src/'bundle_checksums.json')}


def guard():
    limits=PROTOCOL['resources']
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if os.uname().sysname=='Darwin' else 2**20)
    free=shutil.disk_usage(ROOT).free/2**30
    paths=[ROOT/'data',ROOT/'env/step3',*list((ROOT/'output').glob('step[34]_*'))]
    used=sum(p.stat().st_size for parent in paths if parent.is_dir() for p in parent.rglob('*') if p.is_file())/2**30
    if rss>limits['aggregate_memory_gib'] or used>limits['new_storage_gib'] or free<limits['minimum_free_disk_gib']:
        raise MemoryError(f'Resource boundary: RSS={rss:.2f}, storage={used:.2f}, free={free:.2f} GiB')
    return {'peak_process_gib':rss,'fresh_storage_gib':used,'free_disk_gib':free}


def finite(*arrays):
    if any(not np.isfinite(np.asarray(v,dtype=float)).all() for v in arrays):raise ValueError('Nonfinite numerical input')


def write_matrix(path,x):
    a=np.asarray(x,dtype='<f8',order='C');a.tofile(path)


def datekey(d):return pd.Timestamp(d).strftime('%Y-%m-%d')
