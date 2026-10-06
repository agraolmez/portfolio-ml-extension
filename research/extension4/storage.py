"""Immutable stage directories, complete inventories and explicit failed attempts."""
import json
import os
from pathlib import Path
import uuid
from feature_geometry.history import sha


def write_json(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_stage(folder, context=None):
    folder = Path(folder)
    record = json.loads((folder/'commit.json').read_text())
    if record.get('status') != 'COMPLETE' or (context is not None and record['context'] != context):
        raise ValueError('Incomplete or incompatible stage: '+str(folder))
    actual_files = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'commit.json'}
    if actual_files != set(record['artifact_sha256']):
        raise ValueError('Stage inventory mismatch')
    for rel, digest in record['artifact_sha256'].items():
        p = (folder/rel).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p) != digest:
            raise ValueError('Stage hash mismatch: '+rel)
    return record


def commit_stage(folder, context, writer):
    folder = Path(folder)
    if folder.exists():
        verify_stage(folder, context)
        return False
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = folder.with_name('.'+folder.name+'.attempt-'+uuid.uuid4().hex)
    temporary.mkdir()
    # Failure preserves this attempt for inspection. Never delete a prior result.
    writer(temporary)
    inventory = {str(p.relative_to(temporary)): sha(p) for p in temporary.rglob('*') if p.is_file()}
    if not inventory: raise ValueError('Cannot commit an empty stage')
    for p in temporary.rglob('*'):
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    write_json(temporary/'commit.json', dict(status='COMPLETE', context=context, artifact_sha256=inventory))
    verify_stage(temporary, context)
    if folder.exists(): raise FileExistsError('Concurrent stage creation; preserved '+str(temporary))
    temporary.rename(folder)
    fd = os.open(folder.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return True
