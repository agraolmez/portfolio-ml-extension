"""Compact immutable numerical checkpoints, with shared input references.

Each bundle has a canonical context and a SHA-256-verified NPZ payload. Entries
are append-only, transactionally committed, and bound to one run identity.
This does not certify scientific chronology; the caller validates that context.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class CheckpointStore:
    def __init__(self, path, *, run_identity):
        if not isinstance(run_identity, str) or not run_identity:
            raise ValueError('Explicit run identity required')
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS run (
            singleton INTEGER PRIMARY KEY CHECK (singleton=1), identity TEXT NOT NULL)''')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS bundles (
            key TEXT PRIMARY KEY, context TEXT NOT NULL, payload BLOB NOT NULL,
            digest TEXT NOT NULL)''')
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO run VALUES (1,?)', (run_identity,))
            rows = self.connection.execute('SELECT identity FROM run').fetchall()
        if rows != [(run_identity,)]:
            self.close()
            raise ValueError('Checkpoint store belongs to another run')
        self.run_identity = run_identity

    def put(self, key, context, arrays):
        if not isinstance(key, str) or not key or not isinstance(arrays, dict) or not arrays:
            raise ValueError('Explicit key and nonempty numerical arrays required')
        if any(not isinstance(k, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k)
               or k == 'file' for k in arrays):
            raise ValueError('Array names must be identifiers other than file')
        normalized = {}
        for name, value in sorted(arrays.items()):
            array = np.asarray(value)
            if (not isinstance(name, str) or not name or name == 'file'
                    or array.dtype.kind not in 'bifu' or not np.isfinite(array).all()):
                raise ValueError('Finite numeric arrays with explicit names required')
            normalized[name] = np.array(array, copy=True, order='C')
        metadata = canonical(context)
        stream = io.BytesIO()
        np.savez_compressed(stream, **normalized)
        payload = stream.getvalue()
        digest = hashlib.sha256(self.run_identity.encode()+b'\0'+metadata.encode()+b'\0'+payload).hexdigest()
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            old = self.connection.execute('SELECT context,payload,digest FROM bundles WHERE key=?', (key,)).fetchone()
            if old is not None:
                # Verify old content before treating a duplicate as a resume.
                previous = self.get(key, context=context)
                if (set(previous) != set(normalized) or any(
                        previous[k].dtype != normalized[k].dtype or
                        previous[k].shape != normalized[k].shape or
                        not np.array_equal(previous[k], normalized[k]) for k in normalized)):
                    raise ValueError('Refusing to replace an immutable checkpoint')
                return old[2]
            self.connection.execute('INSERT INTO bundles VALUES (?,?,?,?)', (key, metadata, payload, digest))
        return digest

    def get(self, key, *, context=None):
        row = self.connection.execute('SELECT context,payload,digest FROM bundles WHERE key=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        metadata, payload, digest = row
        if hashlib.sha256(self.run_identity.encode()+b'\0'+metadata.encode()+b'\0'+payload).hexdigest() != digest:
            raise ValueError('Checkpoint hash mismatch')
        if context is not None and metadata != canonical(context):
            raise ValueError('Checkpoint context differs from requested source/model/chronology')
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            result = {k:saved[k].copy() for k in saved.files}
        if any(v.dtype.kind not in 'bifu' or not np.isfinite(v).all() for v in result.values()):
            raise ValueError('Invalid checkpoint numerical payload')
        return result

    def inventory(self):
        rows = self.connection.execute('SELECT key,context,digest,length(payload) FROM bundles ORDER BY key').fetchall()
        for key, _, _, _ in rows:
            self.get(key)
        return [dict(key=key, context=json.loads(context), sha256=digest, compressed_bytes=size)
                for key, context, digest, size in rows]

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
