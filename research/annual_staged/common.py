"""Immutable accepted sources, shared constants and durable status updates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/annual_ind_staged_20261001'
CONFIG = ROOT/'config/annual_ind_staged_20261001.json'
RHOS = (.1, .01, .001)
MULTIPLIERS = (100., 10., 1.)
E2 = 'output/feature_geometry_study_20260914'
E3 = 'output/extension3_full_study_20260919'
E4 = 'output/extension4_full_study_20260924'


def now(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def state(**values):
    p = ROOT/'config/annual_ind_execution_state_20260929.json'
    obj = read(p); obj.update(values); atomic(p, obj)


def candidate(rho, multiplier): return f'rho{rho:g}_m{multiplier:g}'


ORDER = tuple(candidate(r, m) for r in RHOS for m in MULTIPLIERS)


class Catalog:
    """Check accepted hashes on first access and file identity on every reuse."""
    def __init__(self):
        self.inventory = {}; self.consumed = {}; self.stamps = {}
        for rel in (E2+'/acceptance_manifest.json', E3+'/acceptance_manifest.json',
                    E4+'/acceptance_manifest.json', 'output/step9_20260913/acceptance_manifest.json',
                    'output/step6_20260911/preparation_manifest.json',
                    'output/step4_20260911/acceptance_manifest.json',
                    'output/industry_projection_screen_20260928/acceptance_manifest.json'):
            manifest = read(ROOT/rel); self.consumed[rel] = sha(ROOT/rel)
            for name, digest in manifest['artifact_sha256'].items():
                self.inventory.setdefault(name, set()).add(digest)

    def path(self, rel):
        p = ROOT/rel; stamp = (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
        if rel not in self.stamps:
            digest = sha(p)
            canonical=str(p.resolve().relative_to(ROOT))
            if digest not in self.inventory.get(rel, set()) | self.inventory.get(canonical, set()):
                raise ValueError('Missing or changed accepted source: '+rel)
            self.consumed[rel] = digest; self.stamps[rel] = stamp
        elif self.stamps[rel] != stamp:
            raise ValueError('Accepted input changed during use: '+rel)
        return p

    def read(self, rel): return read(self.path(rel))

    def unchanged(self):
        for rel in list(self.stamps): self.path(rel)


def finite_equal(left, right, *, atol=1e-11, rtol=1e-10):
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite numerical comparison')
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol)
