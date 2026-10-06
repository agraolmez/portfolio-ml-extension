"""Acceptance gate for the next financial experiment."""
import json
from .io import ROOT,sha


def verify_baseline(root=ROOT):
    path=root/'output/step4_20260911/acceptance_manifest.json'
    if not path.is_file():raise ValueError('Completed Step 4 acceptance is required')
    a=json.loads(path.read_text())
    if a.get('status')!='ACCEPTED_BASELINE_MACHINERY' or a.get('step4_complete') is not True or not a.get('gates') or not all(g.get('passed') is True for g in a['gates']):
        raise ValueError('Completed, passing Step 4 acceptance is required')
    if not a.get('artifact_sha256'):raise ValueError('Accepted artifact inventory is required')
    for rel,digest in a['artifact_sha256'].items():
        p=root/rel
        if not p.is_file() or sha(p)!=digest:raise ValueError(f'Stale/missing accepted baseline artifact: {rel}')
    return {'acceptance_sha256':sha(path),'artifacts_verified':len(a['artifact_sha256']),'status':a['status']}
