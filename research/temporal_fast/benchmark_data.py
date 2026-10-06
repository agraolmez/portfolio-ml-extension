"""Reuse two verified engineering samples without reopening a financial study."""
import json
from pathlib import Path
from feature_geometry.history import sha
from temporal_numerics.spill import OverflowStore
from temporal_study.history import StudyHistory


def accepted_samples(root, preflight_receipt):
    root = Path(root)
    folder = root/'output/extension3_storage_engineering_20260919/real_array_check_v2'
    receipt = json.loads((folder/'storage_check.json').read_text())
    if receipt['status'] != 'PASS' or not receipt['exact_roundtrip']:
        raise ValueError('Verified real-array evidence is required')
    config_path = root/'config/extension3_study_v3_proposed.json'
    if sha(config_path) != receipt['config_sha256']:
        raise ValueError('Storage configuration changed')
    for name, expected in receipt['code_sha256'].items():
        if sha(root/name) != expected: raise ValueError('Measured storage source changed')
    config = json.loads(config_path.read_text())
    history = StudyHistory(root, preflight_receipt, weights=config['reference_weights'])
    def verify(name, expected):
        history.files.path(name)
        if history.files.consumed[name] != expected: raise ValueError('Accepted input changed')
    store = OverflowStore(folder/'operators', source_fingerprint=history.source_fingerprint,
        reference_fingerprint=history.reference_fingerprint, study_lock_sha256=sha(config_path),
        maximum_bytes=512*2**20, verify_input=verify)
    samples = [store.get(day, '2024-12-31') for day in ('2000-03-31','2024-11-30')]
    if any(x is None for x in samples): raise ValueError('Missing accepted benchmark sample')
    return samples, history, config
