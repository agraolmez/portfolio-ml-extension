#!/usr/bin/env python3
"""Bounded exact-storage/neutral-input check; no gain fit or portfolio path."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    import fcntl
    import json
    import shutil
    import time
    from datetime import datetime, timezone
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    from baseline.full import preflight
    from feature_geometry.history import sha
    from geometry_study.kernel import LeanBasis
    from temporal_extension.history import load_parent
    from temporal_extension.streaming import StreamingObjective
    from temporal_study.history import StudyHistory
    from temporal_numerics.spill import OverflowStore, payload_fingerprint
    from temporal_numerics.study import write_new
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():
        raise ValueError('A fresh project output folder is required')
    lock = (ROOT/'output/.extension3_heavy_job.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(parents=True)
    config_path = ROOT/'config/extension3_study_v3_proposed.json'
    config = json.loads(config_path.read_text())
    code = ['temporal_numerics/spill.py', 'temporal_storage/cache.py',
            'scripts/check_extension3_storage.py']
    inventory = {name:sha(ROOT/name) for name in code}
    for name in code:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_new(out/'check_lock.json', dict(code_sha256=inventory, config_sha256=sha(config_path),
        maximum_sample_months=2, gain_optimization_run=False, portfolio_evaluation_run=False))
    thread_limit = threadpool_limits(limits=config['resources']['threads'])
    preflight_receipt, _ = preflight()
    write_new(out/'financial_preflight.json', preflight_receipt)
    history = StudyHistory(ROOT, preflight_receipt, weights=config['reference_weights'])
    parent = load_parent(ROOT, 2025, 1, expected_cutoff='2024-12-31')
    def verify(relative, expected):
        history.files.path(relative)
        if history.files.consumed[relative] != expected:
            raise ValueError('Accepted input changed')
    store = OverflowStore(out/'operators', source_fingerprint=history.source_fingerprint,
        reference_fingerprint=history.reference_fingerprint, study_lock_sha256=sha(config_path),
        maximum_bytes=512*2**20, verify_input=verify)
    rows = []
    # Largest accepted universe and last training month; read only two samples.
    with threadpool_limits(limits=config['resources']['threads']):
        for day in ('2000-03-31', '2024-11-30'):
            begin = time.monotonic()
            sample = history.training_sample(day, parent.settings.cutoff)
            innovations = history.validate_market(sample.market, history.innovations(sample.market))
            basis = LeanBasis(sample)
            cold_seconds = time.monotonic()-begin
            begin = time.monotonic()
            store.put(day, basis, innovations, dict(history.files.consumed))
            write_seconds = time.monotonic()-begin
            begin = time.monotonic()
            reloaded, restored = store.get(day, parent.settings.cutoff)
            read_seconds = time.monotonic()-begin
            assert payload_fingerprint(basis, innovations) == payload_fingerprint(reloaded, restored)
            gains = np.ones(parent.feature_map.gain_count)
            for z, vol, innovation in zip(sample.market.characteristics, sample.market.volatility, innovations):
                np.testing.assert_array_equal(parent.feature_map.signals(z, vol, innovation, gains),
                    parent.model.feature_map.signals(z, vol, parent.model.theta))
            original = StreamingObjective(lambda:iter([(basis, innovations)]), parent.feature_map,
                parent.settings, rho=.01, objective_scale=parent.objective_scale)
            reused = StreamingObjective(lambda:iter([(reloaded, restored)]), parent.feature_map,
                parent.settings, rho=.01, objective_scale=parent.objective_scale)
            timings = []
            for x in (gains, np.linspace(.5, 1.5, len(gains))):
                begin = time.monotonic(); before = original(x); timings.append(time.monotonic()-begin)
                after = reused(x)
                for key in ('value', 'gradient', 'beta', 'risk', 'cost', 'score'):
                    np.testing.assert_array_equal(before[key], after[key])
            rows.append(dict(month=day, stocks=len(sample.market.ids), cold_construction_seconds=cold_seconds,
                write_seconds=write_seconds, read_seconds=read_seconds, warm_two_pass_profile_seconds=timings,
                payload_bytes=(out/'operators'/(day+'.npz')).stat().st_size,
                rss_gib=psutil.Process().memory_info().rss/2**30))
            print(json.dumps(rows[-1]), flush=True)
    history.files.unchanged()
    for relative, expected in dict(history.files.consumed, **parent.hashes).items():
        if sha(ROOT/relative) != expected: raise ValueError('Input changed')
    for relative, expected in inventory.items():
        if sha(ROOT/relative) != expected: raise ValueError('Code changed')
    write_new(out/'storage_check.json', dict(status='PASS', real_accepted_data=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(config_path),
        exact_roundtrip=True, parent_neutral_signals_exact=True,
        portfolio_evaluation_run=False, gain_optimization_run=False,
        conditional_profile_checks_only=True, maximum_sample_months=2, measurements=rows,
        code_sha256=inventory, accepted_input_sha256=dict(history.files.consumed, **parent.hashes),
        limitation='Two sampled months, not a long-history timing or convergence guarantee.'))


if __name__ == '__main__': main()
