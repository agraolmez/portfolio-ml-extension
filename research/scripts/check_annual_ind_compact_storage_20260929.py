"""Roundtrip the two fresh fitting probes through shared-input checkpoints.

This storage exercise does not fit, score, select or replay any model. Full-run
validation/replay histories and their storage requirements are not estimated.
"""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from annual_ind.checkpoints import CheckpointStore
from feature_geometry.history import sha
from extension4.storage import verify_stage, write_json


def main():
    started = time.monotonic()
    out = ROOT / 'output/annual_ind_compact_storage_20260929'
    out.mkdir()
    identity = sha(ROOT/'annual_ind/checkpoints.py')
    database = out/'models.sqlite'
    store = CheckpointStore(database, run_identity=identity)
    expected = {}
    shared_key = 'frequencies/seed1/g0/p512'
    shared_context = dict(kind='fixed_random_frequencies', seed=1, bandwidth_index=0, width=512)
    for index in (50, 0):
        folder = ROOT/f'output/annual_ind_fresh_probe_pipeline_r{index}_20260929'
        for stage in ('fresh_parent', 'fresh_temporal', 'fresh_conditional'):
            verify_stage(folder/stage)
        for name in ('parent', 'temporal'):
            source = folder/f'fresh_{name}/model.npz'
            with np.load(source, allow_pickle=False) as saved:
                arrays = {k:saved[k].copy() for k in saved.files}
            digest = store.put(shared_key, shared_context, dict(frequencies=arrays.pop('frequencies')))
            metadata = json.loads(str(arrays.pop('metadata')))
            context = dict(kind=name, source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                           cutoff='1970-12-31', model_metadata=metadata,
                           frequency_key=shared_key, frequency_sha256=digest)
            key = f'r{index}/{name}'
            store.put(key, context, arrays)
            expected[key] = (source, context)
        source = folder/'fresh_conditional/moments_and_coefficients.npz'
        for multiplier in (100., 10., 1.):
            key = f'r{index}/conditional/{multiplier}'
            context = dict(kind='conditional', source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                           parent_key=f'r{index}/temporal', cutoff='1970-12-31', multiplier=multiplier)
            with np.load(source, allow_pickle=False) as saved:
                store.put(key, context, dict(beta=saved[str(multiplier)]))
            expected[key] = (source, context)
    store.close()
    store = CheckpointStore(database, run_identity=identity)
    for key, (source, context) in expected.items():
        actual = store.get(key, context=context)
        with np.load(source, allow_pickle=False) as saved:
            if context['kind'] == 'conditional':
                np.testing.assert_array_equal(actual['beta'], saved[str(context['multiplier'])])
            else:
                shared = store.get(context['frequency_key'], context=shared_context)
                np.testing.assert_array_equal(shared['frequencies'], saved['frequencies'])
                for k, value in actual.items():
                    np.testing.assert_array_equal(value, saved[k])
                assert context['model_metadata'] == json.loads(str(saved['metadata']))
    inventory = store.inventory()
    store.close()
    result = dict(status='PASS_COMPACT_MODEL_ARRAY_ROUNDTRIP_ONLY',
        source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                            (Path(__file__), ROOT/'annual_ind/checkpoints.py')},
        database_sha256=sha(database), database_bytes=database.stat().st_size,
        models_verified=len(expected), shared_frequency_bundles=1, inventory=inventory,
        seconds=time.monotonic()-started, full_runner_uses_this_store=False,
        source_models_preserved=True, full_study_storage_feasibility_established=False)
    write_json(out/'review.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'inventory'}))


if __name__ == '__main__':
    main()
