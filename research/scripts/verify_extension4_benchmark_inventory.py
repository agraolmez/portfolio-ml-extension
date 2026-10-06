"""Bounded actual-data verification of the accepted F1024 inventory repair.

No fit or selection is recomputed. Resource time is charged to the existing
study; comparison replay is saved in a separate immutable engineering folder.
"""
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension4.run import Study
from extension4.paths import replay
from extension4.storage import verify_stage, write_json
from extension4.verification import independent_month
from extension4_execution.memory_v2 import PressureRuntime, load_authority
from extension4_execution.benchmark_inventory_v1 import include_accepted_benchmarks, SOURCES, GEOMETRY
from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from temporal_fast.backend import single_thread_blas


def main():
    out = ROOT/'output/extension4_benchmark_fix_20260928'
    out.mkdir(exist_ok=True)
    if (out/'verification.json').exists():
        raise FileExistsError('Preserve completed verification; do not overwrite it')
    config, runtime_config, bindings = load_authority(ROOT)
    bindings.update({p:sha(ROOT/p) for p in SOURCES})
    destination = ROOT/config['paths']['proposed_full_root']
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    runtime = None
    with (ROOT/'output/.extension3_heavy_job.lock').open('a+') as heavy, single_thread_blas():
        fcntl.flock(heavy, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            runtime = PressureRuntime(destination, runtime_config, 'benchmark_verification',
                                      project_root=ROOT, bindings=bindings)
            runtime.event('benchmark_repair_verification_started')
            study = Study(ROOT, ROOT/'config/extension4_study_v1_proposed.json',
                          ROOT/'config/extension4_end_to_end_approval.json')
            original_inventory = dict(study.catalog.inventory)
            missing = f'{GEOMETRY}/fits/1981/seed1/fixed1024/model.npz'
            try:study.catalog.path(missing)
            except ValueError:pass
            else:raise AssertionError('Expected original missing-inventory failure not reproduced')
            extension = include_accepted_benchmarks(study.catalog)
            for p,h in original_inventory.items():
                if study.catalog.inventory[p]!=h:raise ValueError('Original accepted inventory changed')
            study.setup(runtime)
            write_json(out/'financial_preflight.json',json.loads((destination/'preflight/receipt.json').read_text()))
            preserved={};groups=economic=neutral=0
            choices={(r['seed'],r['method']):r['multiplier'] for r in
                     json.loads((destination/'selection/selected.json').read_text())}
            for year in range(1981,2026):
                for seed in (1,2,3):
                    runtime.guard();parent=study.parent(year,seed)
                    families={m:config['penalty_search']['multipliers_in_tie_order'] if year<=2015 else [choices[seed,m]]
                              for m in ('C4','CAP4','SIZE4')}
                    folder=destination/'fits'/str(year)/f'seed{seed}'
                    record=verify_stage(folder,dict(lock=study.lock,year=year,seed=seed,
                        parent=parent.model._content_hash(),states=study.book.identity,multipliers=families))
                    for rel,digest in record['artifact_sha256'].items():preserved[str((folder/rel).relative_to(ROOT))]=digest
                    preserved[str((folder/'commit.json').relative_to(ROOT))]=sha(folder/'commit.json')
                    fits=json.loads((folder/'fit.json').read_text())['models']
                    economic+=sum(r['method']!='N4' for r in fits);neutral+=sum(r['method']=='N4' for r in fits);groups+=1
                    rel=f'{GEOMETRY}/fits/{year}/seed{seed}/fixed1024/model.npz'
                    model=GeometryModel.load(study.catalog.path(rel))
                    if model.cutoff!=f'{year-1}-12-31' or model.feature_map.columns!=1025:
                        raise ValueError('Comparison shape/cutoff mismatch')
                    np.testing.assert_array_equal(model.theta,np.zeros(115))
            if (groups,economic,neutral)!=(135,1035,135):raise ValueError('Saved fitting scope incomplete')
            for name in ('states','selection','pilot_gate'):
                folder=destination/name;record=verify_stage(folder)
                preserved[str((folder/'commit.json').relative_to(ROOT))]=sha(folder/'commit.json')
                for p,h in record['artifact_sha256'].items():preserved[str((folder/p).relative_to(ROOT))]=h
            candidates=sorted((destination/'candidate_paths').glob('*/commit.json'))
            if len(candidates)!=420:raise ValueError('Candidate paths incomplete')
            for commit in candidates:
                record=verify_stage(commit.parent);preserved[str(commit.relative_to(ROOT))]=sha(commit)
                for p,h in record['artifact_sha256'].items():preserved[str((commit.parent/p).relative_to(ROOT))]=h
            runtime.event('benchmark_repair_inventory_verified',fit_groups=groups,comparisons=135)
            specs=[(s,m) for s in (1,2,3) for m in config['controls']]
            dates=['1980-12-31','1981-01-31']
            kw=dict(destination=out/'replay',lock_identity=study.lock,policy_identity='BENCHMARK_INVENTORY_REPAIR_CHECK',
                    atol=1e-11,rtol=1e-10,compare=study.compare,
                    progress=lambda n,d:runtime.event('benchmark_repair_replay',months=n,decision=d))
            first=replay(study.source,dates,specs,study.initial,
                         lambda s,m,market,b:study.aim(s,m,market,b,choices),**kw)
            again=replay(study.source,dates,specs,study.initial,
                         lambda *a:(_ for _ in ()).throw(AssertionError('Saved replay recomputed')),**kw)
            pd.testing.assert_frame_equal(first,again)
            errors=[]
            for date in dates:
                folder=out/'replay'/date
                with np.load(folder/'positions.npz',allow_pickle=False) as saved:
                    arrays={k:saved[k].copy() for k in saved.files}
                market=study.source.market(date);outcome=study.source.realized(market)
                errors.append(independent_month(market,outcome,arrays,json.loads((folder/'monthly.json').read_text()),
                    json.loads((folder/'exits.json').read_text()),atol=1e-11,rtol=1e-10))
            study.source.unchanged();study.history.files.unchanged()
            for relative in tuple(study.catalog.consumed):study.catalog.path(relative)
            runtime.sample();runtime.guard()
            write_json(out/'verification.json',dict(status='PASS',original_failure_reproduced=True,
                extension=extension,preserved_artifact_sha256=preserved,fit_groups=groups,economic_fits=economic,
                neutral_fits=neutral,candidate_months=420,frozen_selection_preserved=True,
                all_controls=config['controls'],replay_seeds=[1,2,3],replay_months=2,replay_rows=len(first),
                exact_saved_restart=True,maximum_independent_accounting_error=max(errors),
                original_catalog_preserved=True,financial_code_changed=False,scientific_settings_changed=False,
                scientific_study_complete=False,source_sha256={p:sha(ROOT/p) for p in SOURCES}))
            runtime.event('benchmark_repair_verification_passed')
            runtime.close('COMPLETE')
        except BaseException as error:
            if runtime is not None:runtime.close('STOPPED',error)
            raise


if __name__=='__main__':main()
