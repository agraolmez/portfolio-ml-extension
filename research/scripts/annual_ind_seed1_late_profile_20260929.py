"""Measure the complete 2020 neutral profile with bounded month streaming.

All 649 training months are freshly read from accepted inputs. This is an
identity-geometry moment pass and three neutral solves, not 2020 learned-layer
fitting and not a full-study runtime forecast. Existing operators/results are
not overwritten; no portfolio paths are generated.
"""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from feature_geometry.training import _solve
from geometry_study import kernel
from geometry_study.features import CachedFeatureMap
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from extension4.runtime import prior_seconds
from extension4.storage import write_json
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime
from annual_ind.numerics import ordered_months
from annual_ind.panels import CalendarSourcePanels

OUT = ROOT/'output/annual_ind_seed1_late_profile_20260929'
EARLY = ROOT/'output/annual_ind_seed1_speed_v3_20260929'


def main():
    process_lock = (ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (EARLY/'result.json').exists():
        raise ValueError('Passing preceding benchmark required')
    previous = json.loads((EARLY/'lock.json').read_text())
    for section in ('code_sha256', 'protected_sha256'):
        for rel, expected in previous[section].items():
            if sha(ROOT/rel) != expected:
                raise ValueError('Changed source '+rel)
    ledger_path = ROOT/'output/annual_ind_parallel_review_20260929/resource_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    early_seconds = sum(prior_seconds(ROOT/rel)[0] for rel in ['output/annual_ind_seed1_speed_20260929','output/annual_ind_seed1_speed_v2_20260929','output/annual_ind_seed1_speed_v3_20260929'])
    # Reserve an explicit conservative allowance for unit/review/housekeeping work.
    remaining = ledger['remaining_engineering_seconds'] - early_seconds - 300.
    seconds = min(1500., remaining)
    if seconds <= 0:
        raise TimeoutError('Cumulative engineering cap exhausted')
    limits = dict(total_active_seconds=seconds, pilot_active_seconds=seconds,
                  max_physical_footprint_gib=16., min_system_headroom_gib=4.,
                  max_additional_disk_gib=2., min_free_disk_gib=50.)
    OUT.mkdir()
    write_json(OUT/'lock.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        purpose=__doc__, year=2020, seed=1, width=512, bandwidth_index=0,
        ridge_indices=[0,50,100], first_training_decision='1965-11-30',
        last_training_decision='2019-11-30', cutoff='2019-12-31', training_months=649,
        workers=8, blas_threads=1, source_panel_cache_bytes=256*2**20, budget=limits,
        prior_resource_ledger_sha256=sha(ledger_path), prior_charged_seconds=ledger['charged_seconds'],
        preceding_benchmark_active_seconds=early_seconds, review_allowance_seconds=300.,
        code_sha256={**previous['code_sha256'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'annual_ind/panels.py':sha(ROOT/'annual_ind/panels.py')},
        protected_sha256=previous['protected_sha256'], AI_calls=False, full_study_started=False))
    runtime = None
    try:
        runtime = ProbeRuntime(OUT, {'compute_budget':limits}, 'complete_late_neutral_profile')
        receipt = json.loads((EARLY/'financial_preflight.json').read_text())
        history = StudyHistory(ROOT, receipt, weights=[1/6]*6)
        history.panels = CalendarSourcePanels(history.files,history.panels.columns,256*2**20)
        row_checks = []
        for name, columns in history.panels.columns.items():
            for day in ('1965-11-30','2019-11-30'):
                runtime.guard()
                cached = history.panels.read(name,list(columns),day)
                direct = pd.read_parquet(history.files.path(name),columns=list(columns),
                    filters=[('eom','==',pd.Timestamp(day))])
                pd.testing.assert_frame_equal(cached,direct)
                row_checks.append(dict(source=name,month=day,rows=len(cached),identical=True))
        write_json(OUT/'source_row_checks.json',dict(status='PASS',checks=row_checks))
        preparation = ROOT/'output/step6_20260911/preparation_manifest.json'
        if sha(preparation) != receipt['preparation_manifest_sha256']:
            raise ValueError('Preparation receipt changed')
        history.files.inventory.update(json.loads(preparation.read_text())['artifact_sha256'])
        directions = history.files.path('output/feature_geometry_study_20260914/directions/seed1_g0.bin')
        fmap = CachedFeatureMap(np.fromfile(directions,dtype='<f8').reshape(115,512)[:,:256],
                                'seed1_original_directions_'+sha(directions))
        theta = np.zeros(115)
        months = pd.date_range('1965-11-30','2019-11-30',freq='ME')
        assert len(months) == 649
        source_fingerprints = []
        prepared = []
        def inputs():
            for date in months:
                runtime.guard()
                start = time.monotonic()
                basis = kernel.LeanBasis(history.training_sample(str(date.date()), '2019-12-31'))
                if basis.sample.available_at > '2019-12-31':
                    raise ValueError('Future training outcome')
                source_fingerprints.append(basis.sample.fingerprint)
                prepared.append(dict(date=str(date.date()), n=len(basis.sample.market.ids),
                                     seconds=time.monotonic()-start))
                yield basis
        def one(basis):
            start = time.monotonic()
            record = basis.matrices(fmap,theta)
            # Do not retain the large omega/change/signal arrays after the worker.
            return record['score'],record['risk'],record['cost'],time.monotonic()-start
        start = time.monotonic()
        p = fmap.columns
        score,risk,cost = np.zeros(p),np.zeros((p,p)),np.zeros((p,p))
        computed = 0; monthly_compute = []
        with threadpool_limits(limits=1),single_thread_blas():
            for ss,rr,cc,elapsed in ordered_months(one,inputs(),workers=8,guard=runtime.guard):
                score += ss; risk += rr; cost += cc
                monthly_compute.append(elapsed); computed += 1
                if computed == 1 or computed % 60 == 0:
                    runtime.event('neutral_history_progress',months=computed,total_months=len(months),
                                  last_completed=str(months[computed-1].date()))
            if computed != len(months):
                raise ValueError('Incomplete full-history profile')
            score /= computed; risk /= computed; cost /= computed
            elapsed = time.monotonic()-start
            grid = json.loads((ROOT/'output/step2_20260910/source_settings.json').read_text())['settings']['pf_ml']
            coefficient = history.files.path('data/prepared/step9_v1/baseline/coefficients/g0/2020_p512.npz')
            comparisons = []
            arrays = dict(score=score,risk=risk,cost=cost)
            with np.load(coefficient,allow_pickle=False) as original:
                for index in (0,50,100):
                    beta,residual = _solve(risk+cost,score,float(grid['l_vec'][index]))
                    expected = original['coefficients'][:,index]
                    np.testing.assert_allclose(beta,expected,atol=1e-9,rtol=1e-8)
                    arrays[f'beta_{index}'] = beta
                    comparisons.append(dict(ridge_index=index,relative_residual=residual,
                        maximum_beta_error=float(np.max(abs(beta-expected)))))
        history.files.unchanged()
        with (OUT/'neutral_moments.npz').open('xb') as f:
            np.savez_compressed(f,**arrays)
        pd.DataFrame(prepared).assign(moment_seconds=monthly_compute).to_csv(OUT/'monthly_timing.csv',index=False)
        result = dict(status='PASS_COMPLETE_2020_NEUTRAL_PROFILE_ONLY',training_months=computed,
            source_row_equivalence_checks=len(row_checks),
            profile_wall_seconds=elapsed, preparation_seconds_sum=sum(x['seconds'] for x in prepared),
            worker_moment_seconds_sum=sum(monthly_compute), comparisons=comparisons,
            source_fingerprint=fingerprint(*source_fingerprints),
            source_sha256=dict(history.files.consumed), maximum_security_count=max(x['n'] for x in prepared),
            newly_fitted_learned_models=0,new_baseline_coefficient_solves=3,
            source_cache=history.panels.diagnostics(),new_portfolio_paths=0,full_study_started=False,
            complete_historical_runtime_established=False,active_seconds=time.monotonic()-runtime.start)
        write_json(OUT/'result.json',result)
        runtime.event('complete_late_neutral_profile',seconds=elapsed,months=computed)
        history.panels.release();gc.collect()
        runtime.close('COMPLETE')
    except BaseException as error:
        if runtime is not None:
            runtime.close('STOPPED',error)
        write_json(OUT/'stop.json',dict(status='STOPPED',error_type=type(error).__name__,error=str(error)))
        raise


if __name__ == '__main__':
    main()
