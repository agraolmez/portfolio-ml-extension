"""Predetermined paired worker benchmark, no historical selection or portfolios.

Reference inputs are the two already verified fresh 1971 fitting probes. Temporal
optimization always starts at neutral gains. The middle-ridge order is 4/8/8/4
workers, followed by one eight-worker zero-ridge equivalence check. Conditional
moments use 1/4/8/8/4/1 workers in that order, each followed by all three solves.
No setting or candidate is selected by financial performance.
"""
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import replace
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
from feature_geometry.history import sha
from feature_geometry.training import FitSettings
from geometry_study.kernel import LeanBasis
from temporal_fast_v4.history import StudyHistory
from temporal_fast_v4.settings import VARIANTS
from temporal_fast.backend import single_thread_blas
from temporal_numerics.optimizer import optimize, LBFGSBSettings
from temporal_extension.model import TemporalModel
from extension4.state import StateBook
from extension4.design import validate_history
from extension4.estimation import month_moments
from extension4.storage import verify_stage, write_json
from scripts.annual_ind_pipeline_probe_20260929 import ProbeRuntime
from annual_ind.parallel_layers import ParallelTemporalObjective, conditional_sums
from annual_ind.conditional import solve_conditional
from annual_ind.pipeline_probe import plain

OUT = ROOT/'output/annual_ind_layer_speed_20260929'


def main():
    lockfile = (ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    ledger_path = ROOT/'output/annual_ind_integration_20260929/resource_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    seconds = min(600., ledger['remaining_engineering_seconds'])
    if seconds <= 0:
        raise TimeoutError('Cumulative engineering budget exhausted')
    limits = dict(total_active_seconds=seconds, pilot_active_seconds=seconds,
        max_physical_footprint_gib=16., min_system_headroom_gib=4.,
        max_additional_disk_gib=2., min_free_disk_gib=50.)
    middle = ROOT/'output/annual_ind_fresh_probe_pipeline_r50_20260929'
    previous = json.loads((middle/'lock.json').read_text())
    sources = dict(previous['code_sha256'])
    for rel in ('scripts/benchmark_annual_ind_layers_20260929.py', 'annual_ind/parallel_layers.py'):
        sources[rel] = sha(ROOT/rel)
    for rel, value in sources.items():
        if sha(ROOT/rel) != value:
            raise ValueError('Changed source: '+rel)
    for rel, value in previous['protected_sha256'].items():
        if sha(ROOT/rel) != value:
            raise ValueError('Changed protected decision: '+rel)
    write_json(OUT/'lock.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Execution-only paired benchmark; no portfolio evaluation',
        temporal_order=[4,8,8,4], conditional_order=[1,4,8,8,4,1],
        zero_ridge_eight_worker_check=True, code_sha256=sources,
        protected_sha256=previous['protected_sha256'], budget=limits,
        previous_resource_ledger_sha256=sha(ledger_path),
        prior_charged_seconds=ledger['charged_seconds'], AI_calls=False))
    runtime = ProbeRuntime(OUT, {'compute_budget':limits}, 'layer_speed_benchmark')
    try:
        receipt = json.loads((ROOT/'output/annual_ind_audit_v3_20260929/financial_preflight.json').read_text())
        history = StudyHistory(ROOT, receipt, weights=[1/6]*6)
        bases = []
        for date in pd.date_range('1965-11-30','1970-11-30',freq='ME'):
            runtime.guard()
            bases.append(LeanBasis(history.training_sample(str(date.date()), '1970-12-31')))
        batches = []
        for basis in bases:
            runtime.guard()
            bound = history.innovations(basis.sample.market)
            batches.append((basis, history.validate_market(basis.sample.market, bound), bound))
        history.panels.release(); gc.collect()
        trials = []; conditional = []
        for index in (50, 0):
            folder = ROOT/f'output/annual_ind_fresh_probe_pipeline_r{index}_20260929'
            for stage in ('fresh_parent','fresh_temporal','fresh_conditional'):
                verify_stage(folder/stage)
            metadata = json.loads((folder/'temporal_attempt.json').read_text())
            parent = json.loads((folder/'fresh_parent/fit.json').read_text())
            archived = TemporalModel.load(folder/'fresh_temporal/model.npz',
                expected_parent_fingerprint=metadata['parent_fingerprint'])
            settings = FitSettings(**parent['settings'])
            limits_optimizer = LBFGSBSettings(**metadata['optimizer_settings'])
            def factory():
                for basis, innovation, _ in batches:
                    runtime.guard()
                    yield basis, innovation
            result = None
            for ordinal, workers in enumerate((4,8,8,4) if index==50 else (8,)):
                runtime.event('temporal_benchmark_started', ridge_index=index, workers=workers, ordinal=ordinal)
                objective = ParallelTemporalObjective(factory, archived.feature_map, settings,
                    rho=metadata['rho'], objective_scale=parent['objective_normalizer'],
                    workers=workers, trig_backend='accelerate', **dict(VARIANTS))
                neutral = []
                def checked(gains, with_gradient=True):
                    runtime.guard()
                    value = objective(gains, with_gradient=with_gradient)
                    if not neutral:
                        np.testing.assert_array_equal(gains, np.ones(13))
                        np.testing.assert_allclose(value['beta'], parent['beta'], atol=1e-11, rtol=1e-10)
                        neutral.append(True)
                    runtime.guard()
                    return value
                start = time.monotonic()
                with threadpool_limits(limits=1), single_thread_blas():
                    result = optimize(checked, 13, limits_optimizer)
                elapsed = time.monotonic()-start
                if not result['converged'] or result['accepted_steps'] != metadata['accepted_steps']:
                    raise ValueError('Optimizer convergence or trajectory changed')
                errors = {}
                for key in ('gains','beta','objective_trace','projected_gradient_trace'):
                    actual = np.asarray(result[key]); expected = np.asarray(metadata[key])
                    np.testing.assert_array_equal(actual, expected)
                    errors[key] = float(np.max(abs(actual-expected)))
                trial = dict(ridge_index=index, workers=workers, ordinal=ordinal,
                    seconds=elapsed, identical_to_verified_fit=True, max_errors=errors,
                    evaluations=result['evaluation_count'], updates=result['accepted_steps'])
                write_json(OUT/f'temporal_r{index}_{ordinal}.json', plain(dict(result, workers=workers)))
                trials.append(trial)
                runtime.event('temporal_benchmark_complete', **trial)
            if index == 0:
                continue
            temporal = replace(archived, gains=result['gains'], beta=result['beta'])
            book = StateBook(pd.read_parquet(folder/'fresh_state_history.parquet'),
                             json.loads((folder/'fresh_state_metadata.json').read_text()))
            p = len(temporal.beta)
            def moments(item):
                basis, _, bound = item
                market = basis.sample.market
                innovation = validate_history(market,bound,temporal)
                states = book.bind(market.decision_date,history=13).validate(
                    market.decision_date,history=13,expected_identity=book.identity)
                return month_moments(basis,innovation,temporal,states,
                    state_names=book.names,interaction_columns=tuple(range(1,p)))
            with np.load(folder/'fresh_conditional/moments_and_coefficients.npz',allow_pickle=False) as saved:
                for ordinal,workers in enumerate((1,4,8,8,4,1)):
                    runtime.event('conditional_benchmark_started',workers=workers,ordinal=ordinal)
                    start=time.monotonic()
                    with threadpool_limits(limits=1),single_thread_blas():
                        score,curvature=conditional_sums(batches,moments,columns=2*p-1,
                            workers=workers,guard=runtime.guard)
                        np.testing.assert_array_equal(score,saved['score'])
                        np.testing.assert_array_equal(curvature,saved['curvature'])
                        for multiplier in (100.,10.,1.):
                            beta,_=solve_conditional(curvature,score,parent_width=p,
                                parent_ridge=settings.ridge,multiplier=multiplier)
                            np.testing.assert_array_equal(beta,saved[str(multiplier)])
                    elapsed=time.monotonic()-start
                    trial=dict(workers=workers,ordinal=ordinal,seconds=elapsed,
                        moments_and_all_coefficients_identical=True)
                    conditional.append(trial)
                    runtime.event('conditional_benchmark_complete',**trial)
        history.files.unchanged()
        for rel,value in sources.items():
            if sha(ROOT/rel)!=value:raise ValueError('Source changed during benchmark: '+rel)
        runtime.guard()
        write_json(OUT/'result.json',dict(status='PASS_PAIRED_LAYER_SPEED_BENCHMARK_ONLY',
            temporal=trials,conditional=conditional,new_temporal_numerical_fits=len(trials),
            new_conditional_numerical_solves=3*len(conditional),new_portfolio_paths=0,
            maximum_history_tested=False,full_study_started=False))
        runtime.close('COMPLETE')
        print(json.dumps(dict(status='PASS_PAIRED_LAYER_SPEED_BENCHMARK_ONLY',temporal=trials,conditional=conditional)))
    except BaseException as error:
        runtime.close('STOPPED',error)
        write_json(OUT/'stop.json',dict(status='STOPPED',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':main()
