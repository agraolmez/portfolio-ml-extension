"""Bounded two-month replay of existing E3; no new state or economic fit."""
import fcntl
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.full import preflight
from baseline.core import transition
from feature_geometry.history import sha
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_study.paths import AcceptedPathSource
from extension4.parent import ParentCatalog, PARENT
from extension4.features import disabled_aim


def write_new(path, item):
    with path.open('x') as f:
        json.dump(item, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    out = ROOT/'output/extension4_implementation_20260924'
    if (out/'disabled_replay.json').exists():
        raise FileExistsError('Completed evidence is immutable')
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.monotonic()
    with (ROOT/'output/.extension3_heavy_job.lock').open('a+') as heavy, single_thread_blas():
        fcntl.flock(heavy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt, _ = preflight()
        if receipt != json.loads((ROOT/PARENT/'financial_preflight.json').read_text()):
            raise ValueError('Parent financial dependencies changed')
        saved_preflight = out/'financial_preflight.json'
        if saved_preflight.exists():
            if json.loads(saved_preflight.read_text()) != receipt:
                raise ValueError('Earlier financial preflight differs')
        else:
            write_new(saved_preflight, receipt)
        print('Financial preflight passed; replaying two accepted months.', flush=True)
        history = StudyHistory(ROOT, receipt, weights=[1/6]*6)
        source = AcceptedPathSource(history)
        initial = source.common_initial()
        catalog = ParentCatalog(ROOT)
        models = [catalog.load(1981, seed).model for seed in (1, 2, 3)]
        ids = initial.ids.copy()
        holdings = np.repeat(initial.holdings[:, None], 3, axis=1)
        errors = []
        for day in ('1980-12-31', '1981-01-31'):
            market = source.market(day)
            np.testing.assert_array_equal(ids, market.ids)
            bound = source.innovation(market)
            targets = [disabled_aim(model, market, bound) for model in models]
            outcome = source.realized(market)
            relative = PARENT+'/selected_paths/'+day+'.npz'
            # Selected paths are the accepted post-repair, continuous portfolios.
            with np.load(catalog.path(relative), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['ids'], ids)
                np.testing.assert_array_equal(saved['next_ids'], outcome.next_ids)
                next_holdings = []
                for j, seed in enumerate((1, 2, 3)):
                    choices = np.flatnonzero((saved['seeds'] == seed) & (saved['methods'] == 'selected_temporal'))
                    if len(choices) != 1:
                        raise ValueError('Expected one selected temporal portfolio per seed')
                    col = int(choices[0])
                    result = transition(ids, holdings[:, j], targets[j], market.m,
                        outcome.total_returns, outcome.market_growth, market.impact, market.wealth,
                        outcome.excess_returns, outcome.next_ids)
                    arrays = dict(pretrade=holdings[:, j], target=targets[j],
                        weight=result['weights'], trade=result['trades'],
                        drifted_weight=result['drifted_weights'], next_pretrade=result['next_pretrade'])
                    maximum = 0.
                    for name, value in arrays.items():
                        expected = saved[name][:, col]
                        np.testing.assert_allclose(value, expected, atol=1e-11, rtol=1e-10)
                        maximum = max(maximum, float(np.max(abs(value-expected), initial=0.)))
                    gross = sum(float(w)*float(r) for w, r in zip(result['weights'], outcome.excess_returns))
                    cost = .5*market.wealth*sum(float(q)**2*float(l) for q, l in zip(result['trades'], market.impact))
                    np.testing.assert_allclose([gross, cost],
                        [result['gross_excess'], result['impact_cost']],
                        atol=1e-11, rtol=1e-10)
                    errors.append(dict(date=day, seed=seed, maximum_position_error=maximum,
                                       independent_accounting_passed=True))
                    next_holdings.append(result['next_pretrade'])
                holdings = np.column_stack(next_holdings)
                ids = outcome.next_ids.copy()
            print('Verified '+day, flush=True)
        source.unchanged()
        write_new(out/'disabled_replay.json', dict(status='PASS', financial_preflight_unchanged=True,
            replay_months=2, replay_paths=3, results=errors, all_years_coefficient_refit_checked=False,
            state_data_constructed=False, historical_extension4_fit_run=False,
            timing_label='Pre-2011 selected-parent replay is retrospective, not new validation',
            consumed_inputs_sha256={**source.consumed_inputs, **catalog.consumed},
            elapsed_seconds=time.monotonic()-started))
        print('PASS: bounded actual disabled-parent replay and position accounting.', flush=True)


if __name__ == '__main__':
    main()
