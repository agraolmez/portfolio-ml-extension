"""Resumable own-holdings paths using the unchanged parent transition function."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline.core import transition
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from temporal_continuity.identity import holdings_fingerprint, annual_volatility
from .state import day
from .storage import commit_stage, verify_stage, write_json


def trade_month(market, outcome, ids, holdings, targets, specs, *, atol, rtol):
    """Aims must already be computed before the caller requests realized returns."""
    if (not np.array_equal(ids, market.ids) or not np.array_equal(outcome.ids, ids)
            or outcome.decision_date != market.decision_date or holdings.shape != targets.shape
            or holdings.shape != (len(ids), len(specs))):
        raise ValueError('Ordered holdings, outcomes and targets must match the decision')
    arrays = dict(ids=np.array(ids), next_ids=outcome.next_ids.copy(), pretrade=holdings.copy(),
                  target=targets.copy(), weight=np.empty_like(holdings), trade=np.empty_like(holdings),
                  drifted_weight=np.empty_like(holdings), next_pretrade=np.empty((len(outcome.next_ids),len(specs))),
                  seeds=np.array([s for s, _ in specs]), methods=np.array([m for _, m in specs]))
    rows, exits = [], []
    for column, (seed, method) in enumerate(specs):
        result = transition(ids, holdings[:, column], targets[:, column], market.m,
                            outcome.total_returns, outcome.market_growth, market.impact, market.wealth,
                            outcome.excess_returns, outcome.next_ids)
        for key, result_key in [('weight','weights'),('trade','trades'),('drifted_weight','drifted_weights'),('next_pretrade','next_pretrade')]:
            arrays[key][:,column] = result[result_key]
        gross = sum(float(w)*float(r) for w,r in zip(result['weights'],outcome.excess_returns))
        cost = .5*market.wealth*sum(float(l)*float(q)**2 for l,q in zip(market.impact,result['trades']))
        np.testing.assert_allclose([gross,cost],[result['gross_excess'],result['impact_cost']],atol=atol,rtol=rtol)
        w = result['weights']; absolute = abs(w); exposure = float(absolute.sum())
        rows.append(dict(seed=seed,method=method,decision_eom=market.decision_date,
            return_eom=day(pd.Timestamp(market.decision_date)+pd.offsets.MonthEnd(1)),
            gross_excess=float(gross),impact_cost=float(cost),net_excess=float(gross-cost),
            turnover=float(abs(result['trades']).sum()),gross_exposure=exposure,net_exposure=float(w.sum()),
            largest_absolute_weight=float(absolute.max()),gross_weight_concentration=float(np.sum((absolute/exposure)**2)) if exposure else 0.,
            modeled_annual_volatility=annual_volatility(w,market.covariance),holdings_reset=False,
            pretrade_sha256=holdings_fingerprint(ids,holdings[:,column]),
            next_pretrade_sha256=holdings_fingerprint(outcome.next_ids,result['next_pretrade'])))
        exits += [dict(seed=seed,method=method,decision_eom=market.decision_date,id=int(sid),
                       omitted_drifted_weight=float(value),liquidation_cost_added=False)
                  for sid,value in zip(result['dropped_ids'],result['dropped_drifted_weights'])]
    return arrays, rows, exits


def replay(source, dates, specs, initial, aim_provider, *, destination, lock_identity,
           policy_identity, atol, rtol, progress=None, compare=None):
    dates = tuple(day(d) for d in dates)
    if not dates or dates[0] != initial.decision_date or list(dates) != [day(d) for d in pd.date_range(dates[0],dates[-1],freq='ME')]:
        raise ValueError('Continuous dates from the common initial state required')
    if len(specs) != len(set(specs)): raise ValueError('Duplicate paths')
    destination = Path(destination)
    ids = initial.ids.copy(); holdings = np.repeat(initial.holdings[:,None],len(specs),axis=1)
    previous = fingerprint(initial.ids, initial.holdings, initial.source_fingerprint)
    all_rows = []
    for i,date in enumerate(dates):
        folder = destination/date
        context = dict(lock=lock_identity,day=date,specs=[list(s) for s in specs],previous=previous,policies=policy_identity)
        if folder.exists():
            verify_stage(folder,context)
            with np.load(folder/'positions.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['ids'],ids)
                np.testing.assert_array_equal(saved['pretrade'],holdings)
                ids=saved['next_ids'].copy();holdings=saved['next_pretrade'].copy()
            rows=json.loads((folder/'monthly.json').read_text())
        else:
            market=source.market(date)
            bound=source.innovation(market)
            # Outcome access is deliberately after every dated aim is formed.
            targets=np.column_stack([aim_provider(seed,method,market,bound) for seed,method in specs])
            outcome=source.realized(market)
            arrays,rows,exits=trade_month(market,outcome,ids,holdings,targets,specs,atol=atol,rtol=rtol)
            if compare is not None:compare(date,arrays,rows)
            def writer(folder):
                with (folder/'positions.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                write_json(folder/'monthly.json',rows);write_json(folder/'exits.json',exits)
            commit_stage(folder,context,writer)
            ids=outcome.next_ids.copy();holdings=arrays['next_pretrade']
        previous=sha(folder/'commit.json');all_rows.extend(rows)
        if progress:progress(i+1,date)
    source.unchanged()
    # JSON checkpoints use sorted keys; resumed and uninterrupted tables must match.
    return pd.DataFrame(all_rows).sort_index(axis=1)
