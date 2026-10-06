"""Portable aggregate-results verification; never imports implementation/.

Outputs require a new directory outside this package. All inference conditions
on saved fitted/selected paths. It does not repair or model specification search.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
PERIODS = [(1981, 2020, "primary"), (2021, 2025, "secondary_excludes_primary_overlap")]
METHODS = ("B1", "C4", "IND")
LAGS = (1, 3, 6, 12, 24)
BLOCKS = (12, 24)
DRAWS = 20000
SEED = 20261006


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def utility(gross, net, axis=None):
    return 12 * (np.mean(net, axis=axis) - 5 * np.var(gross, axis=axis, ddof=1))


def verify_integrity():
    manifest = json.loads((BASE / "MANIFEST.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Package content differs from manifest: " + name)
    return len(manifest["files_sha256"])


def calculate(frame):
    expected_dates = pd.date_range("1981-01-31", "2025-12-31", freq="ME")
    required = ["return_eom", "method", "gross_excess", "impact_cost", "net_excess"]
    if frame.columns.tolist() != required or len(frame) != 1620:
        raise ValueError("Unexpected aggregate columns or observation count")
    if set(frame.method) != set(METHODS) or frame.duplicated(["method", "return_eom"]).any():
        raise ValueError("Unexpected methods or duplicate method-month")
    if not np.isfinite(frame[required[2:]].to_numpy()).all() or (frame.impact_cost < 0).any():
        raise ValueError("Nonfinite returns or negative impact cost")
    np.testing.assert_allclose(frame.net_excess, frame.gross_excess-frame.impact_cost, atol=1e-15, rtol=1e-13)
    frame = frame.copy()
    frame["return_eom"] = pd.to_datetime(frame.return_eom)
    for method in METHODS:
        if not np.array_equal(frame[frame.method.eq(method)].sort_values("return_eom").return_eom.to_numpy(), expected_dates.to_numpy()):
            raise ValueError("Incomplete or shifted return calendar")
    metrics, contrasts, diagnostics = [], [], []
    for first, last, role in PERIODS:
        selected = frame[frame.return_eom.between(f"{first}-01-01", f"{last}-12-31")]
        paths = {m: selected[selected.method.eq(m)].sort_values("return_eom") for m in METHODS}
        n = len(paths["B1"])
        levels = {}
        for method in METHODS:
            path = paths[method]
            gross, net, cost = [path[c].to_numpy() for c in ("gross_excess", "net_excess", "impact_cost")]
            level = float(utility(gross, net)); levels[method] = level
            metrics.append(dict(first=first,last=last,period_role=role,method=method,seed=1,months=n,
                annual_utility=level,annual_gross_excess_mean=12*gross.mean(),annual_net_excess_mean=12*net.mean(),
                annual_net_volatility=np.sqrt(12)*net.std(ddof=1),annual_net_sharpe=np.sqrt(12)*net.mean()/net.std(ddof=1),
                annual_modeled_impact_cost=12*cost.mean()))
        def flow(path):
            gross = path.gross_excess.to_numpy()
            return path.net_excess.to_numpy() - 5*(gross-gross.mean())**2
        difference = flow(paths["IND"]) - flow(paths["B1"])
        centered = difference-difference.mean()
        delta = levels["IND"]-levels["B1"]
        z = difference.mean()/(difference.std(ddof=1)/np.sqrt(n))
        p = float(norm.sf(z))
        contrasts.append(dict(first=first,last=last,period_role=role,months=n,baseline_utility=levels["B1"],
            IND_utility=levels["IND"],annual_utility_difference=delta,one_sided_unadjusted_z=z,
            one_sided_unadjusted_p=p,author_score=1-p))
        row = dict(period="primary" if first==1981 else "extension_nonoverlapping",months=n,delta_pp=100*delta,nominal_p=p)
        for lag in LAGS:
            long_variance = centered@centered/n
            for k in range(1,lag+1):
                long_variance += 2*(1-k/(lag+1))*(centered[k:]@centered[:-k]/n)
            if long_variance <= 0:
                raise ValueError("Nonpositive Bartlett long-run variance")
            row[f"hac{lag}_p"] = float(norm.sf(difference.mean()/np.sqrt(long_variance/n)))
        # Reset the generator per reporting period; retain its stream across L=12,24.
        rng = np.random.default_rng(SEED)
        for block in BLOCKS:
            indices = np.empty((DRAWS,n),dtype=np.int32)
            indices[:,0] = rng.integers(n,size=DRAWS)
            for t in range(1,n):
                # Consume both random arrays on every step, including non-restarts.
                restart = rng.random(DRAWS) < 1/block
                fresh = rng.integers(n,size=DRAWS)
                indices[:,t] = np.where(restart,fresh,(indices[:,t-1]+1)%n)
            draws = {}
            for method in ("IND","B1"):
                gross = paths[method].gross_excess.to_numpy()[indices]
                net = paths[method].net_excess.to_numpy()[indices]
                draws[method] = utility(gross,net,axis=1)
            lower, upper = 2*delta-np.quantile(draws["IND"]-draws["B1"],[.975,.025])
            row[f"bootstrap{block}_ci95_low_pp"] = 100*lower
            row[f"bootstrap{block}_ci95_high_pp"] = 100*upper
        diagnostics.append(row)
    return pd.DataFrame(metrics),pd.DataFrame(contrasts),pd.DataFrame(diagnostics)


def compare_table(actual, name):
    expected = read_csv(BASE / "expected" / name)
    if actual.columns.tolist() != expected.columns.tolist() or actual.shape != expected.shape:
        raise ValueError("Expected table schema differs: " + name)
    largest = 0.
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(actual[column],expected[column],atol=2e-11,rtol=2e-11,
                err_msg=name+":"+column)
            largest = max(largest,float(np.max(np.abs(actual[column]-expected[column]))))
        elif actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError("Expected labels differ: " + name+":"+column)
    return largest


def component_checks():
    table = read_csv(BASE / "data/component_controls.csv")
    if len(table)!=19 or set(table.stage)!={"E1","E2","E3","E4","IND"}:
        raise ValueError("Historical control inventory incomplete")
    np.testing.assert_allclose(table.recomputed_delta_bp,table.saved_delta_bp,atol=2e-8,rtol=1e-10)
    if not (table.saved_adjusted_lower_bp <= table.saved_adjusted_upper_bp).all():
        raise ValueError("Invalid archived interval ordering")
    for record in table.to_dict("records"):
        seeds = json.loads(record["seed_deltas_bp"])
        if seeds:
            np.testing.assert_allclose(np.mean(seeds),record["recomputed_delta_bp"],atol=2e-8,rtol=1e-10)
    return dict(rows=len(table),point_and_seed_arithmetic=True,
        limitation="Original intervals are retained results, not newly resampled by this aggregate verifier; component monthly paths are not bundled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path,help="New output directory outside this package")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination==BASE or BASE in destination.parents:
        raise ValueError("Outputs must be outside the immutable reviewer package")
    if destination.exists():
        raise FileExistsError("Choose a new output directory; existing results are never replaced")
    count = verify_integrity()
    tables = calculate(read_csv(BASE / "data/monthly_strategy_returns.csv"))
    names = ("final_metrics.csv","primary_and_secondary_comparisons.csv","retrospective_dependence_sensitivity.csv")
    errors = {name:compare_table(table,name) for name,table in zip(names,tables)}
    bonferroni = json.loads((BASE/"expected/bonferroni_sensitivity.json").read_text())
    six_p = min(6*float(tables[1].iloc[0].one_sided_unadjusted_p),1.)
    np.testing.assert_allclose(six_p,float(bonferroni["adjusted_p"]),atol=2e-14,rtol=2e-14)
    scope = read_csv(BASE/"data/six_comparison_scope.csv")
    if len(scope)!=6:raise ValueError("Six-comparison retrospective scope incomplete")
    controls = component_checks()
    destination.mkdir(parents=True,exist_ok=False)
    for name,table in zip(names,tables):
        table.to_csv(destination/name,index=False)
    report = dict(status="PASS",package_hashes_checked=count,monthly_observations=1620,
        strategies=3,months_per_strategy=540,periods=["1981-2020","2021-2025"],maximum_table_errors=errors,
        bootstrap=dict(draws=DRAWS,seed_per_period=SEED,expected_block_lengths=list(BLOCKS)),
        hac=dict(lags=list(LAGS),kernel="Bartlett",lag_covariance_denominator="N",finite_sample_correction=False),
        component_checks=controls,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        retrospective_six_comparison_bonferroni_p=six_p,complete_research_search_adjustment=False,
        fitted_or_selected_models=False,portfolio_paths_rerun=False,source_data_required=False)
    (destination/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
