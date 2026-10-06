"""One locked development-year engineering exercise, with no portfolio-return path."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .history import CanonicalHistory, sha
from .moments import MomentBasis
from .policy import GeometryModel, act
from .training import FitSettings, fit_geometry, profile


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(x) for x in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item
    with Path(path).open("x") as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def resources(output, plan):
    divisor = 2**30 if sys.platform == "darwin" else 2**20
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    free = shutil.disk_usage(output).free / 2**30
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) / 2**30
    if peak > plan["aggregate_memory_gib"] or free < plan["minimum_free_disk_gib"] or used > plan["maximum_output_gib"]:
        raise MemoryError("Prototype resource boundary exceeded")
    return dict(peak_process_gib=peak, free_disk_gib=free, output_gib=used)


def run(root, plan_path, output):
    root, plan_path, output = map(lambda p: Path(p).resolve(), (root, plan_path, output))
    if output.exists():
        raise FileExistsError("Use a new output destination; historical receipts are immutable")
    plan = json.loads(plan_path.read_text())
    if (plan.get("historical_portfolio_path") is not False
            or plan.get("selection_or_evaluation_outcomes_used") is not False
            or plan.get("input_characteristics") != 115 or plan.get("threads") != 4):
        raise ValueError("This runner supports only the declared bounded prototype")
    dates = pd.date_range(plan["training_first_decision"], plan["training_last_decision"], freq="ME")
    if (len(dates) != 12 or str(dates[0].date()) != "1980-12-31"
            or str(dates[-1].date()) != "1981-11-30" or plan["cutoff"] != "1981-12-31"
            or plan["decision_probe_date"] != "1981-12-31"):
        raise ValueError("Changing the historical exercise requires a separately specified runner/plan")
    settings = FitSettings(**{k: plan[k] for k in FitSettings.__dataclass_fields__})
    if plan["geometry_objectives"] != ["full", "risk_only", "return_prediction"]:
        raise ValueError("Retain all three fixed prototype geometry objectives")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    package = Path(__file__).resolve().parent
    source_files = sorted(package.glob("*.py"))
    source_hashes = {str(p): sha(p) for p in source_files}
    write_json(output / "plan_lock.json", dict(plan=plan, plan_sha256=sha(plan_path),
                                               code_sha256=source_hashes,
                                               started_at_utc=datetime.now(timezone.utc).isoformat(),
                                               pid=os.getpid(), status="LOCKED_BEFORE_HISTORICAL_COMPUTATION"))
    try:
        sys.path.insert(0, str(root))
        from baseline.full import preflight
        from baseline.io import ROOT as baseline_root
        if baseline_root.resolve() != root:
            raise ValueError("Baseline import does not match the requested canonical project")
        print("Checking accepted financial inputs and original machinery.", flush=True)
        with threadpool_limits(limits=plan["threads"]):
            preflight_result, _ = preflight()
            write_json(output / "financial_preflight.json", preflight_result)
            history = CanonicalHistory(root, dates[0], plan["decision_probe_date"], preflight_result)
            fmap = history.feature_map(plan["nonconstant_features"], plan["bandwidth_index"])
            if fmap.dimensions != 115:
                raise ValueError("Expected all 115 original characteristics")
            bases, identity_checks = [], []
            theta0 = np.zeros(fmap.dimensions)
            # First create the fixed lookback matrices, comparing every date to accepted moments.
            for day in dates:
                basis = MomentBasis(history.training_sample(day, settings.cutoff))
                actual = basis.matrices(fmap, theta0)
                expected = history.identity_reference(day, plan["bandwidth_index"])
                p = plan["nonconstant_features"]
                indices = np.r_[0, np.arange(1, 1 + p // 2), np.arange(257, 257 + p // 2)]
                if not np.array_equal(basis.sample.market.ids, expected["ids"]):
                    raise ValueError("Identity-reference security alignment failed")
                differences = {}
                for name, original in [("score", "r_tilde"), ("risk", "risk"), ("cost", "tc"), ("signal", "signal_t")]:
                    target = expected[original]
                    target = target[indices] if name == "score" else target[:, indices] if name == "signal" else target[np.ix_(indices, indices)]
                    np.testing.assert_allclose(actual[name], target, atol=1e-9, rtol=1e-8)
                    differences[name] = float(np.max(abs(actual[name] - target)))
                identity_checks.append(dict(decision_date=str(day.date()), max_absolute_differences=differences))
                bases.append(basis)
                print(f"Original-feature identity verified: {day:%Y-%m-%d}, {len(basis.sample.market.ids)} securities.", flush=True)
                resources(output, plan)
            write_json(output / "identity_checks.json", dict(status="PASS", dates=identity_checks, code_unchanged=True))
            # A real-data directional derivative check, with no geometry optimization yet.
            direction = np.linspace(-1., 1., fmap.dimensions)
            direction /= np.linalg.norm(direction)
            small = [bases[0]]
            identity = profile(bases, fmap, theta0, settings)
            central = profile(small, fmap, theta0, settings)
            epsilon = 1e-5
            fd = (profile(small, fmap, theta0 + epsilon * direction, settings, gradient=False)["value"]
                  - profile(small, fmap, theta0 - epsilon * direction, settings, gradient=False)["value"]) / (2 * epsilon)
            analytic = float(central["gradient"] @ direction)
            np.testing.assert_allclose(analytic, fd, atol=1e-8, rtol=1e-5)
            write_json(output / "historical_gradient_check.json", dict(status="PASS", analytic=analytic,
                        finite_difference=fd, absolute_error=abs(analytic-fd), decision_date=small[0].sample.market.decision_date))
            results = {}
            for objective in plan["geometry_objectives"]:
                def progress(record):
                    resources(output, plan)
                    print(json.dumps(record), flush=True)
                result = fit_geometry(bases, fmap, settings, objective=objective, progress=progress)
                model = GeometryModel.from_fit(fmap, result)
                model.save(output / f"model_{objective}.npz")
                with (output / f"moments_{objective}.npz").open("xb") as handle:
                    np.savez_compressed(handle, **{k: result[k] for k in ("score", "risk", "cost", "beta", "geometry_beta", "theta")})
                summary = {k: v for k, v in result.items() if k not in ("score", "risk", "cost", "beta", "geometry_beta", "theta")}
                summary.update(scale_min=float(np.exp(result["theta"]).min()), scale_max=float(np.exp(result["theta"]).max()),
                               theta_sum=float(result["theta"].sum()), max_abs_theta=float(np.max(abs(result["theta"]))))
                write_json(output / f"fit_{objective}.json", summary)
                pd.DataFrame(dict(characteristic=history.features, theta=result["theta"], scale=np.exp(result["theta"]))).to_csv(output / f"scales_{objective}.csv", index=False)
                results[objective] = summary
            # Dated decision integration only: no future outcomes, return score or portfolio backtest.
            probe = history.market(plan["decision_probe_date"])
            archived = pd.read_parquet(root / "output/step6_baseline_1981_2020/PolicyDecision.parquet",
                                       filters=[("eom", "==", pd.Timestamp(probe.decision_date))])
            archived = archived.set_index("id").loc[probe.ids]
            h = archived.pretrade.to_numpy()
            probes = []
            for objective in plan["geometry_objectives"]:
                loaded = GeometryModel.load(output / f"model_{objective}.npz")
                decision = act(loaded, probe.decision_date, probe.ids, probe.characteristics[0], probe.volatility[0], probe.m, probe.ids, h)
                expected = probe.m @ h + (np.eye(len(h)) - probe.m) @ decision["aim"]
                np.testing.assert_array_equal(decision["weights"], expected)
                probes.append(dict(objective=objective, decision_date=probe.decision_date, security_count=len(h),
                                   original_rule_max_error=float(np.max(abs(decision["weights"]-expected))),
                                   trade_l2_norm=float(np.linalg.norm(decision["trades"])), returns_evaluated=False))
            write_json(output / "decision_integration.json", dict(status="PASS", probes=probes))
            if any(sha(Path(path)) != digest for path, digest in source_hashes.items()):
                raise ValueError("Implementation changed during the locked exercise")
            result = dict(status="BOUNDED_PROTOTYPE_VERIFIED", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic()-started, resources=resources(output, plan),
                          fixed_operator_cache_gib=sum(b.nbytes for b in bases)/2**30,
                          training_months=len(bases), input_characteristics=fmap.dimensions,
                          nonconstant_features=fmap.columns-1, learned_scale_degrees_of_freedom=fmap.dimensions-1,
                          identity_geometry_profile_value=identity["value"], geometry_fits=results,
                          actual_consumed_file_sha256=history.consumed,
                          historical_geometry_fit_run=True, historical_portfolio_path_run=False,
                          original_inertia_and_trading_rule_unchanged=True,
                          performance_improvement_established=False, originality_established=False,
                          interpretation="In-sample development-year optimization and implementation verification only; no net-utility performance finding",
                          pending_comparisons=plan["full_study_comparisons_pending"])
            write_json(output / "prototype_result.json", result)
            files = [p for p in output.iterdir() if p.is_file()]
            write_json(output / "artifact_manifest.json", {p.name: sha(p) for p in files})
            print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "resources", "historical_portfolio_path_run")}), flush=True)
            return result
    except BaseException as exc:
        write_json(output / "failure.json", dict(status="FAILED_RETAINED_FOR_REVIEW", error_type=type(exc).__name__, message=str(exc)))
        raise
