"""Isolated, resumable continuation of three predetermined E2 development fits.

Accepted E2 code and results are read-only. This runner continues their exact
projected Armijo updates from update 8 to at most update 32. Validation returns
never enter fitting or select an intermediate update. The twelve-month paths
are isolated perturbation diagnostics, not a newly fitted continuous strategy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline.core import report_returns, transition
from feature_geometry.features import fingerprint, project_scales
from feature_geometry.history import CanonicalHistory
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings, validate_training
from geometry_study.features import CachedFeatureMap
from geometry_study import kernel

PARENT = ROOT / "output/feature_geometry_study_20260914"
DEFAULT_OUT = ROOT / "output/e2_convergence_pilot_20260928"
YEARS = (1981, 1990, 2000)
INITIAL_UPDATE, FINAL_UPDATE = 8, 32
PROTECTED = ("GOAL.md", "docs/STEP_1_THEORY.md", "docs/RECONSTRUCTION_AMENDMENT.md",
             "config/protocol_v1.json", "config/reconstruction_amendment.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def atomic_json(path, value, *, replace_existing=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(folder, kind, **payload):
    record = dict(at_utc=now(), event=kind, **clean(payload))
    with (folder / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, allow_nan=False), flush=True)


def verify_inventory(inventory):
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError("Changed or out-of-workspace dependency: " + relative)


def accepted_file(path, inventory, consumed):
    relative = str(Path(path).relative_to(ROOT))
    expected = inventory.get(relative)
    if expected is None or sha(path) != expected:
        raise ValueError("Missing/changed accepted artifact: " + relative)
    consumed[relative] = expected
    return Path(path)


def code_inventory():
    paths = [Path(__file__), ROOT / "baseline/core.py", ROOT / "baseline/io.py"]
    paths += sorted((ROOT / "feature_geometry").glob("*.py"))
    paths += [ROOT / "geometry_study/kernel.py", ROOT / "geometry_study/features.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def state_arrays(theta, current):
    return dict(theta=theta, beta=current["beta"], score=current["score"],
                risk=current["risk"], cost=current["cost"],
                value=np.array(current["value"]),
                solve_residual=np.array(current["solve_residual"]))


def current_from_arrays(arrays):
    return dict(beta=arrays["beta"], score=arrays["score"], risk=arrays["risk"],
                cost=arrays["cost"], value=float(arrays["value"]),
                solve_residual=float(arrays["solve_residual"]))


def save_checkpoint(folder, update, theta, current, detail, lock_sha, previous):
    target = folder / "checkpoints" / f"update_{update:03d}.npz"
    atomic_npz(target, **state_arrays(theta, current))
    commit = target.with_suffix(".json")
    atomic_json(commit, dict(update=update, arrays_file=target.name,
                            arrays_sha256=sha(target), lock_sha256=lock_sha,
                            previous_commit_sha256=previous, detail=detail))
    return sha(commit)


def recover_checkpoints(folder, lock_sha):
    directory = folder / "checkpoints"
    directory.mkdir(exist_ok=True)
    # Interrupted writes are retained, never mistaken for accepted checkpoints.
    orphaned = [p for p in directory.iterdir()
                if ".partial_" in p.name or
                (p.suffix == ".npz" and not p.with_suffix(".json").exists())]
    if orphaned:
        archive = folder / "retained_incomplete" / (str(time.time_ns()) + "_checkpoints")
        archive.mkdir(parents=True)
        for path in orphaned:
            path.rename(archive / path.name)
        event(folder, "retained_uncommitted_checkpoint_files", count=len(orphaned),
              directory=str(archive.relative_to(ROOT)))
    previous = None
    latest = None
    records = []
    for index, path in enumerate(sorted(directory.glob("update_*.json"))):
        commit = read(path)
        update = INITIAL_UPDATE + index
        target = path.with_suffix(".npz")
        if (commit["update"] != update or update > FINAL_UPDATE
                or commit["arrays_file"] != target.name
                or commit["lock_sha256"] != lock_sha
                or commit["previous_commit_sha256"] != previous
                or sha(target) != commit["arrays_sha256"]):
            raise ValueError("Checkpoint chain/hash mismatch: " + str(path))
        with np.load(target, allow_pickle=False) as data:
            latest = {key: data[key].copy() for key in data.files}
        previous = sha(path)
        records.append(commit)
    return latest, records, previous


def continue_updates(bases, fmap, settings, normalizer, theta, current,
                     start_update, final_update, on_accept=None, progress=None):
    """The original matrix_fit loop, starting at a previously accepted state.

    Existing scale normalization, gradient, projection, step normalization,
    Armijo test, and exact conditional coefficient solve remain unchanged.
    """
    validate_training(bases, fmap, settings)
    if (isinstance(start_update, bool) or isinstance(final_update, bool)
            or not isinstance(start_update, int) or not isinstance(final_update, int)
            or not 0 <= start_update <= final_update <= FINAL_UPDATE
            or not np.isfinite(normalizer) or normalizer <= 0):
        raise ValueError("Invalid continuation budget or fixed objective normalizer")
    theta = np.asarray(theta, dtype=float).copy()
    if (theta.shape != (fmap.dimensions,) or not np.isfinite(theta).all()
            or abs(float(theta.sum())) > 1e-10
            or float(np.max(abs(theta))) > settings.bound + 1e-12):
        raise ValueError("Initial scales violate the original zero-sum box")
    for key, shape in (("beta", (fmap.columns,)), ("score", (fmap.columns,)),
                       ("risk", (fmap.columns, fmap.columns)),
                       ("cost", (fmap.columns, fmap.columns))):
        if np.asarray(current[key]).shape != shape or not np.isfinite(current[key]).all():
            raise ValueError("Invalid resumed conditional state: " + key)
    if not np.isfinite(current["value"]) or not 0 <= current["solve_residual"] <= 1e-9:
        raise ValueError("Nonfinite or inaccurate resumed conditional solve")
    status = "ITERATION_BUDGET_REACHED"
    evaluations = 0
    trace = []
    first_gradient_norm = None
    update = start_update
    for proposed in range(start_update + 1, final_update + 1):
        raw_gradient = kernel.gradient(bases, fmap, theta, current["beta"], settings, "full")
        gradient = raw_gradient / normalizer
        pg = float(np.max(abs(project_scales(theta + gradient, settings.bound) - theta)))
        if first_gradient_norm is None:
            first_gradient_norm = pg
        if progress:
            progress("gradient", update=update, projected_gradient=pg)
        if pg <= settings.gradient_tolerance:
            status = "PROJECTED_GRADIENT_TOLERANCE"
            break
        step = settings.initial_step / max(float(np.max(abs(gradient))), 1e-30)
        accepted = False
        for backtracks in range(settings.max_backtracks):
            trial_theta = project_scales(theta + step * gradient, settings.bound)
            displacement = trial_theta - theta
            slope = float(raw_gradient @ displacement)
            if slope <= 0:
                break
            trial = kernel.evaluate(bases, fmap, trial_theta, settings, "full")
            evaluations += 1
            if trial["value"] >= current["value"] + settings.armijo * slope:
                accepted = True
                break
            step *= .5
        if not accepted:
            status = "LINE_SEARCH_STALLED"
            break
        theta, current, update = trial_theta, trial, proposed
        detail = dict(iteration=update, value=current["value"], backtracks=backtracks,
                      step=step, projected_gradient_before=pg)
        trace.append(detail)
        if on_accept:
            on_accept(update, theta, current, detail)
        if progress:
            progress("accepted_update", **detail)
    final_gradient = kernel.gradient(bases, fmap, theta, current["beta"], settings, "full") / normalizer
    final_pg = float(np.max(abs(project_scales(theta + final_gradient, settings.bound) - theta)))
    return dict(theta=theta, current=current, update=update, status=status,
                trace=trace, profile_evaluations_this_attempt=evaluations,
                projected_gradient_start=first_gradient_norm,
                projected_gradient_final=final_pg,
                normalized_gradient_final=final_gradient)


def parent_inputs(year, preflight_path, consumed):
    acceptance = read(PARENT / "acceptance_manifest.json")
    if (acceptance.get("accepted") is not True or acceptance.get("study_complete") is not True
            or not all(item["passed"] for item in acceptance["gates"])):
        raise ValueError("Accepted, completed E2 study required")
    for path in (PARENT / "acceptance_manifest.json",
                 ROOT / "output/step3_author_conventions_20260911/bundle_checksums.json",
                 ROOT / "output/step4_20260911/acceptance_manifest.json",
                 ROOT / "output/step6_20260911/preparation_manifest.json",
                 ROOT / "output/step6_20260911/start.json"):
        consumed[str(path.relative_to(ROOT))] = sha(path)
    inventory = acceptance["artifact_sha256"]
    model_dir = PARENT / f"fits/{year}/seed1/full"
    meta = read(accepted_file(model_dir / "fit.json", inventory, consumed))
    parent = GeometryModel.load(accepted_file(model_dir / "model.npz", inventory, consumed))
    parent_lock = read(accepted_file(PARENT / "plan_lock.json", inventory, consumed))
    parent_preflight = PARENT / "financial_preflight.json"
    if sha(parent_preflight) != parent_lock["financial_preflight_sha256"]:
        raise ValueError("Parent preflight changed from its accepted study lock")
    consumed[str(parent_preflight.relative_to(ROOT))] = parent_lock["financial_preflight_sha256"]
    receipt = read(preflight_path)
    if receipt != read(PARENT / "financial_preflight.json"):
        raise ValueError("Fresh financial preflight differs from the accepted parent")
    for name, expected in meta["files_sha256"].items():
        if sha(model_dir / name) != expected:
            raise ValueError("Parent model hash differs from its fit receipt")
    settings = FitSettings(**meta["settings"])
    if (meta["year"] != year or meta["seed"] != 1 or meta["method"] != "full"
            or meta["status"] != "ITERATION_BUDGET_REACHED"
            or settings.max_steps != INITIAL_UPDATE
            or settings.gradient_tolerance != 1e-6
            or meta["trace"][-1]["iteration"] != INITIAL_UPDATE
            or settings.cutoff != f"{year - 1}-12-31"
            or meta["training_first"] != "1965-11-30"
            or meta["training_last"] != f"{year - 1}-11-30"
            or parent.cutoff != settings.cutoff
            or parent.source_fingerprint != meta["source_fingerprint"]):
        raise ValueError("Parent is not the specified eight-update development fit")
    normalizer = float(meta["objective_normalizer"])
    if not normalizer > 0 or not np.isclose(settings.scale_penalty, .01 * normalizer, atol=0, rtol=1e-15):
        raise ValueError("Original objective normalization/penalty changed")
    # Raw panel and volatility scaling are verified before the adapter opens them.
    raw_inventory = read(ROOT / "output/step3_author_conventions_20260911/bundle_checksums.json")
    risk_inventory = read(ROOT / "output/step4_20260911/acceptance_manifest.json")["artifact_sha256"]
    accepted_file(ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet", raw_inventory, consumed)
    accepted_file(ROOT / "data/prepared/step4_v1/risk/scales.parquet", risk_inventory, consumed)
    accepted_file(PARENT / "paths/monthly.parquet", inventory, consumed)
    for day in pd.date_range(settings.cutoff, f"{year}-11-30", freq="ME"):
        accepted_file(PARENT / f"paths/{day.date()}.npz", inventory, consumed)
    return parent, meta, settings, normalizer, receipt, inventory


def make_training(year, receipt, meta, folder):
    source = CanonicalHistory(ROOT, meta["training_first"], meta["training_last"], receipt)
    del source.data  # Market construction uses only per-month frames.
    gc.collect()
    bases = []
    dates = pd.date_range(meta["training_first"], meta["training_last"], freq="ME")
    for index, day in enumerate(dates, 1):
        bases.append(kernel.LeanBasis(source.training_sample(day, meta["cutoff"])))
        if index % 12 == 0 or index == len(dates):
            event(folder, "training_input_progress", year=year, months=index,
                  total_months=len(dates), last_decision=str(day.date()))
    consumed = dict(source.consumed)
    source_key = fingerprint(*(basis.sample.fingerprint for basis in bases))
    if source_key != meta["source_fingerprint"] or len(bases) != meta["training_months"]:
        raise ValueError("Historical training samples differ from accepted E2 inputs")
    source.months.clear()
    del source
    gc.collect()
    return bases, consumed


def fit_year(year, parent, meta, settings, normalizer, receipt, folder, lock_sha):
    completed_path = folder / "fit_result.json"
    diagnostics_path = folder / "fit_diagnostics.npz"
    latest, records, previous = recover_checkpoints(folder, lock_sha)
    if completed_path.exists():
        result = read(completed_path)
        if (result["lock_sha256"] != lock_sha or previous != result["checkpoint_sha256"]
                or not records or result["final_update"] != records[-1]["update"]
                or sha(diagnostics_path) != result["fit_diagnostics_sha256"]):
            raise ValueError("Completed continuation no longer matches its checkpoints")
        verify_inventory(result["consumed_input_sha256"])
        event(folder, "reused_verified_fit", final_update=result["final_update"])
        return latest["theta"], latest["beta"], result
    if diagnostics_path.exists():
        retained = folder / "retained_incomplete"
        retained.mkdir(exist_ok=True)
        destination = retained / (str(time.time_ns()) + "_fit_diagnostics.npz")
        diagnostics_path.rename(destination)
        event(folder, "retained_uncommitted_fit_diagnostics", path=str(destination.relative_to(ROOT)))
    bases, consumed = make_training(year, receipt, meta, folder)
    fmap = CachedFeatureMap(parent.feature_map.frequencies, parent.feature_map.source)
    recovered = kernel.evaluate(bases, fmap, parent.theta, settings, "full")
    np.testing.assert_allclose(recovered["beta"], parent.beta, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(recovered["value"], meta["final_training_value"], atol=1e-12, rtol=1e-10)
    zero = kernel.evaluate(bases, fmap, np.zeros(115), settings, "full")
    np.testing.assert_allclose(zero["value"], normalizer, atol=1e-12, rtol=1e-10)
    identity_path = PARENT / f"fits/{year}/seed1/fixed512/model.npz"
    acceptance = read(PARENT / "acceptance_manifest.json")["artifact_sha256"]
    accepted_file(identity_path, acceptance, consumed)
    identity = GeometryModel.load(identity_path)
    np.testing.assert_allclose(zero["beta"], identity.beta, atol=1e-11, rtol=1e-10)
    parent_gradient = kernel.gradient(bases, fmap, parent.theta, recovered["beta"], settings, "full") / normalizer
    parent_pg = float(np.max(abs(project_scales(parent.theta + parent_gradient,
                                              settings.bound) - parent.theta)))
    neutral = dict(parent_beta_max_abs_error=float(np.max(abs(recovered["beta"] - parent.beta))),
                   parent_profile_value_error=float(abs(recovered["value"] - meta["final_training_value"])),
                   identity_beta_max_abs_error=float(np.max(abs(zero["beta"] - identity.beta))),
                   identity_objective_normalizer_error=float(abs(zero["value"] - normalizer)),
                   source_fingerprint_recovered=True, passed=True)
    event(folder, "neutrality_passed", **neutral)
    if latest is None:
        theta, current, update = parent.theta.copy(), recovered, INITIAL_UPDATE
        previous = save_checkpoint(folder, update, theta, current,
                                   dict(origin="archived_E2_recovered", projected_gradient=parent_pg),
                                   lock_sha, None)
    else:
        theta, current, update = latest["theta"], current_from_arrays(latest), records[-1]["update"]
        fresh = kernel.evaluate(bases, fmap, theta, settings, "full")
        for key in ("beta", "score", "risk", "cost", "value"):
            np.testing.assert_allclose(fresh[key], current[key], atol=1e-12, rtol=1e-10)
        event(folder, "resumed_checkpoint", update=update)

    def commit(iteration, th, state, detail):
        nonlocal previous
        previous = save_checkpoint(folder, iteration, th, state, detail, lock_sha, previous)

    started = time.monotonic()
    result = continue_updates(bases, fmap, settings, normalizer, theta, current,
                              update, FINAL_UPDATE, commit,
                              lambda kind, **item: event(folder, kind, **item))
    atomic_npz(diagnostics_path, theta_update8=parent.theta,
               normalized_gradient_update8=parent_gradient,
               theta_final=result["theta"], normalized_gradient_final=result["normalized_gradient_final"])
    result_meta = dict(year=year, seed=1, status=result["status"], final_update=result["update"],
                      maximum_total_updates=FINAL_UPDATE, original_settings=asdict(settings),
                      objective_normalizer=normalizer, neutral_gate=neutral,
                      projected_gradient_update8=parent_pg,
                      projected_gradient_final=result["projected_gradient_final"],
                      training_value_update8=recovered["value"],
                      training_value_final=result["current"]["value"],
                      relative_training_value_change=(result["current"]["value"] - recovered["value"]) / normalizer,
                      theta_max_abs_change=float(np.max(abs(result["theta"] - parent.theta))),
                      theta_l2_change=float(np.linalg.norm(result["theta"] - parent.theta)),
                      scale_max_abs_change=float(np.max(abs(np.exp(result["theta"]) - np.exp(parent.theta)))),
                      beta_max_abs_change=float(np.max(abs(result["current"]["beta"] - parent.beta))),
                      final_solve_residual=result["current"]["solve_residual"],
                      optimization_seconds_this_attempt=time.monotonic() - started,
                      profile_evaluations_this_attempt=result["profile_evaluations_this_attempt"],
                      checkpoint_sha256=previous, lock_sha256=lock_sha,
                      fit_diagnostics_sha256=sha(diagnostics_path),
                      consumed_input_sha256=consumed,
                      validation_used_to_select_iteration=False,
                      converged=result["status"] == "PROJECTED_GRADIENT_TOLERANCE")
    atomic_json(completed_path, result_meta)
    theta, beta = result["theta"].copy(), result["current"]["beta"].copy()
    del bases, fmap, zero, recovered, current, result
    gc.collect()
    return theta, beta, result_meta


def validate_year(year, parent, theta, beta, receipt, folder, inventory, consumed):
    start, end = f"{year - 1}-12-31", f"{year}-11-30"
    source = CanonicalHistory(ROOT, start, end, receipt)
    del source.data
    gc.collect()
    new = GeometryModel(parent.feature_map, theta, beta, parent.cutoff, parent.bound,
                        parent.source_fingerprint)
    raw = pd.read_parquet(ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet",
                          columns=["id", "eom", "tr_ld1", "mu_ld1"],
                          filters=[("eom", ">=", pd.Timestamp(start)), ("eom", "<=", pd.Timestamp(end))])
    outcomes = {d: g.sort_values("id").set_index("id") for d, g in raw.groupby("eom")}
    archived_monthly = pd.read_parquet(PARENT / "paths/monthly.parquet")
    archived_monthly = archived_monthly[(archived_monthly.seed == 1) & (archived_monthly.method == "full")]
    archived_monthly = archived_monthly.set_index("decision_eom")
    # Interrupted validation attempts are retained; the cheap twelve-month
    # diagnostic can rerun without recomputing a single optimizer update.
    attempt = folder / "validation_attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    holdings = None
    models = (parent, new)
    labels = ("archived8", "continued")
    monthly = []
    errors = dict(parent_weight=0., parent_target=0., parent_pretrade=0.,
                  parent_monthly=0., direct_rule=0., independent_accounting=0., direct_drift=0.)
    target_differences = []
    for day in pd.date_range(start, end, freq="ME"):
        path = accepted_file(PARENT / f"paths/{day.date()}.npz", inventory, consumed)
        with np.load(path, allow_pickle=False) as data:
            saved = {key: data[key].copy() for key in data.files}
        col = np.flatnonzero((saved["seeds"] == 1) & (saved["methods"] == "full"))
        if len(col) != 1:
            raise ValueError("Missing/duplicate archived parent path")
        col = int(col[0])
        market = source.market(day)
        np.testing.assert_array_equal(market.ids, saved["ids"])
        if holdings is None:
            holdings = np.repeat(saved["pretrade"][:, col, None], 2, axis=1)
        np.testing.assert_allclose(holdings[:, 0], saved["pretrade"][:, col], atol=2e-8, rtol=1e-7)
        next_ids = saved["next_ids"]
        row = outcomes[day].loc[market.ids]
        excess = source.months[day].loc[market.ids, "ret_ld1"].to_numpy()
        if row.mu_ld1.nunique() != 1:
            raise ValueError("Inconsistent market total return")
        targets = np.column_stack([model.aim(market.characteristics[0], market.volatility[0], str(day.date())) for model in models])
        next_holdings = np.empty((len(next_ids), 2))
        weights, trades, drifts = [], [], []
        month_records = []
        target_differences.append(float(np.linalg.norm(targets[:, 1] - targets[:, 0])))
        for index, label in enumerate(labels):
            result = transition(market.ids, holdings[:, index], targets[:, index], market.m,
                                row.tr_ld1.to_numpy(), float(row.mu_ld1.iloc[0]),
                                market.impact, market.wealth, excess, next_ids)
            direct = holdings[:, index] + (np.eye(len(market.ids)) - market.m) @ (targets[:, index] - holdings[:, index])
            gross = math.fsum(float(w) * float(r) for w, r in zip(direct, excess))
            cost = .5 * market.wealth * math.fsum(float(l) * float(q) ** 2 for l, q in zip(market.impact, direct - holdings[:, index]))
            np.testing.assert_allclose(direct, result["weights"], atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose([gross, cost], [result["gross_excess"], result["impact_cost"]], atol=1e-12, rtol=1e-10)
            drift = direct * (1 + row.tr_ld1.to_numpy()) / (1 + float(row.mu_ld1.iloc[0]))
            lookup = dict(zip(market.ids, drift))
            independent_next = np.array([lookup.get(sid, 0.) for sid in next_ids])
            np.testing.assert_allclose(independent_next, result["next_pretrade"], atol=1e-12, rtol=1e-10)
            errors["direct_rule"] = max(errors["direct_rule"], float(np.max(abs(direct - result["weights"]))))
            errors["independent_accounting"] = max(errors["independent_accounting"], abs(gross - result["gross_excess"]), abs(cost - result["impact_cost"]))
            errors["direct_drift"] = max(errors["direct_drift"], float(np.max(abs(independent_next - result["next_pretrade"]))))
            next_holdings[:, index] = result["next_pretrade"]
            weights.append(result["weights"])
            trades.append(result["trades"])
            drifts.append(result["drifted_weights"])
            record = dict(year=year, seed=1, method=label, decision_eom=day,
                          return_eom=day + pd.offsets.MonthEnd(1), gross_excess=gross,
                          impact_cost=cost, net_excess=gross - cost,
                          turnover=result["turnover"], gross_exposure=result["gross_exposure"])
            monthly.append(record)
            month_records.append(record)
            if index == 0:
                for key, actual in (("weight", result["weights"]), ("target", targets[:, 0]), ("pretrade", holdings[:, 0])):
                    np.testing.assert_allclose(actual, saved[key][:, col], atol=2e-8, rtol=1e-7)
                    errors["parent_" + key] = max(errors["parent_" + key], float(np.max(abs(actual - saved[key][:, col]))))
                old = archived_monthly.loc[day]
                for key in ("gross_excess", "impact_cost", "net_excess", "turnover", "gross_exposure"):
                    np.testing.assert_allclose(record[key], old[key], atol=1e-10, rtol=1e-8)
                    errors["parent_monthly"] = max(errors["parent_monthly"], abs(record[key] - float(old[key])))
        snapshot = attempt / f"{day.date()}.npz"
        atomic_npz(snapshot, ids=market.ids, next_ids=next_ids, pretrade=holdings,
                   target=targets, weight=np.column_stack(weights), trade=np.column_stack(trades),
                   drifted_weight=np.column_stack(drifts), next_pretrade=next_holdings,
                   methods=np.array(labels))
        atomic_json(snapshot.with_suffix(".json"), dict(arrays_sha256=sha(snapshot), monthly=month_records))
        holdings = next_holdings
        event(folder, "development_validation_month_complete", decision=str(day.date()))
    consumed.update(source.consumed)
    frame = pd.DataFrame(monthly)
    frame.to_csv(attempt / "monthly.csv", index=False)
    metrics = {}
    for label in labels:
        block = frame[frame.method == label]
        metrics[label] = dict(report_returns(block.gross_excess.to_numpy(), block.impact_cost.to_numpy()),
                              annual_mean_gross_excess=float(12 * block.gross_excess.mean()),
                              annual_impact_cost=float(12 * block.impact_cost.mean()),
                              annual_gross_sample_volatility=float(np.sqrt(12 * block.gross_excess.var(ddof=1))),
                              average_turnover=float(block.turnover.mean()),
                              average_gross_exposure=float(block.gross_exposure.mean()))
    differences = {key: metrics["continued"][key] - metrics["archived8"][key]
                   for key in metrics["archived8"] if key != "months"
                   and metrics["continued"][key] is not None and metrics["archived8"][key] is not None}
    report = dict(year=year, seed=1, months=12, metrics=metrics, continued_minus_archived=differences,
                  mean_monthly_target_l2_change=float(np.mean(target_differences)),
                  recovery_and_accounting_max_errors=errors, passed=True,
                  initialization="Both arms use archived E2/full pretrade holdings at the first December decision",
                  holdings_rule="Own holdings carried through all twelve decisions; entry/exit and costs unchanged",
                  scope="Isolated development-year perturbation; not a new continuous historical strategy",
                  statistical_significance_established=False, previously_exposed_development_outcomes=True,
                  intermediate_iteration_selected_using_returns=False,
                  attempt_directory=str(attempt.relative_to(ROOT)))
    atomic_json(attempt / "result.json", report)
    return report, attempt


def run(year, preflight_path, output_root, workers):
    folder = output_root / "years" / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    run_lock = (output_root / ".pilot_run.lock").open("a")
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    completion_path = folder / "completion.json"
    if completion_path.exists():
        completed = read(completion_path)
        verify_inventory(completed["artifact_sha256"])
        verify_inventory(completed["consumed_input_sha256"])
        verify_inventory(read(folder / "lock.json")["code_sha256"])
        print(json.dumps(dict(status="REUSED_VERIFIED_COMPLETED_YEAR", year=year,
                              result=str(completion_path))), flush=True)
        return
    event(folder, "attempt_started", year=year, pid=os.getpid(), workers=workers,
          blas_threads=1, fixed_terminal_update=FINAL_UPDATE)
    consumed = {}
    parent, meta, settings, normalizer, receipt, inventory = parent_inputs(year, preflight_path, consumed)
    protected = read(ROOT / "output/step6_20260911/start.json")["protected"]
    verify_inventory(protected)
    lock = dict(year=year, seed=1, initial_update=INITIAL_UPDATE, final_update=FINAL_UPDATE,
                method="Exact original normalized projected Armijo continuation; no new grid",
                annual_parent_directory=str((PARENT / f"fits/{year}/seed1/full").relative_to(ROOT)),
                original_settings=asdict(settings), objective_normalizer=normalizer,
                historical_validation_used_for_iteration_selection=False,
                economic_definition="Unchanged gamma10, reciprocal wealth, original m and cost accounting",
                workers=workers, blas_threads=1, code_sha256=code_inventory(),
                preflight_sha256=sha(preflight_path), parent_inputs_sha256=dict(consumed),
                protected_sha256=protected)
    if (folder / "lock.json").exists():
        if read(folder / "lock.json") != lock:
            raise ValueError("Existing pilot lock differs; preserve it and review the change")
    else:
        atomic_json(folder / "lock.json", lock)
    lock_sha = sha(folder / "lock.json")
    kernel._POOL.shutdown(wait=True)
    kernel._POOL = ThreadPoolExecutor(max_workers=workers)
    with threadpool_limits(limits=1):
        theta, beta, fit = fit_year(year, parent, meta, settings, normalizer, receipt, folder, lock_sha)
        consumed.update(fit["consumed_input_sha256"])
        if fit["status"] == "LINE_SEARCH_STALLED":
            raise RuntimeError("Continuation line search failed; fit retained for review without validation")
        validation, attempt = validate_year(year, parent, theta, beta, receipt, folder, inventory, consumed)
    verify_inventory(consumed)
    verify_inventory(protected)
    verify_inventory(lock["code_sha256"])
    artifacts = [folder / "lock.json", folder / "fit_result.json", folder / "fit_diagnostics.npz"]
    artifacts += sorted((folder / "checkpoints").glob("update_*"))
    artifacts += sorted(p for p in attempt.iterdir() if p.is_file())
    atomic_json(completion_path, dict(status="COMPLETED_BOUNDED_DEVELOPMENT_DIAGNOSTIC", year=year,
                                     scientific_improvement_or_significance_established=False,
                                     fit=fit, validation=validation,
                                     artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts},
                                     consumed_input_sha256=consumed,
                                     completed_at_utc=now()))
    event(folder, "year_complete", final_update=fit["final_update"],
          source_utility_change=validation["continued_minus_archived"]["source_gross_sample_variance_utility"],
          net_population_utility_change=validation["continued_minus_archived"]["net_population_utility"])


def self_test():
    """Synthetic exact continuation, checkpoint integrity and metric checks only."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_feature_geometry import fixture
    _, fmap, old, settings, _ = fixture(dimensions=6, frequencies=8, months=3)
    bases = [kernel.LeanBasis(basis.sample) for basis in old]
    settings = replace(settings, max_steps=8, gradient_tolerance=1e-20)
    with threadpool_limits(limits=1):
        normalizer = kernel.evaluate(bases, fmap, np.zeros(fmap.dimensions), settings)["value"]
        first = kernel.matrix_fit(bases, fmap, settings, objective_scale=normalizer)
        assert first["trace"][-1]["iteration"] == 8
        current = kernel.evaluate(bases, fmap, first["theta"], settings)
        resumed = continue_updates(bases, fmap, settings, normalizer, first["theta"], current, 8, 32)
        complete = kernel.matrix_fit(bases, fmap, replace(settings, max_steps=32), objective_scale=normalizer)
        np.testing.assert_allclose(resumed["theta"], complete["theta"], atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(resumed["current"]["beta"], complete["beta"], atol=1e-12, rtol=1e-11)
        assert [x["backtracks"] for x in resumed["trace"]] == [x["backtracks"] for x in complete["trace"][9:]]
        with tempfile.TemporaryDirectory(prefix="e2_pilot_checkpoint_test_") as temporary:
            folder = Path(temporary)
            last_hash = save_checkpoint(folder, 8, first["theta"], current, {}, "test-lock", None)
            arrays, records, last = recover_checkpoints(folder, "test-lock")
            assert last == last_hash and records[-1]["update"] == 8
            np.testing.assert_array_equal(arrays["theta"], first["theta"])
            target = folder / "checkpoints/update_008.npz"
            with target.open("ab") as stream:
                stream.write(b"tamper")
            try:
                recover_checkpoints(folder, "test-lock")
            except ValueError:
                pass
            else:
                raise AssertionError("Corrupted checkpoint was not rejected")
    gross, costs = np.array([.01, -.02, .03]), np.array([.001, .002, .003])
    metrics = report_returns(gross, costs)
    np.testing.assert_allclose(metrics["source_gross_sample_variance_utility"], 12 * ((gross-costs).mean() - 5 * gross.var(ddof=1)))
    print(json.dumps(dict(status="PASS", historical_fits_launched=False,
                          checks=["8-to32 continuation matches original32-step fit",
                                  "checkpoint exact recovery and tamper rejection",
                                  "source utility definition"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=YEARS)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, choices=(1, 4), default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.year is None or args.preflight is None:
        parser.error("Historical diagnostic requires --year and a fresh --preflight receipt")
    output_root = args.output_root.resolve()
    if output_root != DEFAULT_OUT.resolve():
        parser.error("This runner only writes its designated isolated pilot directory")
    preflight = args.preflight.resolve()
    if not preflight.is_relative_to(output_root) or not preflight.is_file():
        parser.error("Fresh preflight receipt must exist in the designated pilot directory")
    try:
        run(args.year, preflight, output_root, args.workers)
    except BaseException as error:
        folder = output_root / "years" / str(args.year)
        if folder.exists():
            event(folder, "attempt_stopped", error_type=type(error).__name__, message=str(error),
                  resume="Repeat the same command; verified optimizer checkpoints are retained")
        raise


if __name__ == "__main__":
    main()
