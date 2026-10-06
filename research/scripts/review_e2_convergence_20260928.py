"""Independent, new-only review of the three fixed E2 development diagnostics.

Does not fit a model, select an iteration, search a setting or test significance.
Refuses acceptance unless all three years and the resource supervisor completed.
The Python metrics, matrix accounting and scale projection below do not import
the fitting runner or its financial/reporting helpers. Base R recomputes every
reported metric from the 72 monthly arm observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/e2_convergence_pilot_20260928"
PARENT = ROOT / "output/feature_geometry_study_20260914"
YEARS = (1981, 1990, 2000)
METHODS = ("archived8", "continued")
SOURCE_METRIC = "source_gross_sample_variance_utility"
NET_METRIC = "net_population_utility"
COMPLETE = "COMPLETED_BOUNDED_DEVELOPMENT_DIAGNOSTIC"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def same(actual, expected, *, atol=1e-11, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(inventory, verified):
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), "Out-of-workspace source: " + relative)
        if relative in verified:
            require(verified[relative] == expected, "Conflicting expected hash: " + relative)
        else:
            require(sha(path) == expected, "Changed artifact/source: " + relative)
            verified[relative] = expected


def project_zero_sum_box(vector, bound):
    """Independent scalar bisection of sum(clip(x_i-lambda)) = 0."""
    values = [float(x) for x in vector]
    low, high = min(values) - bound, max(values) + bound
    for _ in range(160):
        middle = low + (high - low) / 2
        trial = [min(bound, max(-bound, x - middle)) for x in values]
        if math.fsum(trial) > 0:
            low = middle
        else:
            high = middle
    return np.array([min(bound, max(-bound, x - (low + high) / 2)) for x in values])


def metrics(frame):
    gross = [float(x) for x in frame.gross_excess]
    cost = [float(x) for x in frame.impact_cost]
    net = [g - c for g, c in zip(gross, cost)]
    n = len(net)
    gm, cm, nm = (math.fsum(x) / n for x in (gross, cost, net))
    gv = math.fsum((x - gm) ** 2 for x in gross) / (n - 1)
    nv = math.fsum((x - nm) ** 2 for x in net) / n
    return dict(months=n, annual_mean_net_excess=12 * nm,
                annual_net_volatility=math.sqrt(12 * nv),
                net_population_utility=12 * (nm - 5 * nv),
                net_sharpe=math.sqrt(12) * nm / math.sqrt(nv),
                source_gross_sample_variance_utility=12 * (nm - 5 * gv),
                source_net_over_gross_sharpe=math.sqrt(12) * nm / math.sqrt(gv),
                annual_mean_gross_excess=12 * gm,
                annual_impact_cost=12 * cm,
                annual_gross_sample_volatility=math.sqrt(12 * gv),
                average_turnover=math.fsum(float(x) for x in frame.turnover) / n,
                average_gross_exposure=math.fsum(float(x) for x in frame.gross_exposure) / n)


def review_fit(year, completed, verified):
    folder = OUT / "years" / str(year)
    lock = read(folder / "lock.json")
    fit = read(folder / "fit_result.json")
    require(completed["fit"] == fit, "Completion embeds a different fit result")
    require(lock["year"] == year and lock["seed"] == 1, "Unexpected fit specification")
    require(lock["initial_update"] == 8 and lock["final_update"] == 32, "Changed update budget")
    require(lock["historical_validation_used_for_iteration_selection"] is False,
            "Returns used to select optimization endpoint")
    for key in ("code_sha256", "parent_inputs_sha256", "protected_sha256"):
        verify_hashes(lock[key], verified)
    require(sha(OUT / "financial_preflight.json") == lock["preflight_sha256"], "Preflight changed")
    require(read(OUT / "financial_preflight.json") == read(PARENT / "financial_preflight.json"),
            "Fresh preflight differs from accepted parent")
    original = read(PARENT / f"fits/{year}/seed1/full/fit.json")
    parent = arrays(PARENT / f"fits/{year}/seed1/full/model.npz")
    settings = fit["original_settings"]
    require(settings == original["settings"] == lock["original_settings"], "Original settings changed")
    require(settings["cutoff"] == f"{year-1}-12-31", "Incorrect training cutoff")
    require(original["training_last"] == f"{year-1}-11-30", "Incorrect last training decision")
    require(original["latest_available_at"] == settings["cutoff"], "Training maturity mismatch")
    require(fit["objective_normalizer"] == original["objective_normalizer"], "Objective normalizer changed")
    same(settings["scale_penalty"], .01 * fit["objective_normalizer"], atol=0, rtol=1e-15)
    require(fit["validation_used_to_select_iteration"] is False, "Validation selected endpoint")
    require(fit["neutral_gate"]["passed"] is True and fit["neutral_gate"]["source_fingerprint_recovered"] is True,
            "Fit neutral/source check failed")
    require(fit["status"] in ("ITERATION_BUDGET_REACHED", "PROJECTED_GRADIENT_TOLERANCE"),
            "Fit did not finish normally")
    final_update = fit["final_update"]
    require(8 <= final_update <= 32, "Endpoint outside protocol")
    require(final_update == 32 or fit["status"] == "PROJECTED_GRADIENT_TOLERANCE",
            "Premature endpoint lacks convergence")
    lock_hash, previous, states = sha(folder / "lock.json"), None, []
    maximum_residual = maximum_objective_error = 0.0
    for update in range(8, final_update + 1):
        checkpoint = folder / "checkpoints" / f"update_{update:03d}.json"
        commit = read(checkpoint)
        source = checkpoint.with_suffix(".npz")
        require(commit["update"] == update and commit["lock_sha256"] == lock_hash,
                "Checkpoint update/lock mismatch")
        require(commit["previous_commit_sha256"] == previous, "Broken checkpoint chain")
        require(commit["arrays_file"] == source.name and commit["arrays_sha256"] == sha(source),
                "Checkpoint arrays changed")
        current = arrays(source)
        require(all(np.isfinite(v).all() for v in current.values()), "Nonfinite checkpoint")
        theta, beta = current["theta"], current["beta"]
        require(theta.shape == (115,) and beta.shape == (513,), "Changed representation size")
        require(abs(math.fsum(theta)) <= 1e-10 and max(abs(theta)) <= settings["bound"] + 1e-12,
                "Infeasible scale vector")
        system = current["risk"] + current["cost"] + settings["ridge"] * np.eye(len(beta))
        residual = float(np.linalg.norm(system @ beta - current["score"]) /
                         max(np.linalg.norm(current["score"]), 1e-30))
        require(residual <= 1e-9, "Independent coefficient equation failed")
        objective = float(current["score"] @ beta - .5 * beta @ system @ beta
                          - .5 * settings["scale_penalty"] * theta @ theta)
        same(current["value"], objective, atol=1e-12)
        maximum_residual = max(maximum_residual, residual)
        maximum_objective_error = max(maximum_objective_error, abs(objective - float(current["value"])))
        if states:
            require(objective >= float(states[-1]["value"]) - 1e-13,
                    "Accepted training objective decreased")
            same(commit["detail"]["value"], objective, atol=1e-12)
        states.append(current)
        previous = sha(checkpoint)
    require(len(list((folder / "checkpoints").glob("update_*.json"))) == len(states),
            "Unexpected checkpoint outside accepted endpoint")
    require(previous == fit["checkpoint_sha256"] and lock_hash == fit["lock_sha256"],
            "Fit terminal checkpoint differs")
    same(states[0]["theta"], parent["theta"], atol=0, rtol=0)
    same(states[0]["beta"], parent["beta"])
    same(states[0]["value"], original["final_training_value"], atol=1e-12)
    same(states[0]["value"], fit["training_value_update8"], atol=1e-12)
    same(states[-1]["value"], fit["training_value_final"], atol=1e-12)
    same(fit["relative_training_value_change"],
         (float(states[-1]["value"]) - float(states[0]["value"])) / fit["objective_normalizer"], atol=1e-12)
    same(fit["theta_max_abs_change"], np.max(abs(states[-1]["theta"] - parent["theta"])), atol=1e-12)
    same(fit["theta_l2_change"], np.linalg.norm(states[-1]["theta"] - parent["theta"]), atol=1e-12)
    same(fit["scale_max_abs_change"], np.max(abs(np.exp(states[-1]["theta"]) - np.exp(parent["theta"]))), atol=1e-12)
    same(fit["beta_max_abs_change"], np.max(abs(states[-1]["beta"] - parent["beta"])), atol=1e-12)
    require(fit["converged"] is (fit["status"] == "PROJECTED_GRADIENT_TOLERANCE"),
            "Convergence label differs from actual stopping condition")
    diagnostics_path = folder / "fit_diagnostics.npz"
    require(sha(diagnostics_path) == fit["fit_diagnostics_sha256"], "Gradient diagnostics changed")
    diagnostic = arrays(diagnostics_path)
    norms = []
    for suffix, state in (("update8", states[0]), ("final", states[-1])):
        theta, gradient = diagnostic["theta_" + suffix], diagnostic["normalized_gradient_" + suffix]
        same(theta, state["theta"], atol=0, rtol=0)
        require(gradient.shape == (115,) and np.isfinite(gradient).all(), "Invalid endpoint gradient")
        norm = float(max(abs(project_zero_sum_box(theta + gradient, settings["bound"]) - theta)))
        same(norm, fit["projected_gradient_" + suffix], atol=1e-12)
        norms.append(norm)
    if fit["converged"]:
        require(norms[-1] <= settings["gradient_tolerance"] + 1e-12, "False convergence claim")
    return dict(year=year, accepted_updates=final_update,
                projected_gradient_update8=norms[0], projected_gradient_final=norms[1],
                gradient_decreased=bool(norms[1] < norms[0]),
                training_value_update8=float(states[0]["value"]),
                training_value_final=float(states[-1]["value"]),
                maximum_independent_solve_residual=maximum_residual,
                maximum_independent_objective_error=maximum_objective_error,
                finite_feasible_scales=True, neutral_receipt_passed=True,
                endpoint_gradient_vectors_reprojected=True,
                endpoint_gradient_from_raw_training_recomputed=False,
                converged=fit["converged"], checkpoints=len(states))


def review_monthly(year, completed, verified):
    validation = completed["validation"]
    directory = (ROOT / validation["attempt_directory"]).resolve()
    require(directory.is_relative_to(OUT / "years" / str(year) / "validation_attempts"),
            "Unexpected validation output directory")
    require(validation == read(directory / "result.json"), "Embedded validation result differs")
    require(validation["passed"] is True and validation["months"] == 12,
            "Validation incomplete or failed")
    require(validation["intermediate_iteration_selected_using_returns"] is False,
            "Validation selected intermediate model")
    table = pd.read_csv(directory / "monthly.csv", parse_dates=["decision_eom", "return_eom"])
    required = ["gross_excess", "impact_cost", "net_excess", "turnover", "gross_exposure"]
    require(len(table) == 24 and not table.duplicated(["decision_eom", "method"]).any(), "Wrong monthly record count")
    require(set(table.method) == set(METHODS) and set(table.seed) == {1} and set(table.year) == {year},
            "Unexpected arm, seed or year")
    require(np.isfinite(table[required].to_numpy()).all() and (table.impact_cost >= 0).all(),
            "Invalid return/accounting values")
    same(table.net_excess, table.gross_excess - table.impact_cost, atol=1e-14)
    dates = pd.date_range(f"{year-1}-12-31", f"{year}-11-30", freq="ME")
    for method in METHODS:
        block = table[table.method == method].sort_values("decision_eom")
        require(list(block.decision_eom) == list(dates), "Nonconsecutive validation decisions")
        require(list(block.return_eom) == list(dates + pd.offsets.MonthEnd(1)), "Return dates differ")
    raw = pd.read_parquet(ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet",
                          columns=["id", "eom", "ret_ld1", "tr_ld1", "mu_ld1", "lambda_diagonal", "wealth_usd"],
                          filters=[("eom", ">=", dates[0]), ("eom", "<=", dates[-1])])
    months = {d: b.set_index("id") for d, b in raw.groupby("eom")}
    accepted_monthly = pd.read_parquet(PARENT / "paths/monthly.parquet", filters=[
        ("seed", "==", 1), ("method", "==", "full"),
        ("decision_eom", ">=", dates[0]), ("decision_eom", "<=", dates[-1])]).set_index("decision_eom")
    previous = None
    max_rule = max_accounting = max_parent = 0.0
    for day in dates:
        p = directory / f"{day.date()}.npz"
        require(read(p.with_suffix(".json"))["arrays_sha256"] == sha(p), "Monthly arrays changed")
        position = arrays(p)
        archive = arrays(PARENT / "paths" / f"{day.date()}.npz")
        col = np.flatnonzero((archive["seeds"] == 1) & (archive["methods"] == "full"))
        require(len(col) == 1, "Missing archived neutral arm")
        col = int(col[0])
        for name in ("ids", "next_ids"):
            np.testing.assert_array_equal(position[name], archive[name])
        np.testing.assert_array_equal(position["methods"], np.array(METHODS))
        if previous is None:
            same(position["pretrade"], np.repeat(archive["pretrade"][:, col, None], 2, axis=1), atol=0, rtol=0)
        else:
            np.testing.assert_array_equal(position["pretrade"], previous)
        matrix = arrays(ROOT / "data/prepared/step6_v1/matrices" / f"{day.date()}.npz")
        np.testing.assert_array_equal(matrix["ids"], position["ids"])
        h, target = position["pretrade"], position["target"]
        w = h + (np.eye(len(h)) - matrix["m"]) @ (target - h)
        q = w - h
        data = months[day].loc[position["ids"]]
        require(data.mu_ld1.nunique() == 1 and data.wealth_usd.nunique() == 1, "Inconsistent market inputs")
        wealth, market = float(data.wealth_usd.iloc[0]), float(data.mu_ld1.iloc[0])
        drift = w * (1 + data.tr_ld1.to_numpy()[:, None]) / (1 + market)
        locations = {sid: i for i, sid in enumerate(position["ids"])}
        next_h = np.stack([drift[locations[sid]] if sid in locations else np.zeros(2) for sid in position["next_ids"]])
        for name, value in (("weight", w), ("trade", q), ("drifted_weight", drift), ("next_pretrade", next_h)):
            same(position[name], value, atol=1e-12)
            max_rule = max(max_rule, float(np.max(abs(position[name] - value))))
        for method_index, method in enumerate(METHODS):
            row = table[(table.decision_eom == day) & (table.method == method)].iloc[0]
            gross = math.fsum(float(x) * float(y) for x, y in zip(w[:, method_index], data.ret_ld1))
            cost = wealth / 2 * math.fsum(float(l) * float(x)**2 for l, x in zip(data.lambda_diagonal, q[:, method_index]))
            expected = [gross, cost, gross - cost, math.fsum(abs(q[:, method_index])), math.fsum(abs(w[:, method_index]))]
            same(row[required].to_numpy(dtype=float), expected, atol=1e-12)
            max_accounting = max(max_accounting, float(np.max(abs(row[required].to_numpy(dtype=float) - expected))))
        for name in ("pretrade", "target", "weight", "trade", "drifted_weight", "next_pretrade"):
            same(position[name][:, 0], archive[name][:, col], atol=2e-8, rtol=1e-7)
            max_parent = max(max_parent, float(np.max(abs(position[name][:, 0] - archive[name][:, col]))))
        archived = table[(table.decision_eom == day) & (table.method == "archived8")].iloc[0]
        same(archived[required].to_numpy(dtype=float), accepted_monthly.loc[day, required].to_numpy(dtype=float),
             atol=1e-10, rtol=1e-8)
        previous = position["next_pretrade"]
    rows = []
    for method in METHODS:
        result = metrics(table[table.method == method].sort_values("decision_eom"))
        for key, value in result.items():
            same(value, validation["metrics"][method][key], atol=1e-12)
        rows.append(dict(year=year, method=method, **result))
    for key, difference in validation["continued_minus_archived"].items():
        same(rows[1][key] - rows[0][key], difference, atol=1e-12)
    return table, rows, dict(months=12, arms=2, maximum_transition_error=max_rule,
                            maximum_accounting_error=max_accounting,
                            maximum_archived_position_error=max_parent,
                            own_holdings_continuity=True, all_months_checked=True,
                            continued_targets_independently_regenerated=False,
                            scope="Independent positions, drift, costs and returns conditional on saved targets")


R_SOURCE = r'''args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=FALSE)
stopifnot(nrow(d)==72L, all(is.finite(d$gross_excess)), all(d$impact_cost>=0))
rows <- list()
for (year in c(1981L,1990L,2000L)) for (method in c("archived8","continued")) {
  x <- d[d$year==year & d$method==method,]
  stopifnot(nrow(x)==12L)
  gross <- x$gross_excess; cost <- x$impact_cost; net <- gross-cost
  gv <- var(gross); nv <- mean((net-mean(net))^2)
  rows[[length(rows)+1L]] <- data.frame(year=year,method=method,months=length(net),
    annual_mean_net_excess=12*mean(net), annual_net_volatility=sqrt(12*nv),
    net_population_utility=12*(mean(net)-5*nv), net_sharpe=sqrt(12)*mean(net)/sqrt(nv),
    source_gross_sample_variance_utility=12*(mean(net)-5*gv),
    source_net_over_gross_sharpe=sqrt(12)*mean(net)/sqrt(gv),
    annual_mean_gross_excess=12*mean(gross), annual_impact_cost=12*mean(cost),
    annual_gross_sample_volatility=sqrt(12*gv), average_turnover=mean(x$turnover),
    average_gross_exposure=mean(x$gross_exposure))
}
write.table(do.call(rbind, rows), args[2], sep=",", row.names=FALSE, col.names=TRUE)
'''


def resource_review():
    execution = read(OUT / "execution.json")
    require(execution["status"] == "THREE_DIAGNOSTIC_YEARS_COMPLETE_PENDING_REVIEW",
            "Supervisor has not completed all three years")
    budget = dict(total_active_seconds=3600, max_physical_footprint_gib=16,
                  min_system_headroom_gib=4, max_additional_disk_gib=5, min_free_disk_gib=50)
    require(read(OUT / "plan.json")["compute_budget"] == budget, "Resource limits changed")
    starts = sorted((OUT / "attempts").glob("*_start.json"))
    require([p.name for p in starts] in (["0001_start.json"], ["0001_start.json", "0002_start.json"]),
            "Only the original attempt and one explicitly reviewed recovery are supported")
    resources = []
    attempts = []
    total_active = 0.0
    for start in starts:
        initial = read(start)
        require(initial["budget"] == budget and initial["clock"] == "mach_absolute_time()",
                "Attempt changed resource limits or accounting clock")
        same(initial["prior_active_seconds"], total_active, atol=1e-9, rtol=0)
        end = start.with_name(start.name.replace("_start", "_end"))
        require(end.exists(), "An execution attempt is still active or incomplete")
        terminal = read(end)
        telemetry = start.with_name(start.name.replace("_start.json", "_resources.jsonl"))
        samples = [json.loads(line) for line in telemetry.read_text().splitlines() if line]
        require(samples, "Attempt has no resource samples")
        require(all(b["active_seconds"] >= a["active_seconds"] for a, b in zip(samples, samples[1:])),
                "Attempt active clock moved backwards")
        require(terminal["active_seconds"] >= samples[-1]["active_seconds"], "Terminal active time lost sampled work")
        for row in samples:
            same(row["cumulative_active_seconds"], initial["prior_active_seconds"] + row["active_seconds"],
                 atol=1e-9, rtol=0)
        resources.extend(samples)
        attempts.append((initial, terminal))
        total_active += terminal["active_seconds"]
    require(attempts[-1][1]["status"] == "THREE_DIAGNOSTIC_YEARS_COMPLETE_PENDING_REVIEW"
            and attempts[-1][1]["error"] is None, "Final supervised attempt did not finish successfully")
    recovery_inputs = {}
    pressure_events = []
    if len(attempts) == 2:
        initial, stopped = attempts[0]
        require(stopped["status"] == "STOPPED"
                and stopped["error"] == "Native memory pressure is not normal: 2; syscall 0",
                "Unreviewed stop reason; only the specifically documented native-pressure event is eligible")
        recovery_path = OUT / "recovery_0002.json"
        binding_path = OUT / "recovery_0002_binding.json"
        require(recovery_path.exists() and binding_path.exists(), "Missing explicit recovery decision/hash binding")
        recovery, binding = read(recovery_path), read(binding_path)
        second_start = attempts[1][0]
        require(recovery["observed_pressure_before_resume"] == 1 and recovery["available_gib"] >= 4
                and recovery["same_code"] is True and recovery["new_resource_limits"] is False
                and recovery["saved_update"] == 23 and recovery["completed_years"] == [1981, 1990],
                "Recovery lacks required safeguards or changes scope")
        same(recovery["prior_active_seconds"], stopped["active_seconds"], atol=1e-9, rtol=0)
        same(recovery["remaining_budget_seconds"], 3600 - stopped["active_seconds"], atol=1e-9, rtol=0)
        same(recovery["swap_gib"], stopped["last_sample"]["swap_used_gib"], atol=0, rtol=0)
        require(recovery["same_plan_sha256"] == sha(OUT / "plan.json"), "Recovery changes frozen plan")
        require(stopped["ended_at_utc"] < recovery["at_utc"] < second_start["started_at_utc"],
                "Recovery decision was not recorded between stop and restart")
        required_paths = [recovery_path, OUT / "attempts/0001_end.json", OUT / "attempts/0002_start.json",
                          OUT / "years/1981/completion.json", OUT / "years/1990/completion.json",
                          OUT / "years/2000/checkpoints/update_023.json", OUT / "years/2000/checkpoints/update_023.npz",
                          OUT / "years/2000/lock.json", OUT / "plan.json"]
        bound = binding["source_sha256"]
        require(all(str(p.relative_to(ROOT)) in bound for p in required_paths), "Incomplete recovery evidence binding")
        require(binding["binding_recorded_after_resume"] is True
                and binding["limits_or_scientific_settings_changed"] is False
                and binding["recorded_at_utc"] > second_start["started_at_utc"],
                "Supplemental recovery binding misstates its chronology or scope")
        verify_hashes(bound, recovery_inputs)
        plan_inventory = read(OUT / "plan.json")["artifact_sha256"]
        require(all(bound.get(relative) == expected for relative, expected in plan_inventory.items()),
                "Recovery failed to bind unchanged launch code and preflight")
        for path in (recovery_path, binding_path):
            recovery_inputs[str(path.relative_to(ROOT))] = sha(path)
        for year in (1981, 1990):
            require(read(OUT / "years" / str(year) / "completion.json")["completed_at_utc"] < stopped["ended_at_utc"],
                    "A purported completed prefix year was not completed before the stop")
        for path in (OUT / "years/1981/completion.json", OUT / "years/1990/completion.json",
                     OUT / "years/2000/lock.json", OUT / "years/2000/checkpoints/update_023.json",
                     OUT / "years/2000/checkpoints/update_023.npz"):
            relative = str(path.relative_to(ROOT))
            stamp = binding["sources_last_modified_utc"][relative]
            require(stamp < stopped["ended_at_utc"], "Preserved prefix artifact was not saved before the stop")
            actual_stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            require(actual_stamp == stamp, "Preserved prefix artifact timestamp changed after recovery binding")
        progress_path = OUT / "years/2000/progress.jsonl"
        progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
        before = [r for r in progress if r["event"] == "accepted_update" and r["at_utc"] < stopped["ended_at_utc"]]
        require(before and max(r["iteration"] for r in before) == 23, "Pre-stop optimizer checkpoint differs")
        resumed = [r for r in progress if r["event"] == "resumed_checkpoint"
                   and r["at_utc"] >= second_start["started_at_utc"]]
        require(len(resumed) == 1 and resumed[0]["update"] == 23, "Second attempt did not recover exact update 23")
        after = [r for r in progress if r["event"] == "accepted_update"
                 and r["at_utc"] >= second_start["started_at_utc"]]
        require(after and after[0]["iteration"] == 24, "Recovered fit restarted or skipped an accepted update")
        recovery_inputs[str(progress_path.relative_to(ROOT))] = sha(progress_path)
        pressure_events.append(dict(attempt=1, detected_native_pressure_level=2,
                                    stopped_at_utc=stopped["ended_at_utc"], worker_terminated=True,
                                    preserved_update=23, resume_attempt=2, normal_pressure_before_resume=True,
                                    completed_prefix_years_unchanged=True, exact_checkpoint_recovered=True,
                                    cumulative_budget_preserved=True, no_resource_limits_increased=True,
                                    recovery_decision_recorded_before_resume=True,
                                    supplemental_hash_binding_recorded_after_resume=True))
    require(resources and total_active <= 3600, "Resource ledger absent or time budget exceeded")
    for row in resources:
        require(row["physical_footprint_gib"] <= 16 and row["system_available_gib"] >= 4
                and row["swap_growth_bytes"] == 0 and row["free_disk_gib"] >= 50
                and row["study_disk_gib"] <= 5 and row["cumulative_active_seconds"] <= 3600,
                "Recorded resource limit violation")
    return dict(passed=True, active_seconds=total_active, samples=len(resources),
                scope="Supervised fitting and validation execution; excludes this independent reviewer",
                interpretation="Numeric limits passed; any recorded native-pressure event was detected, stopped and separately reviewed before one guarded recovery",
                uninterrupted_resource_normality_established=False if pressure_events else True,
                pressure_events=pressure_events, reviewed_recovery_used=bool(pressure_events),
                recovery_evidence_sha256=recovery_inputs,
                peak_process_tree_gib=max(r["physical_footprint_gib"] for r in resources),
                minimum_available_gib=min(r["system_available_gib"] for r in resources),
                maximum_swap_growth_bytes=max(r["swap_growth_bytes"] for r in resources))


def run():
    missing = [year for year in YEARS if not (OUT / "years" / str(year) / "completion.json").exists()]
    require(not missing, f"PARTIAL: cannot review an incomplete pilot; missing years {missing}")
    destination = OUT / "review"
    require(not destination.exists(), "Review directory exists; preserve it rather than overwrite")
    resource = resource_review()
    plan = read(OUT / "plan.json")
    require(plan["years"] == list(YEARS) and plan["seed"] == 1 and plan["total_accepted_updates"] == 32,
            "Prespecified pilot scope changed")
    verified = {}
    verify_hashes(plan["artifact_sha256"], verified)
    completed = {year: read(OUT / "years" / str(year) / "completion.json") for year in YEARS}
    for year, value in completed.items():
        require(value["status"] == COMPLETE and value["year"] == year, "Year has not completed")
        verify_hashes(value["artifact_sha256"], verified)
        verify_hashes(value["consumed_input_sha256"], verified)
    # Capture the accepted parent receipt itself in the review's provenance.
    parent_acceptance = read(PARENT / "acceptance_manifest.json")
    require(parent_acceptance.get("accepted") is True and parent_acceptance.get("study_complete") is True
            and all(g["passed"] is True for g in parent_acceptance["gates"]),
            "Parent is not accepted and completed")
    verify_hashes({str(PARENT.relative_to(ROOT) / "acceptance_manifest.json"): sha(PARENT / "acceptance_manifest.json")}, verified)
    for relative, expected in verified.items():
        if relative.startswith(str(PARENT.relative_to(ROOT)) + "/") and relative.endswith((".npz", ".json", ".parquet")):
            if relative in parent_acceptance["artifact_sha256"]:
                require(parent_acceptance["artifact_sha256"][relative] == expected, "Source differs from accepted parent")
    fitting, accounting, all_monthly, metric_rows = [], {}, [], []
    with threadpool_limits(limits=1):
        for year, value in completed.items():
            fitting.append(review_fit(year, value, verified))
            monthly, rows, checked = review_monthly(year, value, verified)
            all_monthly.append(monthly)
            metric_rows.extend(rows)
            accounting[str(year)] = checked
    # All substantive Python checks passed; only now create new review artifacts.
    destination.mkdir()
    monthly = pd.concat(all_monthly, ignore_index=True)
    monthly.to_csv(destination / "monthly.csv", index=False)
    summary = pd.DataFrame(metric_rows)
    summary.to_csv(destination / "summary.csv", index=False)
    rpath = destination / "verify.R"
    rpath.write_text(R_SOURCE)
    rscript = shutil.which("Rscript") or "/usr/local/bin/Rscript"
    result = subprocess.run([rscript, "--vanilla", str(rpath), str(destination / "monthly.csv"),
                             str(destination / "R_metrics.csv")], capture_output=True, text=True,
                            check=False, timeout=60, cwd=ROOT)
    (destination / "R_output.txt").write_text(result.stdout + result.stderr)
    require(result.returncode == 0, "Independent base R verification failed")
    independent = pd.read_csv(destination / "R_metrics.csv")
    require(list(zip(independent.year, independent.method)) == list(zip(summary.year, summary.method)),
            "R metric rows differ")
    columns = [key for key in summary if key not in ("year", "method")]
    same(independent[columns].to_numpy(), summary[columns].to_numpy(), atol=1e-11, rtol=1e-10)
    rerror = float(np.max(abs(independent[columns].to_numpy() - summary[columns].to_numpy())))
    rverification = dict(status="PASS", independent_language="base R", arm_month_observations=72,
                         year_arm_rows=6, metrics_per_row=len(columns), maximum_absolute_error=rerror,
                         r_source_sha256=sha(rpath), statistics_searched=False)
    write(destination / "R_verification.json", rverification)
    paired = []
    for year in YEARS:
        block = summary[summary.year == year].set_index("method")
        paired.append(dict(year=year, source_utility_difference=float(block.loc["continued", SOURCE_METRIC] - block.loc["archived8", SOURCE_METRIC]),
                           net_utility_difference=float(block.loc["continued", NET_METRIC] - block.loc["archived8", NET_METRIC])))
    mean_source = math.fsum(r["source_utility_difference"] for r in paired) / 3
    mean_net = math.fsum(r["net_utility_difference"] for r in paired) / 3
    source_nonnegative = sum(r["source_utility_difference"] >= 0 for r in paired)
    net_nonnegative = sum(r["net_utility_difference"] >= 0 for r in paired)
    gates = dict(all_three_complete=True, technical_hash_neutral_accounting_checks_passed=True,
                 resource_guard_and_reviewed_recovery_passed=resource["passed"], no_line_search_stall=True,
                 finite_feasible_scales=all(r["finite_feasible_scales"] for r in fitting),
                 endpoint_gradient_decreased_every_year=all(r["gradient_decreased"] for r in fitting),
                 source_utility_positive_mean=mean_source > 0,
                 source_utility_at_least_two_nonnegative=source_nonnegative >= 2,
                 net_utility_positive_mean=mean_net > 0,
                 net_utility_at_least_two_nonnegative=net_nonnegative >= 2,
                 independent_R_metrics_passed=True)
    passed = all(gates.values())
    reviewed = dict(status="COMPLETED_VERIFIED_BOUNDED_DEVELOPMENT_DIAGNOSTIC",
                    gate_passed=passed, gate=gates, years=list(YEARS), seed=1,
                    fitted_year_diagnostics=fitting, accounting=accounting, resources=resource,
                    paired_year_differences=paired, equal_year_mean_source_utility_difference=mean_source,
                    equal_year_mean_net_utility_difference=mean_net,
                    source_nonnegative_years=source_nonnegative, net_nonnegative_years=net_nonnegative,
                    independent_R=rverification, unique_hashes_verified=len(verified),
                    parent_results_unchanged=True, no_new_full_historical_strategy=True,
                    exposed_development_outcomes=True, statistical_significance_established=False,
                    significance_test_performed=False, validation_selected_iteration=False,
                    automatic_optimization_expansion_stopped=not passed,
                    interpretation="Passing is a development consistency screen, not superiority or a guarantee of significance. "
                    "Failure stops automatic expansion of this optimization candidate.",
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(destination / "result.json", reviewed)
    inputs = {**verified, str(OUT.relative_to(ROOT) / "plan.json"): sha(OUT / "plan.json")}
    inputs.update(resource["recovery_evidence_sha256"])
    for year in YEARS:
        relative = str((OUT / "years" / str(year) / "completion.json").relative_to(ROOT))
        inputs[relative] = sha(ROOT / relative)
    for path in (OUT / "attempts").glob("*"):
        if path.is_file():
            inputs[str(path.relative_to(ROOT))] = sha(path)
    inputs[str((OUT / "execution.json").relative_to(ROOT))] = sha(OUT / "execution.json")
    outputs = {str(path.relative_to(ROOT)): sha(path) for path in destination.iterdir() if path.is_file()}
    outputs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    write(destination / "acceptance_manifest.json", dict(
        status="ACCEPTED_COMPLETED_BOUNDED_DIAGNOSTIC", accepted=True, diagnostic_complete=True,
        scientific_study_complete=False,
        improvement_gate_passed=passed, does_not_accept_new_extension=True,
        gates=[dict(name="technical_and_independent_metric_verification", passed=True),
               dict(name="resource_guard_and_reviewed_recovery", passed=True)],
        source_sha256=inputs, artifact_sha256=outputs,
        completed_at_utc=reviewed["completed_at_utc"]))
    print(json.dumps(dict(status=reviewed["status"], gate_passed=passed,
                         mean_source_difference=mean_source, mean_net_difference=mean_net,
                         result=str(destination / "result.json")), allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
