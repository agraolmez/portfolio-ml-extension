"""Independent base-R accounting and inference review of a frozen ensemble screen.

No fitting, target selection, return-dependent mixture tuning or new resampling
is performed. Position packets stream to R without a second large disk copy.
Acceptance is a bounded exploratory screen, never confirmation on fresh data.
"""
from __future__ import annotations

import ctypes
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
import threading
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension4.runtime import Runtime

CONFIG = ROOT / "config/portfolio_ensemble_screen_20260928.json"
OUT = ROOT / "output/portfolio_ensemble_screen_20260928"
BROOT = ROOT / "output/feature_geometry_study_20260914"
CROOT = ROOT / "output/extension4_full_study_20260924"
METHODS = ["B1", "B2", "B3", "C1", "C2", "C3", "BENS", "CENS", "HALF"]
CONTRASTS = ["CENS_vs_BENS", "CENS_vs_B1", "HALF_vs_BENS", "HALF_vs_B1"]
PERIODS = ["primary", "robustness"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


def same(actual, expected, atol=1e-11, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def pressure():
    value = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    result = ctypes.CDLL(None, use_errno=True).sysctlbyname(
        b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
        ctypes.byref(size), None, 0)
    require(result == 0 and value.value == 1,
            f"Native memory pressure is not normal: {value.value}; syscall {result}")


def verify_inventory(inventory, verified):
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), "Out-of-project evidence")
        if relative in verified:
            require(verified[relative] == expected, "Conflicting evidence hash: " + relative)
        else:
            require(sha(path) == expected, "Changed evidence: " + relative)
            verified[relative] = expected


R_SOURCE = r'''args <- commandArgs(trailingOnly=TRUE)
out <- args[1]; dest <- args[2]
methods <- c("B1","B2","B3","C1","C2","C3","BENS","CENS","HALF")
contrasts <- c("CENS_vs_BENS","CENS_vs_B1","HALF_vs_BENS","HALF_vs_B1")
mix <- cbind(c(rep(1/3,3),rep(0,3)),c(rep(0,3),rep(1/3,3)),rep(1/6,6))
con <- file("stdin", "rb")
rd <- function(n) {x <- readBin(con,numeric(),n=n,size=8,endian="little");stopifnot(length(x)==n,all(is.finite(x)));x}
ri <- function(n) {x <- readBin(con,integer(),n=n,size=4,endian="little");stopifnot(length(x)==n);x}
mat <- function(n,p) matrix(rd(n*p),nrow=n,byrow=TRUE)
maxerror <- 0
same <- function(x,y,atol=1e-10,rtol=1e-9) {
  stopifnot(length(x)==length(y),all(is.finite(x)),all(is.finite(y)))
  d <- abs(x-y);stopifnot(all(d<=atol+rtol*abs(y)))
  maxerror <<- max(maxerror,max(d));invisible(TRUE)
}
rows <- vector("list",120); previous <- NULL; previousids <- NULL
for (month in 1:120) {
  sizes <- ri(2); n <- sizes[1]; nn <- sizes[2];stopifnot(n>1,nn>0)
  ids <- rd(n); nextids <- rd(nn); wm <- rd(2); wealth <- wm[1]; market <- wm[2]
  impact <- rd(n); total <- rd(n); excess <- rd(n); inertia <- mat(n,n)
  target <- mat(n,9); h <- mat(n,9); savedw <- mat(n,9); savedq <- mat(n,9)
  savedd <- mat(n,9); savednext <- mat(nn,9)
  stopifnot(!anyDuplicated(ids),!anyDuplicated(nextids),wealth>0,market > -1,all(impact>0))
  if (month==1) same(h,matrix(h[,1],nrow=n,ncol=9),0,0) else {
    same(ids,previousids,0,0);same(h,previous);h <- previous
  }
  same(target[,7:9,drop=FALSE],target[,1:6,drop=FALSE]%*%mix)
  w <- h+(diag(n)-inertia)%*%(target-h); q <- w-h
  drift <- w*((1+total)/(1+market))
  ix <- match(nextids,ids); nh <- matrix(0,nn,9); present <- !is.na(ix)
  nh[present,] <- drift[ix[present],,drop=FALSE]
  same(savedw,w);same(savedq,q);same(savedd,drift);same(savednext,nh)
  for (x in list(h,target,w,q,drift,nh)) same(x[,7:9,drop=FALSE],x[,1:6,drop=FALSE]%*%mix)
  gross <- colSums(w*excess);cost <- wealth/2*colSums(q^2*impact)
  same(gross[7:9],as.numeric(gross[1:6]%*%mix))
  stopifnot(all(cost[7:9]<=as.numeric(cost[1:6]%*%mix)+1e-12))
  rows[[month]] <- data.frame(month_index=month,method=methods,gross_excess=gross,
      impact_cost=cost,net_excess=gross-cost,turnover=colSums(abs(q)),gross_exposure=colSums(abs(w)))
  previous <- nh;previousids <- nextids
}
close(con)
monthly <- do.call(rbind,rows)
write.table(monthly,file.path(dest,"R_monthly.csv"),sep=",",row.names=FALSE)
metrics <- list(); convex <- list(); intervals <- list()
for (period in c("primary","robustness")) {
  dates <- if(period=="primary") 1:60 else 61:120
  g <- matrix(0,60,9);cost <- matrix(0,60,9)
  for (j in 1:9) {
    x <- monthly[monthly$method==methods[j] & monthly$month_index %in% dates,]
    stopifnot(nrow(x)==60);g[,j]<-x$gross_excess;cost[,j]<-x$impact_cost
  }
  net <- g-cost;gv <- apply(g,2,var); nv <- colMeans((net-matrix(colMeans(net),60,9,byrow=TRUE))^2)
  u <- 12*(colMeans(net)-5*gv)
  for(j in 1:9) metrics[[length(metrics)+1]] <- data.frame(period=period,method=methods[j],months=60,
    source_utility=u[j],net_population_utility=12*(mean(net[,j])-5*nv[j]),
    annual_gross_volatility=sqrt(12*gv[j]),annual_net_mean=12*mean(net[,j]),annual_cost=12*mean(cost[,j]))
  for (j in 1:3) {
    gap <- u[6+j]-sum(u[1:6]*mix[,j])
    costgain <- 12*mean(as.numeric(cost[,1:6]%*%mix[,j])-cost[,6+j])
    variancegain <- 60*(sum(gv[1:6]*mix[,j])-gv[6+j])
    same(gap,costgain+variancegain);stopifnot(gap>=-1e-11,costgain>=-1e-11,variancegain>=-1e-11)
    convex[[length(convex)+1]] <- data.frame(period=period,method=methods[6+j],utility_gap=gap,
      cost_saving_contribution=costgain,variance_reduction_contribution=variancegain)
  }
  point <- c(u[8]-u[7],u[8]-u[1],u[9]-u[7],u[9]-u[1])
  for (L in c(12L,24L)) {
    idx <- as.matrix(read.csv(file.path(out,"bootstrap",paste0(period,"_L",L,"_indices.csv")),header=FALSE))
    stopifnot(identical(dim(idx),c(10000L,60L)),all(idx>=1),all(idx<=60),all(idx==floor(idx)))
    draw <- matrix(0,10000,4)
    for(b in 1:10000) {
      ii <- idx[b,];bu <- 12*(colMeans(net[ii,,drop=FALSE])-5*apply(g[ii,,drop=FALSE],2,var))
      draw[b,] <- c(bu[8]-bu[7],bu[8]-bu[1],bu[9]-bu[7],bu[9]-bu[1])
    }
    colnames(draw)<-contrasts
    writeBin(as.double(t(draw)),file.path(dest,paste0("R_",period,"_L",L,"_draws.bin")),size=8,endian="little")
    for(j in 1:4) {
      quantiles <- as.numeric(quantile(draw[,j],c(.05/16,1-.05/16),type=7))
      intervals[[length(intervals)+1]] <- data.frame(period=period,block_months=L,contrast=contrasts[j],
        observed=point[j],lower=2*point[j]-quantiles[2],upper=2*point[j]-quantiles[1])
    }
  }
}
write.table(do.call(rbind,metrics),file.path(dest,"R_metrics.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,convex),file.path(dest,"R_convexity.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,intervals),file.path(dest,"R_intervals.csv"),sep=",",row.names=FALSE)
write.table(data.frame(maximum_accounting_error=maxerror),file.path(dest,"R_accounting.csv"),sep=",",row.names=FALSE)
'''


def send_packet(stream, packet):
    n, nn = len(packet["ids"]), len(packet["next_ids"])
    stream.write(np.asarray([n, nn], dtype="<i4").tobytes())
    chunks = [packet["ids"], packet["next_ids"],
              [float(packet["wealth"]), float(packet["market_growth"])] ]
    chunks += [packet[key] for key in ("impact", "total_returns", "excess_returns", "m",
                                       "targets", "pretrade", "weights", "trades",
                                       "drifted_weights", "next_pretrade")]
    for chunk in chunks:
        stream.write(np.asarray(chunk, dtype="<f8", order="C").tobytes(order="C"))


def r_review(destination, runtime):
    """Stream only already verified checkpoints; actively guard the R child."""
    rpath = destination / "verify.R"
    with rpath.open("x") as stream:
        stream.write(R_SOURCE)
    failure = []
    stop = threading.Event()
    command = [shutil.which("Rscript") or "/usr/local/bin/Rscript", "--vanilla",
               str(rpath), str(OUT / "analysis"), str(destination)]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with (destination / "R_output.txt").open("x") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=ROOT, env=env, start_new_session=True)

        def watch():
            while not stop.wait(2.):
                try:
                    runtime.guard()
                    pressure()
                except BaseException as error:
                    failure.append(error)
                    child.terminate()
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            for day in pd.date_range("2015-12-31", "2025-11-30", freq="ME"):
                runtime.guard()
                pressure()
                packet = arrays(OUT / "checkpoints" / str(day.date()) / "arrays.npz")
                send_packet(child.stdin, packet)
            child.stdin.close()
            while child.poll() is None:
                runtime.guard()
                pressure()
                if failure:
                    raise failure[0]
                time.sleep(1)
            if failure:
                raise failure[0]
            require(child.returncode == 0, "Independent R accounting/inference failed; see R_output.txt")
        finally:
            stop.set()
            watcher.join(timeout=5)
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def accepted(path, inventory, verified):
    relative = str(Path(path).relative_to(ROOT))
    canonical = str(Path(path).resolve().relative_to(ROOT))
    expected = inventory.get(relative, inventory.get(canonical))
    require(expected is not None, "Source absent from accepted inventory: " + relative)
    verify_inventory({relative: expected}, verified)


def check_sources_and_positions(verified, runtime):
    """Bind every supplied economic input and component target to accepted files."""
    baseline = read(BROOT / "acceptance_manifest.json")
    extension = read(CROOT / "acceptance_manifest.json")
    for value in (baseline, extension):
        require(value.get("accepted") is True and all(g["passed"] for g in value["gates"]),
                "Parent has not been accepted")
    inventories = [baseline["artifact_sha256"], extension["artifact_sha256"]]
    step9 = read(ROOT / "output/step9_20260913/acceptance_manifest.json")["artifact_sha256"]
    raw_inventory = read(ROOT / "output/step3_author_conventions_20260911/bundle_checksums.json")
    raw_path = ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet"
    accepted(raw_path, raw_inventory, verified)
    data = pd.read_parquet(raw_path,
        columns=["id", "eom", "valid", "tr_ld1", "mu_ld1", "ret_ld1", "lambda_diagonal", "wealth_usd"],
        filters=[("eom", ">=", pd.Timestamp("2015-12-31")), ("eom", "<=", pd.Timestamp("2025-11-30"))])
    require(not data.duplicated(["id", "eom"]).any(), "Duplicate source stock/month")
    by_month = {date: block.set_index("id") for date, block in data.groupby("eom")}
    previous = None
    maximum_baseline_error = 0.
    for day in pd.date_range("2015-12-31", "2025-11-30", freq="ME"):
        runtime.guard()
        pressure()
        stamp = str(day.date())
        packet = arrays(OUT / "checkpoints" / stamp / "arrays.npz")
        np.testing.assert_array_equal(packet["methods"], METHODS)
        bpath = BROOT / "paths" / f"{stamp}.npz"
        cpath = CROOT / "final_paths" / stamp / "positions.npz"
        for source, inventory in zip((bpath, cpath), inventories):
            accepted(source, inventory, verified)
        b, c = arrays(bpath), arrays(cpath)
        np.testing.assert_array_equal(packet["ids"], b["ids"])
        np.testing.assert_array_equal(packet["ids"], c["ids"])
        np.testing.assert_array_equal(packet["next_ids"], b["next_ids"])
        np.testing.assert_array_equal(packet["next_ids"], c["next_ids"])
        indices = []
        for source, method in ((b, "fixed512"), (c, "C4")):
            for seed in (1, 2, 3):
                found = np.flatnonzero((source["seeds"] == seed) & (source["methods"] == method))
                require(len(found) == 1, "Missing or duplicate accepted target")
                indices.append(int(found[0]))
        expected = np.column_stack([b["target"][:, indices[j]] if j < 3
                                    else c["target"][:, indices[j]] for j in range(6)])
        same(packet["targets"][:, :6], expected, atol=0, rtol=0)
        first = indices[0]
        if previous is None:
            same(packet["pretrade"], np.repeat(b["pretrade"][:, first, None], 9, axis=1), atol=0, rtol=0)
        else:
            same(packet["pretrade"], previous, atol=0, rtol=0)
        for name, archive_name in (("pretrade", "pretrade"), ("weights", "weight"),
                                   ("trades", "trade"), ("drifted_weights", "drifted_weight"),
                                   ("next_pretrade", "next_pretrade")):
            same(packet[name][:, 0], b[archive_name][:, first], atol=2e-8, rtol=1e-7)
            maximum_baseline_error = max(maximum_baseline_error,
                                        float(np.max(abs(packet[name][:, 0] - b[archive_name][:, first]))))
        mpath = ROOT / "data/prepared/step9_v1/baseline/matrices" / f"{stamp}.npz"
        accepted(mpath, step9, verified)
        original_matrix = arrays(mpath)
        np.testing.assert_array_equal(packet["ids"], original_matrix["ids"])
        same(packet["m"], original_matrix["m"], atol=0, rtol=0)
        active_ids = by_month[day].loc[by_month[day].valid].sort_index().index.to_numpy()
        np.testing.assert_array_equal(packet["ids"], active_ids)
        raw = by_month[day].loc[packet["ids"]]
        require(raw.mu_ld1.nunique() == 1 and raw.wealth_usd.nunique() == 1,
                "Inconsistent monthly market return or wealth")
        for name, column in (("impact", "lambda_diagonal"), ("excess_returns", "ret_ld1"),
                             ("total_returns", "tr_ld1")):
            same(packet[name], raw[column].to_numpy(), atol=0, rtol=0)
        same(packet["wealth"], raw.wealth_usd.iloc[0], atol=0, rtol=0)
        same(packet["market_growth"], raw.mu_ld1.iloc[0], atol=0, rtol=0)
        previous = packet["next_pretrade"]
    return dict(months=120, paths=9, accepted_components_per_month=6,
                source_m_returns_impact_wealth_verified=True, targets_recovered_from_accepted_archives=True,
                baseline_seed1_continuous_archive_recovered=True,
                maximum_baseline_position_error=maximum_baseline_error,
                common_initial_holdings=True, continuous_holdings_through_2021=True,
                target_coefficients_refitted=False, target_features_independently_regenerated=False)


def compare_r_outputs(destination):
    monthly = pd.read_csv(OUT / "monthly.csv", parse_dates=["decision_eom", "return_eom"])
    require(len(monthly) == 1080 and not monthly.duplicated(["decision_eom", "method"]).any(),
            "Incomplete or duplicate monthly path rows")
    dates = pd.date_range("2015-12-31", "2025-11-30", freq="ME")
    require(set(monthly.method) == set(METHODS), "Unexpected path methods")
    for method in METHODS:
        block = monthly[monthly.method == method].sort_values("decision_eom")
        require(list(block.decision_eom) == list(dates) and
                list(block.return_eom) == list(dates + pd.offsets.MonthEnd(1)), "Monthly chronology mismatch")
    independent = pd.read_csv(destination / "R_monthly.csv")
    index = monthly.set_index(["decision_eom", "method"])
    columns = ["gross_excess", "impact_cost", "net_excess", "turnover", "gross_exposure"]
    expected = np.array([index.loc[(dates[int(row.month_index)-1], row.method), columns].to_numpy(dtype=float)
                         for row in independent.itertuples()])
    require(len(independent) == 1080, "R has incomplete monthly accounting")
    same(independent[columns].to_numpy(), expected)
    monthly_error = float(np.max(abs(independent[columns].to_numpy() - expected)))
    main = pd.read_csv(OUT / "metrics.csv").set_index(["period", "method"])
    rmetrics = pd.read_csv(destination / "R_metrics.csv").set_index(["period", "method"])
    require(set(main.index) == set(rmetrics.index) and len(main) == 18, "Metric rows differ")
    metric_columns = list(rmetrics.columns)
    same(main.loc[rmetrics.index, metric_columns], rmetrics[metric_columns])
    metric_error = float(np.max(abs(main.loc[rmetrics.index, metric_columns].to_numpy() - rmetrics.to_numpy())))
    maximum_draw_error = 0.
    for period in PERIODS:
        for length in (12, 24):
            draws = pd.read_csv(OUT / "analysis/bootstrap" / f"{period}_L{length}_draws.csv")
            require(list(draws.columns) == CONTRASTS and draws.shape == (10000, 4), "Wrong draw dimensions/order")
            reference = np.fromfile(destination / f"R_{period}_L{length}_draws.bin", dtype="<f8").reshape(10000, 4)
            same(draws.to_numpy(), reference)
            maximum_draw_error = max(maximum_draw_error, float(np.max(abs(draws.to_numpy() - reference))))
    return rmetrics, pd.read_csv(destination / "R_intervals.csv"), dict(
        language="base R", months=120, actual_paths=9, monthly_path_rows=1080,
        period_path_metric_rows=18, paired_bootstrap_resamples=40000, contrast_draws_checked=160000,
        maximum_monthly_error=monthly_error, maximum_metric_error=metric_error,
        maximum_draw_error=maximum_draw_error, bootstrap_conditions_on_saved_fitted_targets=True)


def advancement_gate(rmetrics, intervals):
    decisions = {}
    for candidate in ("CENS", "HALF"):
        points = []
        bounds = []
        for period in PERIODS:
            for control in ("BENS", "B1"):
                delta = float(rmetrics.loc[(period, candidate), "source_utility"] -
                              rmetrics.loc[(period, control), "source_utility"])
                points.append(dict(period=period, control=control, difference=delta))
                if period == "primary":
                    for length in (12, 24):
                        row = intervals[(intervals.period == period) & (intervals.block_months == length) &
                                        (intervals.contrast == f"{candidate}_vs_{control}")]
                        require(len(row) == 1, "Missing or duplicate primary interval")
                        bounds.append(dict(control=control, block_months=length,
                                           lower=float(row.iloc[0].lower)))
        decisions[candidate] = dict(point_gains=points, primary_lower_bounds=bounds,
                                    passed=all(x["difference"] > 0 for x in points)
                                    and all(x["lower"] > 0 for x in bounds))
    preferred = "HALF" if decisions["HALF"]["passed"] else ("CENS" if decisions["CENS"]["passed"] else None)
    return dict(candidates=decisions, eligible_for_separately_specified_larger_test=preferred,
                any_candidate_passed=preferred is not None, both_pass_priority="HALF",
                exhaustive_training_authorized=False, fresh_confirmation_established=False)


def check_checkpoint_chain(lock, verified):
    lock_hash = sha(OUT / "lock.json")
    previous = None
    dates = pd.date_range("2015-12-31", "2025-11-30", freq="ME")
    committed = sorted((OUT / "checkpoints").glob("*/commit.json"))
    require(len(committed) == 120, "Exactly 120 committed monthly checkpoints required")
    for day in dates:
        folder = OUT / "checkpoints" / str(day.date())
        commit = read(folder / "commit.json")
        require(commit["lock_sha256"] == lock_hash and commit["previous_commit_sha256"] == previous
                and commit["decision_eom"] == str(day.date()), "Checkpoint chain differs")
        require(set(commit["files_sha256"]) == {"arrays.npz", "monthly.json"}, "Unexpected checkpoint files")
        verify_inventory({str((folder / name).relative_to(ROOT)): value
                          for name, value in commit["files_sha256"].items()}, verified)
        for source, value in commit["sources_sha256"].items():
            require(lock["artifact_sha256"].get(source) == value, "Checkpoint source differs from frozen lock")
        verify_inventory(commit["sources_sha256"], verified)
        previous = sha(folder / "commit.json")
    analysis = read(OUT / "analysis/commit.json")
    require(analysis["lock_sha256"] == lock_hash and analysis["final_checkpoint_sha256"] == previous,
            "Analysis does not derive from terminal checkpoint")
    verify_inventory(analysis["artifact_sha256"], verified)
    for filename in ("monthly.csv", "metrics.csv", "intervals.json", "convexity.json", "result.json"):
        require(sha(OUT / filename) == sha(OUT / "analysis" / filename), "Public summary differs from committed analysis")


def check_resampling(config, runtime):
    """Recreate prescribed random streams without importing inference helpers."""
    verified_draws = 0
    for period in PERIODS:
        for length in (12, 24):
            runtime.guard()
            pressure()
            path = OUT / "analysis/bootstrap" / f"{period}_L{length}_indices.csv"
            saved = np.loadtxt(path, delimiter=",", dtype=float)
            require(saved.shape == (10000, 60) and np.isfinite(saved).all()
                    and np.all(saved == np.floor(saved)) and saved.min() >= 1 and saved.max() <= 60,
                    "Invalid one-based bootstrap indices")
            for draw in range(10000):
                payload = dict(protocol="portfolio_ensemble_screen_20260928_" + period,
                               master_seed=20260928, component=f"step10_stationary_L{length}",
                               date=config["periods"][period][1], replica=draw+1)
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
                rng = np.random.Generator(np.random.PCG64(int(digest[:32], 16)))
                starts = rng.integers(0, 60, size=60, dtype=np.int32)
                restart = rng.random(60) < 1 / length
                expected = np.empty(60, dtype=np.int32)
                expected[0] = starts[0]
                for index in range(1, 60):
                    expected[index] = starts[index] if restart[index] else (expected[index-1] + 1) % 60
                np.testing.assert_array_equal(saved[draw], expected + 1)
                if draw % 500 == 0:
                    runtime.guard()
                    pressure()
            verified_draws += 10000
    return dict(reconstructed_random_streams=verified_draws, matched_month_indices=verified_draws*60,
                implementation="Independent SHA256 payload, PCG64 primitives and circular recurrence",
                selected_or_rerolled_seeds=False, same_indices_across_all_nine_paths=True)


def check_intervals_and_convexity(destination, rintervals):
    source = pd.DataFrame(read(OUT / "intervals.json"))
    require(len(source) == 16 and set(source.family) == {8} and set(source.draws) == {10000},
            "Wrong comparison family or number of intervals")
    source["contrast"] = source.candidate + "_vs_" + source.control
    keys = ["period", "block_months", "contrast"]
    require(not source.duplicated(keys).any() and not rintervals.duplicated(keys).any(), "Duplicate confidence interval")
    expected = source.set_index(keys)
    reference = rintervals.set_index(keys)
    require(set(expected.index) == set(reference.index), "Interval coverage differs")
    same(expected.loc[reference.index, ["observed", "lower", "upper"]], reference[["observed", "lower", "upper"]])
    convex = pd.DataFrame(read(OUT / "convexity.json")).set_index(["period", "method"])
    rconvex = pd.read_csv(destination / "R_convexity.csv").set_index(["period", "method"])
    require(set(convex.index) == set(rconvex.index) and len(convex) == 6, "Convexity identities incomplete")
    for main, independent in (("utility_gain", "utility_gap"), ("annual_cost_saving", "cost_saving_contribution"),
                              ("annual_risk_penalty_saving", "variance_reduction_contribution")):
        same(convex.loc[rconvex.index, main], rconvex[independent])


def resource_receipt(config):
    budget = config["compute_budget"]
    total, rows = 0., []
    starts = sorted((OUT / "attempts").glob("*_start.json"))
    require(starts, "Missing resource ledger")
    for start in starts:
        initial = read(start)
        end = start.with_name(start.name.replace("_start", "_end"))
        require(end.exists(), "Incomplete supervised attempt")
        terminal = read(end)
        require(initial["budget"] == budget and initial["clock"] == "mach_absolute_time()",
                "Resource configuration or clock changed")
        same(initial["prior_active_seconds"], total, atol=1e-9, rtol=0)
        require(terminal["status"] in ("COMPUTED_PENDING_INDEPENDENT_REVIEW", "INDEPENDENT_REVIEW_COMPLETE")
                and terminal["error"] is None, "Resource stop requires separate review")
        samples = [json.loads(line) for line in start.with_name(
            start.name.replace("_start.json", "_resources.jsonl")).read_text().splitlines() if line]
        require(samples, "No resource samples")
        for row in samples:
            same(row["cumulative_active_seconds"], total + row["active_seconds"], atol=1e-9, rtol=0)
            require(row["physical_footprint_gib"] <= 16 and row["system_available_gib"] >= 4
                    and row["swap_growth_bytes"] == 0 and row["free_disk_gib"] >= 50
                    and row["study_disk_gib"] <= 2 and row["cumulative_active_seconds"] <= 1800,
                    "A recorded resource threshold was exceeded")
        require(terminal["active_seconds"] >= samples[-1]["active_seconds"], "Lost sampled runtime")
        total += terminal["active_seconds"]
        rows.extend(samples)
    require(total <= 1800, "Cumulative execution plus review budget exceeded")
    return dict(passed=True, scope="Runner plus independent review", active_seconds=total,
                samples=len(rows), peak_process_tree_gib=max(r["physical_footprint_gib"] for r in rows),
                minimum_system_headroom_gib=min(r["system_available_gib"] for r in rows),
                maximum_swap_growth_bytes=max(r["swap_growth_bytes"] for r in rows),
                normal_native_pressure_required_by_live_guard=True)


def run():
    config = read(CONFIG)
    require(config["training_authorized"] is False and config["gamma"] == 10 and config["paths"] == METHODS,
            "Changed no-training representation or economic objective")
    require(config["inference"] == read(OUT / "lock.json")["config"]["inference"], "Inference settings changed")
    completion = read(OUT / "execution_complete.json")
    require(completion["status"] == "COMPUTED_PENDING_INDEPENDENT_REVIEW", "Runner is incomplete")
    destination = OUT / "review"
    require(not destination.exists(), "Preserve the existing review; no overwrite allowed")
    mutex = (OUT / "execution.lock").open("a")
    fcntl.flock(mutex.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pressure()
    destination.mkdir()
    audit_path = ROOT / "scripts/audit_portfolio_ensemble_chronology_20260928.py"
    review_lock = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                       source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                           (Path(__file__), CONFIG, OUT / "execution_complete.json", OUT / "lock.json",
                            OUT / "protocol_freeze.json", OUT / "chronology.json", audit_path,
                            ROOT / "extension4/runtime.py", ROOT / "temporal_fast_v4/memory.py")},
                       R_source_sha256=hashlib.sha256(R_SOURCE.encode()).hexdigest(),
                       no_models_fit=True, no_performance_dependent_candidate_choice=True)
    write(destination / "execution_lock.json", review_lock)
    runtime = Runtime(OUT, config, "independent_review")
    closed = False
    try:
        verified = {}
        verify_inventory(review_lock["source_sha256"], verified)
        lock = read(OUT / "lock.json")
        require(lock["config"] == config and lock["path_order"] == METHODS
                and lock["targets_reestimated"] is False, "Frozen screen changed")
        require(read(OUT / "protocol_freeze.json")["config_sha256"] == sha(CONFIG), "Protocol freeze changed")
        chronology = read(OUT / "chronology.json")
        require(chronology["passed"] is True and chronology["protocol_sha256"] == sha(CONFIG)
                and chronology["audit_implementation_sha256"] == sha(audit_path), "Chronology receipt/code differs")
        require(read(OUT / "financial_preflight.json") == read(BROOT / "financial_preflight.json"),
                "Fresh financial preflight differs")
        verify_inventory(lock["artifact_sha256"], verified)
        verify_inventory(chronology["artifact_sha256"], verified)
        verify_inventory(completion["artifact_sha256"], verified)
        check_checkpoint_chain(lock, verified)
        runtime.event("independent_source_hashes_verified", files=len(verified))
        with threadpool_limits(limits=1):
            source_checks = check_sources_and_positions(verified, runtime)
        runtime.event("accepted_targets_and_economic_inputs_verified", months=120)
        randomness = check_resampling(config, runtime)
        runtime.event("all_prescribed_bootstrap_indices_reconstructed", draws=40000)
        r_review(destination, runtime)
        rmetrics, rintervals, comparison = compare_r_outputs(destination)
        check_intervals_and_convexity(destination, rintervals)
        gate = advancement_gate(rmetrics, rintervals)
        main_gate = read(OUT / "result.json")["gate"]
        require(main_gate["candidate_for_larger_study_consideration"] == gate["eligible_for_separately_specified_larger_test"]
                and main_gate["exploratory_gate_passed"] == gate["any_candidate_passed"], "Advancement decision differs")
        for candidate in ("CENS", "HALF"):
            require(main_gate["candidates"][candidate]["passed"] == gate["candidates"][candidate]["passed"],
                    "Candidate gate differs")
        verify_inventory(review_lock["source_sha256"], {})
        runtime.guard()
        pressure()
        runtime.event("independent_R_all_accounting_metrics_draws_intervals_passed", contrast_draws=160000)
        runtime.close("INDEPENDENT_REVIEW_COMPLETE")
        closed = True
        resources = resource_receipt(config)
        result = dict(status="COMPLETED_VERIFIED_BOUNDED_ENSEMBLE_SCREEN", accepted=True,
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      periods=config["periods"], months=120, paths=METHODS, gate=gate,
                      source_checks=source_checks, independent_R=comparison,
                      resampling=randomness, resources=resources,
                      no_models_fitted=True, parent_results_unchanged=True,
                      inference_family=8, block_lengths=[12, 24], all_periods_previously_exposed=True,
                      fresh_confirmation_established=False, exhaustive_training_launched=False,
                      published_paper_outperformance_established=False,
                      limitation="Conditional inference on saved trained targets and exposed outcomes; no model re-estimation uncertainty.")
        write(destination / "result.json", result)
        inputs = dict(verified)
        for path in (OUT / "attempts").iterdir():
            if path.is_file():
                inputs[str(path.relative_to(ROOT))] = sha(path)
        outputs = {str(p.relative_to(ROOT)): sha(p) for p in destination.iterdir() if p.is_file()}
        write(destination / "acceptance_manifest.json", dict(accepted=True, screen_complete=True,
            exhaustive_study_complete=False, new_extension_accepted=False,
            gate_passed=gate["any_candidate_passed"], source_sha256=inputs, artifact_sha256=outputs,
            gates=[dict(name="parent_chronology_hashes_and_neutrality", passed=True),
                   dict(name="independent_R_continuous_accounting_and_all_inference", passed=True),
                   dict(name="combined_resource_budget", passed=True)],
            completed_at_utc=result["completed_at_utc"]))
        print(json.dumps(dict(status=result["status"], gate=gate,
                              result=str(destination / "result.json")), allow_nan=False))
    except BaseException as error:
        if not closed:
            runtime.close("STOPPED", error)
        write(destination / "failure.json", dict(error_type=type(error).__name__, message=str(error),
            at_utc=datetime.now(timezone.utc).isoformat(), accepted=False,
            recovery="Preserve this attempt and review the failure; do not overwrite evidence."))
        raise


if __name__ == "__main__":
    run()
