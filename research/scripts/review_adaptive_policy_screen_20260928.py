"""Independent source, causal policy, base-R accounting and inference review.

No model fitting or candidate selection. Reuses accepted economic packets and
bootstrap date streams, but independently reconstructs risk forecasts, adaptive
targets, continuous positions, costs, every bootstrap draw and family-24 bounds.
Never overwrites an earlier review or accepted parent evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
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
from scripts import review_portfolio_ensemble_screen_20260928_v2 as base
from extension4.runtime import Runtime

read, sha, write, arrays, same = base.read, base.sha, base.write, base.arrays, base.same
require, pressure, verify_inventory, accepted = base.require, base.pressure, base.verify_inventory, base.accepted
CONFIG = ROOT / "config/adaptive_policy_screen_20260928.json"
OUT = ROOT / "output/adaptive_policy_screen_20260928"
PARENT = ROOT / "output/portfolio_ensemble_screen_20260928"
BROOT = ROOT / "output/feature_geometry_study_20260914"
METHODS = ["B1", "BENS", "CENS", "AGREE", "ASIZE", "RISK", "RBASE"]
PAIRS = [("AGREE", x) for x in ("B1", "BENS", "CENS", "ASIZE")] + [("RISK", x) for x in ("B1", "BENS", "CENS", "RBASE")]
CONTRASTS = [a + "_vs_" + b for a, b in PAIRS]
PERIODS = ["primary", "robustness"]
DATES = pd.date_range("2015-12-31", "2025-11-30", freq="ME")
REFDATES = pd.date_range("2010-12-31", "2025-10-31", freq="ME")
HELPER_SHA = "9b6ba66c413be42c1e7102fedc5d30d90c7cd3ecf24713e7d647130761fe21b2"

R_SOURCE = r'''args <- commandArgs(trailingOnly=TRUE)
out <- args[1]; parent <- args[2]; dest <- args[3]
methods <- c("B1","BENS","CENS","AGREE","ASIZE","RISK","RBASE")
ca <- c(4,4,4,4,6,6,6,6); ct <- c(1,2,3,5,1,2,3,7)
contrasts <- paste0(methods[ca],"_vs_",methods[ct])
con <- file("stdin","rb")
rd <- function(n) {x<-readBin(con,numeric(),n=n,size=8,endian="little");stopifnot(length(x)==n,all(is.finite(x)));x}
ri <- function(n) {x<-readBin(con,integer(),n=n,size=4,endian="little");stopifnot(length(x)==n);x}
mat <- function(n,p) matrix(rd(n*p),nrow=n,byrow=TRUE)
maxerror <- 0
same <- function(x,y,atol=1e-10,rtol=1e-9) {
 stopifnot(length(x)==length(y),all(is.finite(x)),all(is.finite(y)))
 d<-abs(x-y);stopifnot(all(d<=atol+rtol*abs(y)));maxerror<<-max(maxerror,max(d));invisible(TRUE)
}
ref <- matrix(0,179,2,dimnames=list(NULL,c("gross_excess","predicted_variance")))
for(j in 1:179) {
 dim<-ri(2);n<-dim[1];p<-dim[2];stopifnot(n>1,p>0)
 w<-rd(n);load<-mat(n,p);fc<-mat(p,p);specific<-rd(n);ret<-rd(n);saved<-rd(2)
 stopifnot(all(specific>=0));same(fc,t(fc))
 fac<-as.numeric(crossprod(load,w)); v<-as.numeric(crossprod(fac,fc%*%fac))+sum(specific*w*w)
 stopifnot(v>=0);gross<-sum(w*ret);same(c(gross,v),saved)
 ref[j,]<-c(gross,v)
}
write.table(data.frame(reference_index=1:179,ref),file.path(dest,"R_reference.csv"),sep=",",row.names=FALSE)
calibrate <- function(x,current) {
 ix<-tail(which(seq_len(nrow(x))<=current),60);stopifnot(length(ix)==60,all(diff(ix)==1))
 vp<-mean(x[ix,2]);vr<-var(x[ix,1]);stopifnot(vp>=0,vr>=0)
 k<-if(vr==0) 1 else min(1,sqrt(vp/vr));list(k=k,vp=vp,vr=vr,ix=ix)
}
# Prespecified neutral/edge contracts; no historical choice is made here.
stopifnot(calibrate(cbind(rep(0,60),rep(1,60)),60)$k==1)
stopifnot(calibrate(cbind(seq_len(60),rep(0,60)),60)$k==0)
stopifnot(calibrate(cbind(seq_len(60),rep(1e9,60)),60)$k==1)
rows<-vector("list",120);diagnostics<-vector("list",120);previous<-NULL;previousids<-NULL
for(month in 1:120) {
 sizes<-ri(2);n<-sizes[1];nn<-sizes[2];stopifnot(n>1,nn>0)
 ids<-rd(n);nextids<-rd(nn);wm<-rd(2);wealth<-wm[1];market<-wm[2]
 impact<-rd(n);total<-rd(n);excess<-rd(n);inertia<-mat(n,n)
 b<-mat(n,3);c<-mat(n,3);savedtarget<-mat(n,7);h<-mat(n,7)
 savedw<-mat(n,7);savedq<-mat(n,7);savedd<-mat(n,7);savednext<-mat(nn,7)
 sg<-rd(n);sd<-rd(n);sv<-rd(n);scalars<-rd(4);refindices<-ri(60)
 stopifnot(!anyDuplicated(ids),!anyDuplicated(nextids),wealth>0,market > -1,all(impact>0))
 if(month==1) same(h,matrix(h[,1],nrow=n,ncol=7),0,0) else {same(ids,previousids,0,0);same(h,previous);h<-previous}
 departure<-c-b;dbar<-rowMeans(departure);v<-rowSums((departure-dbar)^2)/2
 denominator<-dbar^2+v;gate<-rep(1,n);nz<-denominator>0;gate[nz]<-dbar[nz]^2/denominator[nz]
 amp<-if(sum(abs(dbar))==0) 1 else sum(abs(gate*dbar))/sum(abs(dbar))
 z<-calibrate(ref,month+59)
 altered<-ref;future<-which(seq_len(nrow(ref))>month+59)
 if(length(future)>0) {altered[future,1]<-altered[future,1]+12345;altered[future,2]<-altered[future,2]+99999}
 same(unlist(z),unlist(calibrate(altered,month+59)),0,0)
 same(refindices,z$ix-1,0,0);same(sg,gate);same(sd,dbar);same(sv,v)
 same(scalars,c(amp,z$k,z$vp,z$vr));stopifnot(all(gate>=0),all(gate<=1),amp>=0,amp<=1,z$k>=0,z$k<=1)
 bm<-rowMeans(b);cm<-rowMeans(c)
 target<-cbind(b[,1],bm,cm,bm+gate*dbar,bm+amp*dbar,z$k*cm,z$k*bm)
 same(savedtarget,target)
 # Exact input identities: agreement=1 recovers CENS; risk k=1 recovers each parent aim.
 same(bm+dbar,cm);same(1*cm,cm,0,0);same(1*bm,bm,0,0)
 w<-h+(diag(n)-inertia)%*%(target-h);q<-w-h
 drift<-w*((1+total)/(1+market));ix<-match(nextids,ids);nh<-matrix(0,nn,7);present<-!is.na(ix)
 nh[present,]<-drift[ix[present],,drop=FALSE]
 same(savedw,w);same(savedq,q);same(savedd,drift);same(savednext,nh)
 gross<-colSums(w*excess);cost<-wealth/2*colSums(q^2*impact)
 rows[[month]]<-data.frame(month_index=month,method=methods,gross_excess=gross,impact_cost=cost,
  net_excess=gross-cost,turnover=colSums(abs(q)),gross_exposure=colSums(abs(w)))
 diagnostics[[month]]<-data.frame(month_index=month,agreement_h=amp,risk_k=z$k,predicted_var_mean=z$vp,
  realized_var=z$vr,first_reference_index=min(z$ix),last_reference_index=max(z$ix))
 previous<-nh;previousids<-nextids
}
close(con)
monthly<-do.call(rbind,rows)
write.table(monthly,file.path(dest,"R_monthly.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,diagnostics),file.path(dest,"R_policy.csv"),sep=",",row.names=FALSE)
metrics<-list();intervals<-list();priorintervals<-list()
oldpoints<-read.csv(file.path(parent,"metrics.csv"));oldnames<-c("CENS_vs_BENS","CENS_vs_B1","HALF_vs_BENS","HALF_vs_B1")
for(period in c("primary","robustness")) {
 dates<-if(period=="primary") 1:60 else 61:120;g<-matrix(0,60,7);cost<-g
 for(j in 1:7) {x<-monthly[monthly$method==methods[j] & monthly$month_index %in% dates,];stopifnot(nrow(x)==60);g[,j]<-x$gross_excess;cost[,j]<-x$impact_cost}
 net<-g-cost;gv<-apply(g,2,var);nv<-colMeans((net-matrix(colMeans(net),60,7,byrow=TRUE))^2)
 u<-12*(colMeans(net)-5*gv);point<-u[ca]-u[ct]
 for(j in 1:7) metrics[[length(metrics)+1]]<-data.frame(period=period,method=methods[j],months=60,
  source_utility=u[j],net_population_utility=12*(mean(net[,j])-5*nv[j]),annual_gross_volatility=sqrt(12*gv[j]),
  annual_net_mean=12*mean(net[,j]),annual_cost=12*mean(cost[,j]))
 old<-oldpoints[oldpoints$period==period,];ou<-setNames(old$source_utility,old$method)
 op<-c(ou["CENS"]-ou["BENS"],ou["CENS"]-ou["B1"],ou["HALF"]-ou["BENS"],ou["HALF"]-ou["B1"])
 for(L in c(12L,24L)) {
  idx<-as.matrix(read.csv(file.path(parent,"analysis/bootstrap",paste0(period,"_L",L,"_indices.csv")),header=FALSE))
  stopifnot(identical(dim(idx),c(10000L,60L)),all(idx>=1),all(idx<=60),all(idx==floor(idx)))
  draw<-matrix(0,10000,8)
  for(bb in 1:10000) {ii<-idx[bb,];bu<-12*(colMeans(net[ii,,drop=FALSE])-5*apply(g[ii,,drop=FALSE],2,var));draw[bb,]<-bu[ca]-bu[ct]}
  colnames(draw)<-contrasts
  writeBin(as.double(t(draw)),file.path(dest,paste0("R_",period,"_L",L,"_draws.bin")),size=8,endian="little")
  for(j in 1:8) {
   q<-as.numeric(quantile(draw[,j],c(.05/48,1-.05/48),type=7))
   intervals[[length(intervals)+1]]<-data.frame(period=period,block_months=L,contrast=contrasts[j],observed=point[j],lower=2*point[j]-q[2],upper=2*point[j]-q[1])
  }
  olddraw<-read.csv(file.path(parent,"analysis/bootstrap",paste0(period,"_L",L,"_draws.csv")))
  stopifnot(identical(names(olddraw),oldnames),nrow(olddraw)==10000)
  for(j in 1:4) {
   q<-as.numeric(quantile(olddraw[,j],c(.05/48,1-.05/48),type=7))
   priorintervals[[length(priorintervals)+1]]<-data.frame(period=period,block_months=L,contrast=oldnames[j],observed=op[j],lower=2*op[j]-q[2],upper=2*op[j]-q[1])
  }
 }
}
write.table(do.call(rbind,metrics),file.path(dest,"R_metrics.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,intervals),file.path(dest,"R_intervals.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,priorintervals),file.path(dest,"R_prior_intervals.csv"),sep=",",row.names=FALSE)
write.table(data.frame(maximum_accounting_error=maxerror),file.path(dest,"R_accounting.csv"),sep=",",row.names=FALSE)
'''


def send_numbers(stream, chunks):
    for chunk in chunks:
        stream.write(np.asarray(chunk, dtype="<f8", order="C").tobytes(order="C"))


def send_reference(stream, packet, raw_return):
    n, p = packet["loadings"].shape
    stream.write(np.asarray([n, p], dtype="<i4").tobytes())
    send_numbers(stream, [packet[k] for k in ("weights", "loadings", "factor_cov", "specific_variance")] +
                 [raw_return, [float(packet["gross_excess"]), float(packet["predicted_variance"])]] )


def send_month(stream, packet, economic):
    n, nn = len(packet["ids"]), len(packet["next_ids"])
    stream.write(np.asarray([n, nn], dtype="<i4").tobytes())
    chunks = [packet["ids"], packet["next_ids"], [float(economic["wealth"]), float(economic["market_growth"])]]
    chunks += [economic[k] for k in ("impact", "total_returns", "excess_returns", "m")]
    chunks += [packet[k] for k in ("component_baseline", "component_c4", "targets", "pretrade", "weights", "trades", "drifted_weights", "next_pretrade", "agreement_g", "dbar", "v")]
    chunks += [[float(packet[k]) for k in ("h", "risk_k", "predicted_variance_mean", "realized_variance")]]
    send_numbers(stream, chunks)
    stream.write(np.asarray(packet["reference_indices"], dtype="<i4").tobytes())


def r_review(destination, raw_returns, runtime):
    rpath = destination / "verify.R"
    with rpath.open("x") as stream:
        stream.write(R_SOURCE)
    failure, stop = [], threading.Event()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    command = [shutil.which("Rscript") or "/usr/local/bin/Rscript", "--vanilla", str(rpath), str(OUT), str(PARENT), str(destination)]
    with (destination / "R_output.txt").open("x") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
        def watch():
            while not stop.wait(2.):
                try:
                    runtime.guard(); pressure()
                except BaseException as error:
                    failure.append(error); child.terminate(); return
        watcher = threading.Thread(target=watch, daemon=True); watcher.start()
        try:
            observations = pd.read_csv(OUT / "reference/observations.csv").set_index("decision_eom")
            for day in REFDATES:
                runtime.guard(); pressure()
                packet = arrays(OUT / "reference/packets" / f"{day.date()}.npz")
                packet["predicted_variance"] = observations.loc[str(day.date()), "predicted_variance"]
                send_reference(child.stdin, packet, raw_returns[str(day.date())])
            for day in DATES:
                runtime.guard(); pressure()
                stamp = str(day.date())
                send_month(child.stdin, arrays(OUT / "checkpoints" / stamp / "arrays.npz"), arrays(PARENT / "checkpoints" / stamp / "arrays.npz"))
            child.stdin.close()
            while child.poll() is None:
                runtime.guard(); pressure()
                if failure: raise failure[0]
                time.sleep(1)
            if failure: raise failure[0]
            require(child.returncode == 0, "Independent R verification failed; see R_output.txt")
        finally:
            stop.set(); watcher.join(timeout=5)
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()


def merged_parent_inventory(verified):
    inventory = {}
    for study in ("feature_geometry_study_20260914", "extension3_full_study_20260919", "extension4_full_study_20260924"):
        path = ROOT / "output" / study / "acceptance_manifest.json"
        receipt = read(path)
        require(receipt["accepted"] is True and all(g["passed"] for g in receipt["gates"]), "Unaccepted parent")
        verified[str(path.relative_to(ROOT))] = sha(path)
        for relative, expected in receipt["artifact_sha256"].items():
            if relative.startswith("data/prepared/step9_v1/baseline/risk/") or relative.startswith("output/feature_geometry_study_20260914/paths/"):
                require(relative not in inventory or inventory[relative] == expected, "Conflicting accepted parent hashes: " + relative)
                inventory[relative] = expected
    return inventory


def check_reference(verified, runtime):
    inventory = merged_parent_inventory(verified)
    raw_inventory = read(ROOT / "output/step3_author_conventions_20260911/bundle_checksums.json")
    raw_path = ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet"
    accepted(raw_path, raw_inventory, verified)
    raw = pd.read_parquet(raw_path, columns=["id", "eom", "valid", "ret_ld1"], filters=[("eom", ">=", REFDATES[0]), ("eom", "<=", REFDATES[-1])])
    require(not raw.duplicated(["id", "eom"]).any(), "Duplicate reference source stock/month")
    blocks = {day: frame.set_index("id") for day, frame in raw.groupby("eom")}
    observations = pd.read_csv(OUT / "reference/observations.csv", parse_dates=["decision_eom", "return_eom"])
    require(len(observations) == 179 and list(observations.decision_eom) == list(REFDATES) and list(observations.return_eom) == list(REFDATES + pd.offsets.MonthEnd(1)), "Reference chronology must be 179 consecutive exact months")
    raw_returns, maximum_error = {}, 0.
    for day, row in zip(REFDATES, observations.itertuples()):
        runtime.guard(); pressure(); stamp = str(day.date())
        bpath = BROOT / "paths" / f"{stamp}.npz"
        riskpath = ROOT / "data/prepared/step9_v1/baseline/risk" / f"{stamp}.npz"
        for path in (bpath, riskpath): accepted(path, inventory, verified)
        b, risk = arrays(bpath), arrays(riskpath)
        copied = arrays(OUT / "reference/packets" / f"{stamp}.npz")
        found = np.flatnonzero((b["seeds"] == 1) & (b["methods"] == "fixed512"))
        require(len(found) == 1, "Reference B1 absent or duplicated")
        ids, weights = b["ids"], b["weight"][:, int(found[0])]
        for supplied in (copied["ids"], risk["ids"], blocks[day].loc[blocks[day].valid].sort_index().index.to_numpy()): np.testing.assert_array_equal(ids, supplied)
        same(copied["weights"], weights, atol=0, rtol=0)
        for key in ("loadings", "factor_cov", "specific_variance"): same(copied[key], risk[key], atol=0, rtol=0)
        require(np.isfinite(weights).all() and np.isfinite(risk["specific_variance"]).all() and (risk["specific_variance"] >= 0).all(), "Invalid reference risk input")
        returns = blocks[day].loc[ids, "ret_ld1"].to_numpy()
        raw_returns[stamp] = returns
        factor_weight = risk["loadings"].T @ weights
        predicted = float(factor_weight @ risk["factor_cov"] @ factor_weight + np.dot(risk["specific_variance"], weights**2))
        gross = float(np.dot(weights, returns))
        require(predicted >= 0 and np.isfinite(predicted) and np.isfinite(gross), "Invalid reference forecast/return")
        same([row.gross_excess, row.predicted_variance], [gross, predicted])
        same(float(copied["gross_excess"]), gross)
        maximum_error = max(maximum_error, abs(row.gross_excess-gross), abs(row.predicted_variance-predicted))
    return raw_returns, dict(reference_months=179, accepted_B1_executed_weights=True, accepted_factor_covariance=True,
        gross_returns_recomputed_from_raw=True, merged_consistent_E2_E3_E4_inventories=True, maximum_reference_error=maximum_error)


def check_positions(verified, runtime):
    previous, maximum_error = None, 0.
    for day in DATES:
        runtime.guard(); pressure(); stamp = str(day.date())
        packet = arrays(OUT / "checkpoints" / stamp / "arrays.npz")
        oldpath = PARENT / "checkpoints" / stamp / "arrays.npz"
        old = arrays(oldpath)
        np.testing.assert_array_equal(packet["methods"], METHODS)
        for key in ("ids", "next_ids"): np.testing.assert_array_equal(packet[key], old[key])
        same(packet["component_baseline"], old["targets"][:, :3], atol=0, rtol=0)
        same(packet["component_c4"], old["targets"][:, 3:6], atol=0, rtol=0)
        for new_index, old_index in ((0, 0), (1, 6), (2, 7)):
            for key in ("targets", "pretrade", "weights", "trades", "drifted_weights", "next_pretrade"):
                same(packet[key][:, new_index], old[key][:, old_index])
                maximum_error = max(maximum_error, float(np.max(abs(packet[key][:, new_index]-old[key][:, old_index]))))
        if previous is None: same(packet["pretrade"], np.repeat(old["pretrade"][:, 0, None], 7, axis=1), atol=0, rtol=0)
        else: same(packet["pretrade"], previous, atol=0, rtol=0)
        indices = np.asarray(packet["reference_indices"])
        np.testing.assert_array_equal(indices, np.arange(DATES.get_loc(day), DATES.get_loc(day)+60))
        mature_dates = (REFDATES + pd.offsets.MonthEnd(1))[indices]
        require(mature_dates[-1] == day and (mature_dates <= day).all(), "Risk calibration includes unmatured information")
        previous = packet["next_pretrade"]
    return dict(months=120, paths=7, same_common_initial_holdings=True, initial_switch_charged=True,
        continuous_through_2021=True, B1_BENS_CENS_accepted_paths_reproduced=True,
        maximum_parent_position_error=maximum_error, strict_60_month_matured_windows=True,
        original_m_returns_costs_wealth_used_from_hash_bound_accepted_parent=True)


def check_checkpoint_chain(lock, verified):
    previous, lock_hash = None, sha(OUT / "lock.json")
    require(len(list((OUT / "checkpoints").glob("*/commit.json"))) == 120, "Exactly 120 checkpoints required")
    for day in DATES:
        folder = OUT / "checkpoints" / str(day.date()); commit = read(folder / "commit.json")
        require(commit["lock_sha256"] == lock_hash and commit["previous_commit_sha256"] == previous and commit["decision_eom"] == str(day.date()), "Broken monthly checkpoint chain")
        require(set(commit["files_sha256"]) == {"arrays.npz", "monthly.json"}, "Unexpected monthly checkpoint files")
        verify_inventory({str((folder / name).relative_to(ROOT)): digest for name, digest in commit["files_sha256"].items()}, verified)
        parent = str((PARENT / "checkpoints" / str(day.date()) / "arrays.npz").relative_to(ROOT))
        require(commit["parent_packet"] == parent and commit["parent_packet_sha256"] == lock["artifact_sha256"].get(parent), "Parent packet binding differs")
        verify_inventory({parent: commit["parent_packet_sha256"]}, verified)
        require(commit["reference_commit_sha256"] == sha(OUT / "reference/commit.json") and commit["window_last"] == str(day.date()) and commit["window_first"] == str((day - pd.offsets.MonthEnd(59)).date()), "Checkpoint reference/window binding differs")
        previous = sha(folder / "commit.json")
    reference = read(OUT / "reference/commit.json")
    require(reference["lock_sha256"] == lock_hash and reference["reference_months"] == 179, "Reference commit differs")
    verify_inventory({str((OUT / "reference" / name).relative_to(ROOT)): digest for name, digest in reference["files_sha256"].items()}, verified)
    sources = read(OUT / "reference/source_hashes.json")
    require(sources["monthly_sha256"] == lock["artifact_sha256"].get(sources["monthly_file"]), "Reference monthly source differs")
    require(set(sources["months"]) == {str(day.date()) for day in REFDATES}, "Reference source months differ")
    for source_hashes in sources["months"].values():
        for relative, digest in source_hashes.items():
            require(lock["artifact_sha256"].get(relative) == digest, "Reference source absent from frozen lock")
        verify_inventory(source_hashes, verified)
    analysis = read(OUT / "analysis/commit.json")
    require(analysis["lock_sha256"] == lock_hash and analysis["final_checkpoint_sha256"] == previous, "Analysis is not bound to terminal checkpoint")
    verify_inventory({str((OUT / "analysis" / name).relative_to(ROOT)): digest for name, digest in analysis["files_sha256"].items()}, verified)
    for filename in ("monthly.csv", "metrics.csv", "intervals.json", "combined_prior_intervals.json", "result.json"):
        require(sha(OUT / filename) == sha(OUT / "analysis" / filename), "Public artifact differs from committed analysis: " + filename)


def check_resampling(verified, runtime):
    # Regenerate the already accepted streams, never select or reroll them.
    original_out = base.OUT
    try:
        base.OUT = PARENT
        result = base.check_resampling(read(PARENT / "lock.json")["config"], runtime)
    finally: base.OUT = original_out
    for period in PERIODS:
        for length in (12, 24):
            path = PARENT / "analysis/bootstrap" / f"{period}_L{length}_indices.csv"
            require(str(path.relative_to(ROOT)) in verified, "Reused index stream was not source-bound")
    result["same_indices_across_all_seven_paths"] = True
    result.pop("same_indices_across_all_nine_paths", None)
    return result


def compare_intervals(path, reference, expected_rows):
    source = pd.DataFrame(read(path))
    require(len(source) == expected_rows and set(source.family) == {24} and set(source.draws) == {10000}, "Wrong cumulative family or CI count")
    source["contrast"] = source.candidate + "_vs_" + source.control
    keys = ["period", "block_months", "contrast"]
    require(not source.duplicated(keys).any() and not reference.duplicated(keys).any(), "Duplicate interval")
    expected, observed = source.set_index(keys), reference.set_index(keys)
    require(set(expected.index) == set(observed.index), "Interval coverage mismatch")
    same(expected.loc[observed.index, ["observed", "lower", "upper"]], observed[["observed", "lower", "upper"]])


def compare_r_outputs(destination):
    monthly = pd.read_csv(OUT / "monthly.csv", parse_dates=["decision_eom", "return_eom"])
    require(len(monthly) == 840 and not monthly.duplicated(["decision_eom", "method"]).any() and set(monthly.method) == set(METHODS), "Incomplete monthly outcomes")
    for method in METHODS:
        block = monthly[monthly.method == method].sort_values("decision_eom")
        require(list(block.decision_eom) == list(DATES) and list(block.return_eom) == list(DATES + pd.offsets.MonthEnd(1)), "Outcome chronology differs")
    reference = pd.read_csv(destination / "R_monthly.csv")
    require(len(reference) == 840, "Incomplete R accounting")
    columns = ["gross_excess", "impact_cost", "net_excess", "turnover", "gross_exposure"]
    indexed = monthly.set_index(["decision_eom", "method"])
    expected = np.array([indexed.loc[(DATES[int(row.month_index)-1], row.method), columns].to_numpy(dtype=float) for row in reference.itertuples()])
    same(reference[columns], expected)
    main = pd.read_csv(OUT / "metrics.csv").set_index(["period", "method"])
    rm = pd.read_csv(destination / "R_metrics.csv").set_index(["period", "method"])
    require(set(main.index) == set(rm.index) and len(main) == 14, "Metric coverage differs")
    same(main.loc[rm.index, rm.columns], rm)
    maximum_draw_error = 0.
    for period in PERIODS:
        for length in (12, 24):
            draw = pd.read_csv(OUT / "analysis/bootstrap" / f"{period}_L{length}_draws.csv")
            require(list(draw.columns) == CONTRASTS and draw.shape == (10000, 8), "Wrong bootstrap contrast order or count")
            rdraw = np.fromfile(destination / f"R_{period}_L{length}_draws.bin", dtype="<f8").reshape(10000, 8)
            same(draw.to_numpy(), rdraw)
            maximum_draw_error = max(maximum_draw_error, float(np.max(abs(draw.to_numpy()-rdraw))))
    intervals = pd.read_csv(destination / "R_intervals.csv")
    compare_intervals(OUT / "intervals.json", intervals, 32)
    compare_intervals(OUT / "combined_prior_intervals.json", pd.read_csv(destination / "R_prior_intervals.csv"), 16)
    return rm, intervals, dict(language="base R", reference_variances_and_gross_returns=179,
        months=120, paths=7, monthly_rows=840, metric_rows=14, contrast_draws_checked=320000,
        current_intervals=32, previous_family24_intervals=16,
        maximum_monthly_error=float(np.max(abs(reference[columns].to_numpy()-expected))),
        maximum_metric_error=float(np.max(abs(main.loc[rm.index, rm.columns].to_numpy()-rm.to_numpy()))),
        maximum_draw_error=maximum_draw_error, independent_continuous_holdings_propagation=True,
        stock_agreement_and_amplitude_control_recomputed=True,
        strict_maturity_calibration_and_future_perturbation_passed=True,
        neutral_target_formula_contracts_passed=True)


def advancement_gate(metrics, intervals):
    candidates = {}
    for candidate in ("AGREE", "RISK"):
        points, bounds = [], []
        controls = [b for a, b in PAIRS if a == candidate]
        for period in PERIODS:
            for control in controls:
                delta = float(metrics.loc[(period, candidate), "source_utility"]-metrics.loc[(period, control), "source_utility"])
                points.append(dict(period=period, control=control, difference=delta))
                if period == "primary":
                    for length in (12, 24):
                        row = intervals[(intervals.period == period) & (intervals.block_months == length) & (intervals.contrast == f"{candidate}_vs_{control}")]
                        require(len(row) == 1, "Missing primary interval")
                        bounds.append(dict(control=control, block_months=length, lower=float(row.iloc[0].lower)))
        candidates[candidate] = dict(point_gains=points, primary_lower_bounds=bounds, passed=all(x["difference"] > 0 for x in points) and all(x["lower"] > 0 for x in bounds))
    preferred = next((name for name in ("AGREE", "RISK") if candidates[name]["passed"]), None)
    return dict(candidates=candidates, eligible_for_separately_specified_larger_test=preferred,
        any_candidate_passed=preferred is not None, both_pass_priority="AGREE", exhaustive_training_authorized=False,
        fresh_confirmation_established=False)


def resource_receipt(config):
    starts = sorted((OUT / "attempts").glob("*_start.json"))
    require([p.name for p in starts] == ["0001_start.json", "0002_start.json"], "Only one completed runner and one independent review are supported")
    total, rows = 0., []
    for index, start in enumerate(starts):
        initial = read(start); terminal = read(start.with_name(start.name.replace("_start", "_end")))
        require(initial["budget"] == config["compute_budget"] and initial["clock"] == "mach_absolute_time()", "Resource settings changed")
        same(initial["prior_active_seconds"], total, atol=1e-9, rtol=0)
        require(terminal["error"] is None and terminal["status"] == ("COMPUTED_PENDING_INDEPENDENT_REVIEW" if index == 0 else "INDEPENDENT_REVIEW_COMPLETE"), "Unreviewed attempt failure")
        samples = [json.loads(line) for line in start.with_name(start.name.replace("_start.json", "_resources.jsonl")).read_text().splitlines() if line]
        require(samples, "Missing resource samples")
        for row in samples:
            same(row["cumulative_active_seconds"], total + row["active_seconds"], atol=1e-9, rtol=0)
            require(row["physical_footprint_gib"] <= 16 and row["system_available_gib"] >= 4 and row["swap_growth_bytes"] == 0 and row["free_disk_gib"] >= 50 and row["study_disk_gib"] <= 2 and row["cumulative_active_seconds"] <= 1800, "Resource threshold exceeded")
        require(terminal["active_seconds"] >= samples[-1]["active_seconds"], "Lost sampled runtime")
        total += terminal["active_seconds"]; rows.extend(samples)
    require(total <= 1800, "Combined execution/review time exceeded")
    return dict(passed=True, scope="Runner and independent review", active_seconds=total, samples=len(rows),
        peak_process_tree_gib=max(r["physical_footprint_gib"] for r in rows), minimum_system_headroom_gib=min(r["system_available_gib"] for r in rows),
        maximum_swap_growth_bytes=max(r["swap_growth_bytes"] for r in rows), normal_native_pressure_required_by_live_guard=True)


def run():
    config = read(CONFIG)
    require(config["training_authorized"] is False and config["gamma"] == 10 and config["paths"] == METHODS and config["comparisons"] == [list(x) for x in PAIRS], "Changed frozen scientific scope")
    require(config["inference"]["family"] == 24 and config["inference"]["draws"] == 10000 and config["risk_calibration"]["window_months"] == 60, "Changed inference/calibration settings")
    require(sha(Path(base.__file__)) == HELPER_SHA, "Accepted helper reviewer changed")
    completion = read(OUT / "execution_complete.json")
    require(completion["status"] == "COMPUTED_PENDING_INDEPENDENT_REVIEW", "Runner incomplete")
    destination = OUT / "review"
    require(not destination.exists(), "Do not overwrite an existing review")
    mutex = (OUT / "execution.lock").open("a"); fcntl.flock(mutex.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pressure(); destination.mkdir()
    initial_paths = [Path(__file__), Path(base.__file__), CONFIG, OUT / "lock.json", OUT / "execution_complete.json", OUT / "protocol_freeze.json", OUT / "financial_preflight.json", PARENT / "acceptance_manifest.json", PARENT / "chronology.json", ROOT / "scripts/audit_portfolio_ensemble_chronology_20260928.py", ROOT / "extension4/runtime.py", ROOT / "temporal_fast_v4/memory.py"]
    review_lock = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in initial_paths}, R_source_sha256=hashlib.sha256(R_SOURCE.encode()).hexdigest(), no_models_fit=True)
    write(destination / "execution_lock.json", review_lock)
    runtime = Runtime(OUT, config, "independent_review"); closed = False
    try:
        verified = {}; verify_inventory(review_lock["source_sha256"], verified)
        lock = read(OUT / "lock.json")
        require(lock["config"] == config and lock["new_models_fitted"] == 0, "Frozen implementation scope changed")
        require(read(OUT / "protocol_freeze.json")["config_sha256"] == sha(CONFIG), "Configuration freeze changed")
        parent = read(PARENT / "acceptance_manifest.json")
        require(parent["accepted"] is True and all(g["passed"] for g in parent["gates"]), "Prior screen unaccepted")
        chronology = read(PARENT / "chronology.json")
        require(chronology["passed"] is True and chronology["protocol_sha256"] == sha(ROOT / "config/portfolio_ensemble_screen_20260928.json") and chronology["audit_implementation_sha256"] == sha(ROOT / "scripts/audit_portfolio_ensemble_chronology_20260928.py"), "Parent source chronology changed")
        require(read(OUT / "financial_preflight.json") == read(BROOT / "financial_preflight.json"), "Fresh financial preflight differs")
        verify_inventory(lock["artifact_sha256"], verified)
        verify_inventory(parent["source_sha256"], verified); verify_inventory(parent["artifact_sha256"], verified)
        verify_inventory(chronology["artifact_sha256"], verified)
        verify_inventory(completion["artifact_sha256"], verified)
        check_checkpoint_chain(lock, verified)
        runtime.event("independent_source_hashes_verified", files=len(verified))
        with threadpool_limits(limits=1):
            raw_returns, reference_checks = check_reference(verified, runtime)
            position_checks = check_positions(verified, runtime)
        randomness = check_resampling(verified, runtime)
        runtime.event("reference_sources_positions_and_all_date_streams_verified", reference_months=179, months=120)
        r_review(destination, raw_returns, runtime)
        rmetrics, rintervals, comparison = compare_r_outputs(destination)
        gate = advancement_gate(rmetrics, rintervals)
        main_gate = read(OUT / "result.json")["gate"]
        require(main_gate["candidate_for_larger_study_consideration"] == gate["eligible_for_separately_specified_larger_test"] and main_gate["exploratory_gate_passed"] == gate["any_candidate_passed"], "Scientific advancement decision differs")
        for name in ("AGREE", "RISK"): require(main_gate["candidates"][name]["passed"] == gate["candidates"][name]["passed"], "Candidate gate differs")
        verify_inventory(review_lock["source_sha256"], {})
        runtime.guard(); pressure(); runtime.event("independent_R_all_accounting_and_inference_passed", contrast_draws=320000)
        runtime.close("INDEPENDENT_REVIEW_COMPLETE"); closed = True
        resources = resource_receipt(config)
        result = dict(status="COMPLETED_VERIFIED_BOUNDED_ADAPTIVE_POLICY_SCREEN", accepted=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            periods=config["periods"], paths=METHODS, gate=gate, source_checks=position_checks, reference_checks=reference_checks,
            independent_R=comparison, resampling=randomness, resources=resources, no_models_fitted=True, parent_results_unchanged=True,
            inference_family=24, family_scope="Prior ensemble 8 plus current 16 contrasts, not all earlier research", block_lengths=[12, 24],
            all_periods_previously_exposed=True, fresh_confirmation_established=False, exhaustive_training_launched=False,
            published_paper_outperformance_established=False, limitation="Conditional on fitted targets and exposed outcomes; no re-estimation uncertainty, no fresh confirmation, no isolation of timing from generic deleveraging.")
        write(destination / "result.json", result)
        inputs = dict(verified)
        for path in (OUT / "attempts").iterdir():
            if path.is_file(): inputs[str(path.relative_to(ROOT))] = sha(path)
        outputs = {str(p.relative_to(ROOT)): sha(p) for p in destination.iterdir() if p.is_file()}
        write(destination / "acceptance_manifest.json", dict(accepted=True, diagnostic_complete=True, scientific_study_complete=False,
            new_extension_accepted=False, gate_passed=gate["any_candidate_passed"], source_sha256=inputs, artifact_sha256=outputs,
            gates=[dict(name="accepted_sources_chronology_and_parent_neutrality", passed=True), dict(name="independent_R_reference_adaptation_continuous_accounting_all_inference", passed=True), dict(name="combined_resource_budget", passed=True)], completed_at_utc=result["completed_at_utc"]))
        print(json.dumps(dict(status=result["status"], gate=gate, result=str(destination / "result.json")), allow_nan=False))
    except BaseException as error:
        if not closed: runtime.close("STOPPED", error)
        write(destination / "failure.json", dict(error_type=type(error).__name__, message=str(error), at_utc=datetime.now(timezone.utc).isoformat(), accepted=False, recovery="Preserve evidence and inspect the failure; do not overwrite."))
        raise


if __name__ == "__main__":
    run()
