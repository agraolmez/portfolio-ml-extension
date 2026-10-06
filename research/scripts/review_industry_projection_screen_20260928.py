"""Independent source, target industry projection, base-R accounting and inference review.

No model fitting or candidate selection. Reuses accepted economic packets and
bootstrap date streams, but independently reconstructs dense Q projections,
continuous positions, costs, every bootstrap draw and family-32 bounds.
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
CONFIG = ROOT / "config/industry_projection_screen_20260928.json"
OUT = ROOT / "output/industry_projection_screen_20260928"
PARENT = ROOT / "output/portfolio_ensemble_screen_20260928"
BROOT = ROOT / "output/feature_geometry_study_20260914"
METHODS = ["B1", "BENS", "CENS", "IND", "AMPL"]
PAIRS = [("IND", x) for x in ("B1", "BENS", "CENS", "AMPL")]
CONTRASTS = [a + "_vs_" + b for a, b in PAIRS]
PERIODS = ["primary", "robustness"]
DATES = pd.date_range("2015-12-31", "2025-11-30", freq="ME")
PRIOR_ADAPTIVE = ROOT / "output/adaptive_policy_screen_20260928"
INDUSTRIES = ["BusEq", "Chems", "Durbl", "Enrgy", "Hlth", "Manuf", "Money", "NoDur", "Other", "Shops", "Telcm", "Utils"]
HELPER_SHA = "9b6ba66c413be42c1e7102fedc5d30d90c7cd3ecf24713e7d647130761fe21b2"

R_SOURCE = r'''args <- commandArgs(trailingOnly=TRUE)
out<-args[1];parent<-args[2];prior<-args[3];dest<-args[4]
methods<-c("B1","BENS","CENS","IND","AMPL");controls<-c(1,2,3,5)
contrasts<-paste0("IND_vs_",methods[controls]);con<-file("stdin","rb")
rd<-function(n){x<-readBin(con,numeric(),n=n,size=8,endian="little");stopifnot(length(x)==n,all(is.finite(x)));x}
ri<-function(n){x<-readBin(con,integer(),n=n,size=4,endian="little");stopifnot(length(x)==n);x}
mat<-function(n,p)matrix(rd(n*p),nrow=n,byrow=TRUE)
maxerror<-0;maxneutral<-0
same<-function(x,y,atol=1e-10,rtol=1e-9){
 stopifnot(length(x)==length(y),all(is.finite(x)),all(is.finite(y)))
 dd<-abs(x-y);stopifnot(all(dd<=atol+rtol*abs(y)));maxerror<<-max(maxerror,max(dd));invisible(TRUE)
}
solve_spd<-function(a,b){u<-chol(a);backsolve(u,forwardsolve(t(u),b))}
previous<-NULL;previousids<-NULL;rows<-vector("list",120);diagnostics<-vector("list",120)
for(month in 1:120){
 sizes<-ri(3);n<-sizes[1];nn<-sizes[2];p<-sizes[3];stopifnot(n>25,nn>0,p==25)
 ids<-rd(n);nextids<-rd(nn);wm<-rd(2);wealth<-wm[1];market<-wm[2]
 impact<-rd(n);total<-rd(n);excess<-rd(n);inertia<-mat(n,n)
 loadings<-mat(n,p);fc<-mat(p,p);specific<-rd(n)
 b1<-rd(n);bens<-rd(n);cens<-rd(n)
 savedtarget<-mat(n,5);h<-mat(n,5);savedw<-mat(n,5);savedq<-mat(n,5);savedd<-mat(n,5);savednext<-mat(nn,5)
 saveddeparture<-rd(n);savedz<-rd(n);savedcorrection<-rd(n);scalars<-rd(4)
 savedexposurebefore<-rd(12);savedexposureafter<-rd(12);savedmultipliers<-rd(12)
 savedQcorrection<-rd(n);savedresiduals<-rd(4);savedactual<-mat(12,5)
 stopifnot(!anyDuplicated(ids),!anyDuplicated(nextids),wealth>0,market > -1,all(impact>0),all(specific>=0))
 if(month==1)same(h,matrix(h[,1],n,5),0,0)else{same(ids,previousids,0,0);same(h,previous);h<-previous}
 same(fc,t(fc));sigma<-loadings%*%fc%*%t(loadings)+diag(specific)
 Q<-10*sigma+diag(wealth*impact);B<-loadings[,1:12,drop=FALSE];d<-cens-bens
 QB<-solve_spd(Q,B);gram<-crossprod(B,QB);mult<-as.numeric(solve_spd(gram,crossprod(B,d)))
 correction<-as.numeric(QB%*%mult);z<-d-correction
 qbefore<-sum(d*(Q%*%d));qafter<-sum(z*(Q%*%z));qcorr<-sum(correction*(Q%*%correction))
 stopifnot(qbefore>=0,qafter>=0,qafter<=qbefore+1e-10)
 amp<-if(qbefore==0)1 else min(1,sqrt(qafter/qbefore))
 before<-as.numeric(crossprod(B,d));after<-as.numeric(crossprod(B,z))
 same(after,rep(0,12));same(sum(z*(Q%*%correction)),0);same(qbefore,qafter+qcorr)
 z2<-z-as.numeric(QB%*%solve_spd(gram,crossprod(B,z)));same(z,z2)
 same(sum((amp*d)*(Q%*%(amp*d))),qafter)
 same(Q%*%correction,B%*%mult);same(savedQcorrection,Q%*%correction)
 stopifnot(all(abs(savedresiduals)<=1e-8*(1+qbefore)))
 same(saveddeparture,d);same(savedz,z);same(savedcorrection,correction)
 same(scalars,c(amp,qbefore,qafter,qcorr));same(savedexposurebefore,before);same(savedexposureafter,after);same(savedmultipliers,mult)
 target<-cbind(b1,bens,cens,bens+z,bens+amp*d);same(savedtarget,target)
 w<-h+(diag(n)-inertia)%*%(target-h);q<-w-h;drift<-w*((1+total)/(1+market))
 ix<-match(nextids,ids);nh<-matrix(0,nn,5);present<-!is.na(ix);nh[present,]<-drift[ix[present],,drop=FALSE]
 same(savedw,w);same(savedq,q);same(savedd,drift);same(savednext,nh);same(savedactual,crossprod(B,w))
 # Neutral transform must recover the same CENS transition from CENS own state.
 neutral<-h[,3]+(diag(n)-inertia)%*%((bens+d)-h[,3]);same(neutral,w[,3])
 maxneutral<-max(maxneutral,max(abs(neutral-w[,3])))
 gross<-colSums(w*excess);cost<-wealth/2*colSums(q*q*impact)
 rows[[month]]<-data.frame(month_index=month,method=methods,gross_excess=gross,impact_cost=cost,net_excess=gross-cost,
  turnover=colSums(abs(q)),gross_exposure=colSums(abs(w)))
 diagnostics[[month]]<-data.frame(month_index=month,amplitude=amp,qnorm_before=qbefore,qnorm_after=qafter,qnorm_correction=qcorr,
  max_target_industry_exposure=max(abs(after)),max_executed_increment_industry_exposure=max(abs(crossprod(B,(diag(n)-inertia)%*%z))))
 previous<-nh;previousids<-nextids
}
close(con);monthly<-do.call(rbind,rows)
write.table(monthly,file.path(dest,"R_monthly.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,diagnostics),file.path(dest,"R_projection.csv"),sep=",",row.names=FALSE)
metrics<-list();intervals<-list();priorintervals<-list()
oldstudies<-list(parent,prior)
for(period in c("primary","robustness")){
 dates<-if(period=="primary")1:60 else 61:120;g<-matrix(0,60,5);cost<-g
 for(j in 1:5){x<-monthly[monthly$method==methods[j]&monthly$month_index%in%dates,];stopifnot(nrow(x)==60);g[,j]<-x$gross_excess;cost[,j]<-x$impact_cost}
 net<-g-cost;gv<-apply(g,2,var);nv<-colMeans((net-matrix(colMeans(net),60,5,byrow=TRUE))^2)
 u<-12*(colMeans(net)-5*gv);point<-u[4]-u[controls]
 for(j in 1:5)metrics[[length(metrics)+1]]<-data.frame(period=period,method=methods[j],months=60,source_utility=u[j],
  net_population_utility=12*(mean(net[,j])-5*nv[j]),annual_gross_volatility=sqrt(12*gv[j]),annual_net_mean=12*mean(net[,j]),annual_cost=12*mean(cost[,j]))
 for(L in c(12L,24L)){
  idx<-as.matrix(read.csv(file.path(parent,"analysis/bootstrap",paste0(period,"_L",L,"_indices.csv")),header=FALSE))
  stopifnot(identical(dim(idx),c(10000L,60L)),all(idx>=1),all(idx<=60),all(idx==floor(idx)))
  draw<-matrix(0,10000,4)
  for(bb in 1:10000){ii<-idx[bb,];bu<-12*(colMeans(net[ii,,drop=FALSE])-5*apply(g[ii,,drop=FALSE],2,var));draw[bb,]<-bu[4]-bu[controls]}
  colnames(draw)<-contrasts
  writeBin(as.double(t(draw)),file.path(dest,paste0("R_",period,"_L",L,"_draws.bin")),size=8,endian="little")
  for(j in 1:4){q<-as.numeric(quantile(draw[,j],c(.05/64,1-.05/64),type=7));intervals[[length(intervals)+1]]<-data.frame(period=period,block_months=L,contrast=contrasts[j],observed=point[j],lower=2*point[j]-q[2],upper=2*point[j]-q[1])}
  for(oldstudy in oldstudies){
   oldmetrics<-read.csv(file.path(oldstudy,"metrics.csv"));om<-oldmetrics[oldmetrics$period==period,];ou<-setNames(om$source_utility,om$method)
   olddraw<-read.csv(file.path(oldstudy,"analysis/bootstrap",paste0(period,"_L",L,"_draws.csv")));stopifnot(nrow(olddraw)==10000)
   for(j in seq_len(ncol(olddraw))){
    pair<-strsplit(names(olddraw)[j],"_vs_",fixed=TRUE)[[1]];stopifnot(length(pair)==2);op<-ou[pair[1]]-ou[pair[2]]
    qq<-as.numeric(quantile(olddraw[,j],c(.05/64,1-.05/64),type=7))
    priorintervals[[length(priorintervals)+1]]<-data.frame(period=period,block_months=L,contrast=names(olddraw)[j],observed=op,lower=2*op-qq[2],upper=2*op-qq[1])
   }
  }
 }
}
write.table(do.call(rbind,metrics),file.path(dest,"R_metrics.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,intervals),file.path(dest,"R_intervals.csv"),sep=",",row.names=FALSE)
write.table(do.call(rbind,priorintervals),file.path(dest,"R_prior_intervals.csv"),sep=",",row.names=FALSE)
write.table(data.frame(maximum_accounting_error=maxerror,maximum_neutral_position_error=maxneutral),file.path(dest,"R_accounting.csv"),sep=",",row.names=FALSE)
'''


def send_numbers(stream, chunks):
    for chunk in chunks:
        stream.write(np.asarray(chunk, dtype="<f8", order="C").tobytes(order="C"))


def risk_packet(day):
    return arrays(ROOT / "data/prepared/step9_v1/baseline/risk" / f"{day.date()}.npz")


def send_month(stream, packet, economic, risk):
    n, nn = len(packet["ids"]), len(packet["next_ids"])
    stream.write(np.asarray([n, nn, 25], dtype="<i4").tobytes())
    chunks = [packet["ids"], packet["next_ids"], [float(economic["wealth"]), float(economic["market_growth"])]]
    chunks += [economic[k] for k in ("impact", "total_returns", "excess_returns", "m")]
    chunks += [risk[k] for k in ("loadings", "factor_cov", "specific_variance")]
    chunks += [economic["targets"][:, j] for j in (0, 6, 7)]
    chunks += [packet[k] for k in ("targets", "pretrade", "weights", "trades", "drifted_weights", "next_pretrade", "departure", "projected_departure", "hedge_correction")]
    chunks += [[float(packet[k]) for k in ("amplitude", "qnorm_before", "qnorm_after", "qnorm_correction")]]
    chunks += [packet[k] for k in ("industry_exposure_before", "industry_exposure_after", "multipliers")]
    chunks += [packet["q_times_correction"], [float(packet[k]) for k in ("kkt_max_residual", "q_solve_max_residual", "orthogonality_residual", "norm_identity_residual")], packet["actual_industry_exposure"]]
    send_numbers(stream, chunks)


def r_review(destination, runtime):
    rpath = destination / "verify.R"
    with rpath.open("x") as stream:
        stream.write(R_SOURCE)
    failure, stop = [], threading.Event()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    command = [shutil.which("Rscript") or "/usr/local/bin/Rscript", "--vanilla", str(rpath), str(OUT), str(PARENT), str(PRIOR_ADAPTIVE), str(destination)]
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
            for day in DATES:
                runtime.guard(); pressure()
                stamp = str(day.date())
                send_month(child.stdin, arrays(OUT / "checkpoints" / stamp / "arrays.npz"), arrays(PARENT / "checkpoints" / stamp / "arrays.npz"), risk_packet(day))
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


def check_risk_sources(verified, runtime):
    inventory = merged_parent_inventory(verified)
    names_path = ROOT / "output/step3_author_conventions_20260911/factor_calendar.json"
    names = read(names_path)["factors"]
    expected_names = INDUSTRIES + ["low_leverage", "investment", "size", "value", "quality", "low_risk", "seasonality", "accruals", "profitability", "debt_issuance", "short_term_reversal", "profit_growth", "momentum"]
    require(names == expected_names and read(CONFIG)["projection"]["industry_names"] == INDUSTRIES, "Frozen factor names/order differ")
    require(str(names_path.relative_to(ROOT)) in verified, "Current factor metadata is not frozen-source-bound")
    accepted_manifest_path = ROOT / "output/step4_20260911/acceptance_manifest.json"
    require(str(accepted_manifest_path.relative_to(ROOT)) in verified, "Step4 acceptance receipt is not source-bound")
    accepted_manifest = read(accepted_manifest_path)
    require(accepted_manifest["status"] == "ACCEPTED_BASELINE_MACHINERY" and accepted_manifest["step4_complete"] is True
            and all(g["passed"] for g in accepted_manifest["gates"]), "Step4 factor-label source not accepted")
    accepted_names_path = ROOT / "data/prepared/step4_v1/risk/factor_covariances.npz"
    accepted(accepted_names_path, accepted_manifest["artifact_sha256"], verified)
    with np.load(accepted_names_path, allow_pickle=False) as source:
        np.testing.assert_array_equal(source["factors"], expected_names)
    # The calendar receives a new lock hash; the older accepted NPZ establishes
    # the semantic column order independently of that current metadata file.

    for day in DATES:
        runtime.guard(); pressure()
        path = ROOT / "data/prepared/step9_v1/baseline/risk" / f"{day.date()}.npz"
        accepted(path, inventory, verified)
        risk = risk_packet(day)
        old = arrays(PARENT / "checkpoints" / str(day.date()) / "arrays.npz")
        copied = arrays(OUT / "checkpoints" / str(day.date()) / "arrays.npz")
        np.testing.assert_array_equal(risk["ids"], old["ids"])
        np.testing.assert_array_equal(copied["factor_names"], names)
        for key in ("loadings", "factor_cov", "specific_variance"): same(copied[key], risk[key], atol=0, rtol=0)
        require(risk["loadings"].shape == (len(old["ids"]), 25) and risk["factor_cov"].shape == (25, 25), "Risk shape differs")
        require(all(np.isfinite(risk[k]).all() for k in ("loadings", "factor_cov", "specific_variance")) and (risk["specific_variance"] >= 0).all(), "Invalid accepted covariance")
        same(risk["factor_cov"], risk["factor_cov"].T)
    return dict(risk_months=120, first_decision=str(DATES[0].date()), last_decision=str(DATES[-1].date()),
        accepted_factor_covariance=True, merged_consistent_E2_E3_E4_inventories=True,
        all_25_covariance_factors_retained=True, exactly_12_named_target_industry_constraints=True,
        factor_names_verified_from_accepted_Step4_covariance_archive=True, current_calendar_bound_as_metadata=True,
        actual_executed_industry_neutrality_claimed=False, style_exposures_preserved_claimed=False)


def check_positions(verified, runtime):
    previous, maximum_error = None, 0.
    for day in DATES:
        runtime.guard(); pressure(); stamp = str(day.date())
        packet = arrays(OUT / "checkpoints" / stamp / "arrays.npz")
        oldpath = PARENT / "checkpoints" / stamp / "arrays.npz"
        old = arrays(oldpath)
        np.testing.assert_array_equal(packet["methods"], METHODS)
        require(bool(packet["projection_enabled"]), "Historical projection was disabled")
        same(packet["actual_industry_increment_vs_bens"], packet["actual_industry_exposure"]-packet["actual_industry_exposure"][:, 1, None], atol=0, rtol=0)
        for key in ("ids", "next_ids"): np.testing.assert_array_equal(packet[key], old[key])
        for new_index, old_index in ((0, 0), (1, 6), (2, 7)):
            for key in ("targets", "pretrade", "weights", "trades", "drifted_weights", "next_pretrade"):
                same(packet[key][:, new_index], old[key][:, old_index])
                maximum_error = max(maximum_error, float(np.max(abs(packet[key][:, new_index]-old[key][:, old_index]))))
        if previous is None: same(packet["pretrade"], np.repeat(old["pretrade"][:, 0, None], 5, axis=1), atol=0, rtol=0)
        else: same(packet["pretrade"], previous, atol=0, rtol=0)
        previous = packet["next_pretrade"]
    return dict(months=120, paths=5, same_common_initial_holdings=True, initial_switch_charged=True,
        continuous_through_2021=True, B1_BENS_CENS_accepted_paths_reproduced=True,
        maximum_parent_position_error=maximum_error,
        original_m_returns_costs_wealth_used_from_hash_bound_accepted_parent=True)


def check_checkpoint_chain(lock, verified):
    previous, lock_hash = None, sha(OUT / "lock.json")
    require(len(list((OUT / "checkpoints").glob("*/commit.json"))) == 120, "Exactly 120 checkpoints required")
    for day in DATES:
        folder = OUT / "checkpoints" / str(day.date()); commit = read(folder / "commit.json")
        require(commit["lock_sha256"] == lock_hash and commit["previous_commit_sha256"] == previous and commit["decision_eom"] == str(day.date()), "Broken monthly checkpoint chain")
        require(set(commit["files_sha256"]) == {"arrays.npz", "monthly.json"}, "Unexpected monthly checkpoint files")
        require(commit["neutral_control_checked"] is True and commit["target_neutrality_scope_only"] is True, "Checkpoint neutral/scope metadata differs")
        verify_inventory({str((folder / name).relative_to(ROOT)): digest for name, digest in commit["files_sha256"].items()}, verified)
        parent = str((PARENT / "checkpoints" / str(day.date()) / "arrays.npz").relative_to(ROOT))
        require(commit["parent_packet"] == parent and commit["parent_packet_sha256"] == lock["artifact_sha256"].get(parent), "Parent packet binding differs")
        verify_inventory({parent: commit["parent_packet_sha256"]}, verified)
        risk_source = str((ROOT / "data/prepared/step9_v1/baseline/risk" / f"{day.date()}.npz").relative_to(ROOT))
        require(commit["risk_source"] == risk_source and commit["risk_source_sha256"] == lock["artifact_sha256"].get(risk_source), "Risk source binding differs")
        verify_inventory({risk_source: commit["risk_source_sha256"]}, verified)
        previous = sha(folder / "commit.json")
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
    result["same_indices_across_all_five_paths"] = True
    result.pop("same_indices_across_all_nine_paths", None)
    return result


def compare_intervals(path, reference, expected_rows):
    source = pd.DataFrame(read(path))
    require(len(source) == expected_rows and set(source.family) == {32} and set(source.draws) == {10000}, "Wrong cumulative family or CI count")
    source["contrast"] = source.candidate + "_vs_" + source.control
    keys = ["period", "block_months", "contrast"]
    require(not source.duplicated(keys).any() and not reference.duplicated(keys).any(), "Duplicate interval")
    expected, observed = source.set_index(keys), reference.set_index(keys)
    require(set(expected.index) == set(observed.index), "Interval coverage mismatch")
    same(expected.loc[observed.index, ["observed", "lower", "upper"]], observed[["observed", "lower", "upper"]])


def compare_r_outputs(destination):
    monthly = pd.read_csv(OUT / "monthly.csv", parse_dates=["decision_eom", "return_eom"])
    require(len(monthly) == 600 and not monthly.duplicated(["decision_eom", "method"]).any() and set(monthly.method) == set(METHODS), "Incomplete monthly outcomes")
    for method in METHODS:
        block = monthly[monthly.method == method].sort_values("decision_eom")
        require(list(block.decision_eom) == list(DATES) and list(block.return_eom) == list(DATES + pd.offsets.MonthEnd(1)), "Outcome chronology differs")
    reference = pd.read_csv(destination / "R_monthly.csv")
    require(len(reference) == 600, "Incomplete R accounting")
    columns = ["gross_excess", "impact_cost", "net_excess", "turnover", "gross_exposure"]
    indexed = monthly.set_index(["decision_eom", "method"])
    expected = np.array([indexed.loc[(DATES[int(row.month_index)-1], row.method), columns].to_numpy(dtype=float) for row in reference.itertuples()])
    same(reference[columns], expected)
    main = pd.read_csv(OUT / "metrics.csv").set_index(["period", "method"])
    rm = pd.read_csv(destination / "R_metrics.csv").set_index(["period", "method"])
    require(set(main.index) == set(rm.index) and len(main) == 10, "Metric coverage differs")
    same(main.loc[rm.index, rm.columns], rm)
    maximum_draw_error = 0.
    for period in PERIODS:
        for length in (12, 24):
            draw = pd.read_csv(OUT / "analysis/bootstrap" / f"{period}_L{length}_draws.csv")
            require(list(draw.columns) == CONTRASTS and draw.shape == (10000, 4), "Wrong bootstrap contrast order or count")
            rdraw = np.fromfile(destination / f"R_{period}_L{length}_draws.bin", dtype="<f8").reshape(10000, 4)
            same(draw.to_numpy(), rdraw)
            maximum_draw_error = max(maximum_draw_error, float(np.max(abs(draw.to_numpy()-rdraw))))
    intervals = pd.read_csv(destination / "R_intervals.csv")
    compare_intervals(OUT / "intervals.json", intervals, 16)
    compare_intervals(OUT / "combined_prior_intervals.json", pd.read_csv(destination / "R_prior_intervals.csv"), 48)
    return rm, intervals, dict(language="base R", covariance_and_projection_months=120,
        months=120, paths=5, monthly_rows=600, metric_rows=10, contrast_draws_checked=160000,
        current_intervals=16, previous_family32_intervals=48,
        maximum_monthly_error=float(np.max(abs(reference[columns].to_numpy()-expected))),
        maximum_metric_error=float(np.max(abs(main.loc[rm.index, rm.columns].to_numpy()-rm.to_numpy()))),
        maximum_draw_error=maximum_draw_error, independent_continuous_holdings_propagation=True,
        independent_dense_Q_projection_oracle=True, Q_orthogonality_and_contraction=True,
        industry_target_neutrality_and_projection_idempotence=True, AMPL_matches_Q_norm=True,
        neutral_target_transition_contract_passed=True, actual_executed_industry_neutrality_claimed=False)


def advancement_gate(metrics, intervals):
    candidates = {}
    for candidate in ("IND",):
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
    preferred = next((name for name in ("IND",) if candidates[name]["passed"]), None)
    return dict(candidates=candidates, eligible_for_separately_specified_larger_test=preferred,
        any_candidate_passed=preferred is not None, exhaustive_training_authorized=False,
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
    require(config["inference"]["family"] == 32 and config["inference"]["draws"] == 10000 and config["projection"]["industry_names"] == INDUSTRIES and config["projection"]["actual_executed_industry_neutrality_guaranteed"] is False, "Changed inference/projection settings")
    require(sha(Path(base.__file__)) == HELPER_SHA, "Accepted helper reviewer changed")
    completion = read(OUT / "execution_complete.json")
    require(completion["status"] == "COMPUTED_PENDING_INDEPENDENT_REVIEW", "Runner incomplete")
    destination = OUT / "review"
    require(not destination.exists(), "Do not overwrite an existing review")
    mutex = (OUT / "execution.lock").open("a"); fcntl.flock(mutex.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pressure(); destination.mkdir()
    initial_paths = [Path(__file__), Path(base.__file__), CONFIG, OUT / "lock.json", OUT / "execution_complete.json", OUT / "protocol_freeze.json", OUT / "financial_preflight.json", PARENT / "acceptance_manifest.json", PRIOR_ADAPTIVE / "acceptance_manifest.json", PARENT / "chronology.json", ROOT / "scripts/audit_portfolio_ensemble_chronology_20260928.py", ROOT / "extension4/runtime.py", ROOT / "temporal_fast_v4/memory.py"]
    review_lock = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in initial_paths}, R_source_sha256=hashlib.sha256(R_SOURCE.encode()).hexdigest(), no_models_fit=True)
    write(destination / "execution_lock.json", review_lock)
    runtime = Runtime(OUT, config, "independent_review"); closed = False
    try:
        verified = {}; verify_inventory(review_lock["source_sha256"], verified)
        lock = read(OUT / "lock.json")
        require(lock["config"] == config and lock["new_models_fitted"] == 0, "Frozen implementation scope changed")
        require(read(OUT / "protocol_freeze.json")["config_sha256"] == sha(CONFIG), "Configuration freeze changed")
        parent = read(PARENT / "acceptance_manifest.json")
        prior_adaptive = read(PRIOR_ADAPTIVE / "acceptance_manifest.json")
        for receipt in (parent, prior_adaptive):
            require(receipt["accepted"] is True and all(g["passed"] for g in receipt["gates"]), "Prior screen unaccepted")
        chronology = read(PARENT / "chronology.json")
        require(chronology["passed"] is True and chronology["protocol_sha256"] == sha(ROOT / "config/portfolio_ensemble_screen_20260928.json") and chronology["audit_implementation_sha256"] == sha(ROOT / "scripts/audit_portfolio_ensemble_chronology_20260928.py"), "Parent source chronology changed")
        require(read(OUT / "financial_preflight.json") == read(BROOT / "financial_preflight.json"), "Fresh financial preflight differs")
        verify_inventory(lock["artifact_sha256"], verified)
        for receipt in (parent, prior_adaptive):
            verify_inventory(receipt["source_sha256"], verified); verify_inventory(receipt["artifact_sha256"], verified)
        verify_inventory(chronology["artifact_sha256"], verified)
        verify_inventory(completion["artifact_sha256"], verified)
        check_checkpoint_chain(lock, verified)
        runtime.event("independent_source_hashes_verified", files=len(verified))
        with threadpool_limits(limits=1):
            risk_checks = check_risk_sources(verified, runtime)
            position_checks = check_positions(verified, runtime)
        randomness = check_resampling(verified, runtime)
        runtime.event("risk_sources_positions_and_all_date_streams_verified", risk_months=120, months=120)
        r_review(destination, runtime)
        rmetrics, rintervals, comparison = compare_r_outputs(destination)
        gate = advancement_gate(rmetrics, rintervals)
        main_gate = read(OUT / "result.json")["gate"]
        require(main_gate["candidate_for_larger_study_consideration"] == gate["eligible_for_separately_specified_larger_test"] and main_gate["exploratory_gate_passed"] == gate["any_candidate_passed"], "Scientific advancement decision differs")
        for name in ("IND",): require(main_gate["candidates"][name]["passed"] == gate["candidates"][name]["passed"], "Candidate gate differs")
        verify_inventory(review_lock["source_sha256"], {})
        runtime.guard(); pressure(); runtime.event("independent_R_all_accounting_and_inference_passed", contrast_draws=160000)
        runtime.close("INDEPENDENT_REVIEW_COMPLETE"); closed = True
        resources = resource_receipt(config)
        result = dict(status="COMPLETED_VERIFIED_BOUNDED_INDUSTRY_PROJECTION_SCREEN", accepted=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            periods=config["periods"], paths=METHODS, gate=gate, source_checks=position_checks, risk_checks=risk_checks,
            independent_R=comparison, resampling=randomness, resources=resources, no_models_fitted=True, parent_results_unchanged=True,
            inference_family=32, family_scope="Prior ensemble 8 plus adaptive 16 plus current 8 contrasts, not all earlier research", block_lengths=[12, 24],
            all_periods_previously_exposed=True, fresh_confirmation_established=False, exhaustive_training_launched=False,
            published_paper_outperformance_established=False, limitation="Conditional on fitted targets and exposed outcomes; no re-estimation uncertainty, no fresh confirmation, industry neutrality applies to incremental targets, not executed holdings; Q is a target-distance metric, not total dynamic utility.")
        write(destination / "result.json", result)
        inputs = dict(verified)
        for path in (OUT / "attempts").iterdir():
            if path.is_file(): inputs[str(path.relative_to(ROOT))] = sha(path)
        outputs = {str(p.relative_to(ROOT)): sha(p) for p in destination.iterdir() if p.is_file()}
        write(destination / "acceptance_manifest.json", dict(accepted=True, diagnostic_complete=True, scientific_study_complete=False,
            new_extension_accepted=False, gate_passed=gate["any_candidate_passed"], source_sha256=inputs, artifact_sha256=outputs,
            gates=[dict(name="accepted_sources_chronology_and_parent_neutrality", passed=True), dict(name="independent_R_dense_projection_continuous_accounting_all_inference", passed=True), dict(name="combined_resource_budget", passed=True)], completed_at_utc=result["completed_at_utc"]))
        print(json.dumps(dict(status=result["status"], gate=gate, result=str(destination / "result.json")), allow_nan=False))
    except BaseException as error:
        if not closed: runtime.close("STOPPED", error)
        write(destination / "failure.json", dict(error_type=type(error).__name__, message=str(error), at_utc=datetime.now(timezone.utc).isoformat(), accepted=False, recovery="Preserve evidence and inspect the failure; do not overwrite."))
        raise


if __name__ == "__main__":
    run()
