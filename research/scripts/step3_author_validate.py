"""Acceptance checks for the named authors-convention reconstruction only."""
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from step3_author_bundle import ROOT, OUT, PREP, RAW1, RAW2, FEATURES, sha, save_json, resource_guard


def validate():
    gates = []
    def check(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "evidence": detail})
        if not passed:
            save_json(OUT / "failed_validation.json", gates)
            raise AssertionError(f"{name}: {detail}")
    x = pd.read_parquet(PREP / "monthly_inputs.parquet")
    active = x[x.valid]
    check("Unique monthly keys", not x.duplicated(["id", "eom"]).any(), len(x))
    for start, end, name in [("1980-12-31","1982-11-30","Pilot decision coverage"),("1980-12-31","2020-11-30","Baseline decision coverage"),("2020-12-31","2025-11-30","Exposed robustness decision coverage")]:
        observed = active.loc[active.eom.between(start,end), "eom"].drop_duplicates().sort_values().tolist()
        expected = pd.date_range(start,end,freq="ME").tolist()
        check(name, observed==expected, {"months":len(observed),"first":start,"last":end})
    r = pd.read_parquet(PREP / "future_excess_returns.parquet")
    r_actual = pd.read_csv(OUT / "r_actual_future_returns.csv", parse_dates=["eom"])
    keys = ["id","eom"]; returns = [f"ret_ld{k}" for k in range(1,13)]
    r_actual = r_actual.sort_values(keys).reset_index(drop=True)
    r = r.sort_values(keys).reset_index(drop=True)
    check("Actual R source return keys", r[keys].equals(r_actual[keys].astype({"id":r.id.dtype})), len(r))
    ret_diff = float(np.abs(r[returns].to_numpy()-r_actual[returns].to_numpy()).max())
    check("All 12 source return leads agree with actual R", ret_diff<1e-10, {"values":len(r)*12,"max_absolute_difference":ret_diff})
    flags = pd.read_csv(OUT / "r_actual_screens.csv",parse_dates=["eom"]).sort_values(keys).reset_index(drop=True)
    xs = x.sort_values(keys).reset_index(drop=True)
    check("Actual R full-history screen keys", xs[keys].equals(flags[keys].astype({"id":xs.id.dtype})),len(flags))
    mismatches = {f:int(xs[f].ne(flags[f]).sum()) for f in ["valid_data","valid_size","valid"]}
    check("Actual R lookback, percentile, entry and exit rules", sum(mismatches.values())==0,mismatches)
    ranks = pd.read_csv(OUT / "r_actual_rank_samples.csv",parse_dates=["eom"])
    rp = ranks.merge(x[keys+FEATURES],on=keys,validate="one_to_one",suffixes=("_r","_py"))
    rank_diff=max(float((rp[f+"_r"]-rp[f+"_py"]).abs().max()) for f in FEATURES)
    check("R empirical ranks, zero preservation and median imputation",rank_diff<1e-12,{"stock_months":len(rp),"features":len(FEATURES),"values":len(rp)*len(FEATURES),"max_difference":rank_diff})
    v=x[FEATURES].to_numpy()
    check("Complete bounded feature matrix",np.isfinite(v).all() and (v>=0).all() and (v<=1).all(),{"rows":len(x),"features":len(FEATURES)})
    del v,r_actual,ranks,rp
    economic=["ret_ld1","tr_ld0","tr_ld1","dolvol","lambda_diagonal","wealth_usd","mu_ld0","mu_ld1","market_equity_usd"]
    check("Required active economic inputs are finite",np.isfinite(active[economic].to_numpy()).all(),economic)
    check("Return totals use the outcome-month subtraction rate",np.allclose(active.tr_ld1,active.ret_ld1+active.rf_crsp_t30,atol=1e-14,rtol=0),float((active.tr_ld1-active.ret_ld1-active.rf_crsp_t30).abs().max()))
    check("Positive market equity, dollar volume, impact and wealth",active[["market_equity_usd","dolvol","lambda_diagonal","wealth_usd"]].gt(0).all().all(),"USD; USD/day; 0.2/dollar_volume; USD wealth")
    events=pd.read_parquet(PREP / "outcome_and_exit_ledger.parquet")
    check("Every eligible outcome and disappearance has a named convention",len(events)==len(active) and not events.duplicated(keys).any() and events[["ret_ld1","tr_ld1","transition_rule","missing_return_rule"]].notna().all().all(),{"rows":len(events),"imputed_next_excess":int(events.imputed_ld1.sum()),"nonboundary_exits":int((events.exit_after_return & ~events.right_boundary_exit).sum())})
    r1=pd.read_parquet(PREP / "future_excess_returns_r1.parquet")
    check("No unnoticed right-boundary label completion",r1.right_censored_12m.equals(r1.label_maturity_eom.gt("2025-12-31")),{"right_censored_rows":int(r1.right_censored_12m.sum()),"boundary_imputation_is_not_label_maturity":True})
    w=pd.read_parquet(PREP / "market_wealth.parquet")
    original=pd.read_csv(ROOT / "output/step3_20260910/public_sources/[usa]_[mkt]_[monthly]_[vw].csv",parse_dates=["date"])
    wm=w.merge(original.rename(columns={"date":"eom","ret":"source_excess"}),on="eom",validate="one_to_one")
    check("Publisher benchmark used directly with verified reference rate",len(w)==877 and np.array_equal(wm.published_excess,wm.source_excess) and np.allclose(wm.market_total_return,wm.source_excess+wm.rf_crsp_t30,rtol=0,atol=1e-14),{"months":len(wm),"constituent_replication_is_separate_diagnostic":True})
    rec=(w.wealth_usd.shift(1)-w.wealth_usd/(1+w.market_total_return))/w.wealth_usd.shift(1)
    check("Approved reciprocal wealth and exact anchor",float(w.loc[w.eom.eq("2020-12-31"),"wealth_usd"].iloc[0])==1e10 and rec.abs().max()<1e-12 and np.isfinite(w.wealth_usd).all(),float(rec.abs().max()))
    calendar=json.loads((OUT / "factor_calendar.json").read_text())
    check("Daily factor history and full rank",calendar["rank_deficient_days"]==0 and calendar["observations_at_first_covariance_month"]>=2490 and pd.Timestamp(calendar["first_moment_month"])<pd.Timestamp("1970-12-31"),calendar)
    f=pd.read_parquet(PREP / "daily_factor_returns.parquet")
    fr=pd.read_csv(OUT / "factor_R_comparison.csv",parse_dates=["date"])
    fp=f.melt(id_vars="date",value_vars=calendar["factors"],var_name="term",value_name="python_estimate")
    fc=fr.merge(fp,on=["date","term"],validate="one_to_one")
    diff=float((fc.estimate-fc.python_estimate).abs().max())
    check("Independent R factor regression parity",len(fc)==len(fr) and diff<1e-10,{"coefficients":len(fc),"max_difference":diff})
    coverage=pd.read_csv(OUT / "daily_coverage.csv")
    check("All daily partitions available",coverage.year.tolist()==list(range(1952,2026)),{"partitions":len(coverage),"daily_rows":int(coverage.rows.sum()),"missing_returns_preserved":int(coverage.missing_returns_retained.sum())})
    inventory=[]
    for p in sorted([*RAW1.glob("*.parquet"),*RAW2.glob("*.parquet")]):
        receipt=json.loads(p.with_suffix(".receipt.json").read_text())
        valid=sha(p)==receipt["sha256"] and pq.ParquetFile(p).metadata.num_rows==receipt["rows"]
        inventory.append({"path":str(p.relative_to(ROOT)),"sha256":receipt["sha256"],"rows":receipt["rows"],"passed":valid})
    check("Raw captured-source integrity",all(v["passed"] for v in inventory),{"files":len(inventory)})
    save_json(OUT / "raw_inventory.json",inventory)
    protected=json.loads((ROOT / "output/step3_20260911/start.json").read_text())["protected"]
    check("Approved scientific records unchanged",all(sha(ROOT/p)==s for p,s in protected.items()),protected)
    decision=json.loads((ROOT / "config/step3_author_conventions.json").read_text())
    check("Author conventions and correction scope recorded",decision["user_instruction"]=="Finish step 3 please. Do what they do" and decision["bespoke_carried_claim_policy_adopted"] is False,decision["id"])
    manifest={"bundle_id":decision["id"],"status":"ACCEPTED_AUTHOR_CONVENTION_RECONSTRUCTION","step3_complete":True,"accepted_bundle_path":str(PREP.relative_to(ROOT)),"authorized_next_step":"Step 4: baseline machinery and accounting validation","historical_financial_experiments_run":False,"exact_paper_input_vintage":False,"causal_or_realistic_execution_certified":False,"raw_files_verified":len(inventory),"monthly_rows":len(x),"active_stock_months":len(active),"active_securities":int(active.id.nunique()),"decision_range":[str(active.eom.min().date()),str(active.eom.max().date())],"gates":gates,"gates_passed":len(gates),"limitations":decision["limitations"],"resources":resource_guard()}
    save_json(OUT / "acceptance_manifest.json",manifest)
    save_json(PREP / "STATUS.json",{k:v for k,v in manifest.items() if k!="gates"})
    files=[p for p in PREP.rglob("*") if p.is_file()]+list((ROOT / "scripts").glob("step3_author_*"))+[ROOT / "config/step3_author_conventions.json",ROOT/"config/feature_dictionary_v1.csv",ROOT/"output/step2_20260910/source_settings.json"]
    files += [ROOT/"scripts"/name for name in ["step3_io.py","step3_acquire.py","step3_v2_io.py","step3_v2_factor_calendar.py"]]
    files += [p for p in (ROOT/"references/author_code").rglob("*") if p.is_file()]
    save_json(OUT / "bundle_checksums.json",{str(p.relative_to(ROOT)):sha(p) for p in files})
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["gates","limitations"]},indent=2),flush=True)
    return manifest


if __name__=="__main__":
    validate()
