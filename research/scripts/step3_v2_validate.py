"""Validate the new repair vintage without accepting unresolved economics."""
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from step3_v2_io import ROOT, RAW, PREP, OUT, PRIOR_RAW, sha, save_json, utc, resource_guard


def daily_path(year):
    return RAW/f"crsp_ciz_daily_full_ids_{year}.parquet" if year >= 1982 else PRIOR_RAW/f"crsp_ciz_daily_{year}.parquet"


def inventory():
    rows = []
    for base in [PRIOR_RAW, RAW]:
        for p in sorted(base.glob("*.parquet")):
            receipt = json.loads(p.with_suffix(".receipt.json").read_text())
            q = ROOT/receipt["query_path"]
            parameters = json.loads(q.with_suffix(".params.json").read_text())
            fingerprint = hashlib.sha256((q.read_text().removesuffix("\n")+json.dumps(parameters,default=str)).encode()).hexdigest()
            count = pq.ParquetFile(p).metadata.num_rows
            digest = sha(p)
            rows.append({"file":str(p.relative_to(ROOT)),"rows":count,"bytes":p.stat().st_size,"sha256":digest,"hash_pass":digest==receipt["sha256"],"row_count_pass":count==receipt["rows"],"query_pass":fingerprint==receipt["query_fingerprint"]})
    x = pd.DataFrame(rows)
    if not x[["hash_pass","row_count_pass","query_pass"]].all().all():
        raise ValueError("Raw source provenance check failed")
    x.to_csv(OUT/"raw_inventory.csv",index=False)
    if any(list(d.glob("*.partial")) for d in [PRIOR_RAW, RAW]):
        raise ValueError("Unfinished source partition")
    return x


def validate_daily_and_scales():
    scales = pd.read_parquet(OUT/"risk_scale_reconciliationfull.parquet")
    cases = scales[scales.source_missing].copy()
    ids = cases.permno.unique().tolist()
    pieces=[]; coverage=[]
    rates=pd.read_parquet(PRIOR_RAW/"crsp_ciz_t30_index.parquet")
    rates["eom"]=rates.mthcaldt+pd.offsets.MonthEnd(0)
    for year in range(1952,2026):
        p=daily_path(year)
        d=pd.read_parquet(p,columns=["permno","dlycaldt","dlyret"])
        d["eom"]=d.dlycaldt+pd.offsets.MonthEnd(0)
        d=d.merge(rates[["eom","mthtotret"]],on="eom",how="left",validate="many_to_one")
        duplicates=int(d.duplicated(["permno","dlycaldt"]).sum())
        missing_rf=int(d.mthtotret.isna().sum())
        if duplicates or missing_rf or d.dlyret.lt(-1).any():
            raise ValueError("Daily input integrity/rate/domain failure")
        coverage.append({"year":year,"rows":len(d),"first_date":d.dlycaldt.min(),"last_date":d.dlycaldt.max(),"duplicate_keys":duplicates,"missing_reference_rate":missing_rf,"missing_daily_returns":int(d.dlyret.isna().sum())})
        pieces.append(d[d.permno.isin(ids)].copy())
    d=pd.concat(pieces,ignore_index=True)
    d["daily_excess"]=d.dlyret-d.mthtotret/21
    rows=[]
    for row in cases.itertuples():
        sample=d.loc[d.permno.eq(row.permno)&d.dlycaldt.gt(row.eom-pd.offsets.MonthEnd(12))&d.dlycaldt.le(row.eom),"daily_excess"].dropna().to_numpy()
        direct=float(np.std(sample,ddof=1))
        rows.append({"permno":row.permno,"eom":row.eom,"n":len(sample),"direct_numpy_sd":direct,"aggregate_sd":row.corrected_rvol,"difference":direct-row.corrected_rvol})
    check=pd.DataFrame(rows)
    if not check.n.ge(120).all() or check.difference.abs().max()>1e-12:
        raise ValueError("Direct-window volatility verification failed")
    check.to_csv(OUT/"risk_scale_direct_verification.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUT/"daily_coverage_full.csv",index=False)
    old=pd.read_parquet(PRIOR_RAW/"crsp_ciz_daily_1982.parquet",columns=["permno","dlycaldt","dlyret"]).sort_values(["permno","dlycaldt"]).reset_index(drop=True)
    new=pd.read_parquet(daily_path(1982),columns=["permno","dlycaldt","dlyret"])
    new=new[new.permno.isin(old.permno.unique())].sort_values(["permno","dlycaldt"]).reset_index(drop=True)
    if not old.equals(new):
        raise ValueError("Overlapping 1982 daily vintages changed")
    cov=pd.DataFrame(coverage)
    report={"daily_rows_nonoverlapping_partitions":int(cov.rows.sum()),"daily_missing_returns_preserved":int(cov.missing_daily_returns.sum()),"daily_duplicate_keys":int(cov.duplicate_keys.sum()),"missing_daily_reference_rates":int(cov.missing_reference_rate.sum()),"repaired_scales_independently_checked":len(check),"max_direct_sd_difference":float(check.difference.abs().max()),"active_rows_without_positive_scale":int(scales.corrected_missing.sum()),"overlapping_1982_daily_values_identical":True,"observed_scales_changed_by_zero_month_filter_at_1e_12":int((scales.corrected_rvol-scales.old_zero_gate_rvol).abs().gt(1e-12).sum())}
    save_json(OUT/"daily_scale_validation.json",report)
    return report


def factor_parity():
    f=pd.read_parquet(OUT/"full_factor_regressions.parquet")
    r=pd.read_csv(OUT/"full_factor_regression_R_comparison.csv",parse_dates=["date"])
    long=f.melt(id_vars=["date","n_stocks","rank","n_factors","condition"],var_name="term",value_name="python_estimate")
    j=r.merge(long,on=["date","term"],validate="one_to_one")
    j["residual"]=j.estimate-j.python_estimate
    if len(j)!=len(r) or j.residual.abs().gt(1e-10).any() or f['rank'].lt(f.n_factors).any():
        raise ValueError("Factor regression coverage or R parity failure")
    j.to_csv(OUT/"full_factor_regression_parity.csv",index=False)
    result={"days_compared":int(j.date.nunique()),"coefficients_compared":len(j),"max_abs_residual":float(j.residual.abs().max()),"full_daily_regressions":len(f),"rank_deficient_regressions":int(f['rank'].lt(f.n_factors).sum()),"scope":"R lm and Python least squares on identical dated economic inputs; upstream feature-vintage and eligibility acceptance remain separate"}
    save_json(OUT/"full_factor_regression_parity.json",result)
    return result


def main():
    files=inventory()
    daily=validate_daily_and_scales()
    parity=factor_parity()
    start=json.loads((OUT/"start.json").read_text())
    protected={p:{"before":digest,"after":sha(ROOT/p),"unchanged":digest==sha(ROOT/p)} for p,digest in start["protected"].items()}
    if not all(v["unchanged"] for v in protected.values()):
        raise ValueError("Protected scientific record changed")
    save_json(OUT/"protected_after.json",protected)
    m=json.loads((OUT/"monthly_reconstruction.json").read_text()); risk=json.loads((OUT/"risk_scale_repairfull.json").read_text()); market=json.loads((OUT/"market_reconstruction.json").read_text())
    cases=pd.read_csv(OUT/"full_historical_case_ledger.csv")
    gates=[
        {"gate":"Captured raw bytes, rows and query provenance","status":"PASS"},
        {"gate":"Monthly history through December 2025","status":"PASS_COVERAGE"},
        {"gate":"Named CIZ observed-return reconstruction","status":"BUILT_EVENT_ACCEPTANCE_SEPARATE"},
        {"gate":"Required operational volatility scales","status":"PASS_PROVISIONAL_UNIVERSE"},
        {"gate":"Factor regression coverage and first dates","status":"PASS_PROVISIONAL_UNIVERSE"},
        {"gate":"Historical claim valuation and settlement","status":"OPEN_VALUATION_DECISION_PENDING"},
        {"gate":"Published feature vintage and causal eligibility acceptance","status":"OPEN_RECONSTRUCTION_LIMITS"},
        {"gate":"Market constituent reconciliation","status":"OPEN_RESIDUAL"},
        {"gate":"Source excess/total identity and reciprocal arithmetic","status":"PASS_ARITHMETIC_ONLY"}]
    manifest={"recorded_at_utc":utc(),"status":"NOT_ACCEPTED","step3_complete":False,"accepted_bundle_path":None,"historical_financial_experiments_run":False,"raw_files_verified":len(files),"raw_gib":float(files.bytes.sum()/2**30),"raw_rows_across_overlapping_extracts":int(files.rows.sum()),"inventory_integrity_passed":True,"monthly_source_rows":m["jkp_rows"],"monthly_source_months":m["months"],"candidate_rows":m["candidate_rows"],"candidate_ids":m["candidate_ids"],"source_missing_scales_repaired":risk["repaired_cases"],"required_scales_still_missing":risk["corrected_missing_scales"],"full_required_outcome_exceptions":len(cases),"secondary_bridge_cases_recovered":int(cases.resolution.eq("SECONDARY_MONTHLY_BRIDGE_RECOVERED").sum()),"unresolved_required_outcome_cases":int(cases.resolution.eq("UNRESOLVED").sum()),"market_max_reconciliation_gap_bps":market["max_excess_gap_basis_points"],"daily":daily,"factor_validation":parity,"gates":gates,"valuation_proposal":"docs/STEP_3_VALUATION_DECISION.md","valuation_assumptions_adopted":False,"protected_scientific_records_unchanged":True,"resources":resource_guard()}
    save_json(OUT/"acceptance_manifest.json",manifest)
    save_json(PREP/"STATUS.json",{"status":"NOT_ACCEPTED","accepted_bundle_path":None,"manifest":"output/step3_20260911/acceptance_manifest.json"})
    scripts=sorted((ROOT/"scripts").glob("step3*.py"))+sorted((ROOT/"scripts").glob("step3*.R"))
    derived=sorted(PREP.glob("*.parquet"))+sorted(OUT.glob("*.parquet"))
    save_json(OUT/"reproducibility_manifest.json",{"recorded_at_utc":utc(),"scripts":{str(p.relative_to(ROOT)):sha(p) for p in scripts},"derived_files":{str(p.relative_to(ROOT)):sha(p) for p in derived},"data_dependencies":"Only September 10–11 fresh raw/prepared files and pinned source references; no pre-reset implementation or database dependency","financial_execution":False})
    print(json.dumps({k:manifest[k] for k in ["status","raw_files_verified","candidate_rows","source_missing_scales_repaired","unresolved_required_outcome_cases"]}),flush=True)
    return manifest


if __name__=="__main__":
    main()
