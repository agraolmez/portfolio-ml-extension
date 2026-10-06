"""Create a named CIZ return reconstruction and full provisional coverage ledger.

Observed JKP predictors are not overwritten. CIZ values with incomplete event
accounting are observations, not accepted final liquidation proceeds.
"""
import json
import numpy as np
import pandas as pd
from step3_acquire import FEATURES
from step3_prepare import candidate_screens
from step3_v2_io import ROOT, RAW, OUT, PREP, PRIOR_RAW, sha, save_json, resource_guard


def monthly_inputs():
    paths = [PRIOR_RAW / f"jkp_nyse_{y}.parquet" for y in range(1952, 1989)]
    paths.append(RAW / "jkp_nyse_1989_2025.parquet")
    cols = list(dict.fromkeys(["id", "permno", "eom", "common", "exch_main", "primary_sec", "obs_main", "me", "dolvol_126d", "rvol_252d", "sic", "size_grp", "ret", "ret_exc", *FEATURES]))
    j = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    if j.duplicated(["id", "eom"]).any():
        raise ValueError("Duplicate JKP security-month")
    cpaths = [PRIOR_RAW / f"crsp_ciz_monthly_{y}.parquet" for y in range(1952, 1989)]
    cpaths.append(RAW / "crsp_ciz_monthly_1989_2025.parquet")
    cols = ["permno", "mthcaldt", "mthprcdt", "mthret", "mthretx", "mthdelflg", "mthprc", "mthprcflg", "mthcompflg", "mthcompsubflg", "tradingstatusflg"]
    c = pd.concat([pd.read_parquet(p, columns=cols) for p in cpaths], ignore_index=True)
    c["eom"] = c.mthcaldt + pd.offsets.MonthEnd(0)
    if c.duplicated(["permno", "eom"]).any():
        raise ValueError("Duplicate CIZ security-month")
    return j, c, paths + cpaths


def build():
    j, c, paths = monthly_inputs()
    rates = pd.read_parquet(PRIOR_RAW / "crsp_ciz_t30_index.parquet")
    rates["eom"] = rates.mthcaldt + pd.offsets.MonthEnd(0)
    c = c.merge(rates[["eom", "mthtotret"]].rename(columns={"mthtotret": "rf_crsp_t30"}), on="eom", how="left", validate="many_to_one")
    if c.rf_crsp_t30.isna().any():
        raise ValueError("Missing subtraction rate")
    c["return_total_ciz_observed"] = c.mthret
    c["return_excess_ciz_observed"] = c.mthret - c.rf_crsp_t30
    c["accounting_status"] = np.select(
        [c.mthret.isna(), c.mthdelflg.isin(["G", "M", "V"]), c.mthdelflg.isin(["A", "P"])],
        ["missing_observation", "claim_resolution_required", "included_event_terms_to_audit"],
        default="observed_regular_return")
    c["reference_rate_id"] = 1000708
    c["source_imputation_applied"] = False
    c.to_parquet(PREP / "ciz_monthly_outcomes_1952_2025.parquet", index=False, compression="zstd")
    x = j.merge(c, on=["permno", "eom"], how="left", validate="one_to_one")
    del j
    x["source_return_difference"] = x.ret - x.return_total_ciz_observed
    x["decision_eom"] = x.eom
    x["next_realization_eom"] = x.eom + pd.offsets.MonthEnd(1)
    x["market_equity_usd"] = x.me * 1e6
    x["lambda_diagonal"] = 0.2 / x.dolvol_126d.where(x.dolvol_126d.gt(0))
    # Keep the earlier causal screen explicitly provisional while data variants
    # and historical claims are being reconciled. No forward-return filtering.
    x = candidate_screens(x)
    resource_guard()
    x.to_parquet(PREP / "candidate_monthly_1952_2025.parquet", index=False, compression="zstd")
    a = x[x.valid]
    q = a[["id", "permno", "decision_eom", "next_realization_eom"]].merge(
        c.drop(columns=["mthcaldt"]).rename(columns={"eom": "next_realization_eom"}),
        on=["permno", "next_realization_eom"], how="left", validate="one_to_one")
    q.loc[q.next_realization_eom.gt("2025-12-31"), "accounting_status"] = "right_boundary_2026_not_requested"
    q["accounting_status"] = q.accounting_status.fillna("missing_next_security_row")
    q.to_parquet(OUT / "full_candidate_outcome_links.parquet", index=False, compression="zstd")
    problematic = q[q.accounting_status.isin(["missing_observation", "missing_next_security_row", "claim_resolution_required"])]
    problematic.to_csv(OUT / "full_required_outcome_exceptions.csv", index=False)
    differences = x[x.source_return_difference.abs().gt(1e-10)][["id", "permno", "eom", "ret", "return_total_ciz_observed", "source_return_difference", "mthdelflg", "accounting_status"]]
    differences.to_parquet(OUT / "full_return_repair_ledger.parquet", index=False)
    sample = differences[differences.eom.between("1978-01-31", "1982-12-31")]
    old_ids = set(json.loads((ROOT / "output/step3_20260910/required_daily_ids.json").read_text()))
    ids = sorted(a.permno.astype(int).unique().tolist())
    save_json(OUT / "full_required_daily_ids.json", ids)
    cov = x.groupby("eom").agg(source_rows=("id", "size"), active_candidates=("valid", "sum"), raw_source_returns_missing=("ret", lambda s: int(s.isna().sum())))
    cov.to_csv(OUT / "full_monthly_coverage.csv")
    report = {
        "status": "CIZ_RETURN_RECONSTRUCTION_CREATED_NOT_ACCEPTED",
        "jkp_rows": len(x), "ciz_rows": len(c), "months": x.eom.nunique(),
        "first_month": x.eom.min(), "last_month": x.eom.max(),
        "candidate_rows": len(a), "candidate_ids": len(ids), "extra_ids_beyond_pilot_daily_extract": sorted(set(ids)-old_ids),
        "all_source_return_changes": len(differences), "sample_source_return_changes": len(sample),
        "sample_changes_by_flag": sample.mthdelflg.value_counts().to_dict(),
        "outcome_statuses": q.accounting_status.value_counts().to_dict(),
        "required_exception_rows": len(problematic), "required_exception_ids": problematic.permno.nunique(),
        "pilot_exception_rows": int(problematic.next_realization_eom.between("1981-01-31", "1982-12-31").sum()),
        "historical_before_1981_exception_rows": int(problematic.next_realization_eom.lt("1981-01-31").sum()),
        "observed_predictors_preserved": True, "source_imputation_applied": False,
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "resources": resource_guard(),
        "limits": ["Candidate eligibility remains provisional; no R0/R1 historical equivalence is claimed.", "CIZ G/M/V returns do not resolve complete event consideration.", "No future outcomes select current eligibility.", "The full correction of upstream return-derived predictors has not been performed."]}
    save_json(OUT / "monthly_reconstruction.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_sha256", "extra_ids_beyond_pilot_daily_extract", "limits"]}, default=str), flush=True)
    return report


if __name__ == "__main__":
    build()
