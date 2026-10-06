"""Reproduce the historical sample audit, including source failures.

An audit check confirming a discrepancy does NOT accept the affected input.
"""
import json
import numpy as np
import pandas as pd
from step3_io import RAW, OUT, ROOT, save_json, write_prepared
from step3_acquire import FEATURES


def audit():
    j = pd.read_parquet(RAW / "sample_jkp_nyse_1978_1982.parquet")
    old = pd.read_parquet(RAW / "sample_crsp_monthly_1978_1982.parquet")
    old["eom"] = old.date + pd.offsets.MonthEnd(0)
    current = pd.read_parquet(RAW / "sample_crsp_ciz_monthly_1978_1982.parquet")
    current["eom"] = current.mthcaldt + pd.offsets.MonthEnd(0)
    delists = pd.read_parquet(RAW / "crsp_ciz_delistings.parquet")
    delists["eom"] = delists.delistingdt + pd.offsets.MonthEnd(0)
    rates = pd.read_parquet(RAW / "crsp_ciz_t30_index.parquet")
    rates["eom"] = rates.mthcaldt + pd.offsets.MonthEnd(0)
    french = pd.read_parquet(RAW / "ff_monthly.parquet")
    french["eom"] = french.date + pd.offsets.MonthEnd(0)
    checks = []

    def check(name, condition, evidence):
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    for name, frame, key in [("JKP", j, ["id", "eom"]), ("CRSP CIZ", current, ["permno", "eom"]), ("CRSP SIZ", old, ["permno", "eom"]), ("CIZ delistings", delists, ["permno", "eom"]), ("reference rate", rates, ["eom"])]:
        n = int(frame.duplicated(key).sum())
        check(name + " unique keys", n == 0, {"duplicates": n, "rows": len(frame)})
    check("Released dictionary present", len(FEATURES) == 115 and set(FEATURES) <= set(j), {"features": len(FEATURES)})
    check("Security ID equals PERMNO", j.id.eq(j.permno).all(), {"differences": int(j.id.ne(j.permno).sum())})
    check("Dates are month ends", j.eom.dt.is_month_end.all(), {"bad_dates": int((~j.eom.dt.is_month_end).sum())})
    check("Sample covers all 60 months", j.eom.nunique() == 60, {"first": j.eom.min(), "last": j.eom.max()})

    z = j.merge(current, on=["permno", "eom"], how="left", validate="one_to_one")
    z = z.merge(delists, on=["permno", "eom"], how="left", validate="many_to_one")
    z = z.merge(rates[["eom", "mthtotret"]], on="eom", validate="many_to_one")
    z = z.merge(french[["eom", "rf"]], on="eom", validate="many_to_one")
    check("CIZ row join preserves every JKP row", len(z) == len(j) and z.mthcaldt_y.notna().all() if "mthcaldt_y" in z else len(z) == len(j) and z.mthcaldt.notna().all(), {"rows_before": len(j), "rows_after": len(z)})
    z["source_total_return_difference"] = z.ret - z.mthret
    z["source_subtraction_rate_difference"] = z.ret - z.ret_exc - z.mthtotret
    rate_err = float(z.source_subtraction_rate_difference.abs().max())
    check("JKP subtraction rate is CRSP 30-day bill return", rate_err < 1e-10, {"max_absolute_difference": rate_err, "observed_pairs": int(z.source_subtraction_rate_difference.notna().sum())})
    me_error = z.me - z.mthprc * z.shrout / 1000
    check("JKP ME is millions of USD", me_error.abs().max() < 1e-8, {"max_absolute_difference": float(me_error.abs().max())})

    exceptional = z.source_total_return_difference.abs().gt(1e-8)
    y = z[exceptional].copy()
    bad_reason = y.delreasontype.isin(["UNAV", "MVOT", "MTMK", "SHLD", "LP", "INSC", "INSF", "CORQ", "DERE", "BKPY", "OFFRE", "DELQ", "FARG", "EQRQ", "DEEX", "FING"])
    imputed = y.delret.isna() & bad_reason & y.delactiontype.eq("GDR") & y.delstatustype.eq("VCL") & y.delpaymenttype.eq("PRCF")
    y["jkp_effective_delret"] = y.delret.mask(imputed, -0.3)
    y["jkp_return_reconstructed"] = (1 + y.mthret.mask(y.mthret.isna() & y.jkp_effective_delret.notna(), 0)) * (1 + y.jkp_effective_delret.fillna(0)) - 1
    y["source_formula_residual"] = y.ret - y.jkp_return_reconstructed
    check("All exceptional sample returns traced to JKP formula", y.source_formula_residual.abs().max() < 1e-10, {"exceptional_rows": len(y), "max_absolute_residual": float(y.source_formula_residual.abs().max())})
    check("Current JKP total returns agree with CIZ", len(y) == 0, {"exceptions": len(y), "flag_counts": y.mthdelflg.value_counts().to_dict(), "max_absolute_difference": float(y.source_total_return_difference.abs().max())})
    y["issue"] = y.mthdelflg.map({"A": "included_delisting_return_applied_again", "P": "included_delisting_return_applied_again", "V": "partial_value_applied_again_unresolved_remainder", "G": "later_value_attributed_to_delisting_month", "M": "missing_outcome_imputed"})
    y[["permno", "eom", "ret", "ret_exc", "mthret", "mthdelflg", "delret", "delistingdt", "delnextdt", "delamtdt", "delpaymenttype", "delpermno", "source_total_return_difference", "jkp_return_reconstructed", "source_formula_residual", "issue"]].to_parquet(OUT / "sample_return_exception_ledger.parquet", index=False)

    # This sample calendar is a diagnostic slice, not a baseline input selection.
    outcome = current[["permno", "eom", "mthret", "mthdelflg"]].rename(columns={"eom": "realization_eom", "mthret": "next_total_return", "mthdelflg": "next_delisting_flag"})
    outcome["decision_eom"] = outcome.realization_eom - pd.offsets.MonthEnd(1)
    links = j[["id", "permno", "eom", "ret_exc_lead1m"]].rename(columns={"eom": "decision_eom"}).merge(outcome, on=["permno", "decision_eom"], how="left", validate="one_to_one")
    links.to_parquet(OUT / "sample_decision_outcome_links.parquet", index=False)
    check("Next-month join preserves decisions without outcomes", len(links) == len(j), {"decisions": len(j), "output_rows": len(links), "missing_next_outcome_in_slice": int(links.next_total_return.isna().sum()), "slice_right_boundary": "1982-12-31 decisions need 1983-01 outside this sample"})
    check("First pilot decision maps to January 1981", links.loc[links.decision_eom.eq(pd.Timestamp("1980-12-31")), "realization_eom"].dropna().eq(pd.Timestamp("1981-01-31")).all(), "December 1980 decision, January 1981 realization")

    report = {
        "scope": "87,080 source NYSE stock-months, 1978-01 through 1982-12; not an accepted investable sample",
        "status": "NOT_ACCEPTED", "audit_executed": True, "checks": checks,
        "passed_checks": sum(x["passed"] for x in checks), "check_count": len(checks),
        "return_issue_counts": y.issue.value_counts().to_dict(),
        "missingness": {"missing_current_JKP_return": int(j.ret.isna().sum()), "missing_JKP_next_return": int(j.ret_exc_lead1m.isna().sum()), "below_57_features": int(j[FEATURES].notna().sum(axis=1).lt(57).sum()), "nonpositive_or_missing_dollar_volume": int((j.dolvol_126d.isna() | j.dolvol_126d.le(0)).sum())},
        "conclusion": "Keys, dictionary and units reconcile. Return and terminal-claim failures prevent accepting the sample unchanged. Confirming the source formula does not pass the financial input gate.",
    }
    save_json(OUT / "sample_audit.json", report)
    print(json.dumps({k: report[k] for k in ["status", "check_count", "passed_checks", "return_issue_counts"]}))
    return report


if __name__ == "__main__":
    audit()
