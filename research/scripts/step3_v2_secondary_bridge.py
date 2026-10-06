"""Recover a labelled Compustat monthly bridge using a dated CRSP link.

Raw CIZ returns remain unchanged. This bridge is a separate provider convention,
using Compustat's published monthly total return, not reconstructed daily returns.
"""
import json
import numpy as np
import pandas as pd
from step3_v2_io import RAW, PREP, OUT, PRIOR_RAW, save_json, sha, ROOT


def build():
    paths = [RAW / "required_case_compustat_monthly.parquet", RAW / "required_case_crsp_comp_links.parquet"]
    x = pd.read_parquet(paths[0])
    links = pd.read_parquet(paths[1])
    # Historical CCM link, not today's company name or CUSIP.
    lk = links[(links.lpermno.eq(24985)) & links.gvkey.eq("010860") & links.liid.eq("01")]
    if len(lk) != 1:
        raise ValueError("Ambiguous historical Union Electric link")
    d = x[x.gvkey.eq("010860") & x.iid.eq("01") & x.datadate.between("1967-06-01", "1967-09-30")].sort_values("datadate").copy()
    if len(d) != 4 or d.prccm.isna().any() or d.ajexm.nunique() != 1 or not d.curcdm.eq("USD").all():
        raise ValueError("The price bridge is incomplete or its units/adjustment changed")
    link_end = lk.linkenddt.iloc[0]
    if not (d.datadate.ge(lk.linkdt.iloc[0]) & (True if pd.isna(link_end) else d.datadate.le(link_end))).all():
        raise ValueError("Bridge falls outside historical identity interval")
    d["return_total_secondary"] = d.trt1m/100
    # The source's percent scaling is checked against its price/dividend record.
    d["price_dividend_monthly_check"] = (d.prccm+d.dvpsxm.fillna(0))/d.prccm.shift(1)-1
    d["check_residual"] = d.return_total_secondary-d.price_dividend_monthly_check
    d = d.iloc[1:].copy()
    # One quoted unit, without assuming the vendor rounds rather than truncates.
    if not d.check_residual.abs().le(1.01e-6).all():
        raise ValueError("Monthly percent/price/dividend bridge failed")
    d["permno"] = 24985
    d["eom"] = d.datadate+pd.offsets.MonthEnd(0)
    d["source"] = "WRDS comp.secm TRT1M / 100; GVKEY 010860 IID 01; dated CCM link"
    d["source_rule_available_at"] = d.eom
    d["actual_historical_publication_timestamp"] = pd.NaT
    d["status"] = "SECONDARY_REPORTED_RETURN_BRIDGE"
    d["daily_return_recovery_claimed"] = False
    d.to_parquet(PREP / "secondary_monthly_return_bridge.parquet", index=False)
    d.to_csv(OUT / "union_electric_return_bridge.csv", index=False)
    report = {"status": "DATED_SECONDARY_MONTHLY_BRIDGE_RECOVERED", "permno": 24985, "months": [str(z.date()) for z in d.eom], "total_returns_decimal": d.return_total_secondary.tolist(), "max_price_dividend_formula_residual": float(d.check_residual.abs().max()), "compustat_precision": "TRT1M reported to four decimal percentage points; prices retain vendor precision", "required_case_resolved_as_secondary_observation": "July 1967 monthly return", "limits": ["Named secondary-source convention; no exact CIZ daily-compounding equivalence claim.", "No daily prices or returns imputed.", "No invented historical publication timestamp or change to the source feature panel.", "Coastal States Gas's zero TRT1M during an unpriced halt is not accepted by this bridge."], "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}}
    save_json(OUT / "secondary_bridge.json", report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    build()
