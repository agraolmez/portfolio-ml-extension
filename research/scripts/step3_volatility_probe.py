"""Diagnose every missing candidate scale without substituting reconstructed values."""
import json
import pandas as pd
from step3_io import RAW, OUT, save_json


def main():
    cases = pd.read_csv(OUT / "required_risk_scale_exceptions.csv", parse_dates=["eom"])
    ids = cases.permno.astype(int).unique().tolist()
    first = cases.eom.min().year - 1
    last = cases.eom.max().year
    chunks = []
    for year in range(first, last + 1):
        x = pd.read_parquet(RAW / f"crsp_ciz_daily_{year}.parquet",
                            columns=["permno", "dlycaldt", "dlyret", "dlyretmissflg", "dlyretdurflg"],
                            filters=[("permno", "in", ids)])
        chunks.append(x)
    daily = pd.concat(chunks, ignore_index=True)
    if daily.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Duplicate daily keys")
    daily["eom"] = daily.dlycaldt + pd.offsets.MonthEnd(0)
    daily["zero_day"] = daily.dlyret.eq(0).astype(int)
    daily["zero_obs"] = daily.groupby(["permno", "eom"]).zero_day.transform("sum")
    rates = pd.read_parquet(RAW / "crsp_ciz_t30_index.parquet")
    rates["eom"] = rates.mthcaldt + pd.offsets.MonthEnd(0)
    daily = daily.merge(rates[["eom", "mthtotret"]], on="eom", how="left", validate="many_to_one")
    daily["excess_daily_diagnostic"] = daily.dlyret - daily.mthtotret / 21
    rows = []
    for row in cases.itertuples():
        start = row.eom - pd.offsets.MonthEnd(12)
        window = daily[daily.permno.eq(row.permno) & daily.dlycaldt.gt(start) & daily.dlycaldt.le(row.eom)]
        observed = window.excess_daily_diagnostic.dropna()
        filtered = window.loc[window.zero_obs.lt(10), "excess_daily_diagnostic"].dropna()
        rows.append({"permno": int(row.permno), "decision_eom": row.eom,
                     "observed_daily_returns": len(observed), "after_old_zero_month_filter": len(filtered),
                     "raw_120_minimum_met": len(observed) >= 120,
                     "old_filter_120_minimum_met": len(filtered) >= 120,
                     "diagnostic_rvol_without_old_filter": observed.std(ddof=1) if len(observed) >= 120 else None,
                     "affects_pilot_decision": row.affects_pilot_decision})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "volatility_all_required_probe.csv", index=False)
    report = {"status": "DIAGNOSTIC_NO_FEATURE_SUBSTITUTION", "required_missing_scales_probed": len(result),
              "adequate_raw_daily_history": int(result.raw_120_minimum_met.sum()),
              "inadequate_after_old_zero_month_filter": int((~result.old_filter_120_minimum_met).sum()),
              "minimum_observed_daily_returns": int(result.observed_daily_returns.min()),
              "pilot_decision_cases": int(result.affects_pilot_decision.sum()),
              "scope": "12 calendar months, at least 120 observed daily excess returns, sample standard deviation. Does not reconstruct every JKP daily transformation, prove the upstream export commit, or replace any feature. Correcting all affected features may change the universe."}
    save_json(OUT / "volatility_all_required_probe.json", report)
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
