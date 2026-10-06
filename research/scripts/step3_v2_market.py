"""Audit the complete captured market series and construct a diagnostic wealth path.

Full constituent equality has not been established. A correct arithmetic
recurrence alone cannot accept the economic input series.
"""
import json
import numpy as np
import pandas as pd
from step3_v2_io import ROOT, RAW, PREP, OUT, PRIOR_RAW, PRIOR_OUT, save_json, sha, resource_guard


def build():
    market_path = PRIOR_OUT / "public_sources/[usa]_[mkt]_[monthly]_[vw].csv"
    cutoff_path = OUT / "public_sources/return_cutoffs.csv"
    x = pd.read_parquet(RAW / "jkp_usa_market_1951_2025.parquet")
    cut = pd.read_csv(cutoff_path, parse_dates=["eom"])
    pub = pd.read_csv(market_path, parse_dates=["date"]).rename(columns={"date": "eom", "ret": "published_excess"})
    rates = pd.read_parquet(PRIOR_RAW / "crsp_ciz_t30_index.parquet")
    rates["eom"] = rates.mthcaldt+pd.offsets.MonthEnd(0)
    rates = rates[["eom", "mthtotret"]].rename(columns={"mthtotret": "rf_crsp_t30"})
    if any(z.eom.duplicated().any() for z in [cut, pub, rates]) or x.duplicated(["id", "eom"]).any():
        raise ValueError("Duplicate market input keys")
    x = x.merge(cut, on="eom", validate="many_to_one")
    keep = x[["common", "obs_main", "exch_main", "primary_sec"]].eq(1).all(axis=1) & x.ret_lag_dif.le(1) & x.me_lag1.notna() & x.ret_local.notna()
    x = x[keep].copy()
    for column, tag in [("ret_exc", "excess"), ("ret", "total")]:
        clipped = x[column].where(x.source_crsp.eq(1), x[column].clip(lower=x[column+"_0_1"], upper=x[column+"_99_9"]))
        x["weighted_"+tag] = clipped*x.me_lag1
    m = x.groupby("eom").agg(n_stocks_rebuilt=("id", "size"), weight=("me_lag1", "sum"), weighted_excess=("weighted_excess", "sum"), weighted_total=("weighted_total", "sum"))
    m["rebuilt_excess"] = m.weighted_excess/m.weight
    m["rebuilt_total"] = m.weighted_total/m.weight
    m = m.reset_index().merge(pub, on="eom", validate="one_to_one").merge(rates, on="eom", how="left", validate="one_to_one")
    m["excess_reconciliation_gap"] = m.rebuilt_excess-m.published_excess
    m["total_excess_identity_residual"] = m.rebuilt_total-m.rebuilt_excess-m.rf_crsp_t30
    m["published_total_candidate"] = m.published_excess+m.rf_crsp_t30
    m.to_parquet(OUT / "market_reconciliation_full.parquet", index=False)
    rate_check = cut.merge(rates, on="eom", validate="one_to_one")
    cutoff_rate_residual = pd.concat([rate_check.ret_0_1-rate_check.ret_exc_0_1-rate_check.rf_crsp_t30, rate_check.ret_99_9-rate_check.ret_exc_99_9-rate_check.rf_crsp_t30])
    w = m[m.eom.between("1952-12-31", "2025-12-31")][["eom", "published_excess", "rf_crsp_t30", "published_total_candidate"]].copy()
    expected = pd.date_range("1952-12-31", "2025-12-31", freq="ME")
    if not w.eom.reset_index(drop=True).equals(pd.Series(expected)) or w.published_total_candidate.isna().any() or w.published_total_candidate.le(-1).any():
        raise ValueError("Incomplete/invalid wealth candidate calendar")
    protocol = json.loads((ROOT / "config/protocol_v1.json").read_text())["economic"]
    anchor = pd.Timestamp(protocol["wealth_anchor_date"])
    anchor_usd = protocol["wealth_2020_usd"]
    levels = (1+w.published_total_candidate).cumprod()
    w["wealth_usd_candidate"] = anchor_usd*levels/levels[w.eom.eq(anchor)].iloc[0]
    w["recurrence_residual_relative"] = (w.wealth_usd_candidate.shift(1)-w.wealth_usd_candidate/(1+w.published_total_candidate))/w.wealth_usd_candidate.shift(1)
    w["status"] = "ARITHMETIC_VERIFIED_ECONOMIC_INPUT_NOT_ACCEPTED"
    w.to_parquet(PREP / "wealth_path_candidate_1952_2025.parquet", index=False)
    sample = m[m.eom.between("1978-01-31", "1982-12-31")]
    report = {"status": "COMPLETE_SOURCE_BENCHMARK_AUDIT_CONSTITUENT_RECONCILIATION_OPEN", "months_compared": len(m), "stock_count_matches": int(m.n_stocks_rebuilt.eq(m.n_stocks).sum()), "excess_matches_at_1e_10": int(m.excess_reconciliation_gap.abs().le(1e-10).sum()), "max_excess_gap": float(m.excess_reconciliation_gap.abs().max()), "max_excess_gap_basis_points": float(m.excess_reconciliation_gap.abs().max()*10000), "sample_months": len(sample), "sample_stock_count_matches": int(sample.n_stocks_rebuilt.eq(sample.n_stocks).sum()), "sample_max_excess_gap": float(sample.excess_reconciliation_gap.abs().max()), "max_total_excess_identity_residual": float(m.total_excess_identity_residual.abs().max()), "published_cutoff_reference_rate_months": len(rate_check), "max_cutoff_reference_rate_residual": float(cutoff_rate_residual.abs().max()), "wealth_candidate_months": len(w), "wealth_anchor_usd": float(w.loc[w.eom.eq(anchor), "wealth_usd_candidate"].iloc[0]), "max_wealth_recurrence_relative_residual": float(w.recurrence_residual_relative.abs().max()), "accepted_wealth_path": False, "resources": resource_guard(), "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [market_path, cutoff_path, RAW/"jkp_usa_market_1951_2025.parquet", PRIOR_RAW/"crsp_ciz_t30_index.parquet"]}, "limits": ["The public excess-to-total construction uses CRSP's 30-day bill index, not French RF.", "The source cutoffs and stock total/excess identities support the subtraction-rate mapping; they do not prove equality of the captured constituent export and the public benchmark.", "Residuals are retained without attributing them to vintage differences without evidence.", "The reciprocal arithmetic and anchor pass; economic input acceptance remains separate."]}
    save_json(OUT / "market_reconstruction.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_sha256", "limits"]}), flush=True)
    return report


if __name__ == "__main__":
    build()
