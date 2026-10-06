"""Rebuild dated operational volatility scales from CIZ daily returns.

The vendor feature panel is preserved. These separately named scales repair
required numerical inputs; they do not claim a rebuild of all JKP signals.
"""
import argparse
import json
import numpy as np
import pandas as pd
from step3_v2_io import ROOT, RAW, OUT, PREP, PRIOR_RAW, PRIOR_OUT, sha, save_json, resource_guard


def build(full=False):
    end_year = 2025 if full else 1982
    suffix = "full" if full else ""
    rates = pd.read_parquet(PRIOR_RAW / "crsp_ciz_t30_index.parquet")
    rates["eom"] = rates.mthcaldt + pd.offsets.MonthEnd(0)
    if rates.duplicated("eom").any():
        raise ValueError("Duplicate rate months")
    monthly = []
    paths = []
    for year in range(1952, end_year+1):
        path = RAW / f"crsp_ciz_daily_full_ids_{year}.parquet" if full and year >= 1982 else PRIOR_RAW / f"crsp_ciz_daily_{year}.parquet"
        paths.append(path)
        d = pd.read_parquet(path, columns=["permno", "dlycaldt", "dlyret"])
        if d.duplicated(["permno", "dlycaldt"]).any():
            raise ValueError("Duplicate security-day")
        d["eom"] = d.dlycaldt + pd.offsets.MonthEnd(0)
        d = d.merge(rates[["eom", "mthtotret"]], on="eom", how="left", validate="many_to_one")
        if d.mthtotret.isna().any():
            raise ValueError("Missing required rate")
        d["excess"] = d.dlyret - d.mthtotret / 21
        d["square"] = d.excess**2
        d["zero_day"] = d.dlyret.eq(0).astype(int)
        m = d.groupby(["permno", "eom"]).agg(n=("excess", "count"), total=("excess", "sum"), square=("square", "sum"), zero_days=("zero_day", "sum"))
        monthly.append(m)
    m = pd.concat(monthly).sort_index()
    calendar = pd.date_range("1952-12-31", f"{end_year}-12-31", freq="ME")
    ids = sorted(m.index.get_level_values("permno").unique())
    complete = pd.MultiIndex.from_product([ids, calendar], names=["permno", "eom"])
    m = m.reindex(complete, fill_value=0)
    frames = []
    for label, block in [("corrected", m), ("old_zero_gate", m.where(m.zero_days.lt(10), 0))]:
        rolling = block[["n", "total", "square"]].groupby(level="permno").rolling(12, min_periods=12).sum().droplevel(0)
        variance = (rolling.square - rolling.total**2 / rolling.n) / (rolling.n - 1)
        if variance[rolling.n.ge(120)].lt(-1e-14).any():
            raise ValueError("Invalid rolling variance")
        rolling[label + "_rvol"] = np.sqrt(variance.clip(lower=0)).where(rolling.n.ge(120))
        frames.append(rolling[["n", label + "_rvol"]].rename(columns={"n": label + "_days"}))
    result = frames[0].join(frames[1]).reset_index()
    result["volatility_monthly_operational"] = result.corrected_rvol * np.sqrt(21)
    result["available_at_source_rule"] = result.eom
    result["source_rule"] = "CIZ daily total return minus monthly CRSP 30-day bill return/21; 12 calendar months; sample SD; minimum 120 observations"
    result.to_parquet(PREP / f"operational_volatility_through_{end_year}.parquet", index=False, compression="zstd")
    candidate_path = PREP / "candidate_monthly_1952_2025.parquet" if full else ROOT / "data/prepared/step3_v1/candidate_monthly_through_1982.parquet"
    candidate = pd.read_parquet(candidate_path, columns=["permno", "eom", "valid", "rvol_252d"])
    active = candidate[candidate.valid].merge(result, on=["permno", "eom"], how="left", validate="one_to_one")
    active["source_missing"] = active.rvol_252d.isna() | active.rvol_252d.le(0)
    active["corrected_missing"] = active.corrected_rvol.isna() | active.corrected_rvol.le(0)
    active["old_filter_source_residual"] = active.old_zero_gate_rvol - active.rvol_252d
    active["corrected_source_difference"] = active.corrected_rvol - active.rvol_252d
    active.to_parquet(OUT / f"risk_scale_reconciliation{suffix}.parquet", index=False)
    repaired = active[active.source_missing]
    repaired.to_csv(OUT / f"repaired_risk_scale_cases{suffix}.csv", index=False)
    observed = active[active.rvol_252d.notna()]
    report = {"status": "OPERATIONAL_SCALES_REBUILT_SOURCE_FEATURES_PRESERVED", "through_year": end_year, "candidate_rows_checked": len(active),
              "source_missing_scales": int(active.source_missing.sum()), "corrected_missing_scales": int(active.corrected_missing.sum()),
              "repaired_cases": len(repaired[~repaired.corrected_missing]),
              "pilot_decision_repairs": int(repaired.eom.between("1980-12-31", "1982-11-30").sum()),
              "old_gate_reconstruction_observed_pairs": int(observed.old_filter_source_residual.notna().sum()),
              "old_gate_reconstruction_max_abs_residual": float(observed.old_filter_source_residual.abs().max()),
              "old_gate_reconstruction_outside_1e_8": int(observed.old_filter_source_residual.abs().gt(1e-8).sum()),
              "corrected_observed_scale_changes_above_1e_10": int(observed.corrected_source_difference.abs().gt(1e-10).sum()),
              "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
              "resources": resource_guard(),
              "limits": ["This builds a separately named operational normalization scale for the provisional candidate universe.",
                         "Published predictor values and their missing-rank convention remain separately preserved; no full upstream feature-family correction is claimed.",
                         "A changed candidate universe requires an expanded daily-ID extract and a new coverage check.",
                         "Historical event accounting remains an independent acceptance gate."]}
    save_json(OUT / f"risk_scale_repair{suffix}.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["input_sha256", "limits"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    build(full=parser.parse_args().full)
