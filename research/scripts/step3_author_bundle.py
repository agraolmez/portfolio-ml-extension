"""Full-history, explicitly scoped reconstruction using the released conventions.

This is the paper benchmark data route requested on 2026-09-11. It preserves
vendor features/returns, including documented vendor issues; it is not a
corrected reconstruction of every upstream feature or a historical cash ledger.
Run export, R parity, daily coverage, factor calendar, then acceptance separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from step3_acquire import FEATURES
from step3_io import ROOT, sha, save_json, resource_guard as basic_guard
from step3_v2_factor_calendar import industry

OUT = ROOT / "output/step3_author_conventions_20260911"
PREP = ROOT / "data/prepared/step3_author_v3"
RAW1 = ROOT / "data/raw/20260910"
RAW2 = ROOT / "data/raw/20260911"
for p in [OUT, PREP, OUT / "r_inputs", PREP / "daily"]:
    p.mkdir(parents=True, exist_ok=True)


def resource_guard():
    report = basic_guard()
    roots = [ROOT/"data", ROOT/"env/step3", *list((ROOT/"output").glob("step3_*"))]
    size = sum(p.stat().st_size for parent in roots if parent.is_dir() for p in parent.rglob("*") if p.is_file())/2**30
    if size>60:
        raise RuntimeError("Combined fresh Step 3 storage exceeds 60 GiB")
    return {**report, "combined_fresh_step3_storage_gib":size}


def source_monthly():
    paths = [RAW1 / f"jkp_nyse_{y}.parquet" for y in range(1952, 1989)] + [RAW2 / "jkp_nyse_1989_2025.parquet"]
    cols = list(dict.fromkeys(["id", "permno", "eom", "me", "sic", "ff49", "size_grp", "crsp_exchcd", "dolvol_126d", "rvol_252d", *FEATURES]))
    x = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    x = x[x.id.le(99999)].sort_values(["id", "eom"]).reset_index(drop=True)
    m = pd.read_parquet(RAW2 / "jkp_usa_market_1951_2025.parquet", columns=["id", "eom", "ret_exc"])
    m = m[m.id.isin(x.id.unique())].sort_values(["id", "eom"])
    assert not m.duplicated(["id", "eom"]).any() and not x.duplicated(["id", "eom"]).any()
    global_dates = m.loc[m.ret_exc.notna(), "eom"].drop_duplicates().sort_values()
    assert list(global_dates) == list(pd.date_range(global_dates.min(), global_dates.max(), freq="ME"))
    return x, m, paths + [RAW2 / "jkp_usa_market_1951_2025.parquet"]


def long_returns(m):
    pieces = []
    for sid, group in m.groupby("id", sort=True):
        observed = group[group.ret_exc.notna()]
        if observed.empty:
            continue
        dates = pd.date_range(observed.eom.min(), observed.eom.max(), freq="ME")
        v = group.set_index("eom").ret_exc.reindex(dates)
        p = pd.DataFrame({f"ret_ld{k}": v.shift(-k) for k in range(1, 13)})
        missing = p.isna()
        keep = ~missing.all(axis=1)
        p = p[keep].fillna(0)
        for k in range(1, 13):
            p[f"imputed_ld{k}"] = missing.loc[keep, f"ret_ld{k}"]
        p["id"] = sid
        p["eom"] = p.index
        p["last_observed_source_eom"] = observed.eom.max()
        pieces.append(p.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True).sort_values(["id", "eom"])


def source_screens(x):
    """Literal source percentile and row-based entry/exit rules, including gaps."""
    x = x.sort_values(["id", "eom"]).reset_index(drop=True)
    lag = x.groupby("id").eom.shift(12)
    diff = (x.eom.dt.year-lag.dt.year)*12 + x.eom.dt.month-lag.dt.month
    x["valid_data"] = diff.eq(12)
    perc = x.loc[x.valid_data].groupby("eom").me.rank(method="max", pct=True)
    x["me_perc"] = perc
    x["valid_size"] = x.me_perc.gt(.5) & x.me_perc.le(1)
    n_total = x.groupby("eom").valid_data.transform("sum")
    n_size = x.groupby("eom").valid_size.transform("sum")
    n_less = (x.valid_data & x.me_perc.le(.5)).groupby(x.eom).transform("sum")
    # high percentile=100, hence n_more=n_above=0 in the released base case.
    n_missing = (50-n_size).clip(lower=0)
    n_below = np.ceil(np.minimum(n_missing/2, n_less))
    n_below = n_below.where(~((n_below<n_missing) & (n_below>0)), n_missing)
    x["valid_size"] = x.me_perc.gt(.5-n_below/n_total) & x.me_perc.le(1)
    x["valid"] = False
    for _, g in x.groupby("id", sort=False):
        valid_temp = g.valid_data & g.valid_size
        rolling = valid_temp.rolling(12, min_periods=12).sum()
        add = rolling.eq(12).to_numpy()
        delete = rolling.eq(0).to_numpy()
        state = False
        included = np.zeros(len(g), dtype=bool)
        for i in range(1, len(g)):
            if not state and add[i] and not add[i-1]:
                state = True
            if state and delete[i]:
                state = False
            included[i] = state
        x.loc[g.index, "valid"] = included & g.valid_data.to_numpy()
    return x.drop(columns="me_perc")


def export():
    x, m, paths = source_monthly()
    print(f"Captured {len(x):,} NYSE source rows; preparing source returns", flush=True)
    r = long_returns(m)
    r.to_parquet(PREP / "future_excess_returns.parquet", index=False, compression="zstd")
    m.to_csv(OUT / "r_inputs/monthly.csv", index=False)
    ff = pd.read_parquet(RAW1 / "ff_monthly.parquet")
    ff["eom"] = ff.date + pd.offsets.MonthEnd(0)
    ff = ff[["eom", "rf"]]
    bills = pd.read_parquet(RAW1 / "crsp_ciz_t30_index.parquet")
    bills["eom"] = bills.mthcaldt + pd.offsets.MonthEnd(0)
    bills = bills[["eom", "mthtotret"]].rename(columns={"mthtotret": "rf_crsp_t30"})
    rates = bills.merge(ff.rename(columns={"rf": "rf_french"}), on="eom", validate="one_to_one")
    rates.to_parquet(PREP / "monthly_rates.parquet", index=False)
    rates.to_csv(OUT / "r_inputs/rates.csv", index=False)
    ld1 = r[["id", "eom", "ret_ld1", "imputed_ld1"]].merge(ff, on="eom", how="left", validate="many_to_one")
    ld1["tr_ld1_r0"] = ld1.ret_ld1 + ld1.rf
    lag = ld1[["id", "eom", "tr_ld1_r0"]].copy()
    lag["eom"] += pd.offsets.MonthEnd(1)
    ld1 = ld1.merge(lag.rename(columns={"tr_ld1_r0": "tr_ld0_r0"}), on=["id", "eom"], how="left", validate="one_to_one")
    x = x.merge(ld1.drop(columns="rf"), on=["id", "eom"], how="left", validate="one_to_one")
    x["raw_feature_count"] = x[FEATURES].notna().sum(axis=1)
    x["dolvol"] = x.dolvol_126d
    x["rvol_m_source"] = x.rvol_252d*np.sqrt(21)
    x["basic_valid"] = x.me.notna() & x.tr_ld0_r0.notna() & x.tr_ld1_r0.notna() & x.dolvol.gt(0) & x.sic.notna() & x.raw_feature_count.ge(57)
    x[["id", "eom", "me", "sic", "dolvol", "tr_ld0_r0", "tr_ld1_r0", "raw_feature_count", "basic_valid"]].to_csv(OUT / "r_inputs/screen_input.csv", index=False)
    x.to_parquet(PREP / "source_characteristics.parquet", index=False, compression="zstd")
    screened = source_screens(x[x.basic_valid].copy())
    print(f"Source screens: {len(screened):,} rows; {screened.valid.sum():,} active", flush=True)
    # Source ranks are formed over the basic-screen cross-section BEFORE validity.
    parity_dates = pd.to_datetime(["1954-11-30", "1964-11-30", "1980-12-31", "1982-11-30", "2020-11-30", "2025-11-30"])
    samples=screened.loc[screened.eom.isin(parity_dates), ["id", "eom", *FEATURES]].copy()
    # Hexadecimal float literals avoid changing adjacent-double ties during
    # interchange between R and Python. Both receive the exact same raw values.
    for f in FEATURES:
        samples[f]=samples[f].map(lambda v: float(v).hex() if pd.notna(v) else "")
    samples.to_csv(OUT / "r_inputs/raw_rank_samples.csv", index=False)
    for f in FEATURES:
        raw = screened[f].copy()
        screened[f] = screened.groupby("eom")[f].rank(method="max", pct=True).mask(raw.eq(0), 0).fillna(.5)
    screened["ff12"] = industry(pd.to_numeric(screened.sic))
    screened["market_equity_usd"] = screened.me*1e6
    screened["lambda_diagonal"] = .2/screened.dolvol
    screened["realization_eom"] = screened.eom + pd.offsets.MonthEnd(1)
    screened = screened.merge(bills.rename(columns={"eom": "realization_eom"}), on="realization_eom", how="left", validate="many_to_one")
    screened["tr_ld1"] = screened.ret_ld1 + screened.rf_crsp_t30
    # Retain properly dated current-month source total returns separately from R0.
    current = m.merge(bills, on="eom", how="left", validate="many_to_one")
    current["tr_current_source_observed"] = current.ret_exc+current.rf_crsp_t30
    screened = screened.merge(current[["id", "eom", "tr_current_source_observed"]], on=["id", "eom"], how="left", validate="one_to_one")
    screened.to_parquet(PREP / "monthly_inputs.parquet", index=False, compression="zstd")
    screened.to_parquet(PREP / "monthly_inputs_r0.parquet", index=False, compression="zstd")
    ids = sorted(screened.loc[screened.valid, "id"].astype(int).unique().tolist())
    save_json(OUT / "required_daily_ids.json", ids)
    save_json(OUT / "build_summary.json", {"source_rows": len(x), "screened_rows": len(screened), "active_rows": int(screened.valid.sum()), "active_ids": len(ids), "first_active_decision": screened.loc[screened.valid, "eom"].min(), "last_active_decision": screened.loc[screened.valid, "eom"].max(), "return_matrix_rows": len(r), "imputed_return_cells": int(r[[f"imputed_ld{k}" for k in range(1,13)]].sum().sum()), "active_missing_source_risk_scale": int((screened.valid & screened.rvol_m_source.isna()).sum()), "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "resources": resource_guard()})
    print((OUT / "build_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    export()
