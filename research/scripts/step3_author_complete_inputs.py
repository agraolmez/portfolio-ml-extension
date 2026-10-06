"""Finish the author-convention data package with explicit reconstruction fields."""
import json
import numpy as np
import pandas as pd
from step3_author_bundle import ROOT, OUT, PREP, RAW1, RAW2, FEATURES, long_returns, sha, save_json, resource_guard


def finish_monthly():
    m = pd.read_csv(OUT / "r_inputs/monthly.csv", parse_dates=["eom"])
    cpaths=[RAW1/f"crsp_ciz_monthly_{year}.parquet" for year in range(1952,1989)]+[RAW2/"crsp_ciz_monthly_1989_2025.parquet"]
    c=pd.concat([pd.read_parquet(p,columns=["permno","mthcaldt","mthdelflg","mthret"]) for p in cpaths],ignore_index=True)
    c["eom"]=c.mthcaldt+pd.offsets.MonthEnd(0)
    rates=pd.read_parquet(PREP/"monthly_rates.parquet")
    c=c.merge(rates[["eom","rf_crsp_t30"]],on="eom",how="left",validate="many_to_one")
    assert c.rf_crsp_t30.notna().all()
    c["return_total_ciz_observed"]=c.mthret
    c["return_excess_ciz_observed"]=c.mthret-c.rf_crsp_t30
    c=c[["permno","eom","mthdelflg","return_excess_ciz_observed","return_total_ciz_observed"]]
    m = m.merge(c.rename(columns={"permno": "id"}), on=["id", "eom"], how="left", validate="one_to_one")
    m["ret_exc_r0"] = m.ret_exc
    # A/P/V already include the available delisting value in the CIZ return.
    # G/M retain the JKP delayed-value / missing-delisting convention as requested.
    included = m.mthdelflg.isin(["A", "P", "V"]) & m.return_excess_ciz_observed.notna() & m.ret_exc.notna()
    m.loc[included, "ret_exc"] = m.loc[included, "return_excess_ciz_observed"]
    m["r1_minus_r0"] = m.ret_exc-m.ret_exc_r0
    conditions = [m.mthdelflg.eq("G"), m.mthdelflg.eq("M"), m.mthdelflg.eq("V"), included]
    m["convention"] = np.select([v.fillna(False).to_numpy(dtype=bool) for v in conditions], ["JKP_later_delisting_value_in_event_month", "JKP_missing_delisting_rule", "CIZ_partial_value_once_no_separate_residual_claim", "CIZ_included_value_once"], default="JKP_supplied_return")
    m.to_parquet(PREP / "monthly_return_conventions.parquet", index=False, compression="zstd")
    changed = m[m.r1_minus_r0.abs().gt(1e-10)]
    changed.to_parquet(OUT / "r0_r1_return_differences.parquet", index=False)
    r = long_returns(m[["id", "eom", "ret_exc"]])
    r["label_maturity_eom"] = r.eom+pd.offsets.MonthEnd(12)
    r["right_censored_12m"] = r.label_maturity_eom.gt("2025-12-31")
    r.to_parquet(PREP / "future_excess_returns_r1.parquet", index=False, compression="zstd")
    import shutil
    if not (PREP / "monthly_inputs_r0.parquet").exists():
        shutil.copy2(PREP / "monthly_inputs.parquet", PREP / "monthly_inputs_r0.parquet")
    x = pd.read_parquet(PREP / "monthly_inputs_r0.parquet")
    x = x.rename(columns={"ret_ld1": "ret_ld1_r0", "tr_ld1": "tr_ld1_source_rate_corrected"})
    x = x.merge(r[["id", "eom", "ret_ld1"]], on=["id", "eom"], how="left", validate="one_to_one")
    x["tr_ld1"] = x.ret_ld1+x.rf_crsp_t30
    prev = x[["id", "eom", "tr_ld1"]].copy()
    # Build prior return from the complete return matrix, before characteristic
    # screens, as the source does. Otherwise gaps could erase an observed return.
    rates = pd.read_parquet(PREP / "monthly_rates.parquet")
    prev = r[["id", "eom", "ret_ld1"]].copy()
    prev["eom"] += pd.offsets.MonthEnd(1)
    prev = prev.merge(rates[["eom", "rf_crsp_t30"]], on="eom", how="left", validate="many_to_one")
    prev["tr_ld0"] = prev.ret_ld1+prev.rf_crsp_t30
    x = x.merge(prev[["id", "eom", "tr_ld0"]], on=["id", "eom"], how="left", validate="one_to_one")
    public_market=ROOT/"output/step3_20260910/public_sources/[usa]_[mkt]_[monthly]_[vw].csv"
    wealth=pd.read_csv(public_market,parse_dates=["date"]).rename(columns={"date":"eom","ret":"published_excess"})
    wealth=wealth.loc[wealth.eom.between("1952-12-31","2025-12-31"),["eom","published_excess"]]
    wealth=wealth.merge(rates[["eom","rf_crsp_t30"]],on="eom",how="left",validate="one_to_one").sort_values("eom")
    wealth["market_total_return"]=wealth.published_excess+wealth.rf_crsp_t30
    assert wealth.market_total_return.notna().all() and wealth.market_total_return.gt(-1).all()
    protocol=json.loads((ROOT/"config/protocol_v1.json").read_text())["economic"]
    levels=(1+wealth.market_total_return).cumprod()
    wealth["wealth_usd"]=protocol["wealth_2020_usd"]*levels/levels[wealth.eom.eq(protocol["wealth_anchor_date"])].iloc[0]
    wealth["recurrence_residual_relative"]=(wealth.wealth_usd.shift(1)-wealth.wealth_usd/(1+wealth.market_total_return))/wealth.wealth_usd.shift(1)
    wealth["source"] = "Official JKP published USA value-weighted excess market plus verified CRSP 30-day bill reference"
    wealth["mu_ld0"] = wealth.market_total_return
    wealth["mu_ld1"] = wealth.market_total_return.shift(-1)
    wealth.to_parquet(PREP / "market_wealth.parquet", index=False)
    x = x.merge(wealth[["eom", "wealth_usd", "mu_ld0", "mu_ld1"]], on="eom", how="left", validate="many_to_one")
    x["feature_availability_basis"] = "vendor_month_end_rule_not_verified_release_timestamp"
    x["universe_convention"] = "author_future_return_availability_screen"
    x["normalization_scale_source"] = "sqrt_covariance_diagonal_computed_in_step4"
    x["label_maturity_eom"] = x.eom+pd.offsets.MonthEnd(12)
    x["right_censored_12m"] = x.label_maturity_eom.gt("2025-12-31")
    x.to_parquet(PREP / "monthly_inputs.parquet", index=False, compression="zstd")
    # Exhaustive keyed entry/exit ledger, with the authors' treatment identified.
    active = x[x.valid][["id", "eom", "realization_eom", "ret_ld1", "ret_ld1_r0", "imputed_ld1", "tr_ld1"]].copy()
    next_keys = active[["id", "eom"]].copy(); next_keys["eom"] -= pd.offsets.MonthEnd(1); next_keys["active_next"] = True
    active = active.merge(next_keys, on=["id", "eom"], how="left", validate="one_to_one")
    active = active.merge(m[["id", "eom", "mthdelflg", "convention"]].rename(columns={"eom": "realization_eom"}), on=["id", "realization_eom"], how="left", validate="one_to_one")
    active["exit_after_return"] = active.active_next.isna()
    active["right_boundary_exit"] = active.eom.eq("2025-11-30") & active.exit_after_return
    active["transition_rule"] = np.where(active.exit_after_return, "source_omits_id_from_next_weight_vector_no_separate_exit_trade", "source_propagates_weight_by_total_return_over_wealth_growth")
    active["missing_return_rule"] = np.where(active.imputed_ld1.eq(True), "zero_excess_imputation_not_observed_zero_total", "supplied_return_with_named_R1_included_value_correction")
    active.to_parquet(PREP / "outcome_and_exit_ledger.parquet", index=False, compression="zstd")
    active[active.exit_after_return & ~active.right_boundary_exit].to_csv(OUT / "source_exits.csv", index=False)
    cases = pd.read_csv(ROOT / "output/step3_20260911/full_historical_case_ledger.csv")
    # Retain original case evidence intact; the applied keyed outcomes are a
    # separate report so an assumption cannot masquerade as recovered evidence.
    ids = cases.permno.unique() if "permno" in cases else []  # Public export requires the supplied case ledger.
    active[active.id.isin(ids) & (active.imputed_ld1.eq(True) | active.mthdelflg.isin(["G","M","V"]))].to_csv(OUT / "historical_cases_applied_conventions.csv", index=False)
    save_json(OUT / "monthly_completion.json", {"r1_corrected_included_value_rows": len(changed), "r1_corrections_by_flag": changed.mthdelflg.value_counts().to_dict(), "active_zero_excess_imputations": int(active.imputed_ld1.sum()), "source_exits_before_right_boundary": int((active.exit_after_return & ~active.right_boundary_exit).sum()), "source_raw_rvol_not_core_normalization": True, "wealth_months": len(wealth), "wealth_anchor_usd": float(wealth.loc[wealth.eom.eq("2020-12-31"), "wealth_usd"].iloc[0]), "resources": resource_guard()})


def factor_inputs():
    x = pd.read_parquet(PREP / "monthly_inputs.parquet", columns=["id", "eom", "valid", "ff12", "size_grp", *FEATURES])
    x = x[x.valid].copy()
    spec = pd.read_csv(ROOT / "config/feature_dictionary_v1.csv")
    clusters = spec.cluster.unique().tolist()
    for cl in clusters:
        subset = spec[spec.cluster.eq(cl)]
        block = x[subset.characteristic].copy()
        minus = subset.loc[subset.direction.eq(-1), "characteristic"]
        block[minus] = 1-block[minus]
        x[cl] = block.mean(axis=1)
        x[cl] = x.groupby("eom")[cl].transform(lambda s: (s-s.mean())/s.std(ddof=1))
    inds = sorted(x.ff12.unique())
    for ind in inds:
        x[ind] = x.ff12.eq(ind).astype(float)
    factors = inds+clusters
    x["return_eom"] = x.eom+pd.offsets.MonthEnd(1)
    loadings = x[["id", "eom", "return_eom", "size_grp", *factors]].copy()
    loadings.to_parquet(PREP / "factor_loadings.parquet", index=False)
    del x
    rates = pd.read_parquet(PREP / "monthly_rates.parquet")
    ids = json.loads((OUT / "required_daily_ids.json").read_text())
    observations, parity, coverage, paths = [], [], [], []
    old_requested = set(json.loads((ROOT / "output/step3_20260910/required_daily_ids.json").read_text()))
    full_requested = set(json.loads((ROOT / "output/step3_20260911/full_required_daily_ids.json").read_text()))
    for year in range(1952, 2026):
        path = RAW2 / f"crsp_ciz_daily_full_ids_{year}.parquet" if year>=1982 else RAW1 / f"crsp_ciz_daily_{year}.parquet"
        paths.append(path)
        required = set(loadings.loc[loadings.return_eom.dt.year.eq(year), "id"].astype(int))
        if not required.issubset(full_requested if year>=1982 else old_requested):
            raise ValueError(f"Daily extraction omitted required IDs in {year}: {required-old_requested}")
        d = pd.read_parquet(path, columns=["permno", "dlycaldt", "dlyret", "dlydelflg", "dlyretmissflg"])
        d = d[d.permno.isin(ids)].rename(columns={"permno": "id", "dlycaldt": "date", "dlyret": "total_return"})
        assert not d.duplicated(["id", "date"]).any()
        d["return_eom"] = d.date+pd.offsets.MonthEnd(0)
        d = d.merge(rates[["eom", "rf_crsp_t30"]].rename(columns={"eom": "return_eom"}), on="return_eom", how="left", validate="many_to_one")
        assert d.rf_crsp_t30.notna().all()
        d["ret_exc"] = d.total_return-d.rf_crsp_t30/21
        d.to_parquet(PREP / f"daily/{year}.parquet", index=False, compression="zstd")
        coverage.append({"year": year, "rows": len(d), "missing_returns_retained": int(d.ret_exc.isna().sum()), "required_ids": len(required), "source": str(path.relative_to(ROOT)), "sha256": sha(path)})
        joined = d.merge(loadings, on=["id", "return_eom"], validate="many_to_one").dropna(subset=["ret_exc", "size_grp", *factors])
        residuals = []
        for day, g in joined.groupby("date", sort=True):
            a = g[factors].to_numpy(); y = g.ret_exc.to_numpy()
            beta, _, rank, singular = np.linalg.lstsq(a, y, rcond=1e-7)
            observations.append({"date": day, "n_stocks": len(g), "rank": int(rank), "condition": float(singular[0]/singular[-1]), **dict(zip(factors,beta))})
            residuals.append(pd.DataFrame({"id": g.id.to_numpy(), "date": day, "residual": y-a@beta}))
            if not parity or (day.month==12 and day.day<=3 and year in [1964,1965,1966,2020,2025]):
                parity.append(g[["date", "id", "ret_exc", *factors]])
        if residuals:
            pd.concat(residuals, ignore_index=True).to_parquet(PREP / f"daily/residuals_{year}.parquet", index=False, compression="zstd")
        if year%10==0 or year==2025:
            print(f"Daily inputs and factor regressions through {year}", flush=True)
        resource_guard()
    f = pd.DataFrame(observations).sort_values("date")
    f.to_parquet(PREP / "daily_factor_returns.parquet", index=False)
    pd.concat(parity).to_csv(OUT / "factor_R_comparison_inputs.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "daily_coverage.csv", index=False)
    threshold = f.date.iloc[2519].replace(day=1)-pd.Timedelta(days=1)
    first_cov = loadings.loc[loadings.eom.ge(threshold), "eom"].min()
    save_json(OUT / "factor_calendar.json", {"first_daily_factor_date": f.date.min(), "daily_factor_rows": len(f), "factor_count": len(factors), "rank_deficient_days": int(f['rank'].lt(len(factors)).sum()), "date_of_2520th_observation": f.date.iloc[2519], "first_covariance_month": first_cov, "observations_at_first_covariance_month": int(f.date.le(first_cov).sum()), "first_moment_month": first_cov+pd.offsets.MonthEnd(12), "max_condition_number": float(f.condition.max()), "daily_rows": sum(v["rows"] for v in coverage), "missing_daily_returns_retained": sum(v["missing_returns_retained"] for v in coverage), "factors": factors, "resources": resource_guard()})


if __name__ == "__main__":
    import sys
    {"monthly": finish_monthly, "daily": factor_inputs}[sys.argv[1]]()
