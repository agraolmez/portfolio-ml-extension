"""Derive the early factor calendar from actual daily regressions.

This is a coverage diagnostic for the provisional causal universe, not the
complete covariance engine or acceptance of the historical feature vintage.
"""
import argparse
import json
import numpy as np
import pandas as pd
from step3_acquire import FEATURES
from step3_v2_io import ROOT, RAW, OUT, PREP, PRIOR_RAW, save_json, resource_guard


def industry(s):
    groups = {
        "NoDur": [(100,999),(2000,2399),(2700,2749),(2770,2799),(3100,3199),(3940,3989)],
        "Durbl": [(2500,2519),(3630,3659),(3710,3711),(3714,3714),(3716,3716),(3750,3751),(3792,3792),(3900,3939),(3990,3999)],
        "Manuf": [(2520,2589),(2600,2699),(2750,2769),(3000,3099),(3200,3569),(3580,3629),(3700,3709),(3712,3713),(3715,3715),(3717,3749),(3752,3791),(3793,3799),(3830,3839),(3860,3899)],
        "Enrgy": [(1200,1399),(2900,2999)], "Chems": [(2800,2829),(2840,2899)],
        "BusEq": [(3570,3579),(3660,3692),(3694,3699),(3810,3829),(7370,7379)],
        "Telcm": [(4800,4899)], "Utils": [(4900,4949)],
        "Shops": [(5000,5999),(7200,7299),(7600,7699)],
        "Hlth": [(2830,2839),(3693,3693),(3840,3859),(8000,8099)], "Money": [(6000,6999)]}
    labels = np.full(len(s), "Other", dtype=object)
    for label, spans in groups.items():
        flag = np.zeros(len(s), dtype=bool)
        for lo, hi in spans:
            flag |= s.between(lo, hi).to_numpy()
        labels[flag] = label
    return labels


def build(full=False):
    end_year = 2025 if full else 1966
    prefix = "full" if full else "early"
    x = pd.read_parquet(PREP / "candidate_monthly_1952_2025.parquet", columns=["permno", "eom", "sic", "basic_valid", "valid", *FEATURES], filters=[("eom", "<=", pd.Timestamp(f"{end_year}-12-31"))])
    x = x[x.basic_valid].copy()
    for f in FEATURES:
        raw = x[f]
        x[f] = x.groupby("eom")[f].rank(method="max", pct=True).mask(raw.eq(0), 0).fillna(.5)
    x = x[x.valid].copy()
    dictionary = pd.read_csv(ROOT / "config/feature_dictionary_v1.csv")
    clusters = dictionary.cluster.unique().tolist()
    for cl in clusters:
        spec = dictionary[dictionary.cluster.eq(cl)]
        block = x[spec.characteristic].copy()
        minus = spec.loc[spec.direction.eq(-1), "characteristic"]
        block[minus] = 1-block[minus]
        x[cl] = block.mean(axis=1)
        x[cl] = x.groupby("eom")[cl].transform(lambda s: (s-s.mean())/s.std(ddof=1))
    x["ff12"] = industry(pd.to_numeric(x.sic))
    inds = sorted(x.ff12.unique())
    for ind in inds:
        x[ind] = x.ff12.eq(ind).astype(float)
    factors = inds + clusters
    x["return_eom"] = x.eom + pd.offsets.MonthEnd(1)
    loadings = x[["permno", "eom", "return_eom", *factors]].copy()
    loadings.to_parquet(OUT / f"{prefix}_factor_loadings.parquet", index=False)
    del x
    rates = pd.read_parquet(PRIOR_RAW / "crsp_ciz_t30_index.parquet")
    rates["return_eom"] = rates.mthcaldt + pd.offsets.MonthEnd(0)
    observations = []
    parity = []
    for year in range(1954, end_year+1):
        daily_path = RAW / f"crsp_ciz_daily_full_ids_{year}.parquet" if full and year >= 1982 else PRIOR_RAW / f"crsp_ciz_daily_{year}.parquet"
        d = pd.read_parquet(daily_path, columns=["permno", "dlycaldt", "dlyret"])
        d["return_eom"] = d.dlycaldt + pd.offsets.MonthEnd(0)
        d = d.merge(rates[["return_eom", "mthtotret"]], on="return_eom", validate="many_to_one")
        d["ret_exc"] = d.dlyret-d.mthtotret/21
        d = d.merge(loadings, on=["permno", "return_eom"], validate="many_to_one").dropna(subset=["ret_exc", *factors])
        for day, g in d.groupby("dlycaldt", sort=True):
            a = g[factors].to_numpy(); y = g.ret_exc.to_numpy()
            beta, _, rank, singular = np.linalg.lstsq(a, y, rcond=1e-7)
            row = {"date": day, "n_stocks": len(g), "rank": int(rank), "n_factors": len(factors), "condition": float(singular[0]/singular[-1]), **dict(zip(factors, beta))}
            observations.append(row)
            if not parity or (day.month == 12 and day.day <= 3 and year in [1964,1965,1966,2020,2025]):
                parity.append(g[["dlycaldt", "permno", "ret_exc", *factors]])
        resource_guard()
    f = pd.DataFrame(observations).sort_values("date")
    if len(f) < 2520 or f.date.duplicated().any():
        raise ValueError("Insufficient/nonunique factor dates")
    threshold = f.date.iloc[2519].replace(day=1)-pd.Timedelta(days=1)
    first_cov = loadings.loc[loadings.eom.ge(threshold), "eom"].min()
    start_n = int(f.date.le(first_cov).sum())
    f.to_parquet(OUT / f"{prefix}_factor_regressions.parquet", index=False)
    pd.concat(parity).to_csv(OUT / f"{prefix}_factor_regression_R_comparison_inputs.csv", index=False)
    report = {"status": "EARLY_CALENDAR_COMPUTED_FOR_PROVISIONAL_UNIVERSE", "first_daily_factor_date": f.date.min(), "daily_factor_rows": len(f), "factor_count": len(factors), "rank_deficient_days": int(f['rank'].lt(len(factors)).sum()), "date_of_2520th_observation": f.date.iloc[2519], "source_threshold_month": threshold, "first_covariance_month": first_cov, "observations_at_first_covariance_month": start_n, "source_initial_shortfall_warning": start_n < 2490, "first_moment_month": first_cov+pd.offsets.MonthEnd(12), "max_condition_number": float(f.condition.max()), "factors": factors, "limits": ["Uses preserved source-observed predictor ranks and provisional causal screens.", "Does not compute or certify every covariance matrix, idiosyncratic-risk estimate, or Portfolio-ML moment.", "Recompute if the accepted feature vintage or eligibility rule changes."], "resources": resource_guard()}
    report["through_year"] = end_year
    save_json(OUT / f"factor_calendar{'full' if full else ''}.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    build(full=parser.parse_args().full)
