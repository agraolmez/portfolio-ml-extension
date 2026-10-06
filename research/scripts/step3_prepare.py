"""Build explicitly unaccepted candidate screens to locate relevant data failures.

This does not fit portfolios, hide missing outcomes, or accept historical inputs.
"""
import argparse
import json
import numpy as np
import pandas as pd
from step3_io import ROOT, RAW, PREP, OUT, save_json, resource_guard
from step3_acquire import FEATURES


def candidate_screens(frame):
    """Causal version of the released base screen; returns never look forward."""
    x = frame.copy()
    x["feature_count"] = x[FEATURES].notna().sum(axis=1)
    flags = x[["common", "exch_main", "primary_sec", "obs_main"]].eq(1).all(axis=1)
    x["basic_valid"] = flags & x.me.gt(0) & x.dolvol_126d.gt(0) & x.sic.notna() & x.feature_count.ge(57) & x.mthret.notna() & x.mthdelflg.eq("N")
    x = x.sort_values(["id", "eom"]).reset_index(drop=True)
    b = x[x.basic_valid].copy()
    b["month_number"] = b.eom.dt.year * 12 + b.eom.dt.month
    b["valid_data"] = (b.month_number - b.groupby("id").month_number.shift(12)).eq(12)
    b["valid_size"] = False
    for eom, group in b[b.valid_data].groupby("eom", sort=True):
        rank = group.me.rank(method="max", pct=True)
        n = len(group)
        n_size = int(rank.gt(0.5).sum())
        # Literal upper-half/minimum-50 specialization of size_screen_fun.
        n_miss = max(50 - n_size, 0)
        n_less = int(rank.le(0.5).sum())
        n_below = int(np.ceil(min(n_miss / 2, n_less)))
        if n_below > 0:
            n_below += n_miss - n_below
        b.loc[group.index, "valid_size"] = rank.gt(0.5 - n_below / n).to_numpy()
    b["valid"] = False
    for _, group in b.groupby("id", sort=False):
        temp = (group.valid_data & group.valid_size).to_numpy()
        serial = group.month_number.to_numpy()
        included = np.zeros(len(group), dtype=bool)
        state = False
        last_add = False
        for i in range(len(group)):
            # Require a real 12-month window and restart addition after a gap.
            contiguous = i >= 11 and serial[i] - serial[i - 11] == 11
            add = bool(contiguous and temp[i - 11:i + 1].all())
            delete = bool(contiguous and not temp[i - 11:i + 1].any())
            if i and serial[i] != serial[i - 1] + 1:
                state = False
                last_add = False
            if i and not state and add and not last_add:
                state = True
            if state and delete:
                state = False
            included[i] = state and bool(group.valid_data.iloc[i])
            last_add = add
        b.loc[group.index, "valid"] = included
    for col in ["valid_data", "valid_size", "valid"]:
        x[col] = False
        x.loc[b.index, col] = b[col]
    return x


def prepare(end_year):
    required = [RAW / f"jkp_nyse_{y}.parquet" for y in range(1952, end_year + 1)]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Required history missing: {missing}")
    j = pd.concat([pd.read_parquet(p) for p in required], ignore_index=True)
    if j.duplicated(["id", "eom"]).any():
        raise RuntimeError("Duplicate source security-month keys")
    cs = []
    for y in range(1952, end_year + 1):
        c = pd.read_parquet(RAW / f"crsp_ciz_monthly_{y}.parquet", columns=["permno", "mthcaldt", "mthret", "mthretx", "mthdelflg", "mthprc", "mthprcflg", "shrout"])
        c["eom"] = c.mthcaldt + pd.offsets.MonthEnd(0)
        cs.append(c)
    c = pd.concat(cs, ignore_index=True)
    joined = j.merge(c, how="left", on=["permno", "eom"], validate="one_to_one")
    del c, cs, j
    resource_guard()
    x = candidate_screens(joined)
    del joined
    x["decision_eom"] = x.eom
    x["next_realization_eom"] = x.eom + pd.offsets.MonthEnd(1)
    x["market_equity_usd"] = x.me * 1e6
    x["lambda_diagonal"] = 0.2 / x.dolvol_126d.where(x.dolvol_126d.gt(0))
    x["volatility_monthly"] = x.rvol_252d * np.sqrt(21)
    x["source_return_difference"] = x.ret - x.mthret
    candidate_path = PREP / f"candidate_monthly_through_{end_year}.parquet"
    x.to_parquet(candidate_path, index=False, compression="zstd")
    active = x[x.valid]
    ids = sorted(active.id.astype(int).unique().tolist())
    save_json(OUT / "required_daily_ids.json", ids)
    coverage = x.groupby("eom").agg(source_rows=("id", "size"), source_ids=("id", "nunique"), basic_valid=("basic_valid", "sum"), history_valid=("valid_data", "sum"), active_candidates=("valid", "sum"), min_features=("feature_count", "min"), median_features=("feature_count", "median"))
    coverage.to_parquet(OUT / f"monthly_coverage_through_{end_year}.parquet")
    next_ciz = []
    for y in range(1952, end_year + 1):
        c = pd.read_parquet(RAW / f"crsp_ciz_monthly_{y}.parquet", columns=["permno", "mthcaldt", "mthret", "mthdelflg"])
        c["decision_eom"] = c.mthcaldt + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
        next_ciz.append(c)
    q = active[["id", "permno", "decision_eom", "next_realization_eom"]].merge(pd.concat(next_ciz, ignore_index=True), on=["permno", "decision_eom"], how="left", validate="one_to_one")
    q["outcome_status"] = np.where(q.mthret.isna(), "missing_required_next_return", np.where(q.mthdelflg.isin(["G", "M", "V"]), "requires_claim_resolution", "source_return_available_terms_still_to_audit"))
    q.loc[q.decision_eom.eq(pd.Timestamp(f"{end_year}-12-31")), "outcome_status"] = "right_boundary_next_year_outside_slice"
    q.to_parquet(OUT / f"candidate_outcome_links_through_{end_year}.parquet", index=False)
    q[q.outcome_status.isin(["missing_required_next_return", "requires_claim_resolution"])].to_parquet(OUT / f"candidate_required_exceptions_through_{end_year}.parquet", index=False)
    pilot = q[q.next_realization_eom.between("1981-01-31", "1982-12-31")]
    report = {
        "status": "CANDIDATE_NOT_ACCEPTED", "purpose": "locate required history and accounting evidence without fitting portfolios",
        "through_year": end_year, "source_rows": len(x), "source_months": x.eom.nunique(),
        "first_raw_month": x.eom.min(), "last_raw_month": x.eom.max(),
        "first_candidate_active_month": active.eom.min(), "candidate_active_rows": len(active), "candidate_distinct_ids": len(ids),
        "invalid_risk_scale_among_candidates": int((active.rvol_252d.isna() | active.rvol_252d.le(0)).sum()),
        "all_required_outcome_statuses": q.outcome_status.value_counts().to_dict(),
        "pilot_outcome_statuses": pilot.outcome_status.value_counts().to_dict(),
        "boundary": "Candidate screens retain source features and use current total returns. They are for locating failures; they do not certify point-in-time feature lineage or accept corporate-action accounting.",
    }
    save_json(OUT / f"candidate_preparation_through_{end_year}.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-year", type=int, required=True)
    prepare(parser.parse_args().end_year)
