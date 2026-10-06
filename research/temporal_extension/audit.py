"""Bounded, read-only parent verification and neutral archived-path replay.

This command never fits a temporal gain, reconstructs a parent, or evaluates a
new historical strategy. Receipts may only be written to a new output directory.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline.core import transition
from baseline.full import preflight
from feature_geometry.features import fingerprint
from feature_geometry.history import CanonicalHistory, sha
from feature_geometry.policy import GeometryModel
from .features import TemporalFeatureMap
from .reference import Innovation


ROOT = Path(__file__).resolve().parents[1]
PARENT = "output/feature_geometry_study_20260914"
DATES = ("1981-11-30", "1981-12-31")
TOLERANCES = {"inputs_and_signals_atol": 0., "inputs_and_signals_rtol": 0.,
              "archived_path_atol": 1e-11, "archived_path_rtol": 1e-10,
              "accounting_atol": 1e-12, "accounting_rtol": 1e-11}


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


class Integrity:
    def __init__(self):
        self.before = {}

    def track(self, relative, expected=None):
        relative = str(relative)
        value = sha(ROOT / relative)
        if expected is not None and value != expected:
            raise ValueError("Accepted artifact changed: " + relative)
        if relative in self.before and value != self.before[relative]:
            raise ValueError("Artifact changed during audit: " + relative)
        self.before[relative] = value
        return value

    def finish(self):
        for relative, expected in self.before.items():
            if sha(ROOT / relative) != expected:
                raise ValueError("Audit mutated or observed a changed input: " + relative)
        return dict(files_verified_before_and_after=len(self.before),
                    unchanged=True, artifact_sha256=self.before)


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def audit_parents(integrity):
    accepted = read_json(f"{PARENT}/acceptance_manifest.json")
    if not accepted.get("accepted") or not accepted.get("study_complete"):
        raise ValueError("Accepted completed Extension-2 study required")
    inventory = accepted["artifact_sha256"]
    for name in ("acceptance_manifest.json", "fits_manifest.json", "paths_manifest.json", "plan_lock.json"):
        rel = f"{PARENT}/{name}"
        integrity.track(rel, inventory.get(rel))
    current = read_json(f"{PARENT}/plan_lock.json")
    for field in ("code_sha256", "directions_sha256"):
        for relative, expected in current[field].items():
            integrity.track(relative, expected)
    locks = {sha(ROOT / f"{PARENT}/plan_lock.json"): f"{PARENT}/plan_lock.json"}
    for path in sorted((ROOT / PARENT / "engineering_versions").glob("*/plan_lock.json")):
        old = json.loads(path.read_text())
        if old["plan"] != current["plan"] or old["directions_sha256"] != current["directions_sha256"]:
            raise ValueError("Retained execution lock changes the scientific configuration")
        rel = str(path.relative_to(ROOT))
        locks[integrity.track(rel, inventory.get(rel))] = rel

    step9 = read_json("output/step9_20260913/acceptance_manifest.json")
    integrity.track("output/step9_20260913/acceptance_manifest.json")
    selection_rel = "data/prepared/step9_v1/baseline/annual_selection.parquet"
    integrity.track(selection_rel, step9["artifact_sha256"][selection_rel])
    selection = pd.read_parquet(ROOT / selection_rel)
    bundle = read_json("output/step3_author_conventions_20260911/bundle_checksums.json")
    integrity.track("output/step3_author_conventions_20260911/bundle_checksums.json")
    for rel in ("config/feature_dictionary_v1.csv", "output/step2_20260910/source_settings.json"):
        integrity.track(rel, bundle[rel])
    features = read_json("output/step2_20260910/source_settings.json")["features"]
    dictionary = pd.read_csv(ROOT / "config/feature_dictionary_v1.csv")
    if (dictionary.characteristic.tolist() != features or len(features) != 115
            or dictionary.cluster.isna().any() or dictionary.cluster.nunique() != 13):
        raise ValueError("The exact 115-coordinate, 13-theme map is required")
    themes = sorted(dictionary.cluster.unique())
    groups = dictionary.cluster.map({name: i for i, name in enumerate(themes)}).to_numpy(dtype=int)

    records = []
    for year in range(1981, 2026):
        cutoff = f"{year - 1}-12-31"
        rows = selection[selection.return_eom.eq(cutoff)]
        if len(rows) != 1 or int(rows.iloc[0].p) != 512:
            raise ValueError("Unique original annual 512-feature selection required")
        choice = rows.iloc[0]
        for seed in (1, 2, 3):
            folder = f"{PARENT}/fits/{year}/seed{seed}/full"
            for name in ("model.npz", "fit.json"):
                rel = f"{folder}/{name}"
                integrity.track(rel, inventory[rel])
            fit = read_json(f"{folder}/fit.json")
            model = GeometryModel.load(ROOT / folder / "model.npz")
            directions = np.fromfile(ROOT / PARENT / f"directions/seed{seed}_g{int(choice.g_index)}.bin",
                                     dtype="<f8").reshape(115, 512)[:, :256]
            if (model.theta.shape != (115,) or model.beta.shape != (513,)
                    or model.feature_map.frequencies.shape != (115, 256)
                    or fit["method"] != "full" or fit["year"] != year or fit["seed"] != seed
                    or model.cutoff != cutoff or fit["cutoff"] != cutoff
                    or fit["training_first"] != "1965-11-30"
                    or fit["training_last"] != f"{year - 1}-11-30"
                    or fit["training_months"] != 181 + 12 * (year - 1981)
                    or fit["latest_available_at"] != cutoff
                    or fit["source_fingerprint"] != model.source_fingerprint
                    or fit["coefficient_ridge"] != float(choice["lambda"])
                    or fit["bandwidth_index"] != int(choice.g_index)
                    or fit["plan_lock_sha256"] not in locks):
                raise ValueError("Parent model provenance mismatch: " + folder)
            np.testing.assert_array_equal(model.feature_map.frequencies, directions)
            records.append(dict(year=year, seed=seed, model=f"{folder}/model.npz",
                                fit_receipt=f"{folder}/fit.json", cutoff=cutoff,
                                ridge=float(choice["lambda"]), bandwidth_index=int(choice.g_index),
                                scale_min=float(np.exp(model.theta).min()),
                                scale_max=float(np.exp(model.theta).max()),
                                absolute_log_scale_sum=float(abs(model.theta.sum())),
                                stored_solve_residual=fit["solve_residual"],
                                optimizer_status=fit["status"],
                                execution_lock=locks[fit["plan_lock_sha256"]],
                                parameters_sha256=fingerprint(model.theta, model.feature_map.frequencies, model.beta)))
    return dict(parent_family="full", models_verified=len(records), records=records,
                feature_count=115, nonlinear_features=512, coefficients=513,
                themes=themes, characteristic_to_theme=dict(zip(features, dictionary.cluster)),
                optimizer_convergence_established=False), features, groups, bundle, step9


def observation_mask_check(integrity, features, bundle):
    """Audit one bounded source window, not an approved temporal-history policy."""
    raw_rel = "data/prepared/step3_author_v3/source_characteristics.parquet"
    ranks_rel = "data/prepared/step3_author_v3/monthly_inputs.parquet"
    for rel in (raw_rel, ranks_rel):
        integrity.track(rel, bundle[rel])
    filters = [("eom", ">=", pd.Timestamp("1964-05-31")),
               ("eom", "<=", pd.Timestamp("1965-11-30"))]
    raw = pd.read_parquet(ROOT / raw_rel, columns=["id", "eom", *features], filters=filters)
    ranks = pd.read_parquet(ROOT / ranks_rel, columns=["id", "eom", *features], filters=filters)
    keys = ["id", "eom"]
    if raw.duplicated(keys).any() or ranks.duplicated(keys).any():
        raise ValueError("Duplicate source security/month keys")
    raw = raw.set_index(keys)
    ranks = ranks.set_index(keys)
    positions = raw.index.get_indexer(ranks.index)
    if np.any(positions < 0):
        raise ValueError("Prepared rank has no source observation-mask row")
    source = raw.iloc[positions].to_numpy()
    z = ranks.to_numpy()
    if not np.isfinite(z).all() or np.isinf(source).any():
        raise ValueError("Unexpected nonfinite source or prepared characteristics")
    observed = ~np.isnan(source)
    np.testing.assert_array_equal(z[~observed], np.full(int((~observed).sum()), .5))
    dates = ranks.index.get_level_values("eom")
    return dict(first_month=str(dates.min().date()), last_month=str(dates.max().date()),
                months=int(dates.nunique()), expected_months=19, source_rows=len(raw),
                prepared_rows=len(ranks), unmatched_prepared_rows=0,
                observed_cells=int(observed.sum()), missing_source_cells=int((~observed).sum()),
                observed_cells_with_rank_half=int(np.count_nonzero(observed & (z == .5))),
                missing_cells_imputed_to_parent_half=True,
                method="Keyed raw-source NaN mask; never infer missingness from rank 0.5",
                complete_historical_mask_join_verified=False,
                temporal_coverage_or_fallback_policy_approved=False)


def archived_neutral_replay(integrity, groups, receipt, step9):
    source = CanonicalHistory(ROOT, DATES[0], DATES[-1], receipt)
    source.sources.update(step9["artifact_sha256"])
    path_manifest = read_json(f"{PARENT}/paths_manifest.json")["artifact_sha256"]
    fit_inputs = read_json(f"{PARENT}/fits_manifest.json")["consumed_input_sha256"]
    scales_rel = "data/prepared/step4_v1/risk/scales.parquet"
    integrity.track(scales_rel, read_json("output/step4_20260911/acceptance_manifest.json")["artifact_sha256"][scales_rel])
    outcomes = pd.read_parquet(ROOT / "data/prepared/step3_author_v3/monthly_inputs.parquet",
                               columns=["id", "eom", "tr_ld1", "mu_ld1", "valid"],
                               filters=[("eom", ">=", pd.Timestamp(DATES[0])),
                                        ("eom", "<=", pd.Timestamp("1982-01-31"))])
    outcomes = {day: frame.set_index("id") for day, frame in outcomes.groupby("eom")}
    holdings, holding_ids, records = {}, None, []
    with threadpool_limits(limits=4):
        for day_string in DATES:
            day = pd.Timestamp(day_string)
            year = (day + pd.offsets.MonthEnd(1)).year
            market = source.market(day)
            path_rel = f"{PARENT}/paths/{day_string}.npz"
            integrity.track(path_rel, path_manifest[path_rel])
            with np.load(ROOT / path_rel, allow_pickle=False) as data:
                archived = {key: data[key].copy() for key in data.files}
            np.testing.assert_array_equal(market.ids, archived["ids"])
            for kind in ("matrices", "risk"):
                rel = f"data/prepared/step9_v1/baseline/{kind}/{day_string}.npz"
                # Earlier Step9 files are accepted symlinks into Step6/Step4.
                # The final fit receipt records the actual aliases it consumed.
                integrity.track(rel, fit_inputs[rel])
                with np.load(ROOT / rel, allow_pickle=False) as data:
                    np.testing.assert_array_equal(market.ids, data["ids"])
                    if kind == "matrices":
                        np.testing.assert_array_equal(market.m, data["m"])
                    else:
                        sigma = data["loadings"] @ data["factor_cov"] @ data["loadings"].T + np.diag(data["specific_variance"])
                        np.testing.assert_allclose(market.covariance, sigma, atol=1e-14, rtol=1e-12)
            next_date = day + pd.offsets.MonthEnd(1)
            next_ids = outcomes[next_date].query("valid").sort_index().index.to_numpy()
            np.testing.assert_array_equal(next_ids, archived["next_ids"])
            economic_hash = fingerprint(market.ids, market.m, market.covariance, market.impact,
                                        market.wealth, market.gamma, *market.volatility, *market.growth)
            for seed in (1, 2, 3):
                selected = np.flatnonzero((archived["seeds"] == seed) & (archived["methods"] == "full"))
                if len(selected) != 1:
                    raise ValueError("Unique archived full/seed path required")
                col = int(selected[0])
                model_rel = f"{PARENT}/fits/{year}/seed{seed}/full/model.npz"
                model = GeometryModel.load(ROOT / model_rel)
                temporal = TemporalFeatureMap(model.feature_map, model.theta, groups)
                model_hash = temporal.fingerprint
                max_signal_error = 0.
                for z, vol in zip(market.characteristics, market.volatility):
                    missing = Innovation(np.full(z.shape, np.nan), np.zeros(z.shape, dtype=bool),
                                         {"neutral_replay_deliberately_invalid_history": int(z.size)})
                    np.testing.assert_array_equal(temporal.inputs(z, missing, np.ones(13)), z)
                    expected = model.feature_map.signals(z, vol, model.theta)
                    actual = temporal.signals(z, vol, missing, np.ones(13))
                    np.testing.assert_array_equal(actual, expected)
                    max_signal_error = max(max_signal_error, float(np.max(np.abs(actual - expected))))
                z, vol = market.characteristics[0], market.volatility[0]
                missing = Innovation(np.full(z.shape, np.nan), np.zeros(z.shape, dtype=bool), {})
                aim = temporal.signals(z, vol, missing, np.ones(13)) @ model.beta
                np.testing.assert_array_equal(aim, model.aim(z, vol, day_string))
                if seed not in holdings:
                    holdings[seed] = archived["pretrade"][:, col].copy()
                else:
                    np.testing.assert_array_equal(holding_ids, market.ids)
                outcome = outcomes[day].loc[market.ids]
                returns = source.months[day].loc[market.ids, "ret_ld1"].to_numpy()
                result = transition(market.ids, holdings[seed], aim, market.m,
                                    outcome.tr_ld1.to_numpy(), float(outcome.mu_ld1.iloc[0]),
                                    market.impact, market.wealth, returns, next_ids)
                pairs = {"aim": (aim, archived["target"][:, col]),
                         "pretrade": (holdings[seed], archived["pretrade"][:, col]),
                         "weights": (result["weights"], archived["weight"][:, col]),
                         "orders": (result["trades"], archived["trade"][:, col]),
                         "drifted_weights": (result["drifted_weights"], archived["drifted_weight"][:, col]),
                         "next_pretrade": (result["next_pretrade"], archived["next_pretrade"][:, col])}
                errors = {}
                for name, (actual, expected) in pairs.items():
                    np.testing.assert_allclose(actual, expected, atol=TOLERANCES["archived_path_atol"],
                                               rtol=TOLERANCES["archived_path_rtol"])
                    errors[name] = float(np.max(np.abs(actual - expected)))
                gross = float(sum(float(w) * float(r) for w, r in zip(result["weights"], returns)))
                cost = float(.5 * market.wealth * sum(float(l) * float(q)**2
                                                    for l, q in zip(market.impact, result["trades"])))
                np.testing.assert_allclose([gross, cost], [result["gross_excess"], result["impact_cost"]],
                                           atol=TOLERANCES["accounting_atol"], rtol=TOLERANCES["accounting_rtol"])
                if temporal.fingerprint != model_hash or fingerprint(
                        market.ids, market.m, market.covariance, market.impact, market.wealth,
                        market.gamma, *market.volatility, *market.growth) != economic_hash:
                    raise ValueError("Frozen model or economic arrays changed")
                records.append(dict(decision_date=day_string, seed=seed, parent_model=model_rel,
                                    annual_model_year=year, training_cutoff=model.cutoff,
                                    stocks=len(market.ids), nonlinear_features=512, gain_count=13,
                                    feature_slices_compared=13, max_signal_absolute_error=max_signal_error,
                                    path_max_absolute_errors=errors, economic_array_fingerprint=economic_hash,
                                    temporal_parent_fingerprint=model_hash, direct_order_cost=cost,
                                    actual_order_accounting_passed=True,
                                    discarded_exit_count=len(result["dropped_ids"])))
                holdings[seed] = result["next_pretrade"].copy()
            holding_ids = next_ids.copy()
    for relative, expected in source.consumed.items():
        integrity.track(relative, expected)
    return dict(status="PASS", dates=list(DATES), seeds=[1, 2, 3], records=records,
                annual_model_transition="1981 to 1982 between November and December decisions",
                initial_holdings="Archived November 1981 full-path pretrade positions, once per seed",
                own_holdings_carried_between_decisions=True, gains_fixed_to_one=True,
                all_history_deliberately_invalid_to_test_neutral_bypass=True,
                archived_coefficients_reused=True, full_annual_conditional_resolve_performed=False,
                complete_540_month_replay_performed=False,
                accounting_rule="Unchanged original transition, including recorded source exit omissions")


def run_audit(destination):
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT / "output") or destination == ROOT / "output":
        raise ValueError("Choose a new directory under the canonical project's output folder")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    integrity = Integrity()
    engineering_plan = read_json("config/extension3_temporal_audit_v1.json")
    if (engineering_plan["neutral_replay_scope"]["decision_dates"] != list(DATES)
            or engineering_plan["neutral_replay_scope"]["seeds"] != [1, 2, 3]
            or engineering_plan["synthetic_test_tolerances"]["neutral_moments_beta_aims_path"]
            != {"atol": TOLERANCES["archived_path_atol"], "rtol": TOLERANCES["archived_path_rtol"]}):
        raise ValueError("Bounded replay scope/tolerances differ from the recorded engineering plan")
    integrity.track("config/extension3_temporal_audit_v1.json")
    for rel in ("GOAL.md", "docs/STEP_1_THEORY.md", "docs/RECONSTRUCTION_AMENDMENT.md",
                "config/protocol_v1.json", "config/reconstruction_amendment.json"):
        integrity.track(rel)
    print("Verifying accepted financial dependencies before bounded historical checks", flush=True)
    receipt, _ = preflight()
    write_new(destination / "financial_preflight.json", receipt)
    print("Verifying 135 archived learned-scale parents and their tuning", flush=True)
    parents, features, groups, bundle, step9 = audit_parents(integrity)
    write_new(destination / "parent_inventory.json", parents)
    print("Checking raw observation masks in the bounded warm-up audit window", flush=True)
    masks = observation_mask_check(integrity, features, bundle)
    write_new(destination / "observation_mask_audit.json", masks)
    print("Replaying two neutral decisions across an annual refit for three seeds", flush=True)
    replay = archived_neutral_replay(integrity, groups, receipt, step9)
    write_new(destination / "archived_neutral_replay.json", replay)
    unchanged = integrity.finish()
    write_new(destination / "parent_integrity.json", unchanged)
    result = dict(status="PASS_BOUNDED_ENGINEERING_AUDIT", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.monotonic() - started, tolerances=TOLERANCES,
                  parent_models_verified=parents["models_verified"], theme_count=len(parents["themes"]),
                  archived_neutral_decision_seed_cases=len(replay["records"]),
                  parent_artifacts_unchanged=True, optimization_performed=False,
                  full_historical_experiment_performed=False, paid_resources_used=False,
                  performance_or_originality_established=False,
                  limits=["Neutral historical replay reuses archived beta; it does not re-solve every annual training system.",
                          "Two consecutive decisions across a refit are checked; this is not a complete historical replay.",
                          "Source-mask join is checked in one 19-month window; full temporal coverage remains unmeasured.",
                          "Future-availability screening and vendor-vintage limitations of archived ranks remain unchanged.",
                          "Temporal settings, pilot and full historical fitting remain unapproved."],
                  artifacts_sha256={p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()})
    write_new(destination / "audit_manifest.json", result)
    print(json.dumps({k: result[k] for k in ("status", "seconds", "parent_models_verified",
                                            "archived_neutral_decision_seed_cases")}), flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    run_audit(args.output)


if __name__ == "__main__":
    main()
