"""Read-only chronology and accepted-target provenance; no performance calculation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
E2 = "output/feature_geometry_study_20260914"
E3 = "output/extension3_full_study_20260919"
E4 = "output/extension4_full_study_20260924"
CONFIG = "config/portfolio_ensemble_screen_20260928.json"
OUTPUT = ROOT / "output/portfolio_ensemble_screen_20260928/chronology.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    require(not OUTPUT.exists(), "Preserve existing chronology receipt; this audit writes new-only")
    specification = read(ROOT / CONFIG)
    config_sha = sha(ROOT / CONFIG)
    require(specification["decision_dates"] == ["2015-12-31", "2025-11-30"]
            and specification["training_authorized"] is False
            and specification["candidates"] == ["CENS", "HALF"]
            and specification["paths"] == ["B1", "B2", "B3", "C1", "C2", "C3", "BENS", "CENS", "HALF"]
            and specification["common_initial_holdings"] == "accepted fixed512 seed1 pretrade at December 2015 decision",
            "Screen protocol differs from the bounded saved-target specification")
    receipts = {label: read(ROOT / label / "acceptance_manifest.json") for label in (E2, E3, E4)}
    for label, receipt in receipts.items():
        require(receipt["accepted"] is True and receipt["study_complete"] is True
                and all(g["passed"] is True for g in receipt["gates"]), "Parent not accepted: " + label)
    inventory = {}
    for receipt in receipts.values():
        for relative, digest in receipt["artifact_sha256"].items():
            if relative in inventory:
                require(inventory[relative] == digest, "Accepted parent inventories disagree: " + relative)
            inventory[relative] = digest
    consumed = {}
    for label in (E2, E3, E4):
        relative = label + "/acceptance_manifest.json"
        current = sha(ROOT / relative)
        if relative in inventory:
            require(current == inventory[relative], "Parent acceptance changed: " + relative)
        consumed[relative] = current

    def accepted(relative):
        require(relative in inventory, "Unaccepted input: " + relative)
        actual = sha(ROOT / relative)
        require(actual == inventory[relative], "Changed accepted input: " + relative)
        consumed[relative] = actual
        return ROOT / relative

    def metadata(relative):
        return read(accepted(relative))

    # Bind the exact implementation that made the archived targets. These are
    # source reads/hash checks only; no aim or portfolio function is called.
    source_paths = ["extension4/run.py", "extension4/paths.py", "extension4/model.py",
                    "extension4/parent.py", "extension4/selection.py", "geometry_study/run.py",
                    "baseline/engine.py", "temporal_extension/model.py", "feature_geometry/features.py"]
    for relative in source_paths:
        accepted(E4 + "/lock/source_snapshot/" + relative if relative.startswith("extension4/") else relative)
    from temporal_extension.model import TemporalModel

    lock = metadata(E4 + "/lock/commit.json")
    require(lock["status"] == "COMPLETE", "Extension-4 lock incomplete")
    lock_hash = consumed[E4 + "/lock/commit.json"]
    for relative in [p for p in source_paths if p.startswith("extension4/")]:
        require(lock["context"]["implementation_sha256"][relative] == consumed[E4 + "/lock/source_snapshot/" + relative], "Archived dispatch code changed")
    state_identity = metadata(E4 + "/states/identity.json")["identity"]
    selection = metadata(E4 + "/selection/selected.json")
    selected_c4 = [r for r in selection if r["method"] == "C4"]
    require(sorted(r["seed"] for r in selected_c4) == [1, 2, 3]
            and all(r["multiplier"] == 1 and r["available_at"] == "2015-12-31" for r in selected_c4),
            "C4 selection is not the required frozen December-2015 choice")
    parent_selection = metadata(E3 + "/selection_canonical_v1/temporal.json")
    require(parent_selection["status"] == "SELECTION_FROZEN"
            and parent_selection["selection_available_at"] <= "2010-12-31",
            "Temporal parent selection became available too late")
    require(sorted(r["seed"] for r in parent_selection["selected_by_seed"]) == [1, 2, 3]
            and all(r["candidate"] == "temporal__rho_0p1" for r in parent_selection["selected_by_seed"]),
            "Unexpected temporal parent family")
    choices = sorted((r["seed"], r["method"], r["multiplier"]) for r in selection)
    policy_identity = fingerprint(lock_hash, state_identity, choices)
    annual_selection = pd.read_parquet(accepted("data/prepared/step9_v1/baseline/annual_selection.parquet"))
    annual_checks = []
    for year in range(2016, 2026):
        cutoff, last = f"{year-1}-12-31", f"{year-1}-11-30"
        training_months = len(pd.date_range("1965-11-30", last, freq="ME"))
        chosen = annual_selection[annual_selection.return_eom == pd.Timestamp(cutoff)]
        require(len(chosen) == 1 and chosen.iloc[0].p == 512, "Baseline past-year selection missing")
        for seed in (1, 2, 3):
            folder = E4 + f"/fits/{year}/seed{seed}"
            fit, commit = metadata(folder + "/fit.json"), metadata(folder + "/commit.json")
            require(commit["status"] == "COMPLETE" and commit["context"] == fit["context"], "Fit commit/context mismatch")
            require(commit["artifact_sha256"]["fit.json"] == consumed[folder + "/fit.json"], "Fit receipt hash mismatch")
            context = fit["context"]
            require(context["year"] == year and context["seed"] == seed
                    and context["lock"] == lock_hash and context["states"] == state_identity
                    and context["multipliers"]["C4"] == [1.0], "Wrong annual C4 context")
            require(fit["cutoff"] == cutoff and fit["last"] == last and fit["first"] == "1965-11-30"
                    and fit["training_months"] == training_months
                    and fit["parent_selection_available_at"] <= "2010-12-31", "Annual C4 chronology differs")
            p3folder = E3 + f"/fits/{year}/seed{seed}/temporal/rho_0p1"
            p3fit = metadata(p3folder + "/fit.json")
            require(p3fit["fit_spec"] == dict(year=year, seed=seed, kind="temporal", rho=.1, candidate="rho_0p1")
                    and p3fit["parent_settings"]["cutoff"] == cutoff and p3fit["training_last"] == last,
                    "Wrong annual temporal parent chronology")
            parent = TemporalModel.load(accepted(p3folder + "/model.npz"), expected_parent_fingerprint=p3fit["parent_fingerprint"])
            require(parent.cutoff == cutoff and parent._content_hash() == context["parent"], "Wrong bound temporal model")
            c4path = accepted(folder + "/C4__m1.npz")
            require(commit["artifact_sha256"]["C4__m1.npz"] == consumed[folder + "/C4__m1.npz"], "Conditional model hash mismatch")
            with np.load(c4path, allow_pickle=False) as saved:
                model_meta = json.loads(str(saved["metadata"].item()))
                model_identity = str(saved["identity"].item())
                require(fingerprint(saved["beta"], json.dumps(model_meta, sort_keys=True)) == model_identity,
                        "Conditional model metadata/parameter identity mismatch")
            require(model_meta["year"] == year and model_meta["seed"] == seed and model_meta["method"] == "C4"
                    and model_meta["multiplier"] == 1 and model_meta["parent_identity"] == context["parent"]
                    and model_meta["state_identity"] == state_identity and model_meta["protocol_identity"] == lock_hash,
                    "Conditional model metadata differs from annual policy")
            model_rows = [r for r in fit["models"] if r["method"] == "C4"]
            require(len(model_rows) == 1 and model_rows[0]["multiplier"] == 1
                    and model_rows[0]["model_file"] == "C4__m1.npz" and model_rows[0]["model_identity"] == model_identity,
                    "Fit receipt identifies another conditional model")
            baseline_folder = E2 + f"/fits/{year}/seed{seed}/fixed512"
            baseline = metadata(baseline_folder + "/fit.json")
            require(baseline["year"] == year and baseline["seed"] == seed and baseline["method"] == "fixed512"
                    and baseline["cutoff"] == cutoff and baseline["training_last"] == last
                    and baseline["latest_available_at"] == cutoff and baseline["training_months"] == training_months
                    and baseline["bandwidth_index"] == int(chosen.iloc[0].g_index)
                    and baseline["coefficient_ridge"] == float(chosen.iloc[0]["lambda"]), "Baseline chronology/selection differs")
            baseline_path = accepted(baseline_folder + "/model.npz")
            require(baseline["files_sha256"]["model.npz"] == consumed[baseline_folder + "/model.npz"], "Baseline model changed")
            with np.load(baseline_path, allow_pickle=False) as saved:
                bmeta = json.loads(str(saved["metadata"].item()))
            require(bmeta["cutoff"] == cutoff, "Saved baseline model has wrong cutoff")
            annual_checks.append(dict(year=year, seed=seed, cutoff=cutoff, last_training_decision=last,
                                      c4_multiplier=1, temporal_parent_cutoff=parent.cutoff,
                                      c4_model_identity=model_identity, baseline_prior_year_selection=True))
    dates = pd.date_range(*specification["decision_dates"], freq="ME")
    require(len(dates) == 120, "Screen calendar is not 120 months")
    previous_path = E4 + "/final_paths/2015-11-30/commit.json"
    metadata(previous_path)
    previous = consumed[previous_path]
    monthly = []
    common_initial = None
    for day in dates:
        stamp = str(day.date())
        year = (day + pd.offsets.MonthEnd(1)).year
        folder = E4 + "/final_paths/" + stamp
        commit = metadata(folder + "/commit.json")
        context = commit["context"]
        require(commit["status"] == "COMPLETE" and context["day"] == stamp
                and context["lock"] == lock_hash and context["policies"] == policy_identity
                and context["previous"] == previous, "Monthly commit breaks frozen policy/date/chain binding")
        c4specs = [s for s in context["specs"] if s[1] == "C4"]
        require(c4specs == [[1, "C4"], [2, "C4"], [3, "C4"]], "Monthly policy lacks required C4 seeds")
        position_path = accepted(folder + "/positions.npz")
        require(commit["artifact_sha256"]["positions.npz"] == consumed[folder + "/positions.npz"], "Positions differ from dated policy commit")
        with np.load(position_path, allow_pickle=False) as saved:
            require(list(zip(saved["seeds"].tolist(), saved["methods"].tolist())) == [tuple(s) for s in context["specs"]],
                    "Position columns differ from committed model labels")
        baseline_path = accepted(E2 + "/paths/" + stamp + ".npz")
        with np.load(baseline_path, allow_pickle=False) as saved:
            for seed in (1, 2, 3):
                require(np.count_nonzero((saved["seeds"] == seed) & (saved["methods"] == "fixed512")) == 1,
                        "Monthly baseline target columns missing")
            if common_initial is None:
                index = int(np.flatnonzero((saved["seeds"] == 1) & (saved["methods"] == "fixed512"))[0])
                common_initial = dict(decision=stamp, source=E2 + "/paths/" + stamp + ".npz", seed=1,
                                      method="fixed512", security_count=len(saved["ids"]),
                                      holdings_fingerprint=fingerprint(saved["ids"], saved["pretrade"][:, index]))
        require(f"{year-1}-12-31" <= stamp, "Monthly policy predates annual training cutoff")
        monthly.append(dict(decision=stamp, return_year=year, cutoff=f"{year-1}-12-31"))
        previous = consumed[folder + "/commit.json"]
    require(sha(ROOT / CONFIG) == config_sha, "Screen specification changed during audit")
    result = dict(status="PASSED_ACCEPTED_TARGET_CHRONOLOGY_AUDIT", passed=True,
                  artifact_sha256=consumed, protocol_sha256=config_sha,
                  audit_implementation_sha256=sha(__file__), completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  training_launched=False, portfolio_paths_executed=False, performance_computed=False,
                  decision_months=120, configured_paths=9, fixed_candidates=["CENS", "HALF"],
                  c4_selection_available_at="2015-12-31", c4_multipliers_by_seed={"1": 1, "2": 1, "3": 1},
                  temporal_parent_selection_available_at=parent_selection["selection_available_at"],
                  annual_models_checked=annual_checks, monthly_policy_checks=monthly,
                  policy_identity=policy_identity, common_initial_source=common_initial,
                  archived_c4_pretrade_holdings_not_inherited=True,
                  common_initial_application_verified=False,
                  common_initial_scope="Configured source verified; financial runner must initialize all nine new paths from this B1 vector and charge first switch",
                  month_end_selection_and_first_decision_same_date_permitted=True,
                  annual_dispatch_evidence="Dated accepted commits bind global policy; accepted extension4/run.py dispatches year=(decision+MonthEnd(1)).year, and ConditionalModel.aim checks annual year/cutoff. Commit contexts do not contain separate per-month model IDs.",
                  saved_targets_independently_regenerated=False,
                  point_in_time_certified=False, source_publication_timestamps_certified=False,
                  source_revision_vintages_certified=False, all_outcomes_previously_exposed=True,
                  chronology_claim="Annual estimation and frozen-selection chronology at source monthly resolution, conditional on accepted targets; not raw-data publication-timing certification")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, months=120, annual_model_pairs=30, verified_artifacts=len(consumed),
                         portfolio_paths_executed=False, performance_computed=False, receipt=str(OUTPUT))))


if __name__ == "__main__":
    main()
