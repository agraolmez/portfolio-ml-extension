"""Read-only, provenance-bound temporal history; no scientific defaults or fitting.

Ranks are read unchanged, while observation masks come from a keyed join to the
accepted pre-imputation source.  The cache retains at most 13 + L calendar months
and is independent of annual model boundaries.  Creating an adapter is not
authorization to run a historical experiment.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from feature_geometry.features import array, fingerprint
from feature_geometry.history import sha
from feature_geometry.moments import MarketMoment, TrainingSample
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from geometry_study.kernel import LeanBasis
from .features import TemporalFeatureMap
from .reference import CalendarReference, Innovation, month_key

PARENT = "output/feature_geometry_study_20260914"
RANKS = "data/prepared/step3_author_v3/monthly_inputs.parquet"
RAW = "data/prepared/step3_author_v3/source_characteristics.parquet"
SCALES = "data/prepared/step4_v1/risk/scales.parquet"


def _date(value):
    month_key(value)
    return pd.Timestamp(value).date().isoformat()


def _ids(value):
    ids = np.array(value, copy=True)
    if ids.ndim != 1 or pd.isna(ids).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique nonmissing ordered security IDs required")
    ids.setflags(write=False)
    return ids


@dataclass(frozen=True)
class BoundInnovation:
    innovation: Innovation
    date: str
    ids: np.ndarray
    input_fingerprint: str
    source_fingerprint: str
    reference_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "date", _date(self.date))
        object.__setattr__(self, "ids", _ids(self.ids))
        if (not isinstance(self.innovation, Innovation)
                or len(self.ids) != len(self.innovation.values)
                or not all(isinstance(s, str) and s for s in (
                    self.input_fingerprint, self.source_fingerprint, self.reference_fingerprint))):
            raise ValueError("Complete innovation binding required")

    @property
    def fingerprint(self):
        return fingerprint(self.date, self.ids, self.input_fingerprint,
                           self.source_fingerprint, self.reference_fingerprint,
                           self.innovation.values, self.innovation.valid)

    def validate(self, date, ids, z, source_fingerprint, reference_fingerprint):
        """Fail closed on reuse for another date, ordering, input or reference."""
        if (_date(date) != self.date or not np.array_equal(_ids(ids), self.ids)
                or fingerprint(array(z, 2)) != self.input_fingerprint
                or source_fingerprint != self.source_fingerprint
                or reference_fingerprint != self.reference_fingerprint):
            raise ValueError("Innovation date/ID/input/source/reference binding mismatch")
        return self.innovation


class BoundReference:
    """Provenance wrapper for a supplied CalendarReference (also useful in tests)."""
    def __init__(self, reference, *, source_fingerprint):
        if not isinstance(reference, CalendarReference) or not source_fingerprint:
            raise ValueError("A calendar reference and explicit source identity are required")
        self.reference = reference
        self.source_fingerprint = str(source_fingerprint)
        self.reference_fingerprint = fingerprint(
            "strict_calendar_lags_source_mask_bypass_v1", reference.weights,
            reference.dimensions, self.source_fingerprint)
        self._content = self._snapshot()

    def _snapshot(self):
        return fingerprint(self.reference.weights, *(
            fingerprint(key, value, mask)
            for key, (value, mask) in self.reference.rows.items()))

    def at(self, date, ids, z):
        if self._snapshot() != self._content:
            raise ValueError("Bound reference was mutated")
        return self._at(date, ids, z)

    def _at(self, date, ids, z):
        z = array(z, 2)
        for sid, current in zip(_ids(ids), z):
            row = self.reference.rows.get((month_key(date), sid))
            if row is not None and not np.array_equal(row[0], current):
                raise ValueError("Current input differs from archived rank, including imputed cells")
        return BoundInnovation(self.reference.at(date, ids, z), _date(date), ids,
                               fingerprint(z), self.source_fingerprint,
                               self.reference_fingerprint)

    def for_market(self, market):
        if self._snapshot() != self._content:
            raise ValueError("Bound reference was mutated")
        return tuple(self._at(pd.Timestamp(market.decision_date) - pd.offsets.MonthEnd(lag),
                             market.ids, z)
                     for lag, z in enumerate(market.characteristics))

    def validate_market(self, market, bound):
        if len(bound) != 13:
            raise ValueError("Thirteen dated innovation bindings required")
        return tuple(item.validate(pd.Timestamp(market.decision_date) - pd.offsets.MonthEnd(lag),
                                   market.ids, z, self.source_fingerprint,
                                   self.reference_fingerprint)
                     for lag, (item, z) in enumerate(zip(bound, market.characteristics)))


class _AcceptedFiles:
    def __init__(self, root, inventory):
        self.root = Path(root).resolve()
        self.inventory = dict(inventory)
        self.consumed = {}
        self._stats = {}

    def path(self, relative):
        path = self.root / relative
        canonical = str(path.resolve().relative_to(self.root))
        expected = self.inventory.get(relative, self.inventory.get(canonical))
        if expected is None:
            raise ValueError("Unaccepted source: " + relative)
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if relative not in self._stats:
            if sha(path) != expected:
                raise ValueError("Changed accepted source: " + relative)
            self._stats[relative] = stamp
            self.consumed[relative] = expected
        elif stamp != self._stats[relative]:
            raise ValueError("Accepted source changed during use: " + relative)
        return path

    def unchanged(self):
        for relative in tuple(self.consumed):
            self.path(relative)


@dataclass(frozen=True)
class VerifiedParent:
    """Saved Extension-2 D, frequencies, coefficients and economic fit settings."""
    year: int
    seed: int
    model: GeometryModel
    feature_map: TemporalFeatureMap
    settings: FitSettings
    objective_scale: float
    metadata: object
    hashes: object

    @property
    def fingerprint(self):
        return fingerprint(self.year, self.seed, self.feature_map.fingerprint,
                           self.model.beta, self.model.cutoff, self.settings,
                           self.objective_scale, dict(self.hashes))

    def assert_cutoff(self, cutoff):
        if _date(cutoff) != self.model.cutoff:
            raise ValueError("Parent cutoff mismatch; later parent cannot enter an earlier fold")

    def assert_decision(self, decision_date):
        day = pd.Timestamp(_date(decision_date))
        if (day < pd.Timestamp(self.model.cutoff)
                or (day + pd.offsets.MonthEnd(1)).year != self.year):
            raise ValueError("Parent is not the archived annual model for this decision")


def load_parent(root, year, seed, *, expected_cutoff):
    """Load exactly the accepted full learned-scale family; never reconstruct D."""
    if type(year) is not int or year not in range(1981, 2026) or type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError("Archived parent requires year 1981–2025 and seed 1, 2 or 3")
    root = Path(root).resolve()
    if _date(expected_cutoff) != f"{year - 1}-12-31":
        raise ValueError("Parent cutoff mismatch; later parent cannot enter an earlier fold")
    accepted = json.loads((root / PARENT / "acceptance_manifest.json").read_text())
    if not accepted.get("accepted") or not accepted.get("study_complete"):
        raise ValueError("Accepted completed Extension-2 parent required")
    files = _AcceptedFiles(root, accepted["artifact_sha256"])
    lock = json.loads(files.path(f"{PARENT}/plan_lock.json").read_text())
    fits = json.loads(files.path(f"{PARENT}/fits_manifest.json").read_text())
    files.inventory.update(fits["consumed_input_sha256"])
    step9_rel = "output/step9_20260913/acceptance_manifest.json"
    files.inventory.setdefault(step9_rel, lock["step9_acceptance_sha256"])
    step9 = json.loads(files.path(step9_rel).read_text())
    if not all(gate["passed"] for gate in step9["gates"]):
        raise ValueError("Step 9 accepted selection has failed gates")
    files.inventory.update(step9["artifact_sha256"])
    for rel, expected in lock["code_sha256"].items():
        files.inventory.setdefault(rel, expected)
        files.path(rel)
    folder = f"{PARENT}/fits/{year}/seed{seed}/full"
    meta = json.loads(files.path(f"{folder}/fit.json").read_text())
    model = GeometryModel.load(files.path(f"{folder}/model.npz"))
    selection = pd.read_parquet(files.path("data/prepared/step9_v1/baseline/annual_selection.parquet"))
    rows = selection[selection.return_eom.eq(pd.Timestamp(expected_cutoff))]
    if len(rows) != 1:
        raise ValueError("Unique accepted parent annual selection required")
    row = rows.iloc[0]
    dictionary_path = "config/feature_dictionary_v1.csv"
    settings_path = "output/step2_20260910/source_settings.json"
    preflight_rel = f"{PARENT}/financial_preflight.json"
    files.inventory.setdefault(preflight_rel, lock["financial_preflight_sha256"])
    receipt = json.loads(files.path(preflight_rel).read_text())
    bundle_rel = "output/step3_author_conventions_20260911/bundle_checksums.json"
    files.inventory.setdefault(bundle_rel, receipt["dependencies"]["source"]["checksums_sha256"])
    bundle = json.loads(files.path(bundle_rel).read_text())
    files.inventory.update({key: bundle[key] for key in (dictionary_path, settings_path)})
    names = json.loads(files.path(settings_path).read_text())["features"]
    dictionary = pd.read_csv(files.path(dictionary_path))
    if (dictionary.characteristic.tolist() != names or len(names) != 115
            or dictionary.cluster.isna().any() or dictionary.cluster.nunique() != 13):
        raise ValueError("Accepted ordered 115-characteristic / 13-theme map required")
    groups = dictionary.cluster.map({name: i for i, name in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
    direction_rel = f"{PARENT}/directions/seed{seed}_g{int(row.g_index)}.bin"
    files.inventory.setdefault(direction_rel, lock["directions_sha256"][direction_rel])
    directions = np.fromfile(files.path(direction_rel), dtype="<f8").reshape(115, 512)[:, :256]
    valid_locks = {sha(root / PARENT / "plan_lock.json")}
    for path in sorted((root / PARENT / "engineering_versions").glob("*/plan_lock.json")):
        if sha(path) != meta["plan_lock_sha256"]:
            continue
        relative = str(path.relative_to(root))
        files.inventory.setdefault(relative, meta["plan_lock_sha256"])
        old = json.loads(files.path(relative).read_text())
        if old["plan"] != lock["plan"] or old["directions_sha256"] != lock["directions_sha256"]:
            raise ValueError("Parent engineering versions disagree scientifically")
        valid_locks.add(sha(path))
    settings = FitSettings(**meta["settings"])
    if (meta["method"] != "full" or meta["year"] != year or meta["seed"] != seed
            or meta["cutoff"] != expected_cutoff or model.cutoff != expected_cutoff
            or settings.cutoff != expected_cutoff or int(row.p) != 512
            or settings.ridge != float(row["lambda"]) or meta["coefficient_ridge"] != settings.ridge
            or meta["bandwidth_index"] != int(row.g_index)
            or settings.bound != model.bound or meta["bound"] != model.bound
            or meta["source_fingerprint"] != model.source_fingerprint
            or meta["training_first"] != "1965-11-30"
            or meta["training_last"] != f"{year - 1}-11-30"
            or meta["latest_available_at"] != expected_cutoff
            or meta["training_months"] != 181 + 12 * (year - 1981)
            or meta["plan_lock_sha256"] not in valid_locks
            or model.theta.shape != (115,) or model.beta.shape != (513,)
            or not np.array_equal(model.feature_map.frequencies, directions)
            or not np.isfinite(meta["objective_normalizer"]) or meta["objective_normalizer"] <= 0):
        raise ValueError("Archived learned-scale parent provenance mismatch")
    return VerifiedParent(year, seed, model, TemporalFeatureMap(model.feature_map, model.theta, groups),
                          settings, float(meta["objective_normalizer"]),
                          MappingProxyType(meta), MappingProxyType(dict(files.consumed)))


class AcceptedTemporalHistory:
    """Stream accepted months; retain calendar history across annual parent changes."""
    def __init__(self, root, preflight_receipt, *, weights):
        self.root = Path(root).resolve()
        self.weights = array(weights, 1)
        # Reuse the strict weight validation without inventing scientific defaults.
        CalendarReference([], [], np.empty((0, 1)), np.empty((0, 1), bool), self.weights)
        preparation = self.root / "output/step6_20260911/preparation_manifest.json"
        if (preflight_receipt.get("preparation_manifest_sha256") != sha(preparation)
                or preflight_receipt.get("financial_path_run") is not False):
            raise ValueError("Matching accepted financial preflight required")
        accepted = json.loads((self.root / PARENT / "acceptance_manifest.json").read_text())
        if not accepted.get("accepted") or not accepted.get("study_complete"):
            raise ValueError("Accepted completed parent study required")
        inventory = dict(accepted["artifact_sha256"])
        lock_rel = f"{PARENT}/plan_lock.json"
        if sha(self.root / lock_rel) != inventory[lock_rel]:
            raise ValueError("Changed accepted parent lock")
        lock = json.loads((self.root / lock_rel).read_text())
        anchors = {
            "output/step3_author_conventions_20260911/bundle_checksums.json":
                preflight_receipt["dependencies"]["source"]["checksums_sha256"],
            "output/step9_20260913/acceptance_manifest.json": lock["step9_acceptance_sha256"],
        }
        for rel in ("output/step3_author_conventions_20260911/bundle_checksums.json",
                    "output/step4_20260911/acceptance_manifest.json",
                    "output/step9_20260913/acceptance_manifest.json"):
            if rel in anchors and sha(self.root / rel) != anchors[rel]:
                raise ValueError("Changed accepted source receipt: " + rel)
            manifest = json.loads((self.root / rel).read_text())
            if "gates" in manifest and not all(gate["passed"] for gate in manifest["gates"]):
                raise ValueError("Accepted dependency has failed gates: " + rel)
            inventory.update(manifest.get("artifact_sha256", manifest))
        protected = json.loads((self.root / "output/step3_20260911/start.json").read_text())["protected"]
        inventory.update(protected)
        fit_path = self.root / PARENT / "fits_manifest.json"
        if sha(fit_path) != accepted["artifact_sha256"][f"{PARENT}/fits_manifest.json"]:
            raise ValueError("Changed parent source inventory")
        inventory.update(json.loads(fit_path.read_text())["consumed_input_sha256"])
        self.files = _AcceptedFiles(self.root, inventory)
        settings_rel = "output/step2_20260910/source_settings.json"
        self.features = tuple(json.loads(self.files.path(settings_rel).read_text())["features"])
        dictionary = pd.read_csv(self.files.path("config/feature_dictionary_v1.csv"))
        if (dictionary.characteristic.tolist() != list(self.features) or len(self.features) != 115
                or dictionary.cluster.isna().any() or dictionary.cluster.nunique() != 13):
            raise ValueError("Accepted characteristic/theme order mismatch")
        for relative in (RANKS, RAW, SCALES, "config/protocol_v1.json"):
            self.files.path(relative)
        protocol = json.loads(self.files.path("config/protocol_v1.json").read_text())
        self.gamma = protocol["economic"]["risk_aversion"]
        self.source_id = fingerprint(preflight_receipt["preparation_manifest_sha256"], list(self.features),
                                     sha(self.root / "config/protocol_v1.json"))
        self.source_fingerprint = fingerprint("accepted_unscaled_ranks_source_NaN_mask_v1", self.features,
                                              self.files.consumed[RANKS], self.files.consumed[RAW])
        self.reference_fingerprint = fingerprint("strict_calendar_lags_source_mask_bypass_v1",
                                                 self.weights, len(self.features), self.source_fingerprint)
        self.cache_limit = 13 + len(self.weights)
        self._months = OrderedDict()

    @property
    def cached_months(self):
        return tuple(self._months)

    def _month(self, date):
        day = pd.Timestamp(_date(date))
        self.files.unchanged()
        if day in self._months:
            self._months.move_to_end(day)
            return self._months[day]
        filters = [("eom", "==", day)]
        columns = ["id", "eom", "valid", "ret_ld1", "tr_ld0", "mu_ld0", "tr_ld1", "mu_ld1",
                   "lambda_diagonal", "wealth_usd", *self.features]
        ranks = pd.read_parquet(self.files.path(RANKS), columns=columns, filters=filters)
        raw = pd.read_parquet(self.files.path(RAW), columns=["id", "eom", *self.features], filters=filters)
        scales = pd.read_parquet(self.files.path(SCALES), columns=["id", "eom", "vol_scale"], filters=filters)
        frame, observed = _join_month(ranks, raw, scales, self.features)
        self._months[day] = (frame, observed)
        while len(self._months) > self.cache_limit:
            self._months.popitem(last=False)
        return self._months[day]

    def market(self, decision_date):
        day = pd.Timestamp(_date(decision_date)); key = day.date().isoformat()
        base = "data/prepared/step9_v1/baseline"
        with np.load(self.files.path(f"{base}/matrices/{key}.npz"), allow_pickle=False) as data:
            ids, m = data["ids"].copy(), data["m"].copy()
        with np.load(self.files.path(f"{base}/risk/{key}.npz"), allow_pickle=False) as data:
            if not np.array_equal(ids, data["ids"]):
                raise ValueError("Risk/matrix ordered IDs differ")
            covariance = data["loadings"] @ data["factor_cov"] @ data["loadings"].T + np.diag(data["specific_variance"])
        current, _ = self._month(day)
        active = current[current.valid]
        if not np.array_equal(ids, active.id) or active.wealth_usd.nunique() != 1:
            raise ValueError("Parent eligibility/matrix or wealth mismatch")
        z, vol, growth = [], [], []
        for lag in range(13):
            frame, _ = self._month(day - pd.offsets.MonthEnd(lag))
            if np.any(frame.index.get_indexer(ids) < 0):
                raise ValueError("Missing parent current-ID predictor history")
            block = frame.loc[ids]
            z.append(block[list(self.features)].to_numpy()); vol.append(block.vol_scale.to_numpy())
            gt = (1 + block.tr_ld0.to_numpy()) / (1 + block.mu_ld0.to_numpy())
            growth.append(np.where(np.isnan(gt), 1., gt))
        return MarketMoment(key, ids, tuple(z), tuple(vol), tuple(growth), m, covariance,
                            active.lambda_diagonal.to_numpy(), float(active.wealth_usd.iloc[0]),
                            self.gamma, self.source_id)

    def innovations(self, market):
        if market.source_id != self.source_id:
            raise ValueError("Market source differs from accepted temporal history")
        dates, ids, values, masks = [], [], [], []
        day = pd.Timestamp(market.decision_date)
        # Oldest first preserves the most recent months at the MRU cache end.
        for lag in reversed(range(self.cache_limit)):
            date = day - pd.offsets.MonthEnd(lag)
            frame, observed = self._month(date)
            block = frame[frame.id.isin(market.ids)]
            dates.extend([date] * len(block)); ids.extend(block.id.tolist())
            values.extend(block[list(self.features)].to_numpy()); masks.extend(observed.loc[block.id].to_numpy())
        reference = BoundReference(CalendarReference(dates, ids,
            np.asarray(values).reshape(-1, len(self.features)),
            np.asarray(masks, dtype=bool).reshape(-1, len(self.features)), self.weights),
            source_fingerprint=self.source_fingerprint)
        return reference.for_market(market)

    def validate_market(self, market, bound):
        if market.source_id != self.source_id or len(bound) != 13:
            raise ValueError("Accepted market and thirteen bound innovation slices required")
        return tuple(item.validate(pd.Timestamp(market.decision_date) - pd.offsets.MonthEnd(lag),
                                   market.ids, z, self.source_fingerprint, self.reference_fingerprint)
                     for lag, (item, z) in enumerate(zip(bound, market.characteristics)))

    def training_sample(self, decision_date, cutoff):
        day = pd.Timestamp(_date(decision_date)); available = day + pd.offsets.MonthEnd(1)
        if available > pd.Timestamp(_date(cutoff)):
            raise ValueError("Unmatured training outcome at parent cutoff")
        market = self.market(day)
        frame, _ = self._month(day)
        return TrainingSample(market, frame.loc[market.ids].ret_ld1.to_numpy(), available.date().isoformat())

    def batch_factory(self, parent, first, last):
        """Explicit range can be bounded for a later approved pilot, never future data."""
        if not isinstance(parent, VerifiedParent):
            raise ValueError("A verified immutable archived parent is required")
        first, last = pd.Timestamp(_date(first)), pd.Timestamp(_date(last))
        if (first > last or first < pd.Timestamp(parent.metadata["training_first"])
                or last > pd.Timestamp(parent.metadata["training_last"])):
            raise ValueError("Training range exceeds the archived parent cutoff")
        def batches():
            for day in pd.date_range(first, last, freq="ME"):
                sample = self.training_sample(day, parent.model.cutoff)
                bound = self.innovations(sample.market)
                yield LeanBasis(sample), self.validate_market(sample.market, bound)
        return batches


def _join_month(ranks, raw, scales, features):
    """Testable exact-key source mask join; rank 0.5 never implies missingness."""
    keys = ["id", "eom"]
    for frame in (ranks, raw, scales):
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError("Duplicate or missing security/calendar-month keys")
    if not ranks.empty and ranks.eom.nunique() != 1:
        raise ValueError("Month cache must contain exactly one calendar month")
    ranks = ranks.sort_values("id").copy()
    source = raw.set_index(keys)
    positions = source.index.get_indexer(pd.MultiIndex.from_frame(ranks[keys]))
    if np.any(positions < 0):
        raise ValueError("Prepared rank has no matching source-mask key")
    values = source.iloc[positions][list(features)].to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(ranks[list(features)].to_numpy()).all():
        raise ValueError("Nonfinite accepted rank/source value")
    observed = pd.DataFrame(~np.isnan(values), index=ranks.id.to_numpy(), columns=features)
    frame = ranks.merge(scales, on=keys, how="left", validate="one_to_one")
    frame["vol_scale"] = frame.vol_scale.fillna(frame.vol_scale.median())
    if frame.vol_scale.isna().any() or (frame.vol_scale <= 0).any():
        raise ValueError("Unresolved accepted volatility scaling")
    return frame.set_index("id", drop=False), observed
