"""Read-only adapter to the accepted reconstruction; never calls writable Engine caches."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import FeatureMap, fingerprint
from .moments import MarketMoment, TrainingSample


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalHistory:
    """Load only the required dates; explicitly verify before any historical use."""
    def __init__(self, root, first_decision, last_decision, preflight_receipt):
        self.root = Path(root).resolve()
        self.first, self.last = pd.Timestamp(first_decision), pd.Timestamp(last_decision)
        preparation = self.root / "output/step6_20260911/preparation_manifest.json"
        if (preflight_receipt.get("preparation_manifest_sha256") != sha(preparation)
                or preflight_receipt.get("financial_path_run") is not False):
            raise ValueError("A fresh accepted financial preflight is required")
        self.preparation = json.loads(preparation.read_text())["artifact_sha256"]
        source = json.loads((self.root / "output/step3_author_conventions_20260911/bundle_checksums.json").read_text())
        self.sources = {**source, **self.preparation}
        settings = json.loads((self.root / "output/step2_20260910/source_settings.json").read_text())
        self.features = settings["features"]
        self.source_id = fingerprint(preflight_receipt["preparation_manifest_sha256"], self.features,
                                     sha(self.root / "config/protocol_v1.json"))
        self.gamma = json.loads((self.root / "config/protocol_v1.json").read_text())["economic"]["risk_aversion"]
        begin = self.first - pd.offsets.MonthEnd(12)
        filters = [("eom", ">=", begin), ("eom", "<=", self.last)]
        columns = ["id", "eom", "valid", "ret_ld1", "tr_ld0", "mu_ld0", "lambda_diagonal", "wealth_usd", *self.features]
        self.data = pd.read_parquet(self.root / "data/prepared/step3_author_v3/monthly_inputs.parquet", columns=columns, filters=filters)
        scales = pd.read_parquet(self.root / "data/prepared/step4_v1/risk/scales.parquet", columns=["id", "eom", "vol_scale"], filters=filters)
        self.data = self.data.merge(scales, on=["id", "eom"], how="left", validate="one_to_one")
        self.data["vol_scale"] = self.data.vol_scale.fillna(self.data.groupby("eom").vol_scale.transform("median"))
        if self.data.vol_scale.isna().any():
            raise ValueError("Unresolved accepted volatility scaling")
        self.months = {pd.Timestamp(d): frame.sort_values("id").set_index("id", drop=False)
                       for d, frame in self.data.groupby("eom", sort=True)}
        self.consumed = {}

    def _accepted(self, relative):
        path = self.root / relative
        # Step 6's risk symlink points to separately accepted Step 4 files.
        canonical = str(path.resolve().relative_to(self.root))
        expected = self.sources.get(relative, self.sources.get(canonical))
        if expected is None:
            step4 = json.loads((self.root / "output/step4_20260911/acceptance_manifest.json").read_text())["artifact_sha256"]
            expected = step4.get(canonical)
        actual = sha(path)
        if expected is None or actual != expected:
            raise ValueError(f"Missing or changed accepted input: {relative}")
        self.consumed[relative] = actual
        return path

    def feature_map(self, nonconstant_features=512, bandwidth_index=0):
        if nonconstant_features not in (64, 128, 256, 512) or bandwidth_index not in (0, 1):
            raise ValueError("Original directions support only the released feature grid")
        path = self._accepted(f"output/step6_20260911/reference/rff_g{bandwidth_index}.bin")
        original = np.fromfile(path, dtype="<f8").reshape(len(self.features), 256)
        return FeatureMap(original[:, :nonconstant_features // 2],
                          f"authors_R_seed_1_g{bandwidth_index};file_sha256={sha(path)}")

    def market(self, decision_date):
        day = pd.Timestamp(decision_date)
        if day != day + pd.offsets.MonthEnd(0) or not self.first <= day <= self.last:
            raise ValueError("Decision outside the loaded month-end window")
        key = day.strftime("%Y-%m-%d")
        matrix_path = self._accepted(f"data/prepared/step6_v1/matrices/{key}.npz")
        risk_path = self._accepted(f"data/prepared/step6_v1/risk/{key}.npz")
        with np.load(matrix_path, allow_pickle=False) as data:
            ids, m = data["ids"].copy(), data["m"].copy()
        with np.load(risk_path, allow_pickle=False) as data:
            if not np.array_equal(ids, data["ids"]):
                raise ValueError("Accepted matrix/covariance security IDs differ")
            covariance = data["loadings"] @ data["factor_cov"] @ data["loadings"].T + np.diag(data["specific_variance"])
        frame = self.months[day]
        active = frame[frame.valid]
        if not np.array_equal(ids, active.id.to_numpy()):
            raise ValueError("Accepted current universe and matrix IDs differ")
        z, vol, growth = [], [], []
        for lag in range(13):
            history = self.months[day - pd.offsets.MonthEnd(lag)]
            pos = history.index.get_indexer(ids)
            if np.any(pos < 0):
                raise ValueError("Missing current-ID lookback history")
            block = history.iloc[pos]
            z.append(block[self.features].to_numpy())
            vol.append(block.vol_scale.to_numpy())
            gt = (1 + block.tr_ld0.to_numpy()) / (1 + block.mu_ld0.to_numpy())
            growth.append(np.where(np.isnan(gt), 1., gt))
        if active.wealth_usd.nunique() != 1:
            raise ValueError("Inconsistent source wealth within a date")
        return MarketMoment(key, ids, tuple(z), tuple(vol), tuple(growth), m, covariance,
                            active.lambda_diagonal.to_numpy(), float(active.wealth_usd.iloc[0]), self.gamma, self.source_id)

    def training_sample(self, decision_date, cutoff):
        day = pd.Timestamp(decision_date)
        available = day + pd.offsets.MonthEnd(1)
        if available > pd.Timestamp(cutoff):
            raise ValueError("Historical outcome has not matured by the fit cutoff")
        market = self.market(day)
        returns = self.months[day].loc[market.ids].ret_ld1.to_numpy()
        return TrainingSample(market, returns, available.strftime("%Y-%m-%d"))

    def identity_reference(self, decision_date, bandwidth_index):
        day = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
        path = self._accepted(f"data/prepared/step6_v1/moments/g{bandwidth_index}/{day}.npz")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in ("ids", "r_tilde", "risk", "tc", "signal_t")}
