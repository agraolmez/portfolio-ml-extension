"""Frozen geometry model and a dated original-rule decision; no outcome arguments."""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import numpy as np
from .features import FeatureMap, array, fingerprint


@dataclass(frozen=True)
class GeometryModel:
    feature_map: FeatureMap
    theta: np.ndarray
    beta: np.ndarray
    cutoff: str
    bound: float
    source_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "cutoff", date.fromisoformat(self.cutoff).isoformat())
        theta, beta = array(self.theta, 1), array(self.beta, 1)
        if (theta.shape != (self.feature_map.dimensions,) or beta.shape != (self.feature_map.columns,)
                or not np.isfinite(self.bound) or self.bound <= 0
                or abs(theta.sum()) > 1e-10 or np.max(abs(theta)) > self.bound + 1e-12):
            raise ValueError("Invalid frozen geometry model")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "beta", beta)

    @classmethod
    def from_fit(cls, feature_map, result):
        return cls(feature_map, result["theta"], result["beta"], result["settings"]["cutoff"],
                   result["settings"]["bound"], result["source_fingerprint"])

    def aim(self, characteristics, volatility, decision_date):
        if date.fromisoformat(decision_date) < date.fromisoformat(self.cutoff):
            raise ValueError("Model would use future training information")
        return self.feature_map.signals(characteristics, volatility, self.theta) @ self.beta

    def save(self, destination):
        metadata = dict(cutoff=self.cutoff, bound=self.bound, source_fingerprint=self.source_fingerprint,
                        frequency_source=self.feature_map.source, format_version=1)
        metadata["parameter_fingerprint"] = fingerprint(self.feature_map.frequencies, self.theta, self.beta, metadata)
        with Path(destination).open("xb") as handle:
            np.savez_compressed(handle, frequencies=self.feature_map.frequencies, theta=self.theta, beta=self.beta,
                                metadata=np.array(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, source):
        with np.load(source, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            expected = meta.pop("parameter_fingerprint")
            # Canonical metadata order is explicit because fingerprints are not pickle-based.
            canonical = {k: meta[k] for k in ("cutoff", "bound", "source_fingerprint", "frequency_source", "format_version")}
            if meta["format_version"] != 1 or fingerprint(data["frequencies"], data["theta"], data["beta"], canonical) != expected:
                raise ValueError("Changed model parameters or unsupported format")
            return cls(FeatureMap(data["frequencies"], meta["frequency_source"]), data["theta"], data["beta"],
                       meta["cutoff"], meta["bound"], meta["source_fingerprint"])


def act(model, decision_date, ids, characteristics, volatility, m, holding_ids, holdings):
    """Use the supplied strategy's ordered pretrade holdings; never reset at a refit."""
    ids, holding_ids = np.asarray(ids), np.asarray(holding_ids)
    h, m = array(holdings, 1), array(m, 2)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not np.array_equal(ids, holding_ids)
            or h.shape != ids.shape or m.shape != (len(ids), len(ids))):
        raise ValueError("Decision and inherited-holdings IDs must match in order")
    aim = model.aim(characteristics, volatility, decision_date)
    if aim.shape != h.shape:
        raise ValueError("Target/current-universe mismatch")
    weights = m @ h + (np.eye(len(h)) - m) @ aim
    return {"ids": ids.copy(), "aim": aim, "weights": weights, "trades": weights - h,
            "rule": "original m and original recursion; kappa=1"}
