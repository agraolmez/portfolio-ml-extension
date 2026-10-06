"""Explicit-mask, calendar-lagged references for unscaled parent characteristics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from feature_geometry.features import array


def month_key(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day != day.normalize() or day != day + pd.offsets.MonthEnd(0):
        raise ValueError("A calendar month-end date is required")
    return day.to_period("M").ordinal


@dataclass(frozen=True)
class Innovation:
    values: np.ndarray
    valid: np.ndarray
    reasons: dict

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.valid)
        if values.ndim != 2 or valid.shape != values.shape or valid.dtype != bool:
            raise ValueError("Aligned numerical innovations and explicit Boolean mask required")
        if not np.isfinite(values[valid]).all():
            raise ValueError("Nonfinite innovation marked valid")
        # Select before multiplication: NaN * 0 would not be an identity.
        object.__setattr__(self, "values", array(np.where(valid, values, 0.), 2))
        valid = valid.copy()
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "reasons", dict(self.reasons))


class CalendarReference:
    """No default lookback, weights, inferred observation masks, or row shifting.

    Values are the archived *unscaled* parent ranks. `observed` is supplied
    separately from the genuine source-observation mask. This engineering
    implementation is not an approval of a historical missingness protocol.
    """
    def __init__(self, dates, ids, values, observed, weights):
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed)
        weights = array(weights, 1)
        if (values.ndim != 2 or len(dates) != len(ids) or len(ids) != len(values)
                or observed.shape != values.shape or observed.dtype != bool
                or not len(weights) or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-14)
                or not np.isfinite(values[observed]).all()):
            raise ValueError("Invalid calendar panel, explicit mask or reference weights")
        self.weights = weights
        self.dimensions = values.shape[1]
        self.rows = {}
        for day, sid, value, mask in zip(dates, ids, values, observed):
            if pd.isna(sid):
                raise ValueError("Missing security identifier")
            key = (month_key(day), sid)
            if key in self.rows:
                raise ValueError("Duplicate security/calendar-month key")
            self.rows[key] = (value.copy(), mask.copy())

    def at(self, date, ids, parent_z):
        z = array(parent_z, 2)
        ids = np.asarray(ids)
        if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
                or z.shape != (len(ids), self.dimensions)):
            raise ValueError("Current IDs and parent inputs must align")
        month = month_key(date)
        level = np.zeros_like(z)
        current_bad = np.zeros_like(z, dtype=bool)
        history_bad = np.zeros_like(z, dtype=bool)
        missing_row = np.zeros_like(z, dtype=bool)
        for i, sid in enumerate(ids):
            current = self.rows.get((month, sid))
            if current is None:
                current_bad[i] = True
            else:
                value, mask = current
                current_bad[i] = ~mask
                if not np.array_equal(value[mask], z[i, mask]):
                    raise ValueError("Current archived input differs from supplied parent input")
            for lag, weight in enumerate(self.weights, start=1):
                if weight == 0:
                    continue
                past = self.rows.get((month - lag, sid))
                if past is None:
                    history_bad[i] = True
                    missing_row[i] = True
                else:
                    value, mask = past
                    history_bad[i] |= ~mask
                    level[i] += weight * np.where(mask, value, 0.)
        valid = ~(current_bad | history_bad)
        # Counts overlap deliberately; valid + bypassed equals the cell count.
        reasons = dict(cells=z.size, adjusted_eligible=int(valid.sum()),
                       bypassed=int((~valid).sum()), current_unobserved=int(current_bad.sum()),
                       prior_unobserved_or_absent=int(history_bad.sum()),
                       prior_security_month_absent=int(missing_row.sum()))
        return Innovation(z - level, valid, reasons)
