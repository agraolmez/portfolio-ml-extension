"""Dated, unranked opportunity states; proxy timing is never called publication."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from feature_geometry.features import array, fingerprint
from temporal_extension.reference import month_key


def day(value):
    month_key(value)
    return pd.Timestamp(value).date().isoformat()


@dataclass(frozen=True)
class BoundState:
    dates: tuple
    values: np.ndarray
    book_identity: str
    names: tuple

    def __post_init__(self):
        object.__setattr__(self, 'dates', tuple(day(d) for d in self.dates))
        object.__setattr__(self, 'values', array(self.values, 2))
        if self.values.shape != (len(self.dates), len(self.names)) or not self.book_identity:
            raise ValueError('Aligned state dates, names and provenance required')
        object.__setattr__(self, '_identity', fingerprint(self.dates, self.values, self.names, self.book_identity))

    def validate(self, decision_date, *, history, expected_identity):
        expected = tuple(day(pd.Timestamp(decision_date)-pd.offsets.MonthEnd(i)) for i in range(history))
        if (self.dates != expected or self.book_identity != expected_identity
                or self._identity != fingerprint(self.dates, self.values, self.names, self.book_identity)):
            raise ValueError('State date/history/source binding mismatch')
        return self.values


class StateBook:
    def __init__(self, frame, metadata):
        self.frame = frame.copy(deep=True).reset_index(drop=True)
        self.metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
        self.names = tuple(self.metadata['state_names'])
        if len(self.names) != 1 or self.names != ('value_dispersion',):
            raise ValueError('This implementation requires the explicit single-state protocol')
        dates = pd.to_datetime(self.frame.eom)
        if (dates.empty or dates.duplicated().any() or
                list(dates) != list(pd.date_range(dates.iloc[0], dates.iloc[-1], freq='ME'))):
            raise ValueError('A complete increasing calendar, including missing months, is required')
        for date in dates: month_key(date)
        if not np.isfinite(self.frame.normalized).all():
            raise ValueError('Inactive state must be an explicit finite zero')
        self._identity = self._snapshot()
        self._locations = {day(d): i for i, d in enumerate(dates)}

    def _snapshot(self):
        return fingerprint(pd.util.hash_pandas_object(self.frame, index=True).to_numpy(),
                           tuple(self.frame.columns), tuple(map(str, self.frame.dtypes)), self.metadata)

    @property
    def identity(self):
        if self._snapshot() != self._identity:
            raise ValueError('State history or provenance changed after construction')
        return self._identity

    def bind(self, decision_date, *, history=1):
        if type(history) is not int or history not in (1, 13):
            raise ValueError('Use one prediction state or thirteen training slices')
        dates = tuple(day(pd.Timestamp(decision_date)-pd.offsets.MonthEnd(i)) for i in range(history))
        if any(d not in self._locations for d in dates):
            raise ValueError('State calendar does not contain the requested history; no inferred backfill')
        values = np.array([[self.frame.normalized.iloc[self._locations[d]]] for d in dates])
        return BoundState(dates, values, self.identity, self.names)


def build_states(raw, eligible, *, definition, normalizer, source_metadata,
                 first, last, availability_column=None):
    """The caller owns authorization. Tests supply artificial panels.

    In SOURCE_LAG_PROXY_UNCERTIFIED mode, rows are dated by their documented
    source month. No publication timestamp is fabricated. If real availability
    timestamps are supplied, their origin-date eligibility is enforced too.
    """
    if (source_metadata.get('unranked') is not True or
            source_metadata.get('timing_mode') not in ('SOURCE_LAG_PROXY_UNCERTIFIED', 'OBSERVED_AVAILABILITY')):
        raise ValueError('Explicit unranked source and timing qualification required')
    if source_metadata['timing_mode'] == 'OBSERVED_AVAILABILITY' and availability_column is None:
        raise ValueError('Observed availability requires the actual timestamp field')
    field = definition['field']
    if field != 'be_me' or definition['name'] != 'value_dispersion':
        raise ValueError('Unsupported state definition; requires a reviewed implementation')
    keys = ['id', 'eom']
    for frame in (raw, eligible):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('Duplicate or missing source keys')
        for value in frame.eom.unique(): month_key(value)
    if eligible.valid.dtype != bool:
        raise ValueError('Explicit parent eligibility mask required')
    cols = keys + [field, 'me'] + ([] if availability_column is None else [availability_column])
    joined = eligible.loc[eligible.valid, keys].merge(raw[cols], on=keys, how='left',
                                                      validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Eligible parent observation lacks its raw source row')
    groups = {pd.Timestamp(d): g for d, g in joined.groupby('eom', sort=True)}
    rows = []
    for date in pd.date_range(day(first), day(last), freq='ME'):
        g = groups.get(date)
        size = 0 if g is None else len(g)
        count = unavailable = 0
        lo = hi = value = np.nan
        if size:
            x, me = g[field].to_numpy(float), g.me.to_numpy(float)
            valid = np.isfinite(x) & (x > 0) & np.isfinite(me) & (me > 0)
            if availability_column is not None:
                timestamps = pd.to_datetime(g[availability_column], utc=True)
                known = timestamps.notna() & (timestamps <= date.tz_localize('UTC'))
                unavailable = int((~known).sum())
                valid &= known.to_numpy()
            count = int(valid.sum())
            if count:
                lo, hi = np.quantile(x[valid], definition['quantiles'], method=definition['quantile_method'])
        coverage = count/size if size else 0.
        reason = 'OK'
        if count < definition['min_count']: reason = 'INSUFFICIENT_OBSERVATIONS'
        elif coverage < definition['min_coverage']: reason = 'INSUFFICIENT_COVERAGE'
        else: value = float(hi-lo)
        rows.append(dict(eom=date, raw_state=value, q_low=lo, q_high=hi,
                         eligible_count=size, valid_count=count, coverage=coverage,
                         unavailable_count=unavailable, raw_reason=reason))
    frame = pd.DataFrame(rows)
    window, minimum = normalizer['previous_calendar_rows'], normalizer['min_finite_prior']
    if (not normalizer['exclude_current'] or not normalizer['preserve_gaps'] or normalizer['ddof'] != 0
            or normalizer['inactive_value'] != 0. or window < minimum or minimum < 2
            or not 0 < normalizer['scale_floor'] or not 0 < normalizer['clip']):
        raise ValueError('Invalid causal standardization contract')
    values = frame.raw_state.to_numpy()
    normal = []
    for i, current in enumerate(values):
        history = values[max(0, i-window):i]
        history = history[np.isfinite(history)]
        mean = float(history.mean()) if len(history) else np.nan
        scale = float(history.std(ddof=0)) if len(history) else np.nan
        reason = 'OK'; z = unbounded = 0.
        if not np.isfinite(current): reason = 'MISSING_CURRENT'
        elif len(history) < minimum: reason = 'INSUFFICIENT_HISTORY'
        elif not np.isfinite(scale) or scale <= normalizer['scale_floor']: reason = 'DEGENERATE_HISTORY'
        else:
            unbounded = (current-mean)/scale
            z = float(np.clip(unbounded, -normalizer['clip'], normalizer['clip']))
        normal.append(dict(historical_mean=mean, historical_scale=scale, historical_count=len(history),
                           normalized=z, active=reason == 'OK', inactive_reason=reason,
                           clipped=bool(reason == 'OK' and unbounded != z)))
    metadata = dict(source_metadata, state_names=[definition['name']], definition=definition,
                    normalizer=normalizer, first=day(first), last=day(last),
                    actual_availability_field=availability_column,
                    publication_timestamps_created=False)
    return StateBook(pd.concat([frame, pd.DataFrame(normal)], axis=1), metadata)
