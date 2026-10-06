"""Fixed signed-theme Legendre representation; no fitted scales or outcomes.

Production callers must verify the accepted dictionary's source hash before
constructing this immutable map. Missing ranks retain their neutral 0.5 input;
zero ranks are observed values and are never replaced by missing-value logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from baseline.core import normalize_signal

ACCEPTED_DICTIONARY_SHA256='2cf7976d62becc8dfcc3b3dafedad39ee9362d555575584b8c91f072b3f1dcaf'
THEMES=('accruals','debt_issuance','investment','low_leverage','low_risk','momentum',
        'profit_growth','profitability','quality','seasonality','short_term_reversal','size','value')


@dataclass(frozen=True)
class ThemeDictionary:
    characteristics: tuple[str,...]
    themes: tuple[str,...]
    groups: tuple[int,...]
    directions: tuple[float,...]
    sizes: tuple[int,...]

    @classmethod
    def from_rows(cls,rows,characteristic_order):
        """Validate ordered dictionary records without reading any data file."""
        rows=tuple(rows);order=tuple(characteristic_order)
        if (len(rows)!=115 or len(order)!=115 or len(set(order))!=115
                or tuple(int(r['position']) for r in rows)!=tuple(range(1,116))
                or tuple(r['characteristic'] for r in rows)!=order):
            raise ValueError('The ordered 115-characteristic dictionary is required')
        names=tuple(sorted({r['cluster'] for r in rows}))
        if names!=THEMES:raise ValueError('The fixed thirteen theme names are required')
        direction=tuple(float(r['direction']) for r in rows)
        if any(value not in (-1.,1.) for value in direction):raise ValueError('Each characteristic must have fixed direction -1 or +1')
        if any(float(r['rank_missing_value'])!=.5 or str(r['preserve_exact_zero']).lower()!='true' for r in rows):
            raise ValueError('Accepted missing-rank and exact-zero conventions required')
        groups=tuple(names.index(r['cluster']) for r in rows)
        sizes=tuple(groups.count(i) for i in range(13))
        return cls(order,names,groups,direction,sizes)

    def __post_init__(self):
        if (len(self.characteristics)!=115 or len(set(self.characteristics))!=115
                or self.themes!=THEMES or len(self.groups)!=115 or len(self.directions)!=115
                or len(self.sizes)!=13 or any(g not in range(13) for g in self.groups)
                or any(x not in (-1.,1.) for x in self.directions)
                or self.sizes!=tuple(self.groups.count(i) for i in range(13)) or min(self.sizes)<=0):
            raise ValueError('Invalid immutable theme map')


def theme_scores(ranks,dictionary):
    """Average all signed centered ranks in each fixed characteristic group."""
    if not isinstance(dictionary,ThemeDictionary):raise ValueError('Validated ThemeDictionary required')
    ranks=np.asarray(ranks,dtype=float)
    if (ranks.ndim!=2 or ranks.shape[1]!=115 or len(ranks)==0 or np.isinf(ranks).any()
            or np.any(ranks<0) or np.any(ranks>1)):
        raise ValueError('Nonempty stock-by-115 ranks in [0,1], or missing NaN, required')
    imputed=np.where(np.isnan(ranks),.5,ranks)
    oriented=(2*imputed-1)*np.asarray(dictionary.directions)[None,:]
    groups=np.asarray(dictionary.groups)
    return np.column_stack([oriented[:,groups==i].sum(axis=1)/dictionary.sizes[i] for i in range(13)])


def legendre_raw(scores,*,maximum_degree=3):
    """Degree-major columns: thirteen P1, thirteen P2, thirteen P3.

    Degree one is the explicit nested linear-theme control. It does not recover
    the original random-Fourier representation or any Extension-2 model.
    """
    if type(maximum_degree) is not int or maximum_degree not in (1,3):
        raise ValueError('Only the cubic representation or explicit linear control is specified')
    q=np.asarray(scores,dtype=float)
    if (q.ndim!=2 or q.shape[1]!=13 or len(q)==0 or not np.isfinite(q).all()
            or np.any(np.abs(q)>1+8*np.finfo(float).eps)):
        raise ValueError('Finite stock-by-thirteen theme scores in [-1,1] required')
    if maximum_degree==1:return q.copy()
    return np.column_stack((q,(3*q*q-1)/2,(5*q*q*q-3*q)/2))


def theme_signals(ranks,volatility,dictionary,*,maximum_degree=3):
    """Use the unchanged baseline constant, centering, L2 and volatility rules.

    A constant nonlinear column is rejected by normalize_signal. No column is
    silently dropped, perturbed or given a numerical normalization floor.
    """
    scores=theme_scores(ranks,dictionary)
    raw=legendre_raw(scores,maximum_degree=maximum_degree)
    volatility=np.asarray(volatility,dtype=float)
    if volatility.shape!=(len(scores),):raise ValueError('One positive volatility per current stock required')
    return normalize_signal(raw,volatility)


def signal_names(dictionary,*,maximum_degree=3):
    if not isinstance(dictionary,ThemeDictionary) or type(maximum_degree) is not int or maximum_degree not in (1,3):
        raise ValueError('Validated theme map and specified polynomial degree required')
    return ('constant',)+tuple(f'P{degree}:{theme}' for degree in range(1,maximum_degree+1) for theme in dictionary.themes)
