"""Chronological selection guards, not an approved validation protocol.

Callers must supply every fold, candidate, selection cutoff, economic scoring
callback and aggregation rule.  There are deliberately no historical dates,
penalties, stopping defaults or portfolio objective here.  In particular, the
profiled *training* gradient is not used as a validation gradient.

Parent metadata must come from the caller's audited artifact inventory.  These
guards verify consistency of that metadata and of callback receipts; they do
not independently establish that a callback used only the data it declares.
Each plan selects within one seed.  No seed pooling or ensemble is implied.
"""
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import pandas as pd


def _day(value):
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
        raise ValueError("Dates must be finite, timezone-free calendar days")
    return date.strftime("%Y-%m-%d")


def _month_end(value):
    date = _day(value)
    if not pd.Timestamp(date).is_month_end:
        raise ValueError("Training and validation observations must be month ends")
    return date


def _outcome_date(decision_date):
    """The inherited monthly return matures one month after its decision."""
    return (pd.Timestamp(_month_end(decision_date)) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")


def _label(value, kind):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(kind + " must be a nonempty string")
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("An explicit positive integer seed is required")
    return int(value)


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(x not in "0123456789abcdef" for x in value)):
        raise ValueError("An audited lowercase SHA256 is required")
    return value


@dataclass(frozen=True)
class ParentArtifact:
    """Dated identity of one audited learned-scale parent and its settings."""

    artifact_id: str
    seed: int
    cutoff: str
    training_last: str
    outcomes_available_at: str
    settings_available_at: str
    model_sha256: str
    receipt_sha256: str
    settings_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "artifact_id", _label(self.artifact_id, "Parent ID"))
        object.__setattr__(self, "seed", _seed(self.seed))
        for field in ("cutoff", "outcomes_available_at", "settings_available_at"):
            object.__setattr__(self, field, _day(getattr(self, field)))
        object.__setattr__(self, "training_last", _month_end(self.training_last))
        for field in ("model_sha256", "receipt_sha256", "settings_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field)))
        if not _outcome_date(self.training_last) <= self.outcomes_available_at <= self.cutoff:
            raise ValueError("Parent training outcomes are not available at its cutoff")
        if self.settings_available_at > self.cutoff:
            raise ValueError("Parent settings were selected after its cutoff")

    @property
    def fingerprint(self):
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ParentRegistry:
    """Require the specifically declared cutoff; never substitute a later D."""

    def __init__(self, parents: Sequence[ParentArtifact]):
        records = tuple(parents)
        if not records or any(not isinstance(p, ParentArtifact) for p in records):
            raise ValueError("An explicit audited parent inventory is required")
        keys = [(p.seed, p.cutoff) for p in records]
        if len(set(keys)) != len(keys) or len({p.artifact_id for p in records}) != len(records):
            raise ValueError("Duplicate parent IDs or seed/cutoff records")
        self._parents = MappingProxyType(dict(zip(keys, records)))

    def for_fold(self, fold):
        key = (fold.seed, fold.parent_cutoff)
        if key not in self._parents:
            raise ValueError("Required cutoff-specific parent artifact is absent: " + str(key))
        parent = self._parents[key]
        if (parent.cutoff > fold.training_cutoff
                or parent.training_last > fold.training_end
                or parent.outcomes_available_at > fold.training_cutoff
                or parent.settings_available_at > fold.training_cutoff):
            raise ValueError("Parent or its settings used information beyond the fold")
        return parent


@dataclass(frozen=True)
class ValidationFold:
    """Explicit DECISION month-end bounds and information-availability cutoff.

    An earlier parent cutoff is permitted only when the caller explicitly names
    it.  It is not silently replaced with the final annual parent.  Whether such
    a stale-parent design is appropriate still requires a scientific decision.
    Every decision's return matures at the following month end, matching the
    inherited training-sample convention.  These are not return-date bounds.
    """

    name: str
    seed: int
    training_start: str
    training_end: str
    training_cutoff: str
    validation_start: str
    validation_end: str
    parent_cutoff: str

    def __post_init__(self):
        object.__setattr__(self, "name", _label(self.name, "Fold name"))
        object.__setattr__(self, "seed", _seed(self.seed))
        for field in ("training_start", "training_end", "validation_start", "validation_end"):
            object.__setattr__(self, field, _month_end(getattr(self, field)))
        for field in ("training_cutoff", "parent_cutoff"):
            object.__setattr__(self, field, _day(getattr(self, field)))
        if not (self.training_start <= self.training_end <= self.training_cutoff
                < self.validation_start <= self.validation_end):
            raise ValueError("Training and validation dates are not chronological")
        if _outcome_date(self.training_end) > self.training_cutoff:
            raise ValueError("Final training decision's return is not mature at its cutoff")
        if self.parent_cutoff > self.training_cutoff:
            raise ValueError("Parent cutoff is later than the fold training cutoff")

    @property
    def validation_dates(self):
        return tuple(d.strftime("%Y-%m-%d") for d in pd.date_range(
            self.validation_start, self.validation_end, freq="ME"))


@dataclass(frozen=True)
class SelectionPlan:
    """No default folds or evaluation boundary: all choices must be declared."""

    folds: tuple[ValidationFold, ...]
    max_selection_date: str
    criterion_name: str

    def __post_init__(self):
        object.__setattr__(self, "folds", tuple(self.folds))
        object.__setattr__(self, "max_selection_date", _day(self.max_selection_date))
        object.__setattr__(self, "criterion_name", _label(self.criterion_name, "Criterion name"))
        if not self.folds or any(not isinstance(f, ValidationFold) for f in self.folds):
            raise ValueError("Explicit validation folds are required")
        if len({f.name for f in self.folds}) != len(self.folds):
            raise ValueError("Duplicate fold names")
        if len({f.seed for f in self.folds}) != 1:
            raise ValueError("Selection plans must be separate for each seed")
        previous = None
        for fold in self.folds:
            if _outcome_date(fold.validation_end) > self.max_selection_date:
                raise ValueError("Validation crosses the explicitly declared selection boundary")
            if previous is not None and (fold.validation_start <= previous.validation_end
                                        or fold.training_cutoff < previous.training_cutoff):
                raise ValueError("Validation folds overlap or are not in chronological order")
            previous = fold

    def parents(self, registry: ParentRegistry):
        # Resolve every fold before any potentially expensive callback starts.
        return tuple(registry.for_fold(fold) for fold in self.folds)


@dataclass(frozen=True)
class Candidate:
    """One explicitly declared numeric configuration, with no inferred grid."""

    name: str
    parameters: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "name", _label(self.name, "Candidate name"))
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise ValueError("Explicit numeric candidate parameters are required")
        parameters = {}
        for name, value in self.parameters.items():
            _label(name, "Parameter name")
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError("Candidate parameters must be finite numeric values")
            parameters[name] = float(value)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True)
class FoldScore:
    """Validation-only receipt with DECISION dates and subsequent outcome dates."""

    candidate_name: str
    fold_name: str
    criterion_name: str
    criterion_role: str
    score: float
    validation_dates: tuple[str, ...]
    validation_outcomes_available_at: str
    training_last: str
    training_outcomes_available_at: str
    parent_fingerprint: str


@dataclass(frozen=True)
class SelectionResult:
    selected: Candidate
    candidate_scores: tuple[tuple[str, float], ...]
    fold_scores: tuple[FoldScore, ...]
    criterion_name: str
    criterion_role: str
    max_selection_date: str
    maximize: bool


def _validate_score(score, candidate, fold, parent, plan):
    if not isinstance(score, FoldScore):
        raise ValueError("Evaluator must return a FoldScore provenance receipt")
    if (score.candidate_name != candidate.name or score.fold_name != fold.name
            or score.criterion_name != plan.criterion_name or score.criterion_role != "validation"):
        raise ValueError("Receipt must label the specified candidate, fold and validation criterion")
    if score.parent_fingerprint != parent.fingerprint:
        raise ValueError("Validation callback used a different parent or its settings")
    dates = tuple(_month_end(date) for date in score.validation_dates)
    if dates != fold.validation_dates:
        raise ValueError("Validation receipt must include every declared month in exact order")
    if (_month_end(score.training_last) != fold.training_end
            or not _outcome_date(fold.training_end) <= _day(score.training_outcomes_available_at) <= fold.training_cutoff):
        raise ValueError("Training outcomes extend beyond the fold's information cutoff")
    if not _outcome_date(fold.validation_end) <= _day(score.validation_outcomes_available_at) <= plan.max_selection_date:
        raise ValueError("Validation outcomes are unavailable at the selection boundary")
    if isinstance(score.score, bool) or not isinstance(score.score, Real) or not isfinite(score.score):
        raise ValueError("Validation score must be finite")


def select_candidate(*, plan: SelectionPlan, registry: ParentRegistry,
                     candidates: Sequence[Candidate],
                     evaluator: Callable[[Candidate, ValidationFold, ParentArtifact], FoldScore],
                     aggregate: Callable[[tuple[FoldScore, ...]], float],
                     maximize: bool) -> SelectionResult:
    """Coordinate supplied validation callbacks after checking declared timing.

    The caller supplies both the economic score and fold aggregation policy.
    Exact score ties retain the candidate's declared order; no result-dependent
    extra candidates, folds or update counts are introduced by this helper.
    This is not a portfolio evaluator or a validation-based early stopper.
    """
    if not isinstance(maximize, bool):
        raise ValueError("The score direction must be explicitly declared")
    if not callable(evaluator) or not callable(aggregate):
        raise ValueError("Explicit scoring and aggregation callbacks are required")
    candidates = tuple(candidates)
    if not candidates or any(not isinstance(c, Candidate) for c in candidates):
        raise ValueError("An explicit candidate grid is required")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate names")
    keys = set(candidates[0].parameters)
    signatures = []
    for candidate in candidates:
        if set(candidate.parameters) != keys:
            raise ValueError("Candidates must declare the same parameter names")
        signatures.append(tuple(sorted(candidate.parameters.items())))
    if len(set(signatures)) != len(signatures):
        raise ValueError("Duplicate candidate configurations")
    parents = plan.parents(registry)
    scores, receipts = [], []
    for candidate in candidates:
        current = []
        for fold, parent in zip(plan.folds, parents):
            result = evaluator(candidate, fold, parent)
            _validate_score(result, candidate, fold, parent, plan)
            current.append(result)
        total = aggregate(tuple(current))
        if isinstance(total, bool) or not isinstance(total, Real) or not isfinite(total):
            raise ValueError("Aggregate validation score must be finite")
        scores.append((candidate.name, float(total)))
        receipts.extend(current)
    best = 0
    for i in range(1, len(scores)):
        if ((maximize and scores[i][1] > scores[best][1])
                or (not maximize and scores[i][1] < scores[best][1])):
            best = i
    return SelectionResult(candidates[best], tuple(scores), tuple(receipts),
                           plan.criterion_name, "validation", plan.max_selection_date, maximize)
