"""Once-only, per-seed selection from explicitly bounded continuous paths.

No dates, penalty grid, risk aversion or approval are supplied by this module.
The caller provides each candidate's own complete pre-selection path and its
verified artifact inventory.  All candidates remain research exposure.  The
selected development/selection history is retrospective, not a causal record
of a configuration that was already known at the beginning of that history.

Availability follows the inherited month-close ordering: an outcome available
at a month close may inform that close's decision for the following month.
Thus the selection cutoff may equal the first evaluation decision date.
"""
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


def _month(value):
    result = pd.Timestamp(value)
    if (pd.isna(result) or result.tzinfo is not None
            or result != result.normalize() or not result.is_month_end):
        raise ValueError("An explicit timezone-free calendar month end is required")
    return result.strftime("%Y-%m-%d")


def _label(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty explicit labels are required")
    return value


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("A verified lowercase SHA256 is required")
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("Explicit positive integer seeds are required")
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    """An explicitly ordered candidate; exact ties retain this declared order."""

    name: str
    parameters: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "name", _label(self.name))
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise ValueError("Candidate parameters must be declared")
        parameters = {}
        for key, value in self.parameters.items():
            _label(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError("Candidate parameters must be finite real values")
            parameters[key] = float(value)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True)
class SelectionPlan:
    """Explicit once-only selection, with no implicit historical settings.

    ``candidates`` order is the predeclared tie rule (for example strongest
    anchor first, if that is the approved grid). ``family_id`` identifies one
    independently selected model family, such as temporal or current-only.
    Reuse the same dates and budget in the matched diagnostic family.
    """

    protocol_id: str
    family_id: str
    seeds: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    initial_decision: str
    validation_first_return: str
    validation_last_return: str
    selection_available_at: str
    first_evaluation_decision: str
    max_available_decision: str
    max_available_outcome: str
    initial_pretrade_sha256: str
    gamma: float
    exposure_label: str

    def __post_init__(self):
        for name in ("protocol_id", "family_id", "exposure_label"):
            object.__setattr__(self, name, _label(getattr(self, name)))
        object.__setattr__(self, "seeds", tuple(_seed(s) for s in self.seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("A nonempty unique predetermined seed sequence is required")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(not isinstance(c, Candidate) for c in self.candidates):
            raise ValueError("Explicit ordered candidates are required")
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate names must be unique")
        keys = set(self.candidates[0].parameters)
        signatures = []
        for candidate in self.candidates:
            if set(candidate.parameters) != keys:
                raise ValueError("Candidates must declare matching parameter names")
            signatures.append(tuple(sorted(candidate.parameters.items())))
        if len(set(signatures)) != len(signatures):
            raise ValueError("Candidate configurations must be unique")
        for name in ("initial_decision", "validation_first_return", "validation_last_return",
                     "selection_available_at", "first_evaluation_decision",
                     "max_available_decision", "max_available_outcome"):
            object.__setattr__(self, name, _month(getattr(self, name)))
        object.__setattr__(self, "initial_pretrade_sha256", _sha(self.initial_pretrade_sha256))
        if (isinstance(self.gamma, (bool, np.bool_)) or not isinstance(self.gamma, Real)
                or not isfinite(self.gamma) or self.gamma <= 0):
            raise ValueError("The verified parent risk aversion must be explicit and positive")
        object.__setattr__(self, "gamma", float(self.gamma))
        initial_return = _month(pd.Timestamp(self.initial_decision) + pd.offsets.MonthEnd(1))
        last_decision = _month(pd.Timestamp(self.validation_last_return) - pd.offsets.MonthEnd(1))
        if not (initial_return <= self.validation_first_return <= self.validation_last_return
                <= self.max_available_outcome <= self.selection_available_at
                <= self.first_evaluation_decision):
            raise ValueError("Validation outcomes must be available before the evaluation decision")
        if not (self.initial_decision <= last_decision <= self.max_available_decision
                <= self.selection_available_at):
            raise ValueError("Selection decisions exceed the declared available history")

    def as_dict(self):
        result = dict(self.__dict__)
        result["seeds"] = list(self.seeds)
        result["candidates"] = [dict(name=c.name, parameters=dict(c.parameters)) for c in self.candidates]
        return result

    @property
    def fingerprint(self):
        return _digest(self.as_dict())


@dataclass(frozen=True)
class CandidatePath:
    """Bounded path and provenance supplied by the verified portfolio runner.

    The table contains one seed/candidate only; its data are copied at selection.
    ``artifact_sha256`` must identify the consumed path/fit/lock dependencies,
    already verified by the caller. This function cannot hash absent files or
    prove source authorization merely from self-reported metadata.
    """

    seed: int
    candidate: str
    monthly: pd.DataFrame
    artifact_sha256: Mapping[str, str]
    first_trade_charged: bool

    def __post_init__(self):
        object.__setattr__(self, "seed", _seed(self.seed))
        object.__setattr__(self, "candidate", _label(self.candidate))
        if not isinstance(self.monthly, pd.DataFrame):
            raise ValueError("A bounded monthly candidate path is required")
        if not isinstance(self.artifact_sha256, Mapping) or not self.artifact_sha256:
            raise ValueError("Verified candidate input/artifact hashes are required")
        inventory = {_label(name): _sha(value) for name, value in self.artifact_sha256.items()}
        object.__setattr__(self, "artifact_sha256", MappingProxyType(inventory))
        if self.first_trade_charged is not True:
            raise ValueError("The common initial trade must be charged")


DATE_COLUMNS = ("decision_eom", "return_eom", "parent_cutoff", "training_last_decision",
                "training_outcomes_available_at")
VALUE_COLUMNS = ("gross_excess", "impact_cost", "net_excess")
HASH_COLUMNS = ("pretrade_sha256", "next_pretrade_sha256")
REQUIRED_COLUMNS = DATE_COLUMNS + VALUE_COLUMNS + HASH_COLUMNS + ("holdings_reset",)


def _checked_path(path, plan):
    # Inspect dates before copying/reading returns. A future row is rejected,
    # never silently included in candidate scoring or its content fingerprint.
    if not path.monthly.columns.is_unique or not set(REQUIRED_COLUMNS).issubset(path.monthly.columns):
        raise ValueError("A complete, uniquely named monthly path schema is required")
    decisions = tuple(_month(x) for x in path.monthly["decision_eom"])
    returns = tuple(_month(x) for x in path.monthly["return_eom"])
    if (any(d > plan.max_available_decision for d in decisions)
            or any(r > plan.max_available_outcome or r > plan.validation_last_return for r in returns)):
        raise ValueError("Future rows are forbidden in a selection input; provide a bounded artifact")
    last_decision = pd.Timestamp(plan.validation_last_return) - pd.offsets.MonthEnd(1)
    expected = tuple(d.strftime("%Y-%m-%d") for d in pd.date_range(plan.initial_decision, last_decision, freq="ME"))
    expected_returns = tuple(_month(pd.Timestamp(d) + pd.offsets.MonthEnd(1)) for d in expected)
    if decisions != expected or returns != expected_returns:
        raise ValueError("Each candidate must have the same complete ordered continuous monthly path")
    if "seed" in path.monthly.columns and any(_seed(s) != path.seed for s in path.monthly["seed"]):
        raise ValueError("Monthly seed labels differ from the candidate path")
    if "candidate" in path.monthly.columns and any(c != path.candidate for c in path.monthly["candidate"]):
        raise ValueError("Monthly candidate labels differ from the candidate path")
    if "method" in path.monthly.columns and any(c != path.candidate for c in path.monthly["method"]):
        raise ValueError("Monthly method labels differ from the declared candidate")
    table = path.monthly.loc[:, list(REQUIRED_COLUMNS)].copy(deep=True)
    for field in DATE_COLUMNS:
        table[field] = [_month(x) for x in table[field]]
    for field in VALUE_COLUMNS:
        raw = table[field].to_numpy()
        if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("Finite numeric monthly returns and impact costs are required")
        table[field] = raw.astype(float)
    if (table.impact_cost < 0).any():
        raise ValueError("Quadratic impact cost cannot be negative")
    if not np.allclose(table.gross_excess - table.impact_cost, table.net_excess, rtol=1e-11, atol=1e-12):
        raise ValueError("Net return must reconcile with actual-order impact costs")
    for field in HASH_COLUMNS:
        table[field] = [_sha(x) for x in table[field]]
    if table.pretrade_sha256.iloc[0] != plan.initial_pretrade_sha256:
        raise ValueError("Every candidate/seed must start from the same accepted holdings")
    if tuple(table.next_pretrade_sha256.iloc[:-1]) != tuple(table.pretrade_sha256.iloc[1:]):
        raise ValueError("Candidate holdings did not carry continuously between decisions")
    if any(not isinstance(x, (bool, np.bool_)) or bool(x) for x in table.holdings_reset):
        raise ValueError("Annual or period holdings resets are forbidden")
    table["holdings_reset"] = False
    for row in table.itertuples(index=False):
        mature = _month(pd.Timestamp(row.training_last_decision) + pd.offsets.MonthEnd(1))
        if not (mature <= row.training_outcomes_available_at <= row.parent_cutoff <= row.decision_eom):
            raise ValueError("Parent training outcomes must mature before its decision")
        expected_parent = f"{pd.Timestamp(row.return_eom).year - 1}-12-31"
        if row.parent_cutoff != expected_parent:
            raise ValueError("Each decision requires its exact archived annual parent")
        expected_training_last = f"{pd.Timestamp(row.return_eom).year - 1}-11-30"
        if row.training_last_decision != expected_training_last:
            raise ValueError("The candidate must preserve the parent's complete annual training endpoint")
    validation = table[table.return_eom.between(plan.validation_first_return, plan.validation_last_return)]
    expected_n = len(pd.date_range(plan.validation_first_return, plan.validation_last_return, freq="ME"))
    if len(validation) != expected_n:
        raise ValueError("Incomplete declared selection interval")
    values = validation.net_excess.to_numpy()
    score = float(12. * (values.mean() - .5 * plan.gamma * values.var(ddof=0)))
    if not isfinite(score):
        raise ValueError("Nonfinite selection utility")
    rows = table.to_dict("records")
    return dict(seed=path.seed, candidate=path.candidate, annual_net_utility=score,
                selection_months=len(validation), complete_path_months=len(table),
                first_decision=decisions[0], last_decision=decisions[-1],
                first_return=returns[0], last_return=returns[-1],
                first_selection_return=plan.validation_first_return,
                last_selection_return=plan.validation_last_return,
                input_table_sha256=_digest(rows), artifact_sha256=dict(path.artifact_sha256),
                initial_pretrade_sha256=table.pretrade_sha256.iloc[0],
                terminal_pretrade_sha256=table.next_pretrade_sha256.iloc[-1],
                parent_cutoffs=sorted(set(table.parent_cutoff)), first_trade_charged=True,
                holdings_reset=False, annual_parent_cutoffs_verified=True)


def select_candidates(*, plan: SelectionPlan, paths):
    """Return a JSON-serializable frozen selection and exposure receipt.

    Each seed is scored and selected independently. The reported criterion is
    the inherited annual net utility, including realized trading costs. No
    training objective, later-period outcome, seed average, or ensemble is used
    to choose the candidate. Exact ties retain ``plan.candidates`` order.

    This utility does not authorize fitting, approve a grid, verify filesystem
    receipts, or establish that the already exposed evaluation is a fresh test.
    """
    if not isinstance(plan, SelectionPlan):
        raise ValueError("An explicit selection plan is required")
    paths = tuple(paths)
    if any(not isinstance(path, CandidatePath) for path in paths):
        raise ValueError("Verified candidate path containers are required")
    keys = [(path.seed, path.candidate) for path in paths]
    expected = {(seed, candidate.name) for seed in plan.seeds for candidate in plan.candidates}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Exactly every declared seed/candidate path is required")
    inventory = dict(zip(keys, paths))
    selections, exposures = [], []
    for seed in plan.seeds:
        records = [_checked_path(inventory[seed, candidate.name], plan) for candidate in plan.candidates]
        best = 0
        for i in range(1, len(records)):
            if records[i]["annual_net_utility"] > records[best]["annual_net_utility"]:
                best = i
        choice = plan.candidates[best]
        selections.append(dict(seed=seed, candidate=choice.name, parameters=dict(choice.parameters),
                               annual_net_utility=records[best]["annual_net_utility"],
                               terminal_pretrade_sha256=records[best]["terminal_pretrade_sha256"],
                               candidate_scores=[dict(candidate=r["candidate"], annual_net_utility=r["annual_net_utility"])
                                                 for r in records]))
        exposures.extend(records)
    result = dict(status="SELECTION_FROZEN", plan=plan.as_dict(), plan_sha256=plan.fingerprint,
                  selected_by_seed=selections, candidate_exposure=exposures,
                  candidate_seed_paths_inspected=len(exposures),
                  total_candidate_selection_months=sum(r["selection_months"] for r in exposures),
                  selection_available_at=plan.selection_available_at,
                  first_evaluation_decision=plan.first_evaluation_decision,
                  tie_rule="first candidate in the declared order", selection_pools_seeds=False,
                  score="12 * (mean(net_excess) - gamma / 2 * population_variance(net_excess))",
                  availability_convention="outcomes available at a month close may inform that close's next-month decision",
                  development_and_selection_are_retrospective=True,
                  selected_configuration_known_at_initial_decision=False,
                  evaluation_is_previously_exposed=True, historical_execution_authorized=False)
    result["receipt_sha256"] = _digest(result)
    return result
