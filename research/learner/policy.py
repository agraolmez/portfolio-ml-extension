"""Monthly own-holdings policy interface; accepts only contemporaneous inputs."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .core import feasible,decide
from .pipeline import features

@dataclass(frozen=True)
class DecisionInputs:
    decision_eom: str
    ids: np.ndarray
    aim: np.ndarray
    m: np.ndarray
    loadings: np.ndarray
    factor_cov: np.ndarray
    specific_variance: np.ndarray
    impact: np.ndarray
    wealth: float
    ensemble_mean_aim: np.ndarray
    replica_trade_risk_variance: float


def act(fit,method,inputs,holdings):
    """Return implemented weights/trades from this strategy's inherited state.

    Future returns, future aims and labels have no argument in this interface.
    Fit artifacts are created and verified separately using matured labels.
    """
    date=pd.Timestamp(inputs.decision_eom)
    if not pd.Timestamp(fit['first_decision'])<=date<=pd.Timestamp(fit['last_decision']):raise ValueError('Annual fit is not valid on this decision date')
    h=np.asarray(holdings,dtype=float)
    if h.shape!=inputs.aim.shape or h.shape!=(len(inputs.ids),) or not np.isfinite(h).all():raise ValueError('Invalid inherited holdings')
    cap=feasible(inputs.m,inputs.impact)
    q=(inputs.aim-h)-inputs.m@(inputs.aim-h)
    current=dict(m=inputs.m,loadings=inputs.loadings,fc=inputs.factor_cov,specific=inputs.specific_variance,
        impact=inputs.impact,wealth=inputs.wealth,confidence={0:dict(mean=inputs.ensemble_mean_aim,numerator=inputs.replica_trade_risk_variance)})
    state=features(current,h,q,0)
    action=decide(fit,method,state,cap['kappa_min'],cap['kappa_max'])
    trade=action['kappa']*q
    return dict(features=state,decision=action,trade=trade,weight=h+trade,feasibility=cap)
