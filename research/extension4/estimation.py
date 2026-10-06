"""Economic moments reuse the audited parent dynamic operator and ridge units."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from feature_geometry.training import _solve as parent_solve
from .features import dynamic_design


def month_moments(basis, innovations, parent_model, states, *, state_names, interaction_columns):
    design=dynamic_design(basis,innovations,parent_model,states,
                          state_names=state_names,interaction_columns=interaction_columns)
    m=basis.sample.market;omega=design['omega'];change=design['change']
    risk=m.gamma*omega.T@m.covariance@omega
    cost=m.wealth*change.T@(m.impact[:,None]*change)
    return dict(score=omega.T@basis.sample.returns,risk=(risk+risk.T)/2,
                cost=(cost+cost.T)/2,**design)


def solve_coefficients(curvature, score, *, parent_width, parent_ridge, conditional_ridge, disabled):
    score=np.asarray(score,dtype=float);curvature=np.asarray(curvature,dtype=float)
    if (type(disabled) is not bool or curvature.shape!=(len(score),len(score))
            or not 0<parent_width<=len(score) or not np.isfinite(curvature).all()
            or not np.isfinite(score).all() or not np.isfinite(parent_ridge) or parent_ridge<0):
        raise ValueError('Invalid coefficient system')
    if not np.allclose(curvature,curvature.T,atol=1e-12,rtol=1e-10):
        raise ValueError('Asymmetric curvature')
    if disabled:
        if len(score)!=parent_width: raise ValueError('Disabled path requires restricted parent system')
        return parent_solve(curvature,score,parent_ridge)
    if len(score)==parent_width or not np.isfinite(conditional_ridge) or conditional_ridge<=0:
        raise ValueError('Enabled conditional block requires explicit positive penalty')
    penalty=np.r_[np.full(parent_width,parent_ridge),np.full(len(score)-parent_width,conditional_ridge)]
    system=(curvature+curvature.T)/2+np.diag(penalty)
    try:beta=cho_solve(cho_factor(system,lower=True,check_finite=False),score,check_finite=False)
    except np.linalg.LinAlgError as exc:raise ValueError('Non-SPD system; no silent jitter') from exc
    residual=np.linalg.norm(system@beta-score)/max(np.linalg.norm(score),1e-30)
    if not np.isfinite(beta).all() or not np.isfinite(residual) or residual>1e-9:
        raise ValueError('Coefficient residual check failed')
    return beta,float(residual)
