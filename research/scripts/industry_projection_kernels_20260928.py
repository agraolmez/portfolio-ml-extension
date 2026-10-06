"""Fixed risk/impact-weighted projection of incremental industry target risk.

Only target industry exposures are constrained. The unchanged dynamic trading
operator and inherited holdings can retain industry exposures after trading.
"""
from __future__ import annotations

import numpy as np

INDUSTRIES=('BusEq','Chems','Durbl','Enrgy','Hlth','Manuf','Money','NoDur','Other','Shops','Telcm','Utils')
FACTORS=INDUSTRIES+('low_leverage','investment','size','value','quality','low_risk','seasonality',
                  'accruals','profitability','debt_issuance','short_term_reversal','profit_growth','momentum')


def project_industry_targets(baseline_target,c4_target,loadings,factor_cov,specific_variance,
                             impact,wealth,factor_names,*,force_neutral=False):
    """Project C4-minus-baseline targets onto the fixed industry-neutral space.

    Q = 10*(loadings*factor_cov*loadings' + diag(specific_variance))
        + wealth*diag(impact).
    The implementation uses exact Woodbury solves; no fitted penalty, ridge,
    covariance floor, pseudoinverse or exposure renormalization is introduced.
    """
    if tuple(factor_names)!=FACTORS:
        raise ValueError('Accepted factor names/order differ from the fixed 25-factor contract')
    if not isinstance(force_neutral,(bool,np.bool_)):
        raise ValueError('Neutral switch must be boolean')
    baseline=np.asarray(baseline_target,dtype=float);c4=np.asarray(c4_target,dtype=float)
    loads=np.asarray(loadings,dtype=float);factor=np.asarray(factor_cov,dtype=float)
    specific=np.asarray(specific_variance,dtype=float);lam=np.asarray(impact,dtype=float)
    if (baseline.ndim!=1 or baseline.shape!=c4.shape or len(baseline)<12
            or loads.shape!=(len(baseline),25) or factor.shape!=(25,25)
            or specific.shape!=baseline.shape or lam.shape!=baseline.shape):
        raise ValueError('Invalid target or factor-model dimensions')
    if (not all(np.isfinite(x).all() for x in (baseline,c4,loads,factor,specific,lam))
            or not np.isscalar(wealth) or not np.isfinite(wealth) or wealth<=0
            or np.any(specific<0) or np.any(lam<=0)):
        raise ValueError('Finite inputs, positive wealth/impacts and nonnegative specific risk required')
    if not np.allclose(factor,factor.T,atol=1e-12,rtol=1e-10):
        raise ValueError('Asymmetric factor covariance')
    factor=(factor+factor.T)/2
    industry=loads[:,:12]
    # Default matrix_rank tolerance is solely floating-point rank detection;
    # deficient constraints fail instead of receiving an economic regularizer.
    if np.linalg.matrix_rank(industry)!=12:
        raise ValueError('Industry constraints are rank deficient')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            factor_root=np.linalg.cholesky(factor)
            diagonal=10*specific+float(wealth)*lam
            u=np.sqrt(10.)*(loads@factor_root)
            d_inv_u=u/diagonal[:,None]
            middle=np.eye(25)+u.T@d_inv_u
            middle_root=np.linalg.cholesky((middle+middle.T)/2)
            rhs=industry/diagonal[:,None]
            inner=np.linalg.solve(middle_root.T,np.linalg.solve(middle_root,u.T@rhs))
            q_inverse_industry=rhs-d_inv_u@inner
            gram=industry.T@q_inverse_industry
            gram_root=np.linalg.cholesky((gram+gram.T)/2)
            departure=c4-baseline
            exposure=industry.T@departure
            multipliers=np.linalg.solve(gram_root.T,np.linalg.solve(gram_root,exposure))
            correction=q_inverse_industry@multipliers
            projected=departure-correction
            if force_neutral:
                projected=departure.copy();correction=np.zeros_like(departure);multipliers=np.zeros(12)
            def q_apply(x):return diagonal*x+u@(u.T@x)
            before=float(departure@q_apply(departure));after=float(projected@q_apply(projected))
            correction_norm=float(correction@q_apply(correction))
            amplitude=float(np.sqrt(after/before)) if before>0 else 1.
            ind=c4.copy() if force_neutral else baseline+projected
            ampl=c4.copy() if amplitude==1 else baseline+amplitude*departure
            q_correction=q_apply(correction)
            kkt=-q_correction+industry@multipliers
            q_solve_residual=np.max(np.abs(diagonal[:,None]*q_inverse_industry+u@(u.T@q_inverse_industry)-industry))
    except (np.linalg.LinAlgError,FloatingPointError) as error:
        raise ValueError('Positive-definite unregularized projection solve failed') from error
    result=dict(IND=ind,AMPL=ampl,BENS=baseline.copy(),CENS=c4.copy(),departure=departure,
                projected_departure=projected,hedge_correction=correction,amplitude=amplitude,
                qnorm_before=before,qnorm_after=after,qnorm_correction=correction_norm,
                industry_exposure_before=exposure,industry_exposure_after=industry.T@projected,
                multipliers=multipliers,q_times_correction=q_correction,
                kkt_max_residual=float(np.max(np.abs(kkt))),q_solve_max_residual=float(q_solve_residual),
                orthogonality_residual=float(projected@q_correction),
                norm_identity_residual=float(before-after-correction_norm),projection_enabled=not force_neutral)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite projection arithmetic')
    if min(before,after,correction_norm)<0 or after>before+1e-10*(1+before):
        raise ValueError('Quadratic projection norm does not contract')
    return result
