"""One streamed moment pass, shared across explicit conditional ridge candidates."""
import numpy as np
import pandas as pd
from .design import designs
from .estimation import solve_coefficients


class MomentSum:
    def __init__(self, width):
        self.score = np.zeros(width)
        self.risk = np.zeros((width, width))
        self.cost = np.zeros((width, width))
        self.count = 0

    def add(self, sample, omega, change):
        m = sample.market
        rr = m.gamma*omega.T@m.covariance@omega
        cc = m.wealth*change.T@(m.impact[:, None]*change)
        self.score += omega.T@sample.returns
        self.risk += (rr+rr.T)/2
        self.cost += (cc+cc.T)/2
        self.count += 1

    def means(self):
        if not self.count: raise ValueError('Empty moment sample')
        return dict(score=self.score/self.count, risk=self.risk/self.count, cost=self.cost/self.count)


def collect(batches, parent, state_book, extras, *, expected_dates, progress=None):
    p = len(parent.model.beta)
    sums = {name: MomentSum(width) for name, width in [('N4', p), ('C4', 2*p-1), ('CAP4', 2*p-1)]}
    affine = dict(score=0., curvature=0., constant=0.)
    seen = []
    for basis, bound in batches:
        sample, model = basis.sample, parent.model
        date = pd.Timestamp(sample.market.decision_date)
        if (pd.Timestamp(sample.available_at) > pd.Timestamp(model.cutoff)
                or date >= pd.Timestamp(model.cutoff) or date.isoformat() in seen):
            raise ValueError('Training chronology or outcome maturity failed')
        if len(seen) >= len(expected_dates) or date != pd.Timestamp(expected_dates[len(seen)]):
            raise ValueError('Missing, reordered or duplicate training month')
        d = designs(basis, bound, model, state_book.bind(date, history=13), state_book.identity, extras)
        for name in sums:
            omega, change = d['base' if name == 'N4' else name]
            sums[name].add(sample, omega, change)
        m = sample.market
        u0, v0 = (x@model.beta for x in d['base'])
        u1, v1 = d['SIZE4']
        affine['score'] += float(u1@sample.returns-m.gamma*u1@m.covariance@u0-m.wealth*v1@(m.impact*v0))
        affine['curvature'] += float(m.gamma*u1@m.covariance@u1+m.wealth*v1@(m.impact*v1))
        affine['constant'] += float(u0@sample.returns-.5*m.gamma*u0@m.covariance@u0-.5*m.wealth*v0@(m.impact*v0))
        seen.append(date.isoformat())
        if progress: progress(len(seen), sample.market.decision_date)
    if len(seen) != len(expected_dates) or not seen: raise ValueError('Incomplete annual moment sample')
    for key in affine: affine[key] /= len(seen)
    affine['constant'] -= .5*parent.settings.ridge*float(parent.model.beta@parent.model.beta)
    affine['constant'] -= .5*parent.settings.scale_penalty*float(parent.model.feature_map.theta@parent.model.feature_map.theta)
    # Fixed temporal restraint is a constant too, converted back to economic units.
    affine['constant'] -= parent.receipt['objective_scale']*parent.receipt['rho']*float(np.mean((parent.model.gains-1.)**2))
    return dict(moments={k:v.means() for k,v in sums.items()}, affine=affine, months=len(seen))


def fit_moments(collected, parent, multipliers, *, size_bounds, atol, rtol, family_multipliers=None):
    p, ridge = len(parent.model.beta), parent.settings.ridge
    moments = collected['moments']
    neutral = moments['N4']
    beta, residual = solve_coefficients(neutral['risk']+neutral['cost'], neutral['score'],
        parent_width=p, parent_ridge=ridge, conditional_ridge=None, disabled=True)
    np.testing.assert_allclose(beta, parent.model.beta, atol=atol, rtol=rtol,
                               err_msg='Actual archived parent coefficient re-solve failed')
    result = [('N4', None, beta, dict(solve_residual=residual,
               parent_maximum_coefficient_error=float(np.max(abs(beta-parent.model.beta)))))]
    family_multipliers = family_multipliers or {name:multipliers for name in ('C4','CAP4','SIZE4')}
    for method in ('C4', 'CAP4'):
        m = moments[method]
        for multiplier in family_multipliers[method]:
            beta, residual = solve_coefficients(m['risk']+m['cost'], m['score'], parent_width=p,
                parent_ridge=ridge, conditional_ridge=multiplier*ridge, disabled=False)
            system = m['risk']+m['cost']+np.diag(np.r_[np.full(p,ridge),np.full(p-1,multiplier*ridge)])
            # Eigenvalues also provide an inspectable condition-number diagnostic.
            eigenvalues = np.linalg.eigvalsh(system)
            if eigenvalues[0] <= 0: raise ValueError('Nonpositive fitted system')
            result.append((method, multiplier, beta, dict(solve_residual=residual,
                condition_number=float(eigenvalues[-1]/eigenvalues[0]), minimum_eigenvalue=float(eigenvalues[0]))))
    a = collected['affine']
    for multiplier in family_multipliers['SIZE4']:
        penalty = multiplier*ridge*float(parent.model.beta@parent.model.beta)
        curvature = a['curvature']+penalty
        if not np.isfinite(curvature) or curvature <= 0: raise ValueError('Degenerate scalar exposure system')
        alpha = float(np.clip(a['score']/curvature, *size_bounds))
        gradient = a['score']-curvature*alpha
        kkt = abs(alpha-np.clip(alpha+gradient/curvature, *size_bounds))
        if kkt > 1e-12: raise ValueError('Constrained scalar optimality failed')
        result.append(('SIZE4', multiplier, np.array([alpha]), dict(solve_residual=float(kkt),
            condition_number=1., penalty=penalty, affine=a,
            fitted_objective=float(a['constant']+a['score']*alpha-.5*curvature*alpha*alpha))))
    return result
