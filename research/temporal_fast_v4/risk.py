"""Checked exact factor representation of the already accepted covariance."""
import numpy as np
from temporal_fast_v2.kernels import fingerprint


class BoundFactors:
    def __init__(self,market,ids,loadings,factor_cov,specific):
        n=len(market.ids)
        self.arrays=tuple(np.array(x,dtype=np.float64,copy=True) for x in (loadings,factor_cov,specific))
        x,f,s=self.arrays
        if (not np.array_equal(ids,market.ids) or x.ndim!=2 or x.shape[0]!=n
                or f.shape!=(x.shape[1],x.shape[1]) or s.shape!=(n,)
                or any(not np.isfinite(v).all() for v in self.arrays) or np.any(s<=0)):
            raise ValueError('Invalid aligned factor covariance')
        reconstructed=x@f@x.T+np.diag(s)
        np.testing.assert_allclose(reconstructed,market.covariance,atol=1e-12,rtol=1e-10,
                                   err_msg='Factor covariance differs from accepted dense covariance')
        self.maximum_reconstruction_error=float(np.max(abs(reconstructed-market.covariance)))
        for a in self.arrays:a.setflags(write=False)
        self.market_identity=market.fingerprint
        self.identity=fingerprint(self.market_identity,*self.arrays)
        self.nbytes=sum(a.nbytes for a in self.arrays)

    def get(self,market):
        if market.fingerprint!=self.market_identity or fingerprint(market.fingerprint,*self.arrays)!=self.identity:
            raise ValueError('Bound factor covariance changed')
        return self.arrays


def accepted_factors(history,market):
    relative=f'data/prepared/step9_v1/baseline/risk/{market.decision_date}.npz'
    with np.load(history.files.path(relative),allow_pickle=False) as data:
        return BoundFactors(market,data['ids'],data['loadings'],data['factor_cov'],data['specific_variance'])
