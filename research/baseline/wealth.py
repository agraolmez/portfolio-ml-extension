"""Strict reciprocal wealth contract, independent of the preparation implementation."""
import numpy as np
import pandas as pd


def reconstruct(dates,returns,anchor_date,anchor_wealth):
    dates=pd.DatetimeIndex(dates);r=np.asarray(returns,dtype=float);anchor=pd.Timestamp(anchor_date)
    if len(dates)<2 or r.shape!=(len(dates),) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Invalid wealth calendar or aligned returns')
    if not dates.is_month_end.all() or not dates.equals(pd.date_range(dates[0],dates[-1],freq='ME')):
        raise ValueError('Complete month-end wealth calendar required')
    if not np.isfinite(r).all() or np.any(r<=-1) or not np.isfinite(anchor_wealth) or anchor_wealth<=0 or anchor not in dates:
        raise ValueError('Invalid market total return or wealth anchor')
    k=dates.get_loc(anchor);w=np.empty(len(r));w[k]=anchor_wealth
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        for j in range(k,0,-1):w[j-1]=w[j]/(1+r[j])
        for j in range(k+1,len(w)):w[j]=w[j-1]*(1+r[j])
    if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Wealth overflow, underflow or nonpositive value')
    return w


def validate(frame,anchor_date='2020-12-31',anchor_wealth=10_000_000_000.):
    expected=reconstruct(frame.eom,frame.market_total_return,anchor_date,anchor_wealth)
    actual=frame.wealth_usd.to_numpy()
    if not np.isfinite(actual).all() or np.any(actual<=0) or float(frame.loc[frame.eom.eq(anchor_date),'wealth_usd'].iloc[0])!=anchor_wealth:
        raise ValueError('Incorrect wealth values or exact anchor')
    if not np.allclose(actual,expected,atol=0,rtol=1e-12):raise ValueError('Reciprocal wealth mismatch')
    ratio=actual[1:]/actual[:-1];growth=1+frame.market_total_return.to_numpy()[1:]
    if not np.allclose(ratio,growth,atol=1e-12,rtol=1e-12):raise ValueError('Holdings/wealth growth mismatch')
    return {'months':len(frame),'adjacent_links':len(frame)-1,'max_relative_reconstruction_error':float(np.max(abs(actual/expected-1))),'max_growth_ratio_error':float(np.max(abs(ratio-growth))),'anchor_usd':anchor_wealth}
