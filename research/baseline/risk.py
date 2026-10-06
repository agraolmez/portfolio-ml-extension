"""Released factor covariance and specific-risk rules over the accepted history."""
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import lfilter
from .io import ROOT,DATA,OUT,PREP,SETTINGS,verify_data,save_json,sha,guard,datekey,finite


def weighted_cov(values,weights):
    values=np.asarray(values);weights=np.asarray(weights,dtype=float);finite(values,weights)
    weights=weights/weights.sum();centered=values-weights@values
    return (centered.T@(weights[:,None]*centered))/(1-np.sum(weights**2))


def ewma_source(x,lam,start=63):
    x=np.asarray(x,dtype=float);out=np.full(len(x),np.nan)
    if len(x)<=start:return out
    finite(x)
    initial=np.sum(x[:start]**2)/(start-1)
    out[start]=initial
    if len(x)>start+1:out[start+1:]=lfilter([1-lam],[1,-lam],x[start:-1]**2,zi=[lam*initial])[0]
    return np.sqrt(out)


def build():
    verification=verify_data();cfg=SETTINGS['settings']['cov_set']
    factors=json.loads((ROOT/'output/step3_author_conventions_20260911/factor_calendar.json').read_text())['factors']
    f=pd.read_parquet(DATA/'daily_factor_returns.parquet').sort_values('date')
    loads=pd.read_parquet(DATA/'factor_loadings.parquet').sort_values(['eom','id'])
    first=f.date.iloc[cfg['obs']-1].replace(day=1)-pd.Timedelta(days=1)
    dates=sorted(loads.loc[loads.eom.ge(first),'eom'].unique())
    cor_weights=(.5**(1/cfg['hl_cor']))**np.arange(cfg['obs'],0,-1)
    var_weights=(.5**(1/cfg['hl_var']))**np.arange(cfg['obs'],0,-1)
    all_cov=[];covdiag=[]
    for d in dates:
        values=f.loc[f.date.le(d),factors].tail(cfg['obs']).to_numpy();n=len(values)
        if n<cfg['obs']-30:raise ValueError('Insufficient factor covariance history')
        corr_cov=weighted_cov(values,cor_weights[-n:]);std=np.sqrt(np.diag(corr_cov))
        corr=corr_cov/std[:,None]/std[None,:]
        variances=np.diag(weighted_cov(values,var_weights[-n:]))
        cov=corr*np.sqrt(variances[:,None]*variances[None,:])*21
        ev=eigh(cov,eigvals_only=True)
        if ev[0]<=0:raise ValueError('Nonpositive factor covariance')
        all_cov.append(cov);covdiag.append({'eom':pd.Timestamp(d),'n_factor_observations':n,'factor_min_eigenvalue':ev[0],'factor_condition':ev[-1]/ev[0]})
    np.savez_compressed(PREP/'risk/factor_covariances.npz',dates=np.array([datekey(d) for d in dates]),factors=np.array(factors),covariances=np.stack(all_cov))
    # Build original zero-mean, lagged EWMA on each full residual history.
    residuals=pd.concat([pd.read_parquet(p) for p in sorted((DATA/'daily').glob('residuals_*.parquet'))],ignore_index=True).sort_values(['id','date'])
    residuals.to_csv(OUT/'reference/residuals.csv',index=False,float_format='%.17g')
    factor_calendar=pd.Index(f.date);all_months=[]
    for sid,g in residuals.groupby('id',sort=False):
        g=g.copy();g['res_vol']=ewma_source(g.residual.to_numpy(),.5**(1/cfg['hl_stock_var']),cfg['initial_var_obs'])
        positions=factor_calendar.get_indexer(g.date)
        lower=np.full(len(g),np.datetime64('NaT'),dtype='datetime64[ns]')
        eligible=positions>=252;lower[eligible]=factor_calendar.to_numpy()[positions[eligible]-252]
        lag200=g.date.shift(200)
        g=g[lag200.ge(lower)&g.res_vol.notna()]
        if len(g):
            g['eom']=g.date+pd.offsets.MonthEnd(0)
            all_months.append(g.groupby('eom',as_index=False).tail(1)[['id','eom','date','res_vol']])
    monthly=pd.concat(all_months,ignore_index=True)
    monthly.to_parquet(PREP/'risk/observed_specific_risk.parquet',index=False)
    required=loads[loads.eom.ge(first)].merge(monthly,on=['id','eom'],how='left',validate='one_to_one')
    required['specific_missing']=required.res_vol.isna()
    by_size=required.groupby(['eom','size_grp'],observed=True).res_vol.transform('median')
    by_month=required.groupby('eom').res_vol.transform('median')
    required['fallback_all_sizes']=required.specific_missing & by_size.isna()
    required['res_vol_used']=required.res_vol.fillna(by_size).fillna(by_month)
    required['specific_variance_monthly']=21*required.res_vol_used**2
    if required.res_vol_used.isna().any() or required.res_vol_used.le(0).any():raise ValueError('Unresolved or nonpositive specific risk')
    table=[];diagnostics=[]
    for i,d in enumerate(dates):
        g=required[required.eom.eq(d)].sort_values('id');X=g[factors].to_numpy();iv=g.specific_variance_monthly.to_numpy()
        diagonal=np.sum((X@all_cov[i])*X,axis=1)+iv
        finite(diagonal)
        if np.any(diagonal<=0):raise ValueError('Nonpositive stock variance')
        np.savez_compressed(PREP/f'risk/{datekey(d)}.npz',ids=g.id.astype('int64').to_numpy(),loadings=X,factor_cov=all_cov[i],specific_variance=iv)
        table.append(pd.DataFrame({'id':g.id.to_numpy(),'eom':d,'vol_scale':np.sqrt(diagonal),'specific_variance':iv,'specific_missing':g.specific_missing.to_numpy(),'fallback_all_sizes':g.fallback_all_sizes.to_numpy()}))
        diagnostics.append({**covdiag[i],'n_stocks':len(g),'minimum_specific_variance':iv.min(),'minimum_total_variance':diagonal.min(),'missing_specific_imputed':int(g.specific_missing.sum()),'all_size_fallbacks':int(g.fallback_all_sizes.sum())})
    scales=pd.concat(table,ignore_index=True);scales.to_parquet(PREP/'risk/scales.parquet',index=False)
    required[['id','eom','size_grp','res_vol','res_vol_used','specific_variance_monthly','specific_missing','fallback_all_sizes']].to_parquet(PREP/'risk/specific_risk_with_fallbacks.parquet',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'risk_diagnostics.csv',index=False)
    f[['date',*factors]].to_csv(OUT/'reference/factor_returns.csv',index=False,float_format='%.17g')
    loads.loc[loads.eom.ge(first),['id','eom','size_grp']].to_csv(OUT/'reference/risk_universe.csv',index=False)
    save_json(OUT/'risk_build.json',{'status':'BUILT_PENDING_REFERENCE_PARITY','input_verification':verification,'covariance_months':len(dates),'first_covariance_month':datekey(dates[0]),'last_covariance_month':datekey(dates[-1]),'stock_months':len(scales),'specific_missing_imputed':int(required.specific_missing.sum()),'all_size_fallbacks':int(required.fallback_all_sizes.sum()),'daily_residuals':len(residuals),'minimum_specific_variance':float(scales.specific_variance.min()),'resources':guard()})
    print((OUT/'risk_build.json').read_text(),flush=True)


if __name__=='__main__':build()
