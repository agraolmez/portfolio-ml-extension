"""Identical historical inputs for actual pinned R checks; no strategy experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.io import *

DATES=['1965-11-30','1980-12-31','2025-11-30']
MECHANICS=pd.date_range('1980-11-30','1981-02-28',freq='ME')

def fixtures():
    needed=set(MECHANICS)
    for d in map(pd.Timestamp,DATES):needed.update(pd.date_range(d-pd.offsets.MonthEnd(12),d,freq='ME'))
    cols=['id','eom','valid','me','ret_ld1','tr_ld0','tr_ld1','mu_ld0','mu_ld1','lambda_diagonal','wealth_usd',*FEATURES]
    data=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=list(dict.fromkeys(cols)))
    data=data[data.eom.isin(needed)].sort_values(['eom','id']);data['valid']=data.valid.astype(int)
    data.to_csv(OUT/'reference/historical_data.csv',index=False,float_format='%.17g')
    rates=pd.read_parquet(DATA/'monthly_rates.parquet');rates=rates[rates.eom.isin(needed)]
    rates.to_csv(OUT/'reference/historical_rates.csv',index=False,float_format='%.17g')
    meta=[]
    for d in sorted(needed):
        folder=OUT/f'reference/risk/{datekey(d)}';folder.mkdir(parents=True,exist_ok=True)
        with np.load(PREP/f'risk/{datekey(d)}.npz') as z:
            ids=z['ids'];X=z['loadings'];F=z['factor_cov'];iv=z['specific_variance']
        for nm,x in [('loadings',X),('factor',F),('specific',iv)]:write_matrix(folder/f'{nm}.bin',x)
        save_json(folder/'ids.json',ids.tolist());meta.append({'date':datekey(d),'n':len(ids),'k':X.shape[1]})
    save_json(OUT/'reference/historical_metadata.json',{'moment_dates':DATES,'mechanics_dates':[datekey(d) for d in MECHANICS],'risk':meta})
    # Explicit deterministic targets for a bounded accounting test, unrelated to model selection/performance.
    active=data[data.valid.eq(1)&data.eom.isin(MECHANICS)].copy()
    active['w_aim']=active.groupby('eom').me.transform(lambda x:.8*x/x.sum())+1e-4*np.sin(active.id.to_numpy())
    active[['id','eom','w_aim']].to_csv(OUT/'reference/mechanics_aims.csv',index=False,float_format='%.17g')
    print('Exported identical-input historical fixtures.',flush=True)


def moments():
    dates=pd.date_range('1965-11-30','1982-11-30',freq='ME')
    for gi in [0,1]:
        folder=OUT/f'reference/fit/g{gi}';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'denom.bin').open('wb') as a,(folder/'rtilde.bin').open('wb') as r:
            for d in dates:
                with np.load(PREP/f'moments/g{gi}/{datekey(d)}.npz') as z:
                    np.asarray(z['denom'],dtype='<f8').tofile(a);np.asarray(z['r_tilde'],dtype='<f8').tofile(r)
        save_json(folder/'dates.json',[datekey(d) for d in dates])
    print('Exported full original-grid estimation moments.',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='moments':moments()
    else:fixtures()
