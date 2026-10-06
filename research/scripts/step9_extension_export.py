"""Fixed independent-review interchange for the additional five years."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import FEATURES,write_matrix,datekey
from comparisons.io import *

def main():
    begin('extension_export');ref=OUT/'extension_reference';ref.mkdir()
    for gi in [0,1]:
        folder=ref/f'fit/g{gi}';folder.mkdir(parents=True)
        with (folder/'denom.bin').open('wb') as a,(folder/'rtilde.bin').open('wb') as r:
            for d in ADDED:
                with np.load(BASE/f'moments/g{gi}/{datekey(d)}.npz') as z:
                    np.asarray(z['denom'],dtype='<f8').tofile(a);np.asarray(z['r_tilde'],dtype='<f8').tofile(r)
        save_json(folder/'dates.json',[str(d.date()) for d in ADDED])
    dates=pd.date_range(ADDED[0]-pd.offsets.MonthEnd(12),ADDED[-1],freq='ME')
    cols=['id','eom','valid','me','ret_ld1','tr_ld0','tr_ld1','mu_ld0','mu_ld1','lambda_diagonal','wealth_usd',*FEATURES]
    frame=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=list(dict.fromkeys(cols)),filters=[('eom','>=',dates[0]),('eom','<=',dates[-1])]).sort_values(['eom','id'])
    frame['valid']=frame.valid.astype(int);frame.to_csv(ref/'data.csv',index=False,float_format='%.17g')
    rates=pd.read_parquet(DATA/'monthly_rates.parquet');rates[rates.eom.isin(dates)].to_csv(ref/'rates.csv',index=False,float_format='%.17g')
    risk=[]
    for d in dates:
        folder=ref/f'risk/{datekey(d)}';folder.mkdir(parents=True)
        with np.load(BASE/f'risk/{datekey(d)}.npz') as z:
            for name,array in [('loadings',z['loadings']),('factor',z['factor_cov']),('specific',z['specific_variance'])]:write_matrix(folder/f'{name}.bin',array)
            save_json(folder/'ids.json',z['ids'].tolist());risk.append(dict(date=str(d.date()),n=len(z['ids']),k=z['loadings'].shape[1]))
    samples=[d for d in ADDED if d.month in [12,6] or d==ADDED[-1]];assert len(samples)==11
    save_json(ref/'metadata.json',dict(sample_input_dates=[str(d.date()) for d in samples],risk=risk))
    models=pd.read_parquet(CONF/'extension_annual_models.parquet');models.to_json(ref/'conditional_models.json',orient='records',double_precision=15)
    for row in models.itertuples():
        folder=ref/f'conditional/block{row.block_months}';folder.mkdir(parents=True,exist_ok=True)
        with np.load(CONF/f'coefficients/block{row.block_months}/{row.fit_year}.npz') as z:
            np.savetxt(folder/f'{row.fit_year}_counts.csv',z['counts'][[0,49,99]],delimiter=',',fmt='%d')
            np.savetxt(folder/f'{row.fit_year}_coefficients.csv',z['coefficients'][:,[0,49,99]],delimiter=',',fmt='%.17g')
            save_json(folder/f'{row.fit_year}_indices.json',z['indices'].tolist())
    inventory=[];speed=ref/'speed';speed.mkdir()
    for variant in ['primary','horizon36','blocks24']:
        for year in range(2021,2026):
            path=LEARNER/f'fits/{variant}/{year}.json';fit=json.loads(path.read_text());assert fit['status']=='FITTED'
            with np.load(LEARNER/f'fits/{variant}/{year}_training.npz') as z:
                df=pd.DataFrame(z['x1'],columns=['x'+str(j) for j in range(6)])
                df['b']=z['y'][:,0];df['c']=z['y'][:,1];df['weight']=z['weights'];df['decision']=z['decision_dates']
                df['inner_train']=np.isin(z['decision_dates'],z['inner_train_dates']).astype(int);df['validation']=np.isin(z['decision_dates'],z['validation_dates']).astype(int)
                lookup={date:i for i,date in enumerate(z['dates'])};di=np.array([lookup[d] for d in z['decision_dates']])
                for replica in [1,50,100]:df[f'count{replica}']=z['bootstrap_counts'][replica-1,di]
            csv=speed/f'{variant}_{year}.csv';df.to_csv(csv,index=False,float_format='%.17g')
            inventory.append(dict(variant=variant,year=year,csv=str(csv),fit=str(path),input_sha256=sha(csv),fit_sha256=sha(path)))
    save_json(speed/'inputs.json',inventory)
    finish('extension_export',[p for p in ref.rglob('*') if p.is_file()],sample_input_dates=11,new_speed_fits=15,conditional_reference_fits=30)
if __name__=='__main__':main()
