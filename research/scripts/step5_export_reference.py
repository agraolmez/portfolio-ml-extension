"""Export accepted pilot-period inputs for independent actual R execution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.io import ROOT,DATA,PREP,FEATURES,SETTINGS,PROTOCOL,save_json,write_matrix,datekey
import pandas as pd
import numpy as np
OUT=ROOT/'output/step5_20260911';REF=OUT/'reference';REF.mkdir(parents=True,exist_ok=True)

def main():
    first,last=PROTOCOL['periods']['pilot'];decisions=pd.date_range(pd.Timestamp(first)-pd.offsets.MonthEnd(1),pd.Timestamp(last)-pd.offsets.MonthEnd(1),freq='ME')
    dates=pd.date_range(decisions[0]-pd.offsets.MonthEnd(12),decisions[-1],freq='ME')
    cols=['id','eom','valid','me','ret_ld1','tr_ld0','tr_ld1','mu_ld0','mu_ld1','lambda_diagonal','wealth_usd',*FEATURES]
    data=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=list(dict.fromkeys(cols)),filters=[('eom','>=',dates[0]),('eom','<=',dates[-1])]).sort_values(['eom','id'])
    data['valid']=data.valid.astype(int);data.to_csv(REF/'data.csv',index=False,float_format='%.17g')
    rates=pd.read_parquet(DATA/'monthly_rates.parquet');rates[rates.eom.isin(dates)].to_csv(REF/'rates.csv',index=False,float_format='%.17g')
    metadata=[]
    for d in dates:
        folder=REF/f'risk/{datekey(d)}';folder.mkdir(parents=True,exist_ok=True)
        with np.load(PREP/f'risk/{datekey(d)}.npz') as z:
            ids=z['ids'];X=z['loadings'];F=z['factor_cov'];specific=z['specific_variance']
        for name,a in [('loadings',X),('factor',F),('specific',specific)]:write_matrix(folder/f'{name}.bin',a)
        save_json(folder/'ids.json',ids.tolist());metadata.append({'date':datekey(d),'n':len(ids),'k':X.shape[1]})
    save_json(REF/'metadata.json',{'decision_dates':[datekey(d) for d in decisions],'risk':metadata,'full_screened_rows':len(data),'pilot_active_rows':int((data.valid.eq(1)&data.eom.isin(decisions)).sum()),'feature_count':len(FEATURES),'coefficients':'output/step4_20260911/reference/fit/','selection':'output/step4_20260911/reference/r_annual_selection.csv'})
    print('Exported 36 monthly cross-sections, including 12 earlier lookback months.',flush=True)

if __name__=='__main__':main()
