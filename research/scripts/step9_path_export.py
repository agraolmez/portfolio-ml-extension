"""Export all current inputs and expected arrays for independent R path replay."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import write_matrix,datekey
from comparisons.io import *
from comparisons.markets import Markets

def main():
    begin('path_export');ref=OUT/'path_reference';ref.mkdir();source=Markets();metadata=[]
    for period,(first,last) in PERIODS.items():
        path=PREP/'paths'/period;blockmeta=json.loads((path/'path_metadata.json').read_text());save_json(ref/f'{period}_columns.json',blockmeta['columns'])
        for d in pd.date_range(first,last,freq='ME'):
            x,y,cap=source.current(d);folder=ref/f'months/{datekey(d)}';folder.mkdir(parents=True)
            frame=pd.DataFrame(dict(id=x['ids'],aim=x['aim'],impact=x['impact'],returns=y['returns'],growth=y['growth']))
            frame.to_csv(folder/'current.csv',index=False,float_format='%.17g')
            for name,array in [('m',x['m']),('loadings',x['loadings']),('factor',x['fc']),('specific',x['specific'])]:write_matrix(folder/f'{name}.bin',array)
            for block in [12,24]:
                with np.load(CONF/f'aims/block{block}/{datekey(d)}.npz') as z:write_matrix(folder/f'replicas{block}.bin',z['replica_aims'])
            with np.load(path/f'{datekey(d)}.npz') as z:
                for name in ['pretrade','target','trade','weight','drifted_weight']:write_matrix(folder/f'python_{name}.bin',z[name])
                save_json(folder/'next_ids.json',z['next_ids'].tolist())
            if str(d.date())==first:
                initial=source.baseline[d].set_index('id').loc[x['ids'],'pretrade'].to_numpy();write_matrix(ref/f'{period}_initial.bin',initial)
            metadata.append(dict(period=period,date=str(d.date()),n=len(x['ids']),k=x['loadings'].shape[1],wealth=x['wealth'],columns=len(blockmeta['columns']),
                kappa_max=cap['kappa_max'],numerator12=x['confidence'][12]['numerator'],numerator24=x['confidence'][24]['numerator']))
        pd.read_parquet(path/'monthly.parquet').drop(columns='out_of_range_features').to_csv(ref/f'python_{period}_monthly.csv',index=False,float_format='%.17g')
    save_json(ref/'metadata.json',metadata)
    finish('path_export',[p for p in ref.rglob('*') if p.is_file()],months=300,comparison_month_records=8460)
if __name__=='__main__':main()
