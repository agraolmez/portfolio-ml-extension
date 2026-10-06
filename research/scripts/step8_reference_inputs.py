"""Export fixed numerical review inputs; no additional financial experiment."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.io import save_json,sha
from learner.pipeline import OUT,PREP,VARIANTS

def main():
    folder=OUT/'reference';folder.mkdir(exist_ok=True);inventory=[]
    for variant in VARIANTS:
        for year in range(1981,2021):
            path=PREP/f'fits/{variant}/{year}.json';fit=json.loads(path.read_text())
            if fit['status']!='FITTED':continue
            with np.load(PREP/f'fits/{variant}/{year}_training.npz') as z:
                df=pd.DataFrame(z['x1'],columns=['x'+str(j) for j in range(6)])
                df['b']=z['y'][:,0];df['c']=z['y'][:,1];df['weight']=z['weights'];df['decision']=z['decision_dates']
                df['inner_train']=np.isin(z['decision_dates'],z['inner_train_dates']).astype(int)
                df['validation']=np.isin(z['decision_dates'],z['validation_dates']).astype(int)
                lookup={date:i for i,date in enumerate(z['dates'])};di=np.array([lookup[d] for d in z['decision_dates']])
                for replica in [1,50,100]:df[f'count{replica}']=z['bootstrap_counts'][replica-1,di]
            csv=folder/f'{variant}_{year}.csv';df.to_csv(csv,index=False)
            inventory.append(dict(variant=variant,year=year,csv=str(csv),fit=str(path),input_sha256=sha(csv),fit_sha256=sha(path)))
    save_json(folder/'inputs.json',inventory)
    print(f'Exported {len(inventory)} annual variant fits for independent R review.')
if __name__=='__main__':main()
