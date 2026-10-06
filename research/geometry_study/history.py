"""Read-only extension adapter with accepted input hashes and complete original history."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from feature_geometry.history import CanonicalHistory,sha
from feature_geometry.moments import MarketMoment

class History(CanonicalHistory):
    def __init__(self,root,receipt):
        super().__init__(root,'1965-11-30','2025-11-30',receipt)
        for rel in ['output/step4_20260911/acceptance_manifest.json','output/step9_20260913/acceptance_manifest.json']:
            manifest=json.loads((self.root/rel).read_text())
            if not all(g['passed'] for g in manifest['gates']):raise ValueError('Prior acceptance has failed gates')
            self.sources.update(manifest['artifact_sha256'])
        extra=pd.read_parquet(self.root/'data/prepared/step3_author_v3/monthly_inputs.parquet',columns=['id','eom','tr_ld1','mu_ld1'],filters=[('eom','>=',pd.Timestamp('1980-12-31')),('eom','<=',self.last)])
        self.outcomes={d:g.sort_values('id').set_index('id') for d,g in extra.groupby('eom')}

    def _accepted(self,relative):
        path=self.root/relative;canonical=str(path.resolve().relative_to(self.root))
        expected=self.sources.get(relative,self.sources.get(canonical));actual=sha(path)
        if expected is None or expected!=actual:raise ValueError('Unaccepted input '+relative)
        self.consumed[relative]=actual
        return path

    def market(self,decision_date):
        day=pd.Timestamp(decision_date)
        if day!=day+pd.offsets.MonthEnd(0) or not self.first<=day<=self.last:raise ValueError('Decision outside loaded month-end history')
        key=str(day.date());base='data/prepared/step9_v1/baseline'
        with np.load(self._accepted(f'{base}/matrices/{key}.npz'),allow_pickle=False) as data:
            ids,m=data['ids'].copy(),data['m'].copy()
        with np.load(self._accepted(f'{base}/risk/{key}.npz'),allow_pickle=False) as data:
            if not np.array_equal(ids,data['ids']):raise ValueError('Risk/matrix IDs differ')
            covariance=data['loadings']@data['factor_cov']@data['loadings'].T+np.diag(data['specific_variance'])
        active=self.months[day];active=active[active.valid]
        if not np.array_equal(ids,active.id):raise ValueError('Eligibility/matrix IDs differ')
        z=[];vol=[];growth=[]
        for lag in range(13):
            history=self.months[day-pd.offsets.MonthEnd(lag)]
            pos=history.index.get_indexer(ids)
            if np.any(pos<0):raise ValueError('Missing historical current-ID characteristic')
            block=history.iloc[pos];z.append(block[self.features].to_numpy());vol.append(block.vol_scale.to_numpy())
            gt=(1+block.tr_ld0.to_numpy())/(1+block.mu_ld0.to_numpy());growth.append(np.where(np.isnan(gt),1.,gt))
        if active.wealth_usd.nunique()!=1:raise ValueError('Inconsistent monthly wealth')
        return MarketMoment(key,ids,tuple(z),tuple(vol),tuple(growth),m,covariance,active.lambda_diagonal.to_numpy(),float(active.wealth_usd.iloc[0]),self.gamma,self.source_id)
