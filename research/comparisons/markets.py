"""Read accepted and extended current-market inputs, separating outcomes."""
import json
import numpy as np
import pandas as pd
from comparisons.extended_learner import Sources
from .io import LEARNER,BASE,CONF,DATES

class Markets(Sources):
    def __init__(self):
        super().__init__()
        self.caps=pd.read_parquet(LEARNER/'matrix_feasibility.parquet').set_index('decision_eom')
        self.models={}
    def current(self,date):
        rows=self.groups[date];ids=rows.id.to_numpy(np.int64);key=f'{date:%Y-%m-%d}'
        # Baseline authority is independent of the optional extension artifacts.
        with np.load(BASE/f'matrices/{key}.npz') as z:
            assert np.array_equal(ids,z['ids']);m=z['m'].copy()
        with np.load(BASE/f'risk/{key}.npz') as z:
            assert np.array_equal(ids,z['ids']);loadings=z['loadings'].copy();fc=z['factor_cov'].copy();specific=z['specific_variance'].copy()
        full=dict(date=str(date.date()),ids=ids,m=m,loadings=loadings,fc=fc,specific=specific,
            aim=self.baseline[date].set_index('id').loc[ids,'aim'].to_numpy(),impact=rows.lambda_diagonal.to_numpy(),wealth=float(rows.wealth_usd.iloc[0]),confidence={})
        for name in ['aim','m','loadings','fc','specific','impact','wealth']:
            if not np.isfinite(full[name]).all():raise ValueError('Corrupt required baseline input '+name)
        for block in [12,24]:
            try:
                path=CONF/f'aims/block{block}/{key}.npz'
                with np.load(path) as z:
                    if not np.array_equal(ids,z['ids']) or not np.allclose(full['aim'],z['central_aim'],atol=1e-12,rtol=1e-9):raise ValueError('Extension target alignment failure')
                    mean=z['mean_aim'].copy()
                metadata=json.loads(path.with_suffix('.json').read_text());numerator=metadata['features']['dispersion_numerator']
                if not np.isfinite(mean).all() or not np.isfinite(numerator) or numerator<0:raise ValueError('Invalid extension input')
                full['confidence'][block]=dict(mean=mean,numerator=numerator)
            except (OSError,ValueError,KeyError):
                # Policy records the fallback; baseline-only methods stay usable.
                full['confidence'][block]={}
        nextdate=date+pd.offsets.MonthEnd(1)
        if nextdate not in self.groups and date!=DATES[-1]:raise ValueError('Missing required next-universe accounting')
        outcome=dict(returns=rows.ret_ld1.to_numpy(),growth=(1+rows.tr_ld1.to_numpy())/(1+float(rows.mu_ld1.iloc[0])),next_ids=self.groups[nextdate].id.to_numpy(np.int64) if nextdate in self.groups else ids.copy())
        if not np.isfinite(outcome['returns']).all() or not np.isfinite(outcome['growth']).all():raise ValueError('Invalid required accounting input')
        return full,outcome,self.caps.loc[date].to_dict()
    def fit(self,variant,date):
        year=date.year+(date.month==12);key=(variant,year)
        if key not in self.models:
            try:self.models[key]=json.loads((LEARNER/f'fits/{variant}/{year}.json').read_text())
            except (OSError,ValueError):self.models[key]=None
        return self.models[key]
