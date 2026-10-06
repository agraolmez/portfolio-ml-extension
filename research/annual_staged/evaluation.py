"""Past-only candidate scoring, annual selection and uninterrupted portfolio paths."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from annual_ind.selection import author_objective
from baseline.engine import feature_indices
from extension4.estimation import month_moments
from extension4.model import ConditionalModel
from extension4.paths import replay
from extension4.storage import commit_stage, verify_stage, write_json
from temporal_study.paths import AcceptedPathSource
from scripts.industry_projection_kernels_20260928 import project_industry_targets, FACTORS
from .common import RHOS, MULTIPLIERS, candidate, read, sha, finite_equal
from .verification import verify_accounting


def score_year(year,models,history,cache,input_cache,book,out,identity,runtime):
    destination=out/'validation'/str(year)
    context=dict(lock=identity,year=year,models={str(r):m._content_hash() for r,(m,b) in models.items()},
                 states=book.identity)
    def writer(folder):
        rows=[]
        for n,date in enumerate(pd.date_range(f'{year-1}-12-31',f'{year}-11-30',freq='ME'),1):
            day=str(date.date());runtime.guard()
            basis,innovations=input_cache.get_batch(history,day,f'{year}-12-31',cache)
            market=basis.sample.market
            states=book.bind(day,history=13).validate(day,history=13,expected_identity=book.identity)
            for rho,(model,betas) in models.items():
                if model.cutoff!=f'{year-1}-12-31':raise ValueError('Validation model fitted with its own outcome')
                m=month_moments(basis,innovations,model,states,state_names=book.names,
                                interaction_columns=tuple(range(1,len(model.beta))))
                values=author_objective(m['score'],m['risk']+m['cost'],betas)
                weights=m['omega']@betas;trades=m['change']@betas
                direct=(basis.sample.returns@weights-.5*market.gamma*np.sum(weights*(market.covariance@weights),axis=0)
                        -.5*market.wealth*np.sum(market.impact[:,None]*trades**2,axis=0))
                finite_equal(values,direct)
                for j,multiplier in enumerate(MULTIPLIERS):
                    rows.append(dict(candidate=candidate(rho,multiplier),rho=rho,multiplier=multiplier,
                        objective=float(values[j]),decision_eom=day,return_eom=str((date+pd.offsets.MonthEnd(1)).date()),
                        fit_year=year,model_available_at=model.cutoff))
            runtime.event('phase_progress',label=f'{year}: scoring nine candidates',completed=n,total=12,year=year)
        pd.DataFrame(rows).to_parquet(folder/'scores.parquet',index=False)
        write_json(folder/'verification.json',dict(direct_economic_accounting=True,months=12,candidates=9,passed=True))
    commit_stage(destination,context,writer);verify_stage(destination,context)
    return pd.read_parquet(destination/'scores.parquet')


class CheckedPathSource(AcceptedPathSource):
    def market(self,date):
        with self.history.files.preparation():self.current_market=super().market(date)
        return self.current_market
    def innovation(self,market):
        with self.history.files.preparation():return super().innovation(market)
    def realized(self,market):
        with self.history.files.preparation():self.current_outcome=super().realized(market)
        return self.current_outcome


def evaluate_paths(history,book,choices,selections,annual_models,catalog,out,identity,runtime):
    source=CheckedPathSource(history);initial=source.common_initial()
    policies={};base_coefficients={}
    for year in range(1981,2026):
        s=selections[year];name=s['candidate'];rho,multiplier=next((r,m) for r in RHOS for m in MULTIPLIERS if candidate(r,m)==name)
        model,coefficients=annual_models[year][rho]
        policies[year]=ConditionalModel(SimpleNamespace(year=year,seed=1,model=model),'C4',coefficients[:,MULTIPLIERS.index(multiplier)],
                        state_identity=book.identity,protocol_identity=identity,multiplier=multiplier)
        c=choices[year]
        with np.load(catalog.path(f"data/prepared/step9_v1/baseline/coefficients/g{c['gi']}/{year}_p{c['width']}.npz"),allow_pickle=False) as z:
            base_coefficients[year]=z['coefficients'][:,c['ridge_index']].copy()
    target_cache={};projection_checks=[]
    def aims(seed,method,market,bound):
        if seed!=1:raise ValueError('Unapproved seed')
        day=market.decision_date;year=(pd.Timestamp(day)+pd.offsets.MonthEnd(1)).year
        if target_cache.get('day')!=day:
            c=choices[year]
            with np.load(catalog.path(f"data/prepared/step9_v1/baseline/moments/g{c['gi']}/{day}.npz"),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],market.ids)
                base=z['signal_t'][:,feature_indices(c['width'])]@base_coefficients[year]
            # Independently regenerate original features from dated inputs.
            from feature_geometry.features import FeatureMap
            parent=policies[year].parent.model.feature_map.parent
            raw=FeatureMap(parent.frequencies,parent.source).signals(market.characteristics[0],market.volatility[0],np.zeros(115))@base_coefficients[year]
            finite_equal(base,raw)
            conditional=policies[year].aim(market,bound,book.bind(day,history=1))
            with np.load(history.files.path(f'data/prepared/step6_v1/risk/{day}.npz'),allow_pickle=False) as risk:
                np.testing.assert_array_equal(risk['ids'],market.ids)
                projection=project_industry_targets(base,conditional,risk['loadings'],risk['factor_cov'],
                    risk['specific_variance'],market.impact,market.wealth,FACTORS)
            finite_equal(projection['industry_exposure_after'],np.zeros(12),atol=1e-9,rtol=1e-9)
            projection_checks.append(dict(day=day,industry_residual=float(np.max(abs(projection['industry_exposure_after']))),
                kkt_residual=projection['kkt_max_residual']))
            target_cache.clear();target_cache.update(day=day,B1=base,C4=conditional,IND=projection['IND'])
        return target_cache[method]
    def compare(day,arrays,rows):
        verify_accounting(source.current_market,source.current_outcome,arrays,rows)
        # Original seed-1 fixed512 path has exactly these annual baseline coordinates.
        archive=source._archive(day)
        ix=np.flatnonzero((archive['seeds']==1)&(archive['methods']=='fixed512'))
        if len(ix)!=1:raise ValueError('Missing independent accepted baseline path')
        for key in ('pretrade','target','weight','trade','next_pretrade'):
            finite_equal(arrays[key][:,0],archive[key][:,ix[0]],atol=1e-9,rtol=1e-8)
    dates=[str(d.date()) for d in pd.date_range('1980-12-31','2025-11-30',freq='ME')]
    result=replay(source,dates,[(1,'B1'),(1,'C4'),(1,'IND')],initial,aims,destination=out/'paths',
        lock_identity=identity,policy_identity={str(y):p.identity for y,p in policies.items()},atol=1e-11,rtol=1e-10,
        progress=lambda n,d:runtime.event('phase_progress',label='Continuous portfolio evaluation',completed=n,total=540,date=d),compare=compare)
    result.to_parquet(out/'monthly.parquet',index=False)
    return result
