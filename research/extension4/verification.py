"""Independent coefficient solves and accounting from saved actual positions."""
import json
import math
import numpy as np
import pandas as pd
from .storage import verify_stage
from .selection import candidate


def read(path):return json.loads(path.read_text())


def verify_states(book,raw,eligible):
    """Separate sorted-order-statistic and scalar normalization implementation."""
    definition=book.metadata['definition'];normalizer=book.metadata['normalizer']
    rows=eligible.loc[eligible.valid,['id','eom']].merge(raw,on=['id','eom'],validate='one_to_one')
    if len(rows)!=int(eligible.valid.sum()):raise ValueError('Independent state source join is incomplete')
    grouped={pd.Timestamp(d):g for d,g in rows.groupby('eom')};history=[];maximum=0.
    def quantile(values,p):
        pos=(len(values)-1)*p;lower=math.floor(pos);upper=math.ceil(pos)
        return values[lower]+(pos-lower)*(values[upper]-values[lower])
    for saved in book.frame.itertuples(index=False):
        group=grouped.get(saved.eom);values=[];unavailable=0;size=0 if group is None else len(group)
        if group is not None:
            availability=book.metadata['actual_availability_field']
            for row in group.to_dict('records'):
                known=True
                if availability:
                    timestamp=pd.to_datetime(row[availability],utc=True)
                    known=bool(pd.notna(timestamp) and timestamp<=saved.eom.tz_localize('UTC'))
                    unavailable+=int(not known)
                x,me=float(row['be_me']),float(row['me'])
                if known and math.isfinite(x) and x>0 and math.isfinite(me) and me>0:values.append(x)
        values.sort();count=len(values);coverage=count/size if size else 0.
        low,high=(quantile(values,p) for p in definition['quantiles']) if count else (np.nan,np.nan)
        raw_reason=('INSUFFICIENT_OBSERVATIONS' if count<definition['min_count'] else
                    'INSUFFICIENT_COVERAGE' if coverage<definition['min_coverage'] else 'OK')
        current=high-low if raw_reason=='OK' else np.nan
        prior=[v for v in history[-normalizer['previous_calendar_rows']:] if math.isfinite(v)]
        mean=math.fsum(prior)/len(prior) if prior else np.nan
        scale=math.sqrt(math.fsum((v-mean)**2 for v in prior)/len(prior)) if prior else np.nan
        reason=('MISSING_CURRENT' if not math.isfinite(current) else
                'INSUFFICIENT_HISTORY' if len(prior)<normalizer['min_finite_prior'] else
                'DEGENERATE_HISTORY' if not math.isfinite(scale) or scale<=normalizer['scale_floor'] else 'OK')
        unbounded=(current-mean)/scale if reason=='OK' else 0.
        normalized=max(-normalizer['clip'],min(normalizer['clip'],unbounded))
        if (saved.eligible_count,saved.valid_count,saved.unavailable_count,saved.historical_count,
                saved.raw_reason,saved.inactive_reason,bool(saved.active))!=(
                size,count,unavailable,len(prior),raw_reason,reason,reason=='OK'):
            raise ValueError('Independent state eligibility/missingness/normalization classification differs at '+str(saved.eom))
        # Independent reduction order can straddle an exactly-on-boundary clip by an ulp.
        if (abs(abs(unbounded)-normalizer['clip'])>1e-11
                and bool(saved.clipped)!=(unbounded!=normalized)):
            raise ValueError('Independent state clipping classification differs')
        actual=[saved.raw_state,saved.q_low,saved.q_high,saved.coverage,saved.historical_mean,saved.historical_scale,saved.normalized]
        expected=[current,low,high,coverage,mean,scale,normalized]
        np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-10,equal_nan=True)
        maximum=max(maximum,max((abs(a-b) for a,b in zip(actual,expected) if math.isfinite(a)),default=0.))
        history.append(current)
    return dict(months=len(history),maximum_state_error=maximum,publication_timing_certified=False)


def verify_all_fits(study):
    expected=study.config['counts'];economic=neutral=0;maximum=0.
    selected={(r['seed'],r['method']):r['multiplier'] for r in read(study.destination/'selection'/'selected.json')}
    selection_end=pd.Timestamp(study.config['validation_plan']['selection_returns'][1]).year
    for year in range(1981,2026):
        for seed in (1,2,3):
            study.runtime.guard();parent=study.parent(year,seed)
            folder=study.destination/'fits'/str(year)/f'seed{seed}';verify_stage(folder)
            receipt=read(folder/'fit.json')
            if (receipt['training_months']!=181+12*(year-1981) or receipt['first']!='1965-11-30'
                    or receipt['last']!=f'{year-1}-11-30' or receipt['cutoff']!=parent.model.cutoff):
                raise ValueError('Independent annual fit inventory/chronology mismatch')
            wanted={('N4',None)}
            for method in ('C4','CAP4','SIZE4'):
                grid=study.config['penalty_search']['multipliers_in_tie_order'] if year<=selection_end else [selected[seed,method]]
                wanted.update((method,p) for p in grid)
            if {(r['method'],r['multiplier']) for r in receipt['models']}!=wanted or len(receipt['models'])!=len(wanted):
                raise ValueError('Annual fit families/candidates differ from the fixed protocol')
            with np.load(folder/'moments.npz',allow_pickle=False) as moments:
                for record in receipt['models']:
                    method,multiplier=record['method'],record['multiplier']
                    with np.load(folder/record['model_file'],allow_pickle=False) as fitted:beta=fitted['beta']
                    if method=='SIZE4':
                        a=receipt['affine'];ridge=multiplier*parent.settings.ridge*sum(float(b)**2 for b in parent.model.beta)
                        denominator=a['curvature']+ridge
                        if denominator<=0:raise ValueError('Nonpositive scalar objective curvature')
                        optimum=a['score']/denominator;low,high=study.config['size_control']['a_bounds']
                        independent=np.array([min(high,max(low,optimum))]);economic+=1
                    else:
                        risk=moments[method+'_risk'];cost=moments[method+'_cost'];score=moments[method+'_score']
                        penalties=np.full(len(score),parent.settings.ridge)
                        if method!='N4':penalties[len(parent.model.beta):]*=multiplier
                        system=risk+cost+np.diag(penalties)
                        independent=np.linalg.solve(system,score)
                        if method=='N4':
                            neutral+=1
                            np.testing.assert_allclose(independent,parent.model.beta,atol=1e-11,rtol=1e-10)
                        else:economic+=1
                    np.testing.assert_allclose(beta,independent,atol=1e-11,rtol=1e-10)
                    maximum=max(maximum,float(np.max(abs(beta-independent))))
            study.runtime.event('annual_coefficients_independently_verified',year=year,seed=seed)
    if economic!=expected['new_economic_fits'] or neutral!=expected['restricted_parent_checks']:
        raise ValueError('Completed solve counts differ from approved scope')
    return dict(economic_fits=economic,restricted_parent_checks=neutral,maximum_coefficient_error=maximum)


def independent_month(market,outcome,arrays,rows,exits,*,atol,rtol):
    ids=arrays['ids'];next_ids=arrays['next_ids'];h=arrays['pretrade'];target=arrays['target']
    np.testing.assert_array_equal(ids,market.ids);np.testing.assert_array_equal(next_ids,outcome.next_ids)
    weights=market.m@h+(np.eye(len(ids))-market.m)@target
    trades=weights-h
    drift=weights*(1+outcome.total_returns[:,None])/(1+outcome.market_growth)
    lookup={sid:i for i,sid in enumerate(ids)}
    next_h=np.stack([drift[lookup[sid]] if sid in lookup else np.zeros(h.shape[1]) for sid in next_ids])
    for name,expected in [('weight',weights),('trade',trades),('drifted_weight',drift),('next_pretrade',next_h)]:
        np.testing.assert_allclose(arrays[name],expected,atol=atol,rtol=rtol)
    missing=np.flatnonzero(~np.isin(ids,next_ids));expected_exits=[]
    indexed={(int(r['seed']),r['method']):r for r in rows}
    if len(indexed)!=h.shape[1] or len(rows)!=h.shape[1]:raise ValueError('Missing/duplicated accounting rows')
    maximum=0.
    for col,(seed,method) in enumerate(zip(arrays['seeds'],arrays['methods'])):
        row=indexed[int(seed),method]
        gross=sum(float(w)*float(r) for w,r in zip(weights[:,col],outcome.excess_returns))
        cost=market.wealth/2*sum(float(l)*float(q)*float(q) for l,q in zip(market.impact,trades[:,col]))
        if row['decision_eom']!=market.decision_date or row['return_eom']!=str((pd.Timestamp(market.decision_date)+pd.offsets.MonthEnd(1)).date()):
            raise ValueError('Accounting row date mismatch')
        values=np.array([gross,cost,gross-cost,float(np.abs(trades[:,col]).sum()),float(np.abs(weights[:,col]).sum()),float(weights[:,col].sum())])
        actual=np.array([row[k] for k in ['gross_excess','impact_cost','net_excess','turnover','gross_exposure','net_exposure']])
        np.testing.assert_allclose(actual,values,atol=atol,rtol=rtol);maximum=max(maximum,float(np.max(abs(values-actual))))
        absolute=abs(weights[:,col]);gross_exposure=absolute.sum()
        concentration=float(np.sum((absolute/gross_exposure)**2)) if gross_exposure else 0.
        variance=float(np.einsum('i,ij,j->',weights[:,col],market.covariance,weights[:,col]))
        if variance < -1e-12:raise ValueError('Negative independent modeled variance')
        np.testing.assert_allclose([row['largest_absolute_weight'],row['gross_weight_concentration'],row['modeled_annual_volatility']],
            [float(absolute.max()),concentration,np.sqrt(12*max(0.,variance))],atol=atol,rtol=rtol)
        for i in missing:expected_exits.append((int(seed),str(method),int(ids[i]),float(drift[i,col])))
    if len(exits)!=len(expected_exits):raise ValueError('Exit ledger is incomplete')
    for row,expected in zip(exits,expected_exits):
        if (row['seed'],row['method'],row['id'])!=expected[:3] or row['liquidation_cost_added'] is not False:
            raise ValueError('Exit convention changed')
        np.testing.assert_allclose(row['omitted_drifted_weight'],expected[3],atol=atol,rtol=rtol)
    return maximum


def verify_financial_paths(study):
    dates=pd.date_range('1980-12-31','2025-11-30',freq='ME')
    previous_ids=study.initial.ids;previous=np.repeat(study.initial.holdings[:,None],21,axis=1)
    maximum=0.;exit_count=0;verified_rows=[]
    for i,date in enumerate(dates):
        study.runtime.guard();key=str(date.date());folder=study.destination/'final_paths'/key
        verify_stage(folder)
        with np.load(folder/'positions.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        np.testing.assert_array_equal(arrays['ids'],previous_ids)
        np.testing.assert_array_equal(arrays['pretrade'],previous)
        rows=read(folder/'monthly.json');exits=read(folder/'exits.json')
        verified_rows.extend(rows)
        market=study.source.market(key);outcome=study.source.realized(market)
        maximum=max(maximum,independent_month(market,outcome,arrays,rows,exits,atol=1e-11,rtol=1e-10))
        study.compare(key,arrays,rows)
        for seed in (1,2,3):
            n=np.flatnonzero((arrays['seeds']==seed)&(arrays['methods']=='N4'))[0]
            p=np.flatnonzero((arrays['seeds']==seed)&(arrays['methods']=='P3'))[0]
            for name in ['pretrade','target','weight','trade','drifted_weight','next_pretrade']:
                np.testing.assert_allclose(arrays[name][:,n],arrays[name][:,p],atol=1e-11,rtol=1e-10)
        previous_ids=arrays['next_ids'];previous=arrays['next_pretrade'];exit_count+=len(exits)
        if (i+1)%12==0:study.runtime.event('accounting_independently_verified',months=i+1)
    study.source.unchanged()
    # Statistical analysis must consume exactly the independently checked rows.
    verify_stage(study.destination/'final_monthly',dict(lock=study.lock))
    saved=pd.read_parquet(study.destination/'final_monthly'/'monthly.parquet')
    pd.testing.assert_frame_equal(saved.reset_index(drop=True),pd.DataFrame(verified_rows).reset_index(drop=True),check_like=True)
    return dict(months=540,paths=21,path_months=11340,neutral_pairs=1620,maximum_accounting_error=maximum,
                exit_ledger_rows=exit_count,exit_liquidation_charged=False,holdings_resets=0)
