"""Approved past-only initial parent and original annual baseline selections."""
import numpy as np
import pandas as pd
from baseline.core import ridge_path
from baseline.engine import feature_indices
from annual_ind.selection import author_objective, select_prior_history
from extension4.storage import commit_stage, verify_stage, write_json
from .common import ROOT, ORDER, read

BASE = 'data/prepared/step9_v1/baseline'


def choose_baseline(scores, year, first_return):
    frame = scores.copy()
    frame['candidate'] = (frame.g_index.astype(str)+'/'+frame.p.astype(str)+'/'+
                          frame.lambda_index.astype(str))
    frame['model_available_at'] = (frame.fit_year-1).astype(str)+'-12-31'
    order = list(frame[['g_index','p','lambda_index','candidate']].drop_duplicates()
                 .sort_values(['g_index','p','lambda_index']).candidate)
    return select_prior_history(frame, year=year, candidate_order=order,
                                first_return=first_return, tie_tolerance=0.)


def initial_parent(catalog, destination, lock, runtime):
    context = dict(lock=lock, first_return='1966-01-31', last_return='1970-12-31',
                   last_allowed_decision='1970-11-30', excluded_from_baseline_official_validation=True)
    grid = read(ROOT/'output/step2_20260910/source_settings.json')['settings']['pf_ml']
    def writer(folder):
        all_rows = []
        for gi in (0,1):
            moments = {}
            # December 1970 moments contain January 1971 outcomes and are never read here.
            for date in pd.date_range('1965-11-30','1970-11-30',freq='ME'):
                runtime.guard(); day=str(date.date())
                with np.load(catalog.path(f'{BASE}/moments/g{gi}/{day}.npz'),allow_pickle=False) as saved:
                    moments[date] = saved['r_tilde'].copy(), saved['denom'].copy()
            for year in range(1966,1971):
                train = [v for d,v in moments.items() if d<=pd.Timestamp(year-1,11,30)]
                score = np.mean([v[0] for v in train],axis=0)
                curvature = np.mean([v[1] for v in train],axis=0)
                arrays = {}
                for width in grid['p_vec']:
                    indices=feature_indices(width)
                    beta, diagnostic=ridge_path(curvature[np.ix_(indices,indices)],score[indices],grid['l_vec'])
                    arrays[f'p{width}'] = beta
                    for date in pd.date_range(f'{year-1}-12-31',f'{year}-11-30',freq='ME'):
                        r,a = moments[date]
                        values = author_objective(r[indices], a[np.ix_(indices,indices)], beta)
                        all_rows.append(pd.DataFrame(dict(g_index=gi,p=width,
                            lambda_index=np.arange(len(grid['l_vec'])),objective=values,
                            decision_eom=date,return_eom=date+pd.offsets.MonthEnd(1),fit_year=year)))
                with (folder/f'g{gi}_{year}.npz').open('xb') as f: np.savez_compressed(f,**arrays)
                runtime.event('phase_progress',label=f'Initial parent: baseline validation {year}, bandwidth {gi}',
                              completed=(gi*5+year-1965),total=10)
        scores=pd.concat(all_rows,ignore_index=True)
        scores.to_parquet(folder/'scores.parquet',index=False)
        choice=choose_baseline(scores,1971,'1966-01-31')
        gi,width,index=map(int,choice['candidate'].split('/'))
        write_json(folder/'selection.json',dict(choice,year=1971,gi=gi,width=width,
            ridge_index=index,ridge=grid['l_vec'][index],baseline_only_startup=True))
        write_json(folder/'source_sha256.json',catalog.consumed)
    commit_stage(destination, context, writer)
    verify_stage(destination,context)
    return read(destination/'selection.json')


def parent_choices(catalog, first):
    selection=pd.read_parquet(catalog.path(BASE+'/annual_selection.parquet'))
    result={1971:first}
    for year in range(1972,2026):
        rows=selection[selection.return_eom.eq(pd.Timestamp(year-1,12,31))]
        if len(rows)!=1: raise ValueError('Missing unique preceding baseline selection')
        row=rows.iloc[0]
        result[year]=dict(year=year,width=int(row.p),gi=int(row.g_index),
                          ridge_index=int(row.lambda_index),ridge=float(row['lambda']),
                          selection_available_at=f'{year-1}-12-31')
    return result


def check_original_selections(catalog, choices, runtime):
    scores=pd.read_parquet(catalog.path(BASE+'/validation_scores.parquet'))
    for year in range(1981,2026):
        runtime.guard();selected=choose_baseline(scores,year,'1971-01-31')
        c=choices[year]
        if selected['candidate']!=f"{c['gi']}/{c['width']}/{c['ridge_index']}":
            raise ValueError('Baseline selection differs from accepted original procedure')
    return dict(years_checked=45,first=1981,last=2025,original_grid_candidates=808,passed=True)


def select_extensions(scores, year):
    return select_prior_history(scores,year=year,candidate_order=ORDER,
                                first_return='1971-01-31',tie_tolerance=0.)
