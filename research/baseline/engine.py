"""Date-aware, cached baseline estimation and execution; source grids preserved."""
import argparse
import hashlib
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from .io import ROOT,DATA,SOURCE,OUT,PREP,SETTINGS,PROTOCOL,FEATURES,sha,save_json,verify_data,guard,datekey,finite,write_matrix
from .core import inertia,normalize_signal,moment,ridge_path,transition,report_returns


def feature_indices(p,pmax=512):
    if p not in SETTINGS['settings']['pf_ml']['p_vec']:raise ValueError('Feature size outside original grid')
    return np.r_[0,np.arange(1,1+p//2),np.arange(1+pmax//2,1+pmax//2+p//2)]


class Engine:
    def __init__(self,end='2025-11-30',verify=True):
        self.verification=verify_data() if verify else {'validation_fixture':True}
        self.end=pd.Timestamp(end)
        self.spec=SETTINGS['settings'];self.pmax=max(self.spec['pf_ml']['p_vec']);self.first=pd.Timestamp('1965-11-30')
        columns=['id','eom','valid','me','ret_ld1','tr_ld0','tr_ld1','mu_ld0','mu_ld1','lambda_diagonal','wealth_usd',*FEATURES]
        self.data=pd.read_parquet(DATA/'monthly_inputs.parquet',columns=list(dict.fromkeys(columns)),filters=[('eom','>=',pd.Timestamp('1964-11-30')),('eom','<=',self.end)])
        self.scales=pd.read_parquet(PREP/'risk/scales.parquet')
        self.data=self.data.merge(self.scales[['id','eom','vol_scale']],on=['id','eom'],how='left',validate='one_to_one')
        self.data['vol_scale']=self.data.vol_scale.fillna(self.data.groupby('eom').vol_scale.transform('median'))
        if self.data.vol_scale.isna().any():raise ValueError('Unresolved source median covariance scale')
        self.months={pd.Timestamp(d):g.sort_values('id').set_index('id',drop=False) for d,g in self.data.groupby('eom',sort=True)}
        self.rates=pd.read_parquet(DATA/'monthly_rates.parquet').set_index('eom')
        self.rf={pd.Timestamp(d):float(v) for d,v in self.rates.rf_french.items()}
        self.rff_cache=OrderedDict();self.matrix_diagnostics={}
        self.weights=[np.fromfile(OUT/f'reference/rff_g{i}.bin',dtype='<f8').reshape(len(FEATURES),self.pmax//2) for i in range(len(self.spec['pf_ml']['g_vec']))]
        self.code_fingerprint=hashlib.sha256(''.join(sha(ROOT/'baseline'/f) for f in ['core.py','engine.py','risk.py','io.py']).encode()).hexdigest()

    def covariance(self,d):
        with np.load(PREP/f'risk/{datekey(d)}.npz',allow_pickle=False) as z:
            ids=z['ids'];X=z['loadings'];factor=z['factor_cov'];specific=z['specific_variance']
        sigma=X@factor@X.T+np.diag(specific)
        return ids,sigma

    def matrix(self,d):
        d=pd.Timestamp(d);g=self.months[d];active=g[g.valid]
        ids,sigma=self.covariance(d)
        if not np.array_equal(ids,active.id.to_numpy()):raise ValueError('Risk/universe ID misalignment')
        wealth=active.wealth_usd.iloc[0];impact=active.lambda_diagonal.to_numpy()
        key=hashlib.sha256((self.code_fingerprint+sha(PREP/f'risk/{datekey(d)}.npz')+self.verification.get('checksums_sha256','fixture')+sha(ROOT/'config/protocol_v1.json')).encode()).hexdigest()
        path=PREP/f'matrices/{datekey(d)}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['fingerprint'].item())==key:
                    diagnostics=json.loads(str(z['diagnostics'].item()));self.matrix_diagnostics[datekey(d)]=diagnostics
                    return ids,sigma,impact,float(wealth),z['m'].copy()
        m,diag=inertia(sigma,impact,wealth,self.rf[d],mu=PROTOCOL['economic']['source_monthly_growth'],gamma=PROTOCOL['economic']['risk_aversion'])
        diag.update(date=datekey(d),n_stocks=len(ids));self.matrix_diagnostics[datekey(d)]=diag
        np.savez_compressed(path,ids=ids,m=m,fingerprint=np.array(key),diagnostics=np.array(json.dumps(diag)))
        return ids,sigma,impact,float(wealth),m

    def raw_signals(self,d,gidx):
        key=(d,gidx)
        if key in self.rff_cache:
            self.rff_cache.move_to_end(key);return self.rff_cache[key]
        frame=self.months[d];arg=frame[FEATURES].to_numpy()@self.weights[gidx]
        result=np.column_stack([np.cos(arg),np.sin(arg)])
        self.rff_cache[key]=result
        while len(self.rff_cache)>30:self.rff_cache.popitem(last=False)
        return result

    def inputs(self,d,gidx,operators=False):
        d=pd.Timestamp(d);ids,sigma,impact,wealth,m=self.matrix(d)
        signals=[];growth=[]
        for lag in range(13):
            day=d-pd.offsets.MonthEnd(lag);frame=self.months.get(day)
            if frame is None:raise ValueError('Missing lookback month')
            pos=frame.index.get_indexer(ids)
            if np.any(pos<0):raise ValueError('Missing current-ID history')
            block=frame.iloc[pos]
            signals.append(normalize_signal(self.raw_signals(day,gidx)[pos],block.vol_scale.to_numpy()))
            gt=(1+block.tr_ld0.to_numpy())/(1+block.mu_ld0.to_numpy())
            gt=np.where(np.isnan(gt),1.,gt) # explicit source fallback
            growth.append(gt)
        active=self.months[d].loc[ids]
        result=moment(signals,growth,m,sigma,impact,wealth,active.ret_ld1.to_numpy(),gamma=PROTOCOL['economic']['risk_aversion'],return_operators=operators)
        result['ids']=ids
        return result

    def build_moments(self,dates=None):
        dates=list(pd.date_range(self.first,self.end,freq='ME')) if dates is None else list(map(pd.Timestamp,dates))
        fingerprint=hashlib.sha256((self.code_fingerprint+self.verification.get('checksums_sha256','fixture')+sha(PREP/'risk/factor_covariances.npz')+sha(PREP/'risk/scales.parquet')).encode()).hexdigest()
        for gidx,g in enumerate(self.spec['pf_ml']['g_vec']):
            folder=PREP/f'moments/g{gidx}';folder.mkdir(parents=True,exist_ok=True)
            for i,d in enumerate(dates):
                path=folder/f'{datekey(d)}.npz'
                key=fingerprint+sha(OUT/f'reference/rff_g{gidx}.bin')
                reuse=False
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:reuse=str(z['fingerprint'].item())==key
                if not reuse:
                    result=self.inputs(d,gidx)
                    np.savez_compressed(path,**result,fingerprint=np.array(key))
                if i%12==0 or i==len(dates)-1:
                    print(f'Moment inputs: g={g:.6f}, through {datekey(d)} ({i+1}/{len(dates)})',flush=True);guard()
        pd.DataFrame(self.matrix_diagnostics.values()).to_csv(OUT/'matrix_diagnostics.csv',index=False)
        save_json(OUT/'moment_build.json',{'months':len(dates),'bandwidths':len(self.weights),'first':datekey(dates[0]),'last':datekey(dates[-1]),'p_max':self.pmax,'fingerprint':fingerprint,'portfolio_weights_run':False,'resources':guard()})

    def fit(self,end_year=1982):
        pvec=self.spec['pf_ml']['p_vec'];lvec=np.asarray(self.spec['pf_ml']['l_vec']);years=range(1971,end_year+1)
        validations=[];fit_diagnostics=[]
        for gidx,g in enumerate(self.spec['pf_ml']['g_vec']):
            reals={}
            for path in sorted((PREP/f'moments/g{gidx}').glob('*.npz')):
                d=pd.Timestamp(path.stem)
                if d<=pd.Timestamp(end_year,11,30):
                    with np.load(path,allow_pickle=False) as z:reals[d]=(z['r_tilde'].copy(),z['denom'].copy())
            expected=pd.date_range(self.first,pd.Timestamp(end_year,11,30),freq='ME')
            if set(expected)!=set(reals):raise ValueError('Incomplete estimation/validation moment calendar')
            for year in years:
                train=[v for d,v in reals.items() if d<=pd.Timestamp(year-1,11,30)]
                r=np.sum([v[0] for v in train],axis=0)/len(train);a=np.sum([v[1] for v in train],axis=0)/len(train)
                for p in pvec:
                    idx=feature_indices(p,self.pmax);coefs,diag=ridge_path(a[np.ix_(idx,idx)],r[idx],lvec)
                    folder=PREP/f'coefficients/g{gidx}';folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{year}_p{p}.npz',coefficients=coefs,lambdas=lvec,indices=idx,training_months=len(train),training_cutoff=np.array(f'{year-1}-11-30'))
                    fit_diagnostics.append({'g_index':gidx,'year':year,'p':p,'training_months':len(train),**diag})
                    for d in pd.date_range(f'{year-1}-12-31',f'{year}-11-30',freq='ME'):
                        rt,den=reals[d];coeff=coefs
                        values=rt[idx]@coeff-.5*np.sum(coeff*(den[np.ix_(idx,idx)]@coeff),axis=0)
                        validations.append(pd.DataFrame({'decision_eom':d,'return_eom':d+pd.offsets.MonthEnd(1),'g_index':gidx,'g':g,'p':p,'lambda_index':np.arange(len(lvec)),'lambda':lvec,'objective':values,'fit_year':year}))
                print(f'Annual coefficient and validation checks: g={gidx}, {year}',flush=True)
            del reals
        v=pd.concat(validations,ignore_index=True).sort_values(['g_index','p','lambda_index','return_eom'])
        v['cumulative_objective']=v.groupby(['g_index','p','lambda_index']).objective.transform(lambda s:s.expanding().mean())
        december=v[v.return_eom.dt.month.eq(12)]
        choices=[];within=[]
        for year,group in december.groupby(december.return_eom.dt.year):
            for gidx,gg in group.groupby('g_index'):
                best=gg[gg.cumulative_objective.eq(gg.cumulative_objective.max())]
                if len(best)!=1:raise ValueError(f'No unique source within-bandwidth selection for {year}, g{gidx}')
                within.append(best)
            best=group[group.cumulative_objective.eq(group.cumulative_objective.max())]
            if len(best)!=1:raise ValueError(f'No unique source global selection for {year}')
            choices.append(best)
        v.to_parquet(PREP/'validation_scores.parquet',index=False)
        pd.concat(choices).to_parquet(PREP/'annual_selection.parquet',index=False)
        pd.concat(within).to_parquet(PREP/'annual_selection_by_bandwidth.parquet',index=False)
        pd.DataFrame(fit_diagnostics).to_csv(OUT/'coefficient_diagnostics.csv',index=False)
        save_json(OUT/'fit_build.json',{'coefficient_fits':len(fit_diagnostics)*len(lvec),'annual_fit_blocks':len(fit_diagnostics),'validation_rows':len(v),'selection_years':len(choices),'first_fit_year':min(years),'last_fit_year':end_year,'original_grid':{'g':self.spec['pf_ml']['g_vec'],'p':pvec,'lambda_count':len(lvec),'zero_penalty_preserved':bool(lvec[0]==0)},'unique_global_and_within_bandwidth_selection':True,'portfolio_weights_run':False,'resources':guard()})

    def aims(self,dates):
        selections=pd.read_parquet(PREP/'annual_selection.parquet');frames=[]
        for d in map(pd.Timestamp,dates):
            year=(d+pd.offsets.MonthEnd(1)).year
            choice=selections[selections.return_eom.dt.year.eq(year-1)]
            if len(choice)!=1:raise ValueError('Missing unique prior-year selection')
            row=choice.iloc[0];gidx=int(row.g_index);p=int(row.p);li=int(row.lambda_index)
            with np.load(PREP/f'coefficients/g{gidx}/{year}_p{p}.npz',allow_pickle=False) as z:coef=z['coefficients'][:,li];idx=z['indices']
            with np.load(PREP/f'moments/g{gidx}/{datekey(d)}.npz',allow_pickle=False) as z:aim=z['signal_t'][:,idx]@coef;ids=z['ids']
            frames.append(pd.DataFrame({'id':ids,'eom':d,'aim':aim,'g_index':gidx,'p':p,'lambda_index':li,'fit_year':year,'selection_return_year':year-1}))
        return pd.concat(frames,ignore_index=True)

    def run(self,dates,aims,require_step4=True):
        if require_step4:
            accepted=json.loads((OUT/'acceptance_manifest.json').read_text())
            if accepted.get('status')!='ACCEPTED_BASELINE_MACHINERY':raise ValueError('Step 4 acceptance required before financial pilot')
            for rel,digest in accepted['artifact_sha256'].items():
                if sha(ROOT/rel)!=digest:raise ValueError(f'Stale accepted baseline artifact: {rel}')
        dates=list(map(pd.Timestamp,dates));holdings=None;monthly=[];positions=[];exits=[]
        if list(pd.date_range(dates[0],dates[-1],freq='ME'))!=dates:raise ValueError('Execution dates must be consecutive months')
        for i,d in enumerate(dates):
            ids,sigma,impact,wealth,m=self.matrix(d);frame=self.months[d].loc[ids]
            target=aims[aims.eom.eq(d)].set_index('id').reindex(ids)
            if target.aim.isna().any():raise ValueError('Missing aim')
            if holdings is None:holdings=frame.me.to_numpy()/frame.me.sum()
            next_ids=self.months[dates[i+1]].query('valid').id.to_numpy() if i+1<len(dates) else ids
            result=transition(ids,holdings,target.aim.to_numpy(),m,frame.tr_ld1.to_numpy(),frame.mu_ld1.iloc[0],impact,wealth,frame.ret_ld1.to_numpy(),next_ids)
            monthly.append({'decision_eom':d,'return_eom':d+pd.offsets.MonthEnd(1),'gross_excess':result['gross_excess'],'impact_cost':result['impact_cost'],'net_excess':result['gross_excess']-result['impact_cost'],'turnover':result['turnover'],'gross_exposure':result['gross_exposure']})
            positions.append(pd.DataFrame({'id':ids,'eom':d,'pretrade':holdings,'aim':target.aim.to_numpy(),'weight':result['weights'],'trade':result['trades'],'drifted_weight':result['drifted_weights']}))
            if len(result['dropped_ids']):exits.append(pd.DataFrame({'id':result['dropped_ids'],'decision_eom':d,'omitted_next_weight':result['dropped_drifted_weights'],'source_exit_trade_cost':0.}))
            holdings=result['next_pretrade']
        return pd.DataFrame(monthly),pd.concat(positions,ignore_index=True),pd.concat(exits,ignore_index=True) if exits else pd.DataFrame()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['moments','fit','aims']);parser.add_argument('--end',default='1982-11-30');args=parser.parse_args()
    engine=Engine(end=args.end)
    if args.stage=='moments':engine.build_moments()
    elif args.stage=='fit':engine.fit(pd.Timestamp(args.end).year)
    else:
        dates=pd.date_range('1980-12-31',args.end,freq='ME');a=engine.aims(dates);a.to_parquet(PREP/'pilot_aims_preflight.parquet',index=False)
