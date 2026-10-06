"""Annual candidate fitting with explicit, hash-checked accepted-artifact reuse."""
from dataclasses import asdict, replace
import gc
import json
import time
import numpy as np
import pandas as pd
from feature_geometry.features import FeatureMap, fingerprint
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings, _solve
from geometry_study.features import CachedFeatureMap
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.model import TemporalModel
from temporal_numerics.optimizer import optimize, LBFGSBSettings
from temporal_fast_v4.settings import VARIANTS
from extension4.estimation import month_moments
from annual_ind.cached_inputs import ReusedYearInputs
from annual_ind.coordinate_expanded import load_bundle
from annual_ind.streaming_geometry_expanded import GeometryObjective
from annual_ind.temporal_reuse_expanded import SharedTemporalProfiles
from annual_ind.parallel_expanded import conditional_sums
from annual_ind.conditional import solve_conditional
from annual_ind.pipeline_probe import plain
from .common import ROOT, E2, E3, E4, RHOS, MULTIPLIERS, read, atomic, sha, finite_equal


class YearInputs(ReusedYearInputs):
    """Same audited sample iterator, explicitly extended to the approved 2025 cutoff."""
    def __init__(self,history,cache,year,guard,event,*,input_cache):
        if type(year) is not int or not 1971<=year<=2025: raise ValueError('Unapproved annual year')
        self.history,self.cache,self.year,self.guard,self.event=history,cache,year,guard,event
        self.cutoff=f'{year-1}-12-31'
        self.dates=tuple(str(d.date()) for d in pd.date_range('1965-11-30',f'{year-1}-11-30',freq='ME'))
        self.identity=None;self.passes=0;self.input_cache=input_cache


class Archived:
    def __init__(self,catalog,history,book): self.catalog,self.history,self.book=catalog,history,book

    def geometry(self,c,frequencies,settings):
        y=c['year']
        if y<1981:return None
        rel=f'{E2}/fits/{y}/seed1/full'
        fit=self.catalog.read(rel+'/fit.json')
        model=GeometryModel.load(self.catalog.path(rel+'/model.npz'))
        if (fit['bandwidth_index']!=c['gi'] or fit['coefficient_ridge']!=c['ridge']
            or fit['cutoff']!=settings.cutoff or fit['training_last']!=f'{y-1}-11-30'
            or fit['training_months']!=12*(y-1966)+1): raise ValueError('Archived scale chronology differs')
        np.testing.assert_array_equal(model.feature_map.frequencies,frequencies)
        for k,v in asdict(settings).items():
            if k!='scale_penalty' and fit['settings'][k]!=v:raise ValueError('Archived scale procedure differs')
        finite_equal(fit['settings']['scale_penalty'],.01*fit['objective_normalizer'])
        return model,fit,dict(receipt=rel+'/fit.json',receipt_sha256=sha(self.catalog.path(rel+'/fit.json')),
                              model=rel+'/model.npz',model_sha256=sha(self.catalog.path(rel+'/model.npz')))

    def temporal(self,c,rho,theta,frequencies,groups,settings,optimizer):
        if c['year']<1981 or (c['year']>2010 and rho!=.1):return None
        name=str(rho).replace('.','p');rel=f"{E3}/fits/{c['year']}/seed1/temporal/rho_{name}"
        fit=self.catalog.read(rel+'/fit.json');self.catalog.path(rel+'/fit_commit.json')
        model=TemporalModel.load(self.catalog.path(rel+'/model.npz'),expected_parent_fingerprint=fit['parent_fingerprint'])
        if (not fit['converged'] or not fit['eligible_for_portfolio_evaluation'] or not fit['neutral_gate']['passed']
            or fit['rho']!=rho or model.cutoff!=settings.cutoff
            or fit['parent_settings']!=asdict(settings) or fit['optimizer_settings']!=asdict(optimizer)
            or model.source_fingerprint!=self.history.source_fingerprint
            or model.reference_fingerprint!=self.history.reference_fingerprint):
            raise ValueError('Archived temporal specification differs')
        for actual,expected in [(model.feature_map.theta,theta),(model.feature_map.parent.frequencies,frequencies),
                                (model.feature_map.groups,groups)]:np.testing.assert_array_equal(actual,expected)
        for rel_input,digest in fit['accepted_inputs_sha256'].items():
            self.catalog.path(rel_input)
            if self.catalog.consumed[rel_input]!=digest:raise ValueError('Archived training input differs')
        return model,fit,dict(receipt=rel+'/fit.json',model=rel+'/model.npz',
                              model_sha256=sha(self.catalog.path(rel+'/model.npz')))

    def conditional(self,c,rho,model):
        if c['year']<1981 or rho!=.1:return None
        rel=f"{E4}/fits/{c['year']}/seed1"
        fit=self.catalog.read(rel+'/fit.json')
        if (fit['context']['parent']!=model._content_hash() or fit['context']['states']!=self.book.identity
            or fit['cutoff']!=model.cutoff or fit['parent_ridge']!=c['ridge']):
            raise ValueError('Archived conditional parent/state differs')
        with np.load(self.catalog.path(rel+'/moments.npz'),allow_pickle=False) as z:
            score=z['C4_score'].copy();curvature=z['C4_risk']+z['C4_cost']
        return score,curvature,dict(moments=rel+'/moments.npz',moments_sha256=sha(self.catalog.path(rel+'/moments.npz')))


def fit_year(c,history,cache,input_cache,book,catalog,store,runtime,out,verify_system,*,check_archived_neutral=False):
    started=time.monotonic();year=c['year'];guard=runtime.guard
    inputs=YearInputs(history,cache,year,guard,runtime.event,input_cache=input_cache)
    directions=history.files.path(f"{E2}/directions/seed1_g{c['gi']}.bin")
    frequency_source=f"R_MASS_seed_1_g{c['gi']}_nested_512_source_{sha(directions)}"
    fmap=CachedFeatureMap(np.fromfile(directions,dtype='<f8').reshape(115,512)[:,:c['width']//2],frequency_source)
    dictionary=pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'))
    if dictionary.characteristic.tolist()!=list(history.features) or dictionary.cluster.nunique()!=13:
        raise ValueError('Accepted characteristic/theme ordering differs')
    groups=dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
    gp=read(ROOT/'config/feature_geometry_study_v1.json')['geometry_settings']
    settings=FitSettings(inputs.cutoff,ridge=c['ridge'],scale_penalty=0.,**{k:v for k,v in gp.items() if k!='relative_scale_penalty'})
    context=dict(coordinate=c,source=history.source_fingerprint,reference=history.reference_fingerprint,
                 frequency=fmap.fingerprint,geometry_settings=asdict(settings),training_months=len(inputs))
    archive=Archived(catalog,history,book);key=str(year);saved=load_bundle(store,key+'/E2',context)
    if saved is None:
        tick=time.monotonic();old=archive.geometry(c,fmap.frequencies,settings)
        if old is not None:
            model,receipt,origin=old
            arrays=dict(theta=model.theta,beta=model.beta)
            meta=dict(context,normalizer=receipt['objective_normalizer'],origin=origin,status=receipt['status'],new_fit=False)
        else:
            objective=GeometryObjective(inputs.bases,len(inputs),fmap,settings,workers=12,guard=guard)
            zero=objective.evaluate(np.zeros(115));normalizer=zero['value']
            if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('Invalid objective normalizer')
            with np.load(history.files.path(f"data/prepared/step9_v1/baseline/coefficients/g{c['gi']}/{year}_p{c['width']}.npz"),allow_pickle=False) as z:
                expected=z['coefficients'][:,c['ridge_index']]
            finite_equal(zero['beta'],expected,atol=1e-9,rtol=1e-8)
            settings=replace(settings,scale_penalty=.01*normalizer);objective.settings=settings
            result=objective.fit(normalizer,lambda row:runtime.event('phase_progress',
                label=f'{year}: learning characteristic scales',completed=row['iteration'],total=8,year=year))
            arrays=dict(theta=result['theta'],beta=result['beta'])
            meta=dict(context,normalizer=normalizer,new_fit=True,status=result['status'],
                      result=plain({k:v for k,v in result.items() if k not in ('theta','beta','geometry_beta')}))
            del objective,result,zero;gc.collect()
        meta['seconds']=time.monotonic()-tick;store.put(key+'/E2',meta,arrays);saved=meta,arrays
    gm,ga=saved;settings=replace(settings,scale_penalty=.01*gm['normalizer'])
    temporal_map=TemporalFeatureMap(FeatureMap(fmap.frequencies,fmap.source),ga['theta'],groups)
    parent=fingerprint(ga['theta'],ga['beta'],temporal_map.fingerprint,settings.cutoff)
    optimizer=LBFGSBSettings(**read(ROOT/'config/extension3_study_v3_proposed.json')['optimizer'])
    bank=None;models={};summaries=[]
    for rho in RHOS:
        guard();runtime.event('phase_progress',label=f'{year}: temporal candidate {rho}',year=year,completed=None,total=None)
        tc=dict(context,parent=parent,rho=rho,optimizer=asdict(optimizer))
        saved=load_bundle(store,key+f'/rho{rho}/E3',tc)
        if saved is None:
            tick=time.monotonic();old=archive.temporal(c,rho,ga['theta'],fmap.frequencies,groups,settings,optimizer)
            if old is not None:
                model,receipt,origin=old
                tm=dict(tc,new_fit=False,origin=origin,model_metadata=model._metadata(),
                        converged=True,neutral_gate=receipt['neutral_gate'],evaluations=receipt['evaluation_count'])
                ta=dict(gains=model.gains,beta=model.beta,lower=model.lower,upper=model.upper)
            else:
                if bank is None:
                    bank=SharedTemporalProfiles(inputs.batches,temporal_map,settings,maximum_bytes=128*2**20,
                        guard=guard,objective_scale=gm['normalizer'],workers=12,trig_backend='accelerate',**dict(VARIANTS,reuse_bytes=128*2**20))
                evaluate=bank.for_restraint(rho);neutral={}
                def checked(gains,with_gradient=True):
                    payload=evaluate(gains,with_gradient=with_gradient)
                    if not neutral:
                        np.testing.assert_array_equal(gains,np.ones(13));finite_equal(payload['beta'],ga['beta'])
                        neutral.update(passed=True,maximum_beta_error=float(np.max(abs(payload['beta']-ga['beta']))))
                    if payload['training_months']!=len(inputs):raise ValueError('Temporal history differs')
                    return payload
                result=optimize(checked,13,optimizer,progress=lambda row:runtime.event('temporal_optimizer',year=year,rho=rho,details=plain(row)))
                atomic(out/f'optimizer_{year}_{rho}_{time.time_ns()}.json',plain(result));guard()
                if not result['converged'] or not neutral.get('passed'):raise ValueError('Temporal convergence gate failed')
                model=TemporalModel(temporal_map,result['gains'],result['beta'],settings.cutoff,settings.cutoff,parent,
                    history.source_fingerprint,history.reference_fingerprint,np.full(13,optimizer.lower),np.full(13,optimizer.upper))
                tm=dict(tc,new_fit=True,model_metadata=model._metadata(),converged=True,neutral_gate=neutral,evaluations=result['evaluation_count'])
                ta=dict(gains=model.gains,beta=model.beta,lower=model.lower,upper=model.upper)
            tm['seconds']=time.monotonic()-tick;store.put(key+f'/rho{rho}/E3',tm,ta);saved=tm,ta
        tm,ta=saved;mm=tm['model_metadata']
        model=TemporalModel(temporal_map,ta['gains'],ta['beta'],mm['cutoff'],mm['parent_cutoff'],mm['parent_fingerprint'],
                            mm['source_fingerprint'],mm['reference_fingerprint'],ta['lower'],ta['upper'])
        cc=dict(context,rho=rho,parent=model._content_hash(),state=book.identity,multipliers=list(MULTIPLIERS))
        saved=load_bundle(store,key+f'/rho{rho}/C4',cc)
        if saved is None:
            tick=time.monotonic();old=archive.conditional(c,rho,model);p=len(model.beta)
            if old is not None: score,curvature,origin=old
            else:
                def moments(item):
                    basis,innovations=item;m=basis.sample.market
                    states=book.bind(m.decision_date,history=13).validate(m.decision_date,history=13,expected_identity=book.identity)
                    return month_moments(basis,innovations,model,states,state_names=book.names,interaction_columns=tuple(range(1,p)))
                score,curvature=conditional_sums(inputs.batches(),moments,columns=2*p-1,workers=12,guard=guard)
                origin=None
            neutral,residual=_solve(curvature[:p,:p],score[:p],settings.ridge);finite_equal(neutral,model.beta)
            values=[solve_conditional(curvature,score,parent_width=p,parent_ridge=settings.ridge,multiplier=m) for m in MULTIPLIERS]
            coefficients=np.column_stack([v[0] for v in values])
            independent=verify_system(curvature,score,coefficients,p,settings.ridge,MULTIPLIERS)
            cm=dict(cc,seconds=time.monotonic()-tick,origin=origin,new_moments=old is None,
                    neutral_beta_error=float(np.max(abs(neutral-model.beta))),residuals=[v[1] for v in values],independent=independent)
            ca=dict(coefficients=coefficients,score=score,curvature=curvature)
            store.put(key+f'/rho{rho}/C4',cm,ca);saved=cm,ca
        cm,ca=saved;models[rho]=(model,ca['coefficients'])
        summaries.append(dict(rho=rho,temporal_new=tm['new_fit'],temporal_seconds=tm['seconds'],conditional_seconds=cm['seconds'],new_moments=cm['new_moments']))
        runtime.event('candidate_complete',year=year,rho=rho,temporal_reused=not tm['new_fit'],conditional_moments_reused=not cm['new_moments'])
    if check_archived_neutral:
        probe=SharedTemporalProfiles(inputs.batches,temporal_map,settings,maximum_bytes=128*2**20,
            guard=guard,objective_scale=gm['normalizer'],workers=12,trig_backend='accelerate',**dict(VARIANTS,reuse_bytes=128*2**20))
        neutral=probe.for_restraint(.1)(np.ones(13),with_gradient=True)
        finite_equal(neutral['beta'],ga['beta']);del probe
        runtime.event('archived_neutral_recomputed',year=year,maximum_beta_error=float(np.max(abs(neutral['beta']-ga['beta']))))
    history.files.unchanged();catalog.unchanged();guard()
    summary=dict(year=year,width=c['width'],geometry_new=gm['new_fit'],geometry_seconds=gm['seconds'],
        candidates=summaries,elapsed_seconds=time.monotonic()-started,training_months=len(inputs))
    runtime.event('year_fits_complete',**summary)
    return dict(models=models,summary=summary)
