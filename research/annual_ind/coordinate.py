"""Fresh annual E2/E3/C4 coordinate with immutable layer checkpoints.

This module chooses neither validation winners nor a historical launch budget.
It accepts only coordinates in the preserved author grid. Geometry always starts
at zero and temporal optimization at one; a checkpoint is a completed layer,
never an optimizer warm start from another year or candidate.
"""
from dataclasses import asdict,replace
import gc
import json
import time
import numpy as np
import pandas as pd
from feature_geometry.features import FeatureMap,fingerprint
from feature_geometry.history import sha
from feature_geometry.training import FitSettings,_solve
from geometry_study.features import CachedFeatureMap
from temporal_extension.features import TemporalFeatureMap
from temporal_extension.model import TemporalModel
from temporal_extension.history import RAW,RANKS
from temporal_numerics.optimizer import optimize,LBFGSBSettings
from temporal_fast_v4.settings import VARIANTS
from extension4.state import build_states
from extension4.estimation import month_moments
from extension4.storage import write_json
from annual_ind.inputs import YearInputs
from annual_ind.streaming_geometry import GeometryObjective
from annual_ind.temporal_reuse import SharedTemporalProfiles
from annual_ind.parallel_layers import conditional_sums
from annual_ind.conditional import solve_conditional
from annual_ind.pipeline_probe import plain


def load_bundle(store,key,expected):
    row=store.connection.execute('SELECT context FROM bundles WHERE key=?',(key,)).fetchone()
    if row is None:return None
    metadata=json.loads(row[0]);arrays=store.get(key,context=metadata)
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Checkpoint belongs to another annual coordinate/source')
    return metadata,arrays


def causal_state(root,history,cutoff):
    config=json.loads((root/'config/extension4_study_v1_proposed.json').read_text())
    filters=[('eom','<=',pd.Timestamp(cutoff))]
    raw=pd.read_parquet(history.files.path(RAW),columns=['id','eom','be_me','me'],filters=filters)
    eligible=pd.read_parquet(history.files.path(RANKS),columns=['id','eom','valid'],filters=filters)
    definition=config['state_definitions'][0]
    metadata=dict(unranked=True,timing_mode='SOURCE_LAG_PROXY_UNCERTIFIED',
        source_sha256=history.files.consumed[RAW],eligibility_sha256=history.files.consumed[RANKS],
        source_lag_months=definition['source_lag_months'],source_max_age_months=definition['source_max_age_months'],
        actual_accounting_age_observed=False,revision_vintages_available=False)
    return build_states(raw,eligible,definition=definition,normalizer=config['state_standardization'],
        source_metadata=metadata,first=str(raw.eom.min().date()),last=cutoff)


def fit_coordinate(root,out,history,cache,store,runtime,*,year,width,gi,ridge_index,
                   restraints,workers=8,reference=None):
    """Return verified compact models; failed candidates stop rather than disappear."""
    if width not in (64,128,256,512) or gi not in (0,1) or type(ridge_index) is not int or not 0<=ridge_index<=100:
        raise ValueError('Coordinate outside original candidate grid')
    if not restraints or tuple(restraints)!=tuple(x for x in (.1,.01,.001) if x in restraints):
        raise ValueError('Explicit ordered inherited temporal restraints required')
    started=time.monotonic();guard=runtime.guard
    inputs=YearInputs(history,cache,year,guard,runtime.event)
    grid=json.loads((root/'output/step2_20260910/source_settings.json').read_text())['settings']['pf_ml']
    ridge=float(grid['l_vec'][ridge_index])
    directions=history.files.path(f'output/feature_geometry_study_20260914/directions/seed1_g{gi}.bin')
    fmap=CachedFeatureMap(np.fromfile(directions,dtype='<f8').reshape(115,512)[:,:width//2],
                          'annual_ind_author_seed1_'+sha(directions))
    dictionary=pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'))
    if (dictionary.characteristic.tolist()!=list(history.features) or
            dictionary.cluster.nunique()!=13 or dictionary.cluster.isna().any()):
        raise ValueError('Exact accepted characteristic/theme dictionary required')
    groups=dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
    gp=json.loads((root/'config/feature_geometry_study_v1.json').read_text())['geometry_settings']
    settings=FitSettings(inputs.cutoff,ridge=ridge,scale_penalty=0.,
        **{k:v for k,v in gp.items() if k!='relative_scale_penalty'})
    context=dict(year=year,width=width,gi=gi,ridge_index=ridge_index,ridge=ridge,
        cutoff=inputs.cutoff,training_first=inputs.dates[0],training_last=inputs.dates[-1],
        training_months=len(inputs),source=history.source_fingerprint,
        reference=history.reference_fingerprint,frequency=fmap.fingerprint,
        geometry_settings=asdict(settings),relative_scale_penalty=gp['relative_scale_penalty'])
    key=f'{year}/p{width}/g{gi}/r{ridge_index}'
    stage=lambda label,complete=None,total=None:runtime.event('phase_progress',
        label=f'{year}: {label}',completed=complete,total=total,year=year)
    stage('learning characteristic scales',0,settings.max_steps)
    saved=load_bundle(store,key+'/E2',context)
    if saved is None:
        start=time.monotonic()
        objective=GeometryObjective(inputs.bases,len(inputs),fmap,settings,workers=workers,guard=guard)
        zero=objective.evaluate(np.zeros(115));normalizer=zero['value']
        if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('Nonpositive objective normalizer')
        oracle=history.files.path(f'data/prepared/step9_v1/baseline/coefficients/g{gi}/{year}_p{width}.npz')
        with np.load(oracle,allow_pickle=False) as f:expected=f['coefficients'][:,ridge_index]
        np.testing.assert_allclose(zero['beta'],expected,atol=1e-9,rtol=1e-8)
        runtime.event('neutral_baseline_passed',year=year,maximum_beta_error=float(np.max(abs(zero['beta']-expected))))
        settings=replace(settings,scale_penalty=gp['relative_scale_penalty']*normalizer)
        objective.settings=settings
        result=objective.fit(normalizer,lambda row:stage('learning characteristic scales',row['iteration'],settings.max_steps))
        if reference is not None:
            expected=json.loads((reference/'fresh_parent/fit.json').read_text())
            for k in ('theta','beta','trace','profile_evaluations','status','final_training_value'):
                if k=='trace':
                    if result[k]!=expected[k]:raise ValueError('Streaming geometry trace differs from accepted reference')
                else:np.testing.assert_array_equal(result[k],expected[k])
        metadata=dict(context,normalizer=normalizer,source_identity=inputs.identity,
            seconds=time.monotonic()-start,result=plain({k:v for k,v in result.items() if k not in ('theta','beta','geometry_beta')}),
            neutral_oracle_sha256=sha(oracle))
        arrays=dict(theta=result['theta'],beta=result['beta'],baseline_beta=zero['beta'])
        store.put(key+'/E2',metadata,arrays);saved=metadata,arrays
        runtime.event('layer_checkpoint',year=year,layer='E2',key=key+'/E2')
        del objective,result,zero;gc.collect()
    gm,ga=saved;normalizer=gm['normalizer']
    settings=replace(settings,scale_penalty=gp['relative_scale_penalty']*normalizer)
    temporal_map=TemporalFeatureMap(FeatureMap(fmap.frequencies,fmap.source),ga['theta'],groups)
    parent=fingerprint(ga['theta'],ga['beta'],temporal_map.fingerprint,settings.cutoff)
    optimizer=LBFGSBSettings(**json.loads((root/'config/extension3_study_v3_proposed.json').read_text())['optimizer'])
    variants=dict(VARIANTS,reuse_bytes=128*2**20)
    bank=SharedTemporalProfiles(inputs.batches,temporal_map,settings,maximum_bytes=128*2**20,
        guard=guard,objective_scale=normalizer,workers=workers,trig_backend='accelerate',**variants)
    book=None;models={};summaries=[]
    for rho in restraints:
        guard();stage(f'learning temporal gains, restraint {rho}')
        tc=dict(context,parent=parent,rho=rho,optimizer=asdict(optimizer),variants=variants)
        tkey=key+f'/rho{rho}/E3';saved=load_bundle(store,tkey,tc)
        if saved is None:
            start=time.monotonic();evaluate=bank.for_restraint(rho);neutral={}
            def checked(gains,with_gradient=True):
                payload=evaluate(gains,with_gradient=with_gradient)
                if not neutral:
                    np.testing.assert_array_equal(gains,np.ones(13))
                    np.testing.assert_allclose(payload['beta'],ga['beta'],atol=1e-11,rtol=1e-10)
                    neutral.update(passed=True,maximum_beta_error=float(np.max(abs(payload['beta']-ga['beta']))))
                if payload['training_months']!=len(inputs):raise ValueError('Temporal training count changed')
                return payload
            result=optimize(checked,13,optimizer,progress=lambda row:runtime.event(
                'temporal_optimizer',year=year,rho=rho,details=plain(row)))
            # Preserve the exact failed diagnostic even when the optimizer caught a guard.
            attempt=out/f'raw_{year}_{width}_{gi}_{ridge_index}_{rho}_{time.time_ns()}.json'
            write_json(attempt,plain(dict(result,neutral_gate=neutral)))
            guard()
            if not result['converged'] or not neutral.get('passed'):
                raise ValueError('Temporal convergence gate failed; raw diagnostics retained; no candidate removal')
            if reference is not None and rho==.1:
                expected=json.loads((reference/'temporal_attempt.json').read_text())
                for k in ('gains','beta','objective_trace','projected_gradient_trace'):
                    np.testing.assert_array_equal(result[k],expected[k])
            tm=dict(tc,seconds=time.monotonic()-start,neutral_gate=neutral,result=plain(
                {k:v for k,v in result.items() if k not in ('gains','beta','gradient')}))
            ta={k:result[k] for k in ('gains','beta','gradient')}
            store.put(tkey,tm,ta);saved=tm,ta
            runtime.event('layer_checkpoint',year=year,layer='E3',rho=rho,key=tkey)
        tm,ta=saved
        model=TemporalModel(temporal_map,ta['gains'],ta['beta'],settings.cutoff,settings.cutoff,parent,
            history.source_fingerprint,history.reference_fingerprint,
            np.broadcast_to(optimizer.lower,(13,)),np.broadcast_to(optimizer.upper,(13,)))
        stage(f'fitting conditional coefficients, restraint {rho}')
        if book is None:book=causal_state(root,history,inputs.cutoff)
        cc=dict(context,parent=model._content_hash(),state=book.identity,multipliers=[100.,10.,1.],rho=rho)
        ckey=key+f'/rho{rho}/C4';saved=load_bundle(store,ckey,cc)
        if saved is None:
            start=time.monotonic();p=len(model.beta)
            def moments(item):
                basis,innovation=item;m=basis.sample.market
                states=book.bind(m.decision_date,history=13).validate(m.decision_date,
                    history=13,expected_identity=book.identity)
                return month_moments(basis,innovation,model,states,
                    state_names=book.names,interaction_columns=tuple(range(1,p)))
            score,curvature=conditional_sums(inputs.batches(),moments,columns=2*p-1,
                                            workers=workers,guard=guard)
            neutral,residual=_solve(curvature[:p,:p],score[:p],settings.ridge)
            np.testing.assert_allclose(neutral,model.beta,atol=1e-11,rtol=1e-10)
            values=[solve_conditional(curvature,score,parent_width=p,parent_ridge=ridge,multiplier=x)
                    for x in (100.,10.,1.)]
            coefficients=np.column_stack([v[0] for v in values])
            if reference is not None and rho==.1:
                with np.load(reference/'fresh_conditional/moments_and_coefficients.npz',allow_pickle=False) as f:
                    for j,x in enumerate((100.,10.,1.)):np.testing.assert_array_equal(coefficients[:,j],f[str(x)])
            cm=dict(cc,seconds=time.monotonic()-start,residuals=[v[1] for v in values],
                neutral_beta_max_error=float(np.max(abs(neutral-model.beta))),neutral_residual=residual)
            ca=dict(coefficients=coefficients)
            store.put(ckey,cm,ca);saved=cm,ca
            runtime.event('layer_checkpoint',year=year,layer='C4',rho=rho,key=ckey)
        cm,ca=saved;models[rho]=(model,ca['coefficients'])
        summaries.append(dict(rho=rho,temporal_seconds=tm['seconds'],conditional_seconds=cm['seconds'],
            convergence=tm['result']['status'],evaluations=tm['result']['evaluation_count']))
    history.files.unchanged();guard()
    summary=dict(year=year,width=width,gi=gi,ridge_index=ridge_index,training_months=len(inputs),
        geometry_seconds=gm['seconds'],restraints=summaries,elapsed_seconds=time.monotonic()-started,
        cache=cache.diagnostics(),source_panels=history.panels.diagnostics(),
        neutral_gates_passed=True,old_models_used_for_initialization=False,
        exact_early_reference_passed=reference is not None)
    runtime.event('coordinate_complete',**summary)
    return dict(summary=summary,geometry=(gm,ga),feature_map=fmap,models=models)
