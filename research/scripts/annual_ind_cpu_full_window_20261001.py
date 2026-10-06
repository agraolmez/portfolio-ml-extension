"""One fixed neutral evaluation, within the remaining 10-minute CPU diagnostic.

This measures the production input path; it does not train or select a model.
The completed pilot scale checkpoint and old neutral evaluation are read-only.
"""
from scripts.annual_ind_cpu_diagnostic_20261001 import *
from annual_ind.cached_inputs import ReusedYearInputs
from annual_ind.temporal_reuse import SharedTemporalProfiles
from annual_ind.pipeline_probe import plain
from extension4.runtime import prior_seconds


def main():
    mutex=(ROOT/'output/.annual_ind_probe.lock').open('a')
    fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=OUT/'full_window'
    if out.exists():raise FileExistsError('Preserve all earlier measurements')
    previous=prior_seconds(OUT)[0]
    lock=json.loads((OUT/'lock.json').read_text())
    for block in ('code_sha256','protected_sha256'):
        for name,digest in lock[block].items():
            if sha(ROOT/name)!=digest:raise ValueError('Locked source changed: '+name)
    limits=dict(lock['budget'],total_active_seconds=600.-previous,pilot_active_seconds=600.-previous)
    out.mkdir()
    write_json(out/'lock.json',dict(lock,budget=limits,
        diagnostic_seconds_previously_used=previous,
        purpose=__doc__,workers=8,
        code_sha256={**lock['code_sha256'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}))
    runtime=None;error=None;status='STOPPED'
    try:
        runtime=ProbeRuntime(out,{'compute_budget':limits},'diagnostic')
        runtime.event('phase_progress',label='Measuring the faster path across all 649 training months',completed=0,total=649)
        receipt,_=preflight();write_json(out/'financial_preflight.json',receipt)
        history=StudyHistory(ROOT,receipt,weights=[1/6]*6)
        history.files.inventory.update(json.loads((ROOT/'output/step6_20260911/preparation_manifest.json').read_text())['artifact_sha256'])
        history.files=PreparationFiles(history.files)
        history.panels=CalendarSourcePanels(history.files,history.panels.columns,256*2**20)
        old=ROOT/'output/annual_ind_runner_pilot_v3_20260930'
        oldlock=json.loads((old/'lock.json').read_text())
        disk=open_parent_store(ROOT,history,oldlock['input_cache_study_lock_sha256'])
        store=CheckpointStore.__new__(CheckpointStore)
        store.path=old/'models.sqlite';store.run_identity=sha(old/'lock.json')
        store.connection=sqlite3.connect(store.path.resolve().as_uri()+'?mode=ro',uri=True)
        row=next(x for x in store.inventory() if x['key']=='2020/p512/g0/r50/E2')
        arrays=store.get(row['key']);store.close();context=row['context']
        freq=np.fromfile(history.files.path('output/feature_geometry_study_20260914/directions/seed1_g0.bin'),dtype='<f8').reshape(115,512)[:,:256]
        dictionary=pd.read_csv(history.files.path('config/feature_dictionary_v1.csv'))
        groups=dictionary.cluster.map({v:i for i,v in enumerate(sorted(dictionary.cluster.unique()))}).to_numpy(int)
        fmap=TemporalFeatureMap(FeatureMap(freq,'cpu_full_window_20261001'),arrays['theta'],groups)
        settings=FitSettings(**dict(context['geometry_settings'],scale_penalty=.01*context['normalizer']))
        operators=OperatorCache(maximum_bytes=5*2**30,guard=runtime.guard)
        inputs_cache=PreparationCheckedInputCache(4*2**30,runtime.guard,disk)
        inputs=ReusedYearInputs(history,operators,2020,runtime.guard,runtime.event,input_cache=inputs_cache)
        if len(inputs)!=649:raise ValueError('Historical input count changed')
        with threadpool_limits(limits=1),single_thread_blas():
            bank=SharedTemporalProfiles(inputs.batches,fmap,settings,maximum_bytes=128*2**20,
                guard=runtime.guard,objective_scale=context['normalizer'],workers=8,
                trig_backend='accelerate',**dict(VARIANTS,reuse_bytes=128*2**20))
            evaluate=bank.for_restraint(.1)
            start=time.monotonic();cpu=time.process_time()
            result=evaluate(np.ones(13));elapsed=time.monotonic()-start;cpu=time.process_time()-cpu
        prior_path=old/'raw_2020_512_0_50_0.1_1790823670615698000.json'
        prior=json.loads(prior_path.read_text())
        for name in ('beta','gradient','value'):np.testing.assert_array_equal(result[name],prior[name])
        np.testing.assert_array_equal(result['beta'],arrays['beta'])
        if result['training_months']!=649:raise ValueError('Temporal profile omitted a month')
        history.files.unchanged();runtime.guard()
        write_json(out/'neutral_profile.json',plain(result))
        write_json(out/'result.json',dict(status='COMPLETE_FULL_WINDOW_TIMING_CHECK',
            seconds=elapsed,cpu_seconds=cpu,average_cpu_cores=cpu/elapsed,
            months=649,passes=inputs.passes,exact_beta_gradient_value_match=True,
            previous_neutral_sha256=sha(prior_path),previous_neutral_seconds=927.9577694999753,
            historical_speed_ratio=927.9577694999753/elapsed,
            comparison_limitation='Historical timing, not contemporaneous paired runs; new run starts with empty in-memory caches.',
            input_cache=inputs_cache.diagnostics(),model_updates=0,
            full_pilot_restart=False,full_study_started=False,
            consumed_inputs_sha256=history.files.consumed))
        status='COMPLETE';runtime.event('cpu_full_window_complete',seconds=elapsed,exact_match=True)
    except BaseException as exc:
        error=exc;write_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if runtime is not None:runtime.close(status,error)


if __name__=='__main__':main()
