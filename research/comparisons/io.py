"""Immutable source/stage records for the complete Step 9 study."""
import json
from datetime import datetime,timezone
import pandas as pd
from baseline.io import ROOT,DATA,PROTOCOL,SETTINGS,sha,save_json
from baseline.full import PROTECTED,preflight as financial_preflight
OUT=ROOT/'output/step9_20260913';PREP=ROOT/'data/prepared/step9_v1'
BASE=PREP/'baseline';CONF=PREP/'confidence';LEARNER=PREP/'learner';RUN=PREP/'continuous_R1'
OLD_BASE=ROOT/'data/prepared/step6_v1';OLD_CONF=ROOT/'data/prepared/step7_v1';OLD_LEARNER=ROOT/'data/prepared/step8_v1'
DATES=pd.date_range('1980-12-31','2025-11-30',freq='ME');ADDED=DATES[480:]
PERIODS={'selection':('2000-12-31','2010-11-30'),'evaluation':('2010-12-31','2020-11-30'),'robustness':('2020-12-31','2025-11-30')}


def initialize():
    if (OUT/'start.json').exists():raise ValueError('Step 9 already initialized')
    receipt,_=financial_preflight();accepted={}
    for stage in [6,7,8]:
        date={6:'20260911',7:'20260912',8:'20260913'}[stage];path=ROOT/f'output/step{stage}_{date}/acceptance_manifest.json';item=json.loads(path.read_text())
        if not item[f'step{stage}_complete'] or not all(g['passed'] for g in item['gates']):raise ValueError('Prior stage is not accepted')
        for p,h in item['artifact_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed accepted source '+p)
        accepted[str(path.relative_to(ROOT))]=sha(path)
    sources={ROOT/p for p in accepted}
    sources.update([DATA/'monthly_inputs.parquet',DATA/'monthly_rates.parquet',ROOT/'output/step2_20260910/source_settings.json'])
    for folder in [OLD_BASE,OLD_CONF,OLD_LEARNER]:sources.update(p for p in folder.rglob('*') if p.is_file() and p.name!='README.md')
    sources.update(p for p in (ROOT/'data/prepared/step4_v1/risk').glob('*') if p.is_file())
    sources.update(p for p in (ROOT/'baseline').glob('*.py'));sources.update(p for p in (ROOT/'learner').glob('*.py'));sources.update(p for p in (ROOT/'confidence').glob('*.py'))
    sources.update(ROOT/f'output/step6_20260911/reference/rff_g{g}.bin' for g in [0,1])
    sources.add(ROOT/'output/step6_baseline_1981_2020/PolicyDecision.parquet');sources.add(ROOT/'output/step6_baseline_1981_2020/monthly_performance.parquet')
    OUT.mkdir(parents=True,exist_ok=True);PREP.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'start.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),authorization='User: ok. let us finish step 9',protocol_id=PROTOCOL['protocol_id'],
        protected={p:sha(ROOT/p) for p in PROTECTED},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},prior_acceptances=accepted,
        financial_preflight=receipt,plan_sha256=sha(ROOT/'docs/STEP_9_PLAN.md'),periods=PERIODS,performance_inference_run=False))
    print('Step 9 source lock and approved full-period plan verified.',flush=True)


def preflight():
    record=json.loads((OUT/'start.json').read_text())
    for section in ['protected','source_sha256']:
        for p,h in record[section].items():
            if sha(ROOT/p)!=h:raise ValueError('Changed locked source '+p)
    if sha(ROOT/'docs/STEP_9_PLAN.md')!=record['plan_sha256']:raise ValueError('Execution plan changed after source lock')
    return record


def code_paths(stage):
    common=['comparisons/__init__.py','comparisons/io.py','docs/STEP_9_PLAN.md']
    if stage in ['moments','base_fits','confidence']:common+=['comparisons/extension.py']
    elif stage in ['states','labels','fits']:common+=['comparisons/extended_learner.py']
    elif stage in ['selection','evaluation','robustness']:common+=['comparisons/core.py','comparisons/markets.py','comparisons/pipeline.py']
    return common


def begin(stage):
    if (OUT/'acceptance_manifest.json').exists() or (OUT/f'{stage}_manifest.json').exists():raise ValueError('Accepted/completed stage is immutable')
    preflight()
    if (OUT/f'{stage}_code.json').exists():raise ValueError('Partial stage exists; diagnose without silent overwrite')
    save_json(OUT/f'{stage}_code.json',{p:sha(ROOT/p) for p in code_paths(stage)})


def finish(stage,files,**details):
    for p,h in json.loads((OUT/f'{stage}_code.json').read_text()).items():
        if sha(ROOT/p)!=h:raise ValueError('Code changed during stage')
    save_json(OUT/f'{stage}_manifest.json',dict(status='COMPLETE',start_sha256=sha(OUT/'start.json'),
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file() and not p.is_symlink()},**details))


def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.resolve()!=source.resolve():raise ValueError('Existing path is not the accepted dependency')
    else:target.symlink_to(source)
