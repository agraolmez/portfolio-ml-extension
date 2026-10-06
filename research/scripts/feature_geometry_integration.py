"""Verify the completed 1981 model set without choosing settings or reporting performance."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from feature_geometry.history import sha
parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
import numpy as np,pandas as pd
from geometry_study.run import ROOT,OUT,locked,feature_map
from geometry_study.history import History
from feature_geometry.policy import GeometryModel
from baseline.core import transition
hashes={}
for p in (OUT/'fits/1981').glob('*/*/fit.json'):
 j=json.loads(p.read_text())
 for name,digest in j['files_sha256'].items():
  file=p.parent/name
  if sha(file)!=digest:raise ValueError('Changed fitted model '+str(file))
  hashes[str(file.relative_to(ROOT))]=digest
if len(hashes)!=33:raise ValueError('Complete first-year model set is required')
plan=locked()['plan'];source=History(ROOT,json.loads((OUT/'financial_preflight.json').read_text()))
base={d:g.sort_values('id').set_index('id') for d,g in pd.read_parquet(ROOT/'data/prepared/step9_v1/continuous_R1/PolicyDecision.parquet',filters=[('eom','>=',pd.Timestamp('1980-12-31')),('eom','<=',pd.Timestamp('1981-11-30'))]).groupby('eom')}
specs=[(s,m) for s in plan['frequency_seeds'] for m in plan['methods']+['fixed1024_double_ridge']];models={}
for seed,method in specs:
 fm='joint' if method=='joint_refit' else method;name='geometry_model' if method=='joint' else 'model'
 models[seed,method]=GeometryModel.load(OUT/f'fits/1981/seed{seed}/{fm}/{name}.npz')
h=None;max_weight_error=0.;max_accounting_error=0.;max_rule_error=0.;count=0
for d in pd.date_range('1980-12-31','1981-11-30',freq='ME'):
 x=source.market(d);ids=x.ids;nextd=d+pd.offsets.MonthEnd(1);nextids=source.months[nextd].query('valid').id.to_numpy()
 if h is None:h=np.repeat(base[d].loc[ids,'pretrade'].to_numpy()[:,None],len(specs),axis=1)
 out=source.outcomes[d].loc[ids];ret=source.months[d].loc[ids,'ret_ld1'].to_numpy();nh=np.empty((len(nextids),len(specs)))
 for j,(seed,method) in enumerate(specs):
  model=models[seed,method];aim=model.aim(x.characteristics[0],x.volatility[0],str(d.date()))
  result=transition(ids,h[:,j],aim,x.m,out.tr_ld1.to_numpy(),float(out.mu_ld1.iloc[0]),x.impact,x.wealth,ret,nextids)
  direct=h[:,j]+(np.eye(len(ids))-x.m)@(aim-h[:,j]);rule_error=float(max(abs(direct-result['weights'])));max_rule_error=max(max_rule_error,rule_error)
  gross=math.fsum(float(w)*float(r) for w,r in zip(direct,ret));cost=.5*x.wealth*math.fsum(float(l)*float(q)**2 for l,q in zip(x.impact,direct-h[:,j]))
  max_accounting_error=max(max_accounting_error,abs(gross-result['gross_excess']),abs(cost-result['impact_cost']))
  if seed==1 and method=='fixed512':max_weight_error=max(max_weight_error,float(max(abs(result['weights']-base[d].loc[ids,'weight'].to_numpy()))))
  drift=direct*(1+out.tr_ld1.to_numpy())/(1+float(out.mu_ld1.iloc[0]));manual=dict(zip(ids,drift));expected=np.array([manual.get(sid,0.) for sid in nextids]);np.testing.assert_allclose(expected,result['next_pretrade'],atol=1e-12,rtol=1e-10)
  nh[:,j]=result['next_pretrade'];count+=1
 h=nh
assert max_weight_error<1e-9 and max_accounting_error<1e-12 and max_rule_error<1e-12
for path,digest in hashes.items():
 if sha(ROOT/path)!=digest:raise ValueError('Model changed during integration check')
record=dict(source_model_sha256=hashes,status='PASS',scope='First development return year integration only; no performance aggregation, tuning, selection or inference',return_year=1981,monthly_decisions=12,paths=24,path_month_checks=count,original_path_max_weight_error=max_weight_error,independent_accounting_max_error=max_accounting_error,original_rule_max_error=max_rule_error,own_holdings_and_entry_exit_alignment_checked=True,all_source_model_files_unchanged=True,complete_historical_study=False)
if args.receipt:
 with args.receipt.open('x') as handle:json.dump(record,handle,indent=2);handle.write('\n')
print(json.dumps({k:v for k,v in record.items() if k!='source_model_sha256'}),flush=True)
