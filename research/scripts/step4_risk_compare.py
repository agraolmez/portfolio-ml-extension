"""Full-history comparison against independently executed R risk construction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.io import *
p=np.load(PREP/'risk/factor_covariances.npz')['covariances']
r=np.fromfile(OUT/'reference/r_factor_covariances.bin',dtype='<f8').reshape(p.shape)
a=pd.read_parquet(PREP/'risk/specific_risk_with_fallbacks.parquet')
b=pd.read_csv(OUT/'reference/r_specific_risk.csv',parse_dates=['eom'],float_precision='round_trip')
m=a.merge(b,on=['id','eom'],validate='one_to_one')
res={'factor_covariance_values':int(p.size),'factor_covariance_max_abs':float(abs(p-r).max()),'factor_covariance_match':bool(np.allclose(p,r,atol=1e-12,rtol=1e-10)),'specific_rows':len(m),'specific_risk_max_abs':float(abs(m.specific_variance_monthly-m.specific_variance).max()),'specific_risk_match':bool(len(m)==len(a)==len(b) and np.allclose(m.specific_variance_monthly,m.specific_variance,atol=1e-12,rtol=1e-10)),'source_missing_match':bool((m.specific_missing==m.source_missing).all())}
save_json(OUT/'risk_parity.json',res)
assert res['factor_covariance_match'] and res['specific_risk_match'] and res['source_missing_match']
print('Full-history risk parity passed.')
